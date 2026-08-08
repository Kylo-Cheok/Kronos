"""Phase 9 R10 — r5+r17 direction logits ensemble.

R9 showed r5 (CE, high prec/low cov) and r17 (focal, low prec/high cov)
have complementary characteristics. This script averages their TTA-averaged
direction logits and evaluates with the same gate configs.

If the ensemble outperforms r5 alone, the two models capture different
patterns and averaging improves robustness.

Output: outputs/eval_p9_panel_r10_ensemble.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import (  # noqa: E402
    FEATURES, LOOKBACK, PREDICT_WINDOW, WINDOW, load_csv,
)
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p4_r7_expanded_tta import (  # noqa: E402
    average_logits, derive_time_features, normalize_with_lookback,
)
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_tta_eval import CLIP, DEVICE, HORIZONS, PREDICT_WINDOW_LEN  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
GATE_MAG = 0.002
MODELS = ["r5_frozen_pool48", "r17_frozen_focal_g2"]
IDX_N = 5
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]

R5_BASELINES = {
    "base": {"ret": 0.065, "beat": 8, "prec": 0.3552, "cov": 0.41},
    "h3_agree_thr0.45": {"ret": 0.074, "beat": 8, "prec": 0.3896, "cov": 0.3320},
    "h5_agree_thr0.45": {"ret": 0.086, "beat": 7, "prec": 0.3723, "cov": 0.3475},
    "h3_selfade0.02": {"ret": 0.078, "beat": 7, "prec": 0.4146, "cov": 0.2771},
    "fade": {"ret": 0.068, "beat": 7, "prec": 0.4210, "cov": 0.2059},
}


def load_index_returns():
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return {n: np.log(close / close.shift(n)) for n in (5, 10, 20)}, close


def get_index_ret(idx_returns, date):
    try:
        loc = idx_returns.index.get_indexer([date], method="pad")[0]
    except (KeyError, IndexError):
        return float("nan")
    if loc < 0:
        return float("nan")
    val = idx_returns.iloc[loc]
    return float(val) if pd.notna(val) else float("nan")


def make_test_windows(df, lookbacks):
    lo = pd.Timestamp(TEST_LO)
    max_lb = max(lookbacks)
    needed = max_lb + PREDICT_WINDOW + 1
    out = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        ced = df["timestamps"].iloc[start + LOOKBACK - 1]
        if not (ced > lo):
            continue
        extra = max_lb - LOOKBACK
        if start - extra < 0:
            continue
        big_start = start - extra
        window = df.iloc[big_start: big_start + needed].copy()
        out.append({
            "start": start,
            "context_end_date": ced,
            "features": window[FEATURES].to_numpy(dtype=np.float32),
            "raw_close": window["close"].to_numpy(dtype=np.float32),
            "timestamps": window["timestamps"],
        })
    return out


@torch.no_grad()
def predict_all_horizons(loaded, w, lookbacks):
    max_lb = max(lookbacks)
    per_lb_logits = []
    target_dir = target_ret = None
    for lb in lookbacks:
        drop = max_lb - lb
        x_full = w["features"][drop: drop + lb + PREDICT_WINDOW_LEN + 1]
        close_full = w["raw_close"][drop: drop + lb + PREDICT_WINDOW_LEN + 1]
        ts_full = w["timestamps"].iloc[drop: drop + lb + PREDICT_WINDOW_LEN + 1]
        x_norm = normalize_with_lookback(x_full, lb)
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
        stamp = derive_time_features(ts_full)
        stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
        raw_close_tensor = torch.from_numpy(close_full).unsqueeze(0).to(DEVICE)
        token_seq_0, token_seq_1 = loaded["tokenizer"].encode(x_tensor, half=True)
        _, _, hidden_states = loaded["model"](
            token_seq_0[:, :-1], token_seq_1[:, :-1],
            stamp_tensor[:, :-1, :], return_context=True,
        )
        outputs = loaded["head"](hidden_states, context_length=lb)
        logits = outputs["direction_logits"][0].cpu().numpy()
        per_lb_logits.append(logits.astype(np.float64))
        if target_dir is None:
            targets = make_multihorizon_targets(
                raw_close_tensor, context_length=lb,
                horizons=loaded["horizons"],
                min_deadzone=loaded["min_deadzone"],
                volatility_multiplier=loaded["vol_mult"],
            )
            target_dir = targets["direction"][0].cpu().numpy()
            target_ret = targets["returns"][0].cpu().numpy()
    return {"per_lb_logits": per_lb_logits,
            "target_direction": target_dir,
            "target_return": target_ret}


def gate_base(hard, score, trh1, thr, mag):
    gated = hard.copy()
    keep = (score >= thr) & (np.abs(trh1) >= mag)
    gated[~keep] = FLAT_CLASS
    return gated


def gate_hX_agree(hard_h1, score_h1, trh1, hard_hX, thr, mag):
    gated = gate_base(hard_h1, score_h1, trh1, thr, mag)
    up_mask = (gated == UP_CLASS)
    disagree = up_mask & (hard_hX != UP_CLASS)
    gated[disagree] = FLAT_CLASS
    return gated


def apply_fade(g, idx_arr):
    g = g.copy()
    valid = ~np.isnan(idx_arr)
    g[(g == UP_CLASS) & (idx_arr >= 0)] = FLAT_CLASS
    g[(g == UP_CLASS) & ~valid] = FLAT_CLASS
    return g


def apply_sel_fade(g, idx_arr, pos_thr):
    g = g.copy()
    valid = ~np.isnan(idx_arr)
    g[(g == UP_CLASS) & (idx_arr > pos_thr)] = FLAT_CLASS
    g[(g == UP_CLASS) & ~valid] = FLAT_CLASS
    return g


def backtest(g, trh1, tdh1):
    pos = (g == UP_CLASS).astype(np.float32)
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    ret = float(np.sum(pos * trh1) - turnover * COST)
    gm = gated_actionable_metrics(g, tdh1)
    return {"ret": ret, "n_calls": int(gm["n_calls"]),
            "prec": gm["precision_on_calls"],
            "cov": gm["coverage"], "hit": gm["gated_nonflat_acc"]}


def eval_config(store, bh, gate_fn):
    bt = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in store]
    ret = np.array([b["ret"] for b in bt])
    prec = np.array([b["prec"] for b in bt])
    cov = np.array([b["cov"] for b in bt])
    return {"ret_mean": float(ret.mean()), "beat_bh": int((ret > bh).sum()),
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
            "per_symbol": [{"symbol": s["symbol"], **b} for s, b in zip(store, bt)]}


def main():
    t0 = time.time()
    idx_returns_dict, _ = load_index_returns()
    idx_ret_series = idx_returns_dict[IDX_N]
    lookbacks = (124, 126, 128, 130, 132, 134)

    # Load both models
    loaded = {}
    for name in MODELS:
        print(f"Loading {name}...", flush=True)
        loaded[name] = load_model(model_dir(name))

    store = []
    print(f"\n=== Predicting panel (ensemble {MODELS}, idx_n={IDX_N}) ===", flush=True)
    for sym in PANEL:
        df = load_csv(sym)
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, idx_rets = [], [], []
        # per_lb_logits[model_name][lb_idx] = list of [H, 3]
        per_lb_logits = {n: [[] for _ in lookbacks] for n in MODELS}
        for w in windows:
            drop = max(lookbacks) - REF_CTX
            close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
            tgt = make_multihorizon_targets(
                torch.from_numpy(close_ref).unsqueeze(0),
                context_length=REF_CTX, horizons=HORIZONS,
                min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
            td_all.append(tgt["direction"][0].cpu().numpy())
            tr_all.append(tgt["returns"][0].cpu().numpy())
            idx_rets.append(get_index_ret(idx_ret_series, w["context_end_date"]))
            for n in MODELS:
                pred = predict_all_horizons(loaded[n], w, lookbacks)
                for lb_i in range(len(lookbacks)):
                    per_lb_logits[n][lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        idx_arr = np.asarray(idx_rets)

        # Per-model TTA-averaged logits, then ensemble average
        confs = {}
        for h_idx in range(4):
            model_avg = []
            for n in MODELS:
                avg_lg = average_logits(
                    [np.asarray(per_lb_logits[n][i])[:, h_idx, :] for i in range(len(lookbacks))], None)
                model_avg.append(avg_lg)
            ens_lg = np.mean(model_avg, axis=0)  # ensemble average
            confs[h_idx] = direction_confidence_from_logits(ens_lg)

        tdh1 = td[:, 0]
        trh1 = tr[:, 0]
        bh = float(np.sum(trh1))
        store.append({
            "symbol": sym, "n": len(windows), "bh": bh,
            "hard": {h: confs[h]["hard_pred"] for h in range(4)},
            "score": confs[0]["actionable_score"],
            "trh1": trh1, "tdh1": tdh1, "idx_arr": idx_arr,
        })
        print(f"  {sym}: bh={bh:.3f}", flush=True)

    bh = np.array([s["bh"] for s in store])
    results = {}

    configs = [
        ("base", lambda s: gate_base(s["hard"][0], s["score"], s["trh1"], 0.45, GATE_MAG)),
        ("fade", lambda s: apply_fade(gate_base(s["hard"][0], s["score"], s["trh1"], 0.45, GATE_MAG), s["idx_arr"])),
        ("h3_agree_thr0.45", lambda s: gate_hX_agree(s["hard"][0], s["score"], s["trh1"], s["hard"][1], 0.45, GATE_MAG)),
        ("h5_agree_thr0.45", lambda s: gate_hX_agree(s["hard"][0], s["score"], s["trh1"], s["hard"][2], 0.45, GATE_MAG)),
        ("h3_selfade0.02", lambda s: apply_sel_fade(
            gate_hX_agree(s["hard"][0], s["score"], s["trh1"], s["hard"][1], 0.45, GATE_MAG),
            s["idx_arr"], 0.02)),
    ]

    print(f"\n=== Config comparison (ensemble vs r5 CE) ===", flush=True)
    print(f"  {'config':<22} {'ens ret':>8} {'r5 ret':>8} {'ens beat':>8} {'r5 beat':>8} {'ens prec':>9} {'r5 prec':>9}", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        r5 = R5_BASELINES.get(name, {})
        print(f"  {name:<22} {r['ret_mean']:8.3f} {r5.get('ret', 0):8.3f} "
              f"{r['beat_bh']:>4}/9   {r5.get('beat', 0):>4}/9   "
              f"{r['prec_mean']:8.2%} {r5.get('prec', 0):8.2%}", flush=True)

    # Per-symbol for best ret config
    best_ret = max(results.values(), key=lambda x: x["ret_mean"])
    best_name = [k for k, v in results.items() if v is best_ret][0]
    print(f"\n=== Best ret: {best_name} ===", flush=True)
    for ps in best_ret["per_symbol"]:
        print(f"  {ps['symbol']}: ret={ps['ret']:.3f} prec={ps['prec']:.2%} "
              f"cov={ps['cov']:.2%} n_calls={ps['n_calls']}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r10_ensemble.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
