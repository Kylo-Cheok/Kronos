"""Phase 9 R13 — r5 confidence threshold sweep (leak-free, inference-side).

R12 confirmed r5's hard-label CE is the loss sweet spot. Before more training
experiments, R13 checks whether the fixed conf threshold 0.45 is optimal for
r5. Sweeps 0.35-0.55 across base/h3_agree/h5_agree/fade gates.

Output: outputs/eval_p9_panel_r13_confsweep.json
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
from run_tta_eval import DEVICE, HORIZONS, PREDICT_WINDOW_LEN  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
MODEL_NAME = "r5_frozen_pool48"
IDX_N = 5
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]
CONF_SWEEP = [0.35, 0.40, 0.45, 0.50, 0.55]


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


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def gate_hX_agree(hard_h1, score, hard_hX, thr):
    gated = gate_conf(hard_h1, score, thr)
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


def backtest(g, trh1, tdh1):
    pos = (g == UP_CLASS).astype(np.float32)
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    ret = float(np.sum(pos * trh1) - turnover * COST)
    gm = gated_actionable_metrics(g, tdh1)
    return {"ret": ret, "n_calls": int(gm["n_calls"]),
            "prec": gm["precision_on_calls"],
            "cov": gm["coverage"]}


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

    print(f"Loading {MODEL_NAME}...", flush=True)
    loaded = load_model(model_dir(MODEL_NAME))

    store = []
    print(f"\n=== Predicting panel ({MODEL_NAME}) ===", flush=True)
    for sym in PANEL:
        df = load_csv(sym)
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, idx_rets = [], [], []
        per_lb_logits = [[] for _ in lookbacks]
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
            pred = predict_all_horizons(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        idx_arr = np.asarray(idx_rets)

        confs = {}
        for h_idx in range(4):
            avg_lg = average_logits(
                [np.asarray(per_lb_logits[i])[:, h_idx, :] for i in range(len(lookbacks))], None)
            confs[h_idx] = direction_confidence_from_logits(avg_lg)

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

    print(f"\n=== Conf sweep (r5 leak-free) ===", flush=True)
    print(f"  {'gate/thr':<20} {'ret':>7} {'beat':>6} {'prec':>7} {'cov':>7}", flush=True)
    for thr in CONF_SWEEP:
        configs = [
            (f"base_{thr}", lambda s, t=thr: gate_conf(s["hard"][0], s["score"], t)),
            (f"h3_agree_{thr}", lambda s, t=thr: gate_hX_agree(s["hard"][0], s["score"], s["hard"][1], t)),
            (f"h5_agree_{thr}", lambda s, t=thr: gate_hX_agree(s["hard"][0], s["score"], s["hard"][2], t)),
            (f"fade_{thr}", lambda s, t=thr: apply_fade(gate_conf(s["hard"][0], s["score"], t), s["idx_arr"])),
        ]
        for name, gate_fn in configs:
            r = eval_config(store, bh, gate_fn)
            results[name] = r
            print(f"  {name:<20} {r['ret_mean']:7.3f} {r['beat_bh']:>3}/9 {r['prec_mean']:6.2%} {r['cov_mean']:6.2%}", flush=True)

    # Find best by ret and by beat
    best_ret = max(results.values(), key=lambda x: x["ret_mean"])
    best_ret_name = [k for k, v in results.items() if v is best_ret][0]
    best_beat = max(results.values(), key=lambda x: (x["beat_bh"], x["ret_mean"]))
    best_beat_name = [k for k, v in results.items() if v is best_beat][0]
    print(f"\n=== Best ret: {best_ret_name} (ret={best_ret['ret_mean']:.3f}, beat={best_ret['beat_bh']}/9) ===", flush=True)
    print(f"=== Best beat: {best_beat_name} (beat={best_beat['beat_bh']}/9, ret={best_beat['ret_mean']:.3f}) ===", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r13_confsweep.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
