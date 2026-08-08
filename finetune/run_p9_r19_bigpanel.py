"""Phase 9 R19 — Expanded 30-stock panel: r5 generalization stress test.

18 rounds of loss/gate/data tweaks failed to beat r5. Before investing in
costly training-side changes (CSI300 feature injection needs 5-file rewrite),
stress-test whether r5's 9-stock panel alpha (beat 8/9 B&H) generalizes to a
broader 30-stock panel. If it holds, r5 is a real production config. If it
collapses, the 9-stock alpha was luck and no amount of further tuning matters.

Panel: 30 stocks spanning SSE main board, SZSE main board, ChiNext, STAR.
Gates: base / csi300_fade / h3_agree / h3_selfade0.02 (R8 Pareto best).
Zero training cost; prediction runs once.

Output: outputs/eval_p9_panel_r19_bigpanel.json
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
MODEL = "r5_frozen_pool48"
IDX_N = 5
# 30-stock panel: original 9 + 21 new, spanning boards.
PANEL = [
    "000001", "000002", "000063", "000333", "000651", "000858", "000938",
    "002001", "002007", "002415", "002466", "002594", "002714",
    "300003", "300015", "300059",
    "600036", "600276", "600519", "600887", "601318", "601398",
    "603259", "603288",
    "688169", "688185", "688235",
    "000568", "002129", "002230",
]


def load_index_returns():
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return np.log(close / close.shift(IDX_N))


def get_index_ret(series, date):
    try:
        loc = series.index.get_indexer([date], method="pad")[0]
    except (KeyError, IndexError):
        return float("nan")
    if loc < 0:
        return float("nan")
    val = series.iloc[loc]
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


def gate_hX_agree(hard_h1, score, hard_hX, thr):
    gated = gate_conf(hard_h1, score, thr)
    up_mask = (gated == UP_CLASS)
    disagree = up_mask & (hard_hX != UP_CLASS)
    gated[disagree] = FLAT_CLASS
    return gated


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
            "n_symbols": len(store),
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
            "per_symbol": [{"symbol": s["symbol"], **b} for s, b in zip(store, bt)]}


def main():
    t0 = time.time()
    idx_series = load_index_returns()
    lookbacks = (124, 126, 128, 130, 132, 134)

    md = model_dir(MODEL)
    print(f"Loading {MODEL}...", flush=True)
    loaded = load_model(md)

    store = []
    skipped = []
    print(f"=== Predicting 30-stock panel ({MODEL}) ===", flush=True)
    for sym in PANEL:
        try:
            df = load_csv(sym)
        except Exception as e:
            skipped.append((sym, str(e)))
            continue
        windows = make_test_windows(df, lookbacks)
        if not windows:
            skipped.append((sym, "no test windows"))
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
            idx_rets.append(get_index_ret(idx_series, w["context_end_date"]))
            pred = predict_all_horizons(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)

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
            "trh1": trh1, "tdh1": tdh1, "idx_arr": np.asarray(idx_rets),
        })
        print(f"  {sym}: bh={bh:.3f} n={len(windows)}", flush=True)

    if skipped:
        print(f"\nSkipped {len(skipped)} symbols: {skipped}", flush=True)

    bh = np.array([s["bh"] for s in store])
    n_sym = len(store)

    configs = [
        ("base", lambda s: gate_conf(s["hard"][0], s["score"], 0.45)),
        ("csi300_fade", lambda s: apply_fade(
            gate_conf(s["hard"][0], s["score"], 0.45), s["idx_arr"])),
        ("h3_agree", lambda s: gate_hX_agree(s["hard"][0], s["score"], s["hard"][1], 0.45)),
        ("h3_selfade0.02", lambda s: apply_sel_fade(
            gate_hX_agree(s["hard"][0], s["score"], s["hard"][1], 0.45),
            s["idx_arr"], 0.02)),
    ]

    results = {}
    print(f"\n=== {MODEL} 30-stock panel configs ===", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        print(f"  {name:<16} ret={r['ret_mean']:.3f} beat={r['beat_bh']}/{n_sym} "
              f"prec={r['prec_mean']:.2%} cov={r['cov_mean']:.2%}", flush=True)

    # Regime breakdown: up vs down symbols (bh > 0 vs bh <= 0).
    up_syms = [s for s in store if s["bh"] > 0]
    down_syms = [s for s in store if s["bh"] <= 0]
    print(f"\n=== Regime split: {len(up_syms)} up-symbols, {len(down_syms)} down-symbols ===", flush=True)
    bh_up = np.array([s["bh"] for s in up_syms])
    bh_dn = np.array([s["bh"] for s in down_syms])
    for name, gate_fn in configs:
        bt_up = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in up_syms]
        bt_dn = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in down_syms]
        ret_up = np.mean([b["ret"] for b in bt_up])
        ret_dn = np.mean([b["ret"] for b in bt_dn])
        beat_up = int(sum(b["ret"] > b_b for b, b_b in zip(bt_up, bh_up)))
        beat_dn = int(sum(b["ret"] > b_b for b, b_b in zip(bt_dn, bh_dn)))
        print(f"  {name:<16} up: ret={ret_up:.3f} beat={beat_up}/{len(up_syms)} | "
              f"down: ret={ret_dn:.3f} beat={beat_dn}/{len(down_syms)}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r19_bigpanel.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
