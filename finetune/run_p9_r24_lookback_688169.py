"""Phase 9 R24 — cond thr on 688169 + CSI300 lookback sweep.

R21 cond_0.40_0.50 is panel-optimal. R24 verifies it on 688169 (original
benchmark) and sweeps CSI300 lookback (3d/5d/10d/20d) to see if regime
signal timing matters.

Hypothesis: 5d may not be optimal. Shorter (3d) = more responsive but
noisy; longer (10d/20d) = smoother but lagged.

Output: outputs/eval_p9_panel_r24_lookback.json
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

from evaluate_gated_ensemble import load_csv  # noqa: E402
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p4_r7_expanded_tta import average_logits  # noqa: E402
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p9_r19_bigpanel import (  # noqa: E402
    PANEL, REF_CTX, REF_DZ, REF_VOL, COST,
    make_test_windows, predict_all_horizons,
)
from run_tta_eval import DEVICE, HORIZONS  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)


def load_index_returns(n: int):
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return np.log(close / close.shift(n))


def get_index_ret(series, date):
    try:
        loc = series.index.get_indexer([date], method="pad")[0]
    except (KeyError, IndexError):
        return float("nan")
    if loc < 0:
        return float("nan")
    val = series.iloc[loc]
    return float(val) if pd.notna(val) else float("nan")


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def gate_cond_thr(hard, score, idx_arr, thr_lo, thr_hi):
    g = hard.copy()
    dn = (idx_arr < 0) & ~np.isnan(idx_arr)
    up = (idx_arr >= 0) & ~np.isnan(idx_arr)
    nan = np.isnan(idx_arr)
    g[score < thr_lo] = FLAT_CLASS
    g[up & (score < thr_hi)] = FLAT_CLASS
    g[nan & (score < thr_hi)] = FLAT_CLASS
    return g


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
            "prec": gm["precision_on_calls"], "cov": gm["coverage"]}


def eval_config(store, bh, gate_fn):
    bt = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in store]
    ret = np.array([b["ret"] for b in bt])
    prec = np.array([b["prec"] for b in bt])
    cov = np.array([b["cov"] for b in bt])
    return {"ret_mean": float(ret.mean()), "beat_bh": int((ret > bh).sum()),
            "n_symbols": len(store),
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
            "total_ret": float(ret.sum()),
            "per_symbol": [{"symbol": s["symbol"], **b} for s, b in zip(store, bt)]}


def main():
    t0 = time.time()
    lookbacks = (124, 126, 128, 130, 132, 134)

    md = model_dir("r5_frozen_pool48")
    print(f"Loading r5_frozen_pool48...", flush=True)
    loaded = load_model(md)

    # Load all index lookback series
    idx_series_map = {n: load_index_returns(n) for n in [3, 5, 10, 20]}

    # Predict once per symbol (predictions don't depend on index lookback)
    base_store = []
    print(f"=== Predicting panel ({len(PANEL)} symbols) ===", flush=True)
    for sym in PANEL:
        try:
            df = load_csv(sym)
        except Exception as e:
            print(f"  SKIP {sym}: {e}", flush=True)
            continue
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, dates = [], [], []
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
            dates.append(w["context_end_date"])
            pred = predict_all_horizons(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        avg_lg = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        conf = direction_confidence_from_logits(avg_lg)
        # Precompute idx arrays for each lookback
        idx_arrs = {n: np.array([get_index_ret(idx_series_map[n], d) for d in dates])
                    for n in idx_series_map}
        base_store.append({
            "symbol": sym, "bh": float(np.sum(tr[:, 0])),
            "hard": conf["hard_pred"], "score": conf["actionable_score"],
            "trh1": tr[:, 0], "tdh1": td[:, 0], "dates": dates,
            "idx_arrs": idx_arrs,
        })
        print(f"  {sym}: bh={base_store[-1]['bh']:.3f}", flush=True)

    n_sym = len(base_store)
    bh = np.array([s["bh"] for s in base_store])

    # For each index lookback, test cond_0.40_0.50
    print(f"\n=== R24: cond_0.40_0.50 with different CSI300 lookback ===", flush=True)
    print(f"{'lookback':<10} {'ret_mean':>9} {'total':>8} {'beat':>7} {'prec':>7} {'cov':>7}", flush=True)
    results = {}
    for n in [3, 5, 10, 20]:
        # Build store with this lookback's idx_arr
        store_n = []
        for s in base_store:
            s_n = {k: v for k, v in s.items() if k != "idx_arrs"}
            s_n["idx_arr"] = s["idx_arrs"][n]
            store_n.append(s_n)
        # fixed 0.45
        r_fixed = eval_config(store_n, bh, lambda s: gate_conf(s["hard"], s["score"], 0.45))
        # fade
        r_fade = eval_config(store_n, bh, lambda s: apply_fade(
            gate_conf(s["hard"], s["score"], 0.45), s["idx_arr"]))
        # cond_0.40_0.50
        r_cond = eval_config(store_n, bh, lambda s: gate_cond_thr(
            s["hard"], s["score"], s["idx_arr"], 0.40, 0.50))
        results[f"lb{n}_fixed"] = r_fixed
        results[f"lb{n}_fade"] = r_fade
        results[f"lb{n}_cond"] = r_cond
        print(f"lb{n}_fixed  {r_fixed['ret_mean']:>+9.3f} {r_fixed['total_ret']:>+8.3f} "
              f"{r_fixed['beat_bh']:>3}/{n_sym}  {r_fixed['prec_mean']:>6.2%} {r_fixed['cov_mean']:>6.2%}", flush=True)
        print(f"lb{n}_fade   {r_fade['ret_mean']:>+9.3f} {r_fade['total_ret']:>+8.3f} "
              f"{r_fade['beat_bh']:>3}/{n_sym}  {r_fade['prec_mean']:>6.2%} {r_fade['cov_mean']:>6.2%}", flush=True)
        print(f"lb{n}_cond   {r_cond['ret_mean']:>+9.3f} {r_cond['total_ret']:>+8.3f} "
              f"{r_cond['beat_bh']:>3}/{n_sym}  {r_cond['prec_mean']:>6.2%} {r_cond['cov_mean']:>6.2%}", flush=True)
        print(flush=True)

    # 688169 single-symbol detail
    print(f"\n=== 688169 single-symbol detail ===", flush=True)
    sym688 = next((s for s in base_store if s["symbol"] == "688169"), None)
    if sym688:
        print(f"688169 bh={sym688['bh']:.3f} n={len(sym688['trh1'])}", flush=True)
        for n in [3, 5, 10, 20]:
            s_n = {"hard": sym688["hard"], "score": sym688["score"],
                   "trh1": sym688["trh1"], "tdh1": sym688["tdh1"],
                   "idx_arr": sym688["idx_arrs"][n]}
            r_fixed = backtest(gate_conf(s_n["hard"], s_n["score"], 0.45), s_n["trh1"], s_n["tdh1"])
            r_fade = backtest(apply_fade(gate_conf(s_n["hard"], s_n["score"], 0.45), s_n["idx_arr"]),
                              s_n["trh1"], s_n["tdh1"])
            r_cond = backtest(gate_cond_thr(s_n["hard"], s_n["score"], s_n["idx_arr"], 0.40, 0.50),
                              s_n["trh1"], s_n["tdh1"])
            print(f"  lb{n}: fixed ret={r_fixed['ret']:+.3f} prec={r_fixed['prec']:.2%} cov={r_fixed['cov']:.2%} | "
                  f"fade ret={r_fade['ret']:+.3f} prec={r_fade['prec']:.2%} cov={r_fade['cov']:.2%} | "
                  f"cond ret={r_cond['ret']:+.3f} prec={r_cond['prec']:.2%} cov={r_cond['cov']:.2%}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r24_lookback.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
