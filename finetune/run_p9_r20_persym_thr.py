"""Phase 9 R20 — Per-symbol threshold sensitivity (zero training cost).

R19 showed r5 generalizes (beat 23/28) but model has only weak spontaneous
regime sensitivity (UP cov 55% vs DN cov 45%). Before heavy training-side
CSI300 injection (5-file rewrite), check if per-symbol adaptive threshold
has headroom: sweep thr per symbol, see if optimal thr is heterogeneous.

If all symbols peak at thr=0.45, r5 is at its ceiling. If dispersed, an
adaptive gate (e.g. higher thr for down-symbols) could lift returns.

Reuses R19 prediction logic; adds per-symbol thr sweep + idx-conditional
sweep (down-symbols get higher thr).

Output: outputs/eval_p9_panel_r20_persym_thr.json
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

from evaluate_gated_ensemble import FEATURES, LOOKBACK, WINDOW, load_csv  # noqa: E402
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p4_r7_expanded_tta import (  # noqa: E402
    average_logits, derive_time_features, normalize_with_lookback,
)
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p9_r19_bigpanel import (  # noqa: E402
    PANEL, REF_CTX, REF_DZ, REF_VOL, TEST_LO, COST, IDX_N,
    get_index_ret, load_index_returns, make_test_windows, predict_all_horizons,
)
from run_tta_eval import DEVICE, HORIZONS, PREDICT_WINDOW_LEN  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

THR_SWEEP = [round(x, 2) for x in np.arange(0.30, 0.75, 0.05)]


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def backtest_ret(g, trh1, tdh1):
    pos = (g == UP_CLASS).astype(np.float32)
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    ret = float(np.sum(pos * trh1) - turnover * COST)
    gm = gated_actionable_metrics(g, tdh1)
    return ret, gm["precision_on_calls"], gm["coverage"], gm["n_calls"]


def main():
    t0 = time.time()
    idx_series = load_index_returns()
    lookbacks = (124, 126, 128, 130, 132, 134)

    md = model_dir("r5_frozen_pool48")
    print(f"Loading r5_frozen_pool48...", flush=True)
    loaded = load_model(md)

    store = []
    print(f"=== Predicting panel ({len(PANEL)} symbols) ===", flush=True)
    for sym in PANEL:
        try:
            df = load_csv(sym)
        except Exception as e:
            print(f"  SKIP {sym}: {e}", flush=True)
            continue
        windows = make_test_windows(df, lookbacks)
        if not windows:
            print(f"  SKIP {sym}: no test windows", flush=True)
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
            "hard": confs[0]["hard_pred"],
            "score": confs[0]["actionable_score"],
            "trh1": trh1, "tdh1": tdh1, "idx_arr": np.asarray(idx_rets),
        })
        print(f"  {sym}: bh={bh:.3f} n={len(windows)}", flush=True)

    n_sym = len(store)
    print(f"\n=== Per-symbol threshold sweep (thr in {THR_SWEEP}) ===", flush=True)
    print(f"{'SYMBOL':<10} {'BH':>7} ", end="", flush=True)
    for thr in THR_SWEEP:
        print(f"thr{thr:.2f} ", end="", flush=True)
    print("  OPT_THR  OPT_RET  RET@0.45", flush=True)

    per_sym_opt = []
    for s in store:
        rets = []
        for thr in THR_SWEEP:
            g = gate_conf(s["hard"], s["score"], thr)
            ret, _, _, _ = backtest_ret(g, s["trh1"], s["tdh1"])
            rets.append(ret)
        opt_idx = int(np.argmax(rets))
        opt_thr = THR_SWEEP[opt_idx]
        opt_ret = rets[opt_idx]
        # ret at fixed 0.45
        thr_045_idx = THR_SWEEP.index(0.45) if 0.45 in THR_SWEEP else None
        ret_045 = rets[thr_045_idx] if thr_045_idx is not None else float("nan")
        per_sym_opt.append({"symbol": s["symbol"], "bh": s["bh"],
                            "opt_thr": opt_thr, "opt_ret": opt_ret,
                            "ret_045": ret_045, "rets": rets})
        print(f"{s['symbol']:<10} {s['bh']:>+7.3f} ", end="", flush=True)
        for r in rets:
            print(f"{r:>+6.3f} ", end="", flush=True)
        print(f"  {opt_thr:>6.2f}  {opt_ret:>+6.3f}  {ret_045:>+6.3f}", flush=True)

    # Heterogeneity check
    opt_thrs = [p["opt_thr"] for p in per_sym_opt]
    print(f"\n=== Heterogeneity ===", flush=True)
    print(f"  opt_thr distribution: { {t: opt_thrs.count(t) for t in sorted(set(opt_thrs))} }", flush=True)
    print(f"  opt_thr mean={np.mean(opt_thrs):.3f} std={np.std(opt_thrs):.3f}", flush=True)

    # Global: sum ret across all symbols per thr
    global_rets = [sum(p["rets"][i] for p in per_sym_opt) for i in range(len(THR_SWEEP))]
    best_global_idx = int(np.argmax(global_rets))
    print(f"\n=== Global (sum ret across {n_sym} symbols) ===", flush=True)
    for i, thr in enumerate(THR_SWEEP):
        marker = " <-- BEST" if i == best_global_idx else ""
        print(f"  thr={thr:.2f}: total_ret={global_rets[i]:+.3f}{marker}", flush=True)
    print(f"  fixed 0.45 total_ret={sum(p['ret_045'] for p in per_sym_opt):+.3f}", flush=True)

    # Oracle: per-symbol best thr (upper bound of adaptive)
    oracle_total = sum(p["opt_ret"] for p in per_sym_opt)
    fixed_total = sum(p["ret_045"] for p in per_sym_opt)
    print(f"\n=== Adaptive ceiling ===", flush=True)
    print(f"  Oracle (per-symbol best): {oracle_total:+.3f}", flush=True)
    print(f"  Fixed 0.45:               {fixed_total:+.3f}", flush=True)
    print(f"  Adaptive headroom:        {oracle_total - fixed_total:+.3f}", flush=True)

    # Regime-conditional: down-symbols (bh<=0) vs up-symbols (bh>0) optimal thr
    dn_opts = [p for p in per_sym_opt if p["bh"] <= 0]
    up_opts = [p for p in per_sym_opt if p["bh"] > 0]
    print(f"\n=== Regime-conditional opt thr ===", flush=True)
    print(f"  Down-symbols (n={len(dn_opts)}): mean opt_thr={np.mean([p['opt_thr'] for p in dn_opts]):.3f}", flush=True)
    print(f"  Up-symbols   (n={len(up_opts)}): mean opt_thr={np.mean([p['opt_thr'] for p in up_opts]):.3f}", flush=True)

    # Index-conditional: thr varies by csi300 5d ret
    # For each symbol, split windows by idx_arr sign, find best thr per regime
    print(f"\n=== Index-conditional per-symbol (within-symbol split) ===", flush=True)
    print(f"{'SYMBOL':<10} {'dn_opt':>7} {'up_opt':>7} {'dn_n':>5} {'up_n':>5}", flush=True)
    idx_cond_summary = []
    for s in store:
        idx = s["idx_arr"]
        valid = ~np.isnan(idx)
        dn_mask = valid & (idx < 0)
        up_mask = valid & (idx >= 0)
        dn_rets, up_rets = [], []
        for thr in THR_SWEEP:
            g = gate_conf(s["hard"], s["score"], thr)
            if dn_mask.any():
                r, _, _, _ = backtest_ret(g[dn_mask], s["trh1"][dn_mask], s["tdh1"][dn_mask])
                dn_rets.append(r)
            else:
                dn_rets.append(float("nan"))
            if up_mask.any():
                r, _, _, _ = backtest_ret(g[up_mask], s["trh1"][up_mask], s["tdh1"][up_mask])
                up_rets.append(r)
            else:
                up_rets.append(float("nan"))
        dn_opt = THR_SWEEP[int(np.nanargmax(dn_rets))] if not all(np.isnan(x) for x in dn_rets) else float("nan")
        up_opt = THR_SWEEP[int(np.nanargmax(up_rets))] if not all(np.isnan(x) for x in up_rets) else float("nan")
        idx_cond_summary.append({"symbol": s["symbol"], "dn_opt": dn_opt, "up_opt": up_opt,
                                 "dn_n": int(dn_mask.sum()), "up_n": int(up_mask.sum())})
        print(f"{s['symbol']:<10} {dn_opt:>7.2f} {up_opt:>7.2f} {int(dn_mask.sum()):>5} {int(up_mask.sum()):>5}", flush=True)

    dn_opts_idx = [x["dn_opt"] for x in idx_cond_summary if not np.isnan(x["dn_opt"])]
    up_opts_idx = [x["up_opt"] for x in idx_cond_summary if not np.isnan(x["up_opt"])]
    print(f"\n  Mean within-symbol dn_opt_thr={np.mean(dn_opts_idx):.3f}  up_opt_thr={np.mean(up_opts_idx):.3f}", flush=True)

    out = {
        "n_symbols": n_sym,
        "thr_sweep": THR_SWEEP,
        "per_symbol": per_sym_opt,
        "global_rets": global_rets,
        "oracle_total": oracle_total,
        "fixed_045_total": fixed_total,
        "idx_conditional": idx_cond_summary,
    }
    out_path = ROOT / "outputs" / "eval_p9_panel_r20_persym_thr.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
