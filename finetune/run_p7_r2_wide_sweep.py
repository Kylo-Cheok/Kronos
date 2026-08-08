"""Phase 7 — Round R2: wide-lookback TTA sweep (numpy only, on p7 wide store).

1) Sanity: reproduce P5-8 baseline numbers from the wide store (must match
   p6 cache exactly: nf=72.51%, gate prec=75%/gnf=96%).
2) Per-lookback nonflat curve (model x horizon x lb) to see the landscape.
3) Sweep ALL contiguous TTA subsets (widths 1..11) of the wide grid per
   (model, horizon); direction model per horizon fixed to P5-8 mapping
   (h=1,10->r10; h=3,5->r5). Pick per-horizon best config by ungated nonflat.
4) Build candidate (per-h best config), re-sweep h=1 gate (thr x strict_h5),
   dual-bar vs baseline computed from the SAME wide store.

Output: outputs/eval_p7_r2_wide_sweep.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons  # noqa: E402
from run_p4_r7_expanded_tta import average_logits  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from run_p6_eval_zoo import gate_sweep  # noqa: E402
from run_p7_r1_build_wide_store import WIDE_LOOKBACKS  # noqa: E402
from selective_prediction import (  # noqa: E402
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
PRIMARY_W_BY_H = {1: 0.925, 3: 0.775, 5: 1.0, 10: 0.65}
P5_BASE_MAP = {1: PRIMARY, 3: SECONDARY, 5: SECONDARY, 10: PRIMARY}
P5_TTA_BY_H = {
    1: (124, 126, 128, 130),
    3: (124, 126, 128, 130),
    5: (124, 126, 128, 130, 132, 134),
    10: (126, 128, 130, 132),
}
GATE_MAG = 0.002


def tta_logits(store, model, h, lookbacks):
    lb_idx = [WIDE_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [store[model][h]["per_lb_logits"][i] for i in lb_idx]
    return average_logits(per_lb, None)


def contiguous_configs():
    """All contiguous subsets of WIDE_LOOKBACKS with width 1..11."""
    lbs = list(WIDE_LOOKBACKS)
    out = []
    for w in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        for i in range(0, len(lbs) - w + 1):
            out.append(tuple(lbs[i:i + w]))
    return out


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p7_wide_per_lb_store.npz", WIDE_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    n = len(tta[PRIMARY][1]["td"])
    print(f"Loaded wide store: {list(tta.keys())}, {n} windows, "
          f"{len(WIDE_LOOKBACKS)} lookbacks")

    result = {"n_windows": n, "lookbacks": list(WIDE_LOOKBACKS)}

    # ---- 1. Sanity: reproduce P5-8 baseline ----
    bp_dir, bt_dir, bt_ret, bpred_ret, base_score = {}, {}, {}, {}, {}
    for h in HORIZONS:
        m = P5_BASE_MAP[h]
        conf = direction_confidence_from_logits(tta_logits(tta, m, h, P5_TTA_BY_H[h]))
        bp_dir[h] = conf["hard_pred"]
        base_score[h] = conf["actionable_score"]
        bt_dir[h] = tta[m][h]["td"]
        bt_ret[h] = tta[m][h]["tr"]
        bpred_ret[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                     PRIMARY_W_BY_H[h])
    base_summary = summarize_horizons(bp_dir, bpred_ret, bt_dir, bt_ret, HORIZONS)
    base_gated = apply_consistency_and_magnitude_gate(
        bp_dir[1], base_score[1], sl[PRIMARY][1]["pret"],
        confidence_threshold=0.45, min_abs_return=GATE_MAG, require_sign_agree=False)
    base_gated = apply_consistency_filter(base_gated, bp_dir[1], bp_dir[5], "strict")
    bm = gated_actionable_metrics(base_gated, bt_dir[1])
    base_bt = absolute_direction_backtest(base_gated, bt_ret[1], transaction_cost=0.0005)
    base_summary.update(h1_gated_precision=bm["precision_on_calls"],
                        h1_gated_nonflat=bm["gated_nonflat_acc"],
                        h1_gated_coverage=bm["coverage"],
                        h1_gated_backtest=base_bt)
    print(f"SANITY baseline: nf={base_summary['nonflat_accuracy_overall']:.2%} "
          f"(expect 72.51%) mae={base_summary['return_mae_overall']:.6f} "
          f"gate prec={bm['precision_on_calls']:.2%} gnf={bm['gated_nonflat_acc']:.2%} "
          f"bt={base_bt['total_return']:.2%}")
    result["baseline_summary"] = base_summary

    # ---- 2. Per-lookback nonflat curve ----
    curves = {}
    for m in (PRIMARY, SECONDARY):
        curves[m] = {}
        for h in HORIZONS:
            td = tta[m][h]["td"]
            row = []
            for i, lb in enumerate(WIDE_LOOKBACKS):
                conf = direction_confidence_from_logits(tta_logits(tta, m, h, (lb,)))
                row.append(float(nonflat_accuracy(conf["hard_pred"], td)))
            curves[m][str(h)] = row
    result["per_lb_nonflat_curve"] = curves
    for m in (PRIMARY, SECONDARY):
        for h in HORIZONS:
            c = np.array(curves[m][str(h)])
            bi = int(np.argmax(c))
            print(f"curve {m} h={h}: best lb={WIDE_LOOKBACKS[bi]} "
                  f"nf={c[bi]:.2%} | lb128 nf={c[WIDE_LOOKBACKS.index(128)]:.2%}")

    # ---- 3. Contiguous subset sweep per horizon (P5-8 model mapping) ----
    configs = contiguous_configs()
    best_by_h = {}
    sweep_tables = {}
    for h in HORIZONS:
        m = P5_BASE_MAP[h]
        td = tta[m][h]["td"]
        rows = []
        for lbs in configs:
            conf = direction_confidence_from_logits(tta_logits(tta, m, h, lbs))
            nf = float(nonflat_accuracy(conf["hard_pred"], td))
            rows.append((lbs, nf))
        rows.sort(key=lambda t: t[1], reverse=True)
        best_by_h[h] = rows[0]
        sweep_tables[str(h)] = [{"lbs": list(l), "nf": v} for l, v in rows[:10]]
        p5_nf = [v for l, v in rows if l == P5_TTA_BY_H[h]][0]
        print(f"h={h} ({m}): best={rows[0][0]} nf={rows[0][1]:.2%} | "
              f"P5-8 cfg nf={p5_nf:.2%} | delta={rows[0][1] - p5_nf:+.2%}")
    result["sweep_top10_by_horizon"] = sweep_tables
    result["best_config_by_horizon"] = {str(h): list(best_by_h[h][0]) for h in HORIZONS}

    # ---- 4. Candidate + gate sweep + dual-bar ----
    cp_dir, ct_dir, ct_ret, cpred_ret, cand_score = {}, {}, {}, {}, {}
    for h in HORIZONS:
        m = P5_BASE_MAP[h]
        conf = direction_confidence_from_logits(tta_logits(tta, m, h, best_by_h[h][0]))
        cp_dir[h] = conf["hard_pred"]
        cand_score[h] = conf["actionable_score"]
        ct_dir[h] = tta[m][h]["td"]
        ct_ret[h] = tta[m][h]["tr"]
        cpred_ret[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                     PRIMARY_W_BY_H[h])
    cand_summary = summarize_horizons(cp_dir, cpred_ret, ct_dir, ct_ret, HORIZONS)
    cand_best, cand_rows = gate_sweep(cp_dir, cand_score, sl[PRIMARY][1]["pret"],
                                      ct_dir[1], True)
    if cand_best:
        cm = gated_actionable_metrics(cand_best["gated"], ct_dir[1])
        cand_bt = absolute_direction_backtest(cand_best["gated"], ct_ret[1],
                                              transaction_cost=0.0005)
        cand_summary.update(h1_gated_precision=cm["precision_on_calls"],
                            h1_gated_nonflat=cm["gated_nonflat_acc"],
                            h1_gated_coverage=cm["coverage"],
                            h1_gated_backtest=cand_bt,
                            h1_gate={"thr": cand_best["thr"], "mag": GATE_MAG,
                                     "consistency": "strict_h5"})
        print(f"CANDIDATE: nf={cand_summary['nonflat_accuracy_overall']:.2%} "
              f"gate thr={cand_best['thr']} cov={cm['coverage']:.2%} "
              f"prec={cm['precision_on_calls']:.2%} gnf={cm['gated_nonflat_acc']:.2%} "
              f"bt={cand_bt['total_return']:.2%}")
    decision = dual_bar_decision(cand_summary, base_summary)
    print(f"DECISION: promote={decision['promote']} reason={decision['reason']} "
          f"d_nf={decision['nonflat_delta']:+.4f}")
    result["candidate_summary"] = cand_summary
    result["decision_vs_p5_8"] = decision

    out = ROOT / "outputs" / "eval_p7_r2_wide_sweep.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
