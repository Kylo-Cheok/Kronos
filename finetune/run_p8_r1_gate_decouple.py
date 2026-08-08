"""Phase 8 — Round R1: h=1 gate confidence decoupling (numpy only).

All previous phases derived the gate's confidence score from the SAME TTA
logits as the hard prediction. Here the hard prediction stays on the P5-8
sweet spot (4lb_left) — so ungated nonflat and the h=5 consistency reference
are untouched — while the confidence score is computed from DIFFERENT,
wider lookback ensembles (the p7 wide store, 25 lookbacks 104..152). A
better-calibrated score re-ranks which of the SAME hard calls pass the gate,
potentially lifting precision/gated-nonflat in-band without moving anything
else.

Sweep: score_source x thr(0.40..0.55) x mag(0,0.002,0.004) x strict_h5.
Baseline anchor (must reproduce): 4lb_left + thr=0.45 + mag=0.002 +
strict_h5 -> cov=22.22% prec=75.00% gnf=96.00% bt=102.63%.

Output: outputs/eval_p8_r1_gate_decouple.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p7_r1_build_wide_store import WIDE_LOOKBACKS  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from selective_prediction import (  # noqa: E402
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"

SCORE_SOURCES = {
    "4lb_left(baseline)": (124, 126, 128, 130),
    "7lb_full": (122, 124, 126, 128, 130, 132, 134),
    "wide_all_25lb": tuple(WIDE_LOOKBACKS),
    "wide_left_13lb": tuple(lb for lb in WIDE_LOOKBACKS if lb <= 128),
    "wide_right_13lb": tuple(lb for lb in WIDE_LOOKBACKS if lb >= 128),
    "wide_core_9lb": tuple(range(112, 145, 4)),
    "5lb_left": (122, 124, 126, 128, 130),
    "single_128": (128,),
    "wide_far_left": (104, 108, 112, 116, 120),
    "wide_far_right": (136, 140, 144, 148, 152),
}
THR_GRID = np.round(np.arange(0.40, 0.551, 0.01), 2)
MAG_GRID = (0.0, 0.002, 0.004)


def avg_logits(store, lookbacks_all, model, h, lbs):
    idx = [lookbacks_all.index(lb) for lb in lbs]
    return average_logits([store[model][h]["per_lb_logits"][i] for i in idx], None)


def main() -> int:
    wide = load_per_lb_store(ROOT / "outputs" / "p7_wide_per_lb_store.npz", WIDE_LOOKBACKS)
    p6 = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")

    # hard h=1 from 4lb_left (fixed); h=5 hard from baseline 6lb (fixed)
    hard1 = direction_confidence_from_logits(
        avg_logits(p6, ALL_LOOKBACKS, PRIMARY, 1, (124, 126, 128, 130)))["hard_pred"]
    hard5 = direction_confidence_from_logits(
        avg_logits(p6, ALL_LOOKBACKS, SECONDARY, 5, (124, 126, 128, 130, 132, 134)))["hard_pred"]
    td1 = p6[PRIMARY][1]["td"]
    tr1 = p6[PRIMARY][1]["tr"]
    gate_ret = sl[PRIMARY][1]["pret"]

    rows = []
    for src_name, lbs in SCORE_SOURCES.items():
        score = direction_confidence_from_logits(
            avg_logits(wide, WIDE_LOOKBACKS, PRIMARY, 1, lbs))["actionable_score"]
        for thr in THR_GRID:
            for mag in MAG_GRID:
                g = apply_consistency_and_magnitude_gate(
                    hard1, score, gate_ret, confidence_threshold=float(thr),
                    min_abs_return=mag, require_sign_agree=False)
                g = apply_consistency_filter(g, hard1, hard5, "strict")
                m = gated_actionable_metrics(g, td1)
                if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                    continue
                if m["n_calls"] < 8:
                    continue
                bt = absolute_direction_backtest(g, tr1, transaction_cost=0.0005)
                rows.append({"src": src_name, "thr": float(thr), "mag": mag,
                             "cov": m["coverage"], "prec": m["precision_on_calls"],
                             "gnf": m["gated_nonflat_acc"], "n": m["n_calls"],
                             "bt": bt["total_return"]})

    rows.sort(key=lambda r: (r["prec"], r["gnf"]), reverse=True)
    print("Top 12 by (prec, gnf):")
    for r in rows[:12]:
        print(f"  {r['src']:20s} thr={r['thr']:.2f} mag={r['mag']:.3f} "
              f"cov={r['cov']:.2%} prec={r['prec']:.2%} gnf={r['gnf']:.2%} "
              f"n={r['n']} bt={r['bt']:.2%}")
    base = [r for r in rows if r["src"] == "4lb_left(baseline)"
            and abs(r["thr"] - 0.45) < 1e-9 and r["mag"] == 0.002]
    print("Baseline anchor:", base[0] if base else "NOT FOUND (check)")

    out = ROOT / "outputs" / "eval_p8_r1_gate_decouple.json"
    out.write_text(json.dumps({"rows_in_band": rows[:50], "n_in_band": len(rows)},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}  ({len(rows)} in-band configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
