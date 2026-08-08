"""Phase 6 — Round P6-2: gating / TTA refinement sweep (CPU-only).

Goal: find a h=1 gate configuration that maximises the *trading* metric
(gated backtest return) while staying in the 20-40% coverage band, WITHOUT
changing any model. Pure numpy over the existing P6 caches.

We sweep:
  * h=1 source TTA config (9 options) for the gate confidence logits
  * confidence key: actionable_score / max_prob / margin / nonflat_prob
  * threshold grid 0.30..0.55 (step 0.01)
  * magnitude filter: 0.0 / 0.001 / 0.002 / 0.003 / 0.004
  * consistency: none / soft / strict (vs h=5)
  * require_sign_agree: False / True

The baseline P5-8 gate (actionable_score, r10 h=1 P5-TTA (124,126,128,130),
thr=0.45, mag=0.002, strict_h5, sign_agree=False) must reproduce ~102.63%
backtest — used as a validation anchor.

Output: outputs/eval_p6_p2_gate_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS,
    UP_CLASS,
    DOWN_CLASS,
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

HORIZONS = (1, 3, 5, 10)

LOOKBACK_CONFIGS = [
    ("3lb_tight", (126, 128, 130)),
    ("4lb_left", (124, 126, 128, 130)),
    ("4lb_right", (126, 128, 130, 132)),
    ("5lb_current", (124, 126, 128, 130, 132)),
    ("5lb_left", (122, 124, 126, 128, 130)),
    ("5lb_right", (126, 128, 130, 132, 134)),
    ("6lb", (124, 126, 128, 130, 132, 134)),
    ("7lb_full", (122, 124, 126, 128, 130, 132, 134)),
    ("single_128", (128,)),
]
LB_NAME_TO_TUPLE = dict(LOOKBACK_CONFIGS)

# P5-8 reference (baseline gate)
P5_H1_TTA = (124, 126, 128, 130)
P5_H5_TTA = (124, 126, 128, 130, 132, 134)
P5_BASE_MAP = {1: "r10_joint_splitlr", 3: "r5_frozen_pool48",
               5: "r5_frozen_pool48", 10: "r10_joint_splitlr"}

CONF_KEYS = ["actionable_score", "max_prob", "margin", "nonflat_prob"]
THR_GRID = [round(x, 2) for x in np.arange(0.30, 0.56, 0.01)]
MAGS = [0.0, 0.001, 0.002, 0.003, 0.004]
CONSISTENCY = ["none", "soft", "strict"]
SIGN_AGREE = [False, True]

COV_MIN, COV_MAX = 0.20, 0.40


def tta_logits(store, model, h, lookbacks):
    lb_idx = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [store[model][h]["per_lb_logits"][i] for i in lb_idx]
    return average_logits(per_lb, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-lb", default=str(ROOT / "outputs" / "p6_per_lb_store.npz"))
    ap.add_argument("--sl", default=str(ROOT / "outputs" / "p6_sl_store.npz"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "eval_p6_p2_gate_sweep.json"))
    args = ap.parse_args()

    tta = load_per_lb_store(Path(args.per_lb), ALL_LOOKBACKS)
    sl = load_sl_store(Path(args.sl))
    models = sorted(tta.keys())
    n = len(tta[models[0]][1]["td"])
    print(f"Loaded {len(models)} models, {n} windows")

    m_h1 = P5_BASE_MAP[1]  # r10_joint_splitlr
    m_h5 = P5_BASE_MAP[5]  # r5_frozen_pool48

    # h=5 hard pred (consistency reference) — fixed P5 TTA.
    h5_logits = tta_logits(tta, m_h5, 5, P5_H5_TTA)
    h5_hard = direction_confidence_from_logits(h5_logits)["hard_pred"]

    # gate return (predicted h=1 return) + true return + true direction for h=1.
    gate_ret = sl[m_h1][1]["pret"]
    t_dir1 = tta[m_h1][1]["td"]
    t_ret1 = tta[m_h1][1]["tr"]

    # ---- Baseline P5-8 gate (validation anchor) ----
    base_logits = tta_logits(tta, m_h1, 1, P5_H1_TTA)
    base_conf = direction_confidence_from_logits(base_logits)
    base_hard = base_conf["hard_pred"]
    base_score = base_conf["actionable_score"]
    base_gated = apply_consistency_and_magnitude_gate(
        base_hard, base_score, gate_ret,
        confidence_threshold=0.45, min_abs_return=0.002, require_sign_agree=False)
    base_gated = apply_consistency_filter(base_gated, base_hard, h5_hard, "strict")
    bm = gated_actionable_metrics(base_gated, t_dir1)
    bbt = absolute_direction_backtest(base_gated, t_ret1, transaction_cost=0.0005)
    print(f"BASELINE P5-8 gate: cov={bm['coverage']:.2%} prec={bm['precision_on_calls']:.2%} "
          f"gnf={bm['gated_nonflat_acc']:.2%} bt_ret={bbt['total_return']:.2%} "
          f"(anchor should be ~102.63%)")

    # ---- Sweep ----
    rows = []
    for cfg_name, lbs in LOOKBACK_CONFIGS:
        logits = tta_logits(tta, m_h1, 1, lbs)
        conf = direction_confidence_from_logits(logits)
        hard = conf["hard_pred"]
        for ckey in CONF_KEYS:
            score = conf[ckey]
            for thr in THR_GRID:
                for mag in MAGS:
                    for cons in CONSISTENCY:
                        for sign in SIGN_AGREE:
                            gated = apply_consistency_and_magnitude_gate(
                                hard, score, gate_ret,
                                confidence_threshold=thr, min_abs_return=mag,
                                require_sign_agree=sign)
                            if cons != "none":
                                gated = apply_consistency_filter(gated, hard, h5_hard, cons)
                            m = gated_actionable_metrics(gated, t_dir1)
                            if not (COV_MIN - 1e-9 <= m["coverage"] <= COV_MAX + 1e-9):
                                continue
                            if m["n_calls"] < 8:
                                continue
                            bt = absolute_direction_backtest(gated, t_ret1, transaction_cost=0.0005)
                            rows.append({
                                "h1_tta": cfg_name, "conf_key": ckey, "thr": thr,
                                "mag": mag, "consistency": cons, "sign_agree": sign,
                                "coverage": m["coverage"],
                                "precision": m["precision_on_calls"],
                                "gated_nonflat": m["gated_nonflat_acc"],
                                "n_calls": m["n_calls"],
                                "bt_total_return": bt["total_return"],
                                "bt_hit": bt["hit_rate"],
                                "d_bt_vs_base": bt["total_return"] - bbt["total_return"],
                                "d_prec_vs_base": m["precision_on_calls"] - bm["precision_on_calls"],
                            })
    print(f"\nSwept {len(rows)} in-band configs.")

    rows.sort(key=lambda r: (r["bt_total_return"], r["precision"]), reverse=True)
    top = rows[:20]
    print("\n=== TOP 20 by gated backtest (within 20-40% coverage) ===")
    for r in top[:10]:
        print(f"  {r['h1_tta']:>11} {r['conf_key']:>16} thr={r['thr']:.2f} "
              f"mag={r['mag']:.3f} {r['consistency']:>6} sign={int(r['sign_agree'])} "
              f"| cov={r['coverage']:.2%} prec={r['precision']:.2%} "
              f"bt={r['bt_total_return']:.2%} (d_bt={r['d_bt_vs_base']:+.2%})")

    # Also best by precision (within band) for completeness.
    by_prec = sorted(rows, key=lambda r: (r["precision"], r["bt_total_return"]), reverse=True)[:10]

    result = {
        "symbol": "688169",
        "n_windows": n,
        "baseline_p5_8_gate": {
            "h1_tta": "P5(124,126,128,130)", "conf_key": "actionable_score",
            "thr": 0.45, "mag": 0.002, "consistency": "strict", "sign_agree": False,
            "coverage": bm["coverage"], "precision": bm["precision_on_calls"],
            "gated_nonflat": bm["gated_nonflat_acc"], "n_calls": bm["n_calls"],
            "bt_total_return": bbt["total_return"], "bt_hit": bbt["hit_rate"],
        },
        "n_in_band_configs": len(rows),
        "best_by_backtest": top,
        "best_by_precision": by_prec,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
