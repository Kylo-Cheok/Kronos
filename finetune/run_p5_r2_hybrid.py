"""Round P5-2: Hybrid per-horizon TTA lookback strategy.

P5-1 found per-horizon optimal lookbacks boost ungated nonflat +1.89pt but
hurt h=1 gate precision (-1.25pt) because h=1's optimal 5lb_left changes
the confidence distribution.

This round tests a hybrid: keep h=1 at Phase-4 5lb_current (preserve gate),
apply P5-1 optimal lookbacks only to h=3/5/10 (boost ungated nonflat).

Also tests alternative h=1 configs that might preserve gate precision:
  - 5lb_current (Phase-4, baseline)
  - 4lb_left (close to 5lb_left but tighter)
  - 3lb_tight (tightest around 128)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons
from promoted_config import (
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
)
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits
from run_p4_r8_h1_only_tta import load_per_lb_store, PER_LB_CACHE
from run_p4_r9_blend_sweep import load_sl_store, SL_CACHE_PATH as SL_CACHE
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

HORIZONS = (1, 3, 5, 10)

# P5-1 optimal lookbacks for h=3/5/10 (h=1 to be swept)
P5_1_OPTIMAL = {
    3: (124, 126, 128, 130),        # 4lb_left
    5: (124, 126, 128, 130, 132, 134),  # 6lb
    10: (126, 128, 130, 132),       # 4lb_right
}

# h=1 lookback candidates to test (with h=3/5/10 fixed at P5-1 optimal)
H1_CANDIDATES = [
    ("5lb_current", (124, 126, 128, 130, 132)),  # Phase-4 baseline
    ("4lb_left", (124, 126, 128, 130)),
    ("3lb_tight", (126, 128, 130)),
    ("5lb_left", (122, 124, 126, 128, 130)),     # P5-1 optimal (hurt gate)
    ("4lb_right", (126, 128, 130, 132)),
    ("single_128", (128,)),
]


def evaluate_hybrid(
    tta_store: dict,
    sl_store: dict,
    h1_lookbacks: tuple[int, ...],
    h_lookbacks: dict[int, tuple[int, ...]],
) -> dict:
    """Evaluate hybrid config: h=1 with h1_lookbacks, h=3/5/10 with their optimal."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    per_h_weights = PROMOTED_RETURN_BLEND["primary_weight_by_horizon"]

    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    h1_conf = None
    h1_hard = None
    for h in HORIZONS:
        model_name = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = h_lookbacks[h]
        lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
        per_lb = [tta_store[model_name][h]["per_lb_logits"][idx] for idx in lb_indices]
        avg_logits = average_logits(per_lb, None)
        conf = direction_confidence_from_logits(avg_logits)
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = tta_store[model_name][h]["td"]
        t_ret[h] = tta_store[model_name][h]["tr"]
        w = float(per_h_weights[h])
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w
        )
        if h == 1:
            h1_conf = conf[PROMOTED_H1_GATE["confidence_key"]]
            h1_hard = conf["hard_pred"]

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)
    gate_ret = sl_store[primary][1]["pret"]
    gated = apply_consistency_and_magnitude_gate(
        h1_hard, h1_conf, gate_ret,
        confidence_threshold=float(PROMOTED_H1_GATE["confidence_threshold"]),
        min_abs_return=float(PROMOTED_H1_GATE["min_abs_return"]),
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, t_dir[1])
    summary["h1_gated_precision"] = m["precision_on_calls"]
    summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = m["coverage"]
    summary["h1_gate_metrics"] = m
    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    return summary, bt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "eval_p5_r2_hybrid.json"
    )
    args = parser.parse_args()

    print("=== Loading cached stores ===")
    tta_store = load_per_lb_store(PER_LB_CACHE, ALL_LOOKBACKS)
    sl_store = load_sl_store(SL_CACHE)
    n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded: {n_windows} windows")

    # Phase-4 baseline (all 5lb_current)
    p4_lookbacks = {h: (124, 126, 128, 130, 132) for h in HORIZONS}
    print("\n=== Phase-4 baseline (uniform 5lb_current) ===")
    p4_summary, p4_bt = evaluate_hybrid(tta_store, sl_store, p4_lookbacks[1], p4_lookbacks)
    print(
        f"nf={p4_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={p4_summary['return_mae_overall']:.5f} "
        f"g_prec={p4_summary['h1_gated_precision']:.2%} "
        f"g_nf={p4_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={p4_summary['h1_gated_coverage']:.2%}"
    )

    # Sweep h=1 lookback candidates with h=3/5/10 fixed at P5-1 optimal
    print("\n=== Hybrid: h=3/5/10 at P5-1 optimal, sweep h=1 lookback ===")
    results = {}
    for h1_name, h1_lb in H1_CANDIDATES:
        h_lookbacks = {1: h1_lb, **P5_1_OPTIMAL}
        summary, bt = evaluate_hybrid(tta_store, sl_store, h1_lb, h_lookbacks)
        decision = dual_bar_decision(summary, p4_summary)
        results[h1_name] = {
            "h1_lookbacks": list(h1_lb),
            "h3_lookbacks": list(P5_1_OPTIMAL[3]),
            "h5_lookbacks": list(P5_1_OPTIMAL[5]),
            "h10_lookbacks": list(P5_1_OPTIMAL[10]),
            "summary": summary,
            "backtest": bt,
            "decision": decision,
        }
        print(
            f"\n  h1={h1_name:12s} lb={h1_lb}: "
            f"nf={summary['nonflat_accuracy_overall']:.2%} "
            f"mae={summary['return_mae_overall']:.5f} "
            f"g_prec={summary['h1_gated_precision']:.2%} "
            f"g_nf={summary['h1_gated_nonflat']:.2%} "
            f"g_cov={summary['h1_gated_coverage']:.2%} "
            f"-> {decision['promote']} ({decision['reason']})"
        )

    # Find best hybrid
    print("\n=== Finding best hybrid ===")
    # Prioritize: gate precision >= 70% (no regression), then max nonflat
    in_band = [
        (name, r) for name, r in results.items()
        if 0.20 - 1e-9 <= r["summary"]["h1_gated_coverage"] <= 0.40 + 1e-9
    ]
    no_gate_regression = [
        (name, r) for name, r in in_band
        if r["summary"]["h1_gated_precision"] >= p4_summary["h1_gated_precision"] - 0.005
    ]
    if no_gate_regression:
        best_name, best_r = max(
            no_gate_regression, key=lambda x: x[1]["summary"]["nonflat_accuracy_overall"]
        )
        print(f"Best (no gate regression): h1={best_name}")
    elif in_band:
        best_name, best_r = max(
            in_band, key=lambda x: (
                x[1]["summary"]["h1_gated_precision"],
                x[1]["summary"]["nonflat_accuracy_overall"],
            )
        )
        print(f"Best (in band, may have gate regression): h1={best_name}")
    else:
        best_name, best_r = None, None
        print("No in-band config found")

    if best_r:
        print(
            f"  nf={best_r['summary']['nonflat_accuracy_overall']:.2%} "
            f"mae={best_r['summary']['return_mae_overall']:.5f} "
            f"g_prec={best_r['summary']['h1_gated_precision']:.2%} "
            f"g_nf={best_r['summary']['h1_gated_nonflat']:.2%} "
            f"g_cov={best_r['summary']['h1_gated_coverage']:.2%}"
        )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "phase4_baseline": p4_summary,
        "phase4_backtest": p4_bt,
        "hybrid_results": results,
        "best_hybrid": {"name": best_name, "result": best_r} if best_r else None,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
