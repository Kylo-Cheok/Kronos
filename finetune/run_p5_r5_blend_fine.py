"""Round P5-5: Per-horizon return blend weight refinement at 0.025 step.

P4-9 swept coarse weights [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] and
found per-horizon optima at h=1:0.9, h=3:0.7, h=5:1.0, h=10:0.7.

P5-5 sweeps at 0.025 step around each horizon's optimal to find finer minima.
Per-horizon MAE curves from P4-9 suggest:
  - h=1: very flat near 0.85-0.95 (diff < 0.0002); check 0.875, 0.925
  - h=3: very flat near 0.7-0.85 (diff < 0.00003); check 0.65, 0.75, 0.775
  - h=5: monotone decreasing toward 1.0; check 0.975
  - h=10: minimum at 0.7; check 0.625, 0.65, 0.725, 0.75

Direction & h=1 gate are blend-independent (direction from TTA logits,
gate uses TTA confidence + SL R10 raw |ret|), so this round only affects MAE.
Dual-bar: need overall MAE strictly lower AND no horizon MAE worsens > 5%.
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
    PROMOTED_TTA_LOOKBACKS_BY_HORIZON,
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

# Per-horizon fine sweep grids (0.025 step around P4-9 optimal)
BLEND_GRID_BY_HORIZON: dict[int, tuple[float, ...]] = {
    1: (0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975, 1.000),
    3: (0.625, 0.650, 0.675, 0.700, 0.725, 0.750, 0.775, 0.800),
    5: (0.900, 0.925, 0.950, 0.975, 1.000),
    10: (0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725, 0.750),
}


def horizon_mae(pred_ret: np.ndarray, target_ret: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_ret - target_ret)))


def build_dir_and_gate(
    tta_store: dict, sl_store: dict
) -> tuple[
    dict[int, np.ndarray],  # pred_dir
    dict[int, np.ndarray],  # t_dir
    dict[int, np.ndarray],  # t_ret
    np.ndarray,             # h1_hard
    np.ndarray,             # h1_score
    np.ndarray,             # h1_gate_ret
    np.ndarray,             # h1_td
]:
    """Build direction preds (per-h TTA lookbacks) + h=1 gate inputs.

    Direction/gate are blend-independent, so compute once and reuse across
    all blend configs.
    """
    pred_dir: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        model = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[h]
        lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
        per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
        avg_logits = average_logits(per_lb, None)
        conf = direction_confidence_from_logits(avg_logits)
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = tta_store[model][h]["td"]
        t_ret[h] = tta_store[model][h]["tr"]

    # h=1 gate inputs: confidence from 4lb_left TTA + magnitude from SL R10 raw |ret|
    h1_model = PROMOTED_MODEL_BY_HORIZON[1]
    h1_lookbacks = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[1]
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in h1_lookbacks]
    per_lb_h1 = [tta_store[h1_model][1]["per_lb_logits"][idx] for idx in lb_indices]
    avg_logits_h1 = average_logits(per_lb_h1, None)
    conf1 = direction_confidence_from_logits(avg_logits_h1)
    h1_hard = conf1["hard_pred"]
    h1_score = conf1[PROMOTED_H1_GATE["confidence_key"]]
    h1_gate_ret = sl_store[PROMOTED_RETURN_BLEND["primary"]][1]["pret"]
    h1_td = tta_store[h1_model][1]["td"]

    return pred_dir, t_dir, t_ret, h1_hard, h1_score, h1_gate_ret, h1_td


def build_summary(
    pred_dir: dict[int, np.ndarray],
    t_dir: dict[int, np.ndarray],
    t_ret: dict[int, np.ndarray],
    blend_weights: dict[int, float],
    sl_store: dict,
    primary: str,
    secondary: str,
    h1_hard: np.ndarray,
    h1_score: np.ndarray,
    h1_gate_ret: np.ndarray,
    h1_td: np.ndarray,
) -> dict:
    pred_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"],
            sl_store[secondary][h]["pret"],
            blend_weights[h],
        )

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)

    # h=1 gate: P5-3 config (4lb_left + thr=0.45 + mag=0.002)
    gated = apply_consistency_and_magnitude_gate(
        h1_hard, h1_score, h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, h1_td)
    summary["h1_gated_precision"] = m["precision_on_calls"]
    summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = m["coverage"]
    summary["h1_gate_metrics"] = m
    summary["blend_weights"] = {str(k): v for k, v in blend_weights.items()}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p5_r5_blend_fine.json",
    )
    args = parser.parse_args()

    print("=== Loading cached stores ===")
    tta_store = load_per_lb_store(PER_LB_CACHE, ALL_LOOKBACKS)
    sl_store = load_sl_store(SL_CACHE)
    n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded: {n_windows} windows")

    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]

    # 1. Precompute direction + h=1 gate inputs (blend-independent)
    (
        pred_dir, t_dir, t_ret,
        h1_hard, h1_score, h1_gate_ret, h1_td,
    ) = build_dir_and_gate(tta_store, sl_store)

    # 2. P5-3 baseline (current per-horizon weights from P4-9)
    p5_3_weights = dict(PROMOTED_RETURN_BLEND["primary_weight_by_horizon"])
    print(f"\nP5-3 baseline weights: {p5_3_weights}")
    p5_3_summary = build_summary(
        pred_dir, t_dir, t_ret, p5_3_weights, sl_store, primary, secondary,
        h1_hard, h1_score, h1_gate_ret, h1_td,
    )
    print(
        f"  nf={p5_3_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={p5_3_summary['return_mae_overall']:.6f} "
        f"g_prec={p5_3_summary['h1_gated_precision']:.2%} "
        f"g_nf={p5_3_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={p5_3_summary['h1_gated_coverage']:.2%}"
    )
    for h in HORIZONS:
        print(
            f"  h={h}: nf={p5_3_summary['by_horizon'][str(h)]['nonflat_accuracy']:.2%} "
            f"mae={p5_3_summary['by_horizon'][str(h)]['return_mae']:.6f}"
        )

    # 3. Per-horizon fine sweep
    print("\n=== Per-horizon fine blend sweep (0.025 step) ===")
    sweep: dict[str, list[dict]] = {}
    optimal: dict[int, float] = {}
    for h in HORIZONS:
        ret_r10 = sl_store[primary][h]["pret"]
        ret_r5 = sl_store[secondary][h]["pret"]
        target = sl_store[primary][h]["tr"]
        rows = []
        for w in BLEND_GRID_BY_HORIZON[h]:
            blended = blend_returns(ret_r10, ret_r5, w)
            mae = horizon_mae(blended, target)
            rows.append({
                "weight_r10": w,
                "mae": mae,
                "mae_rel_vs_p5_3": (mae - p5_3_summary["by_horizon"][str(h)]["return_mae"])
                                   / max(p5_3_summary["by_horizon"][str(h)]["return_mae"], 1e-12),
            })
        sweep[str(h)] = rows
        best = min(rows, key=lambda r: r["mae"])
        optimal[h] = best["weight_r10"]
        print(
            f"  h={h}: best w_r10={best['weight_r10']} mae={best['mae']:.6f} "
            f"(P5-3 mae={p5_3_summary['by_horizon'][str(h)]['return_mae']:.6f}, "
            f"rel delta={best['mae_rel_vs_p5_3']:+.4%})"
        )

    print(f"\nFine-sweep optimal weights: {optimal}")

    # 4. Build combined config with fine-sweep optima
    fine_summary = build_summary(
        pred_dir, t_dir, t_ret, optimal, sl_store, primary, secondary,
        h1_hard, h1_score, h1_gate_ret, h1_td,
    )
    print(
        f"\nFine-sweep combined: "
        f"nf={fine_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={fine_summary['return_mae_overall']:.6f} "
        f"g_prec={fine_summary['h1_gated_precision']:.2%} "
        f"g_nf={fine_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={fine_summary['h1_gated_coverage']:.2%}"
    )
    for h in HORIZONS:
        print(
            f"  h={h}: nf={fine_summary['by_horizon'][str(h)]['nonflat_accuracy']:.2%} "
            f"mae={fine_summary['by_horizon'][str(h)]['return_mae']:.6f} "
            f"(delta={fine_summary['by_horizon'][str(h)]['return_mae'] - p5_3_summary['by_horizon'][str(h)]['return_mae']:+.6f})"
        )

    # 5. Dual-bar decision
    decision = dual_bar_decision(fine_summary, p5_3_summary)
    print(f"\n=== Dual-bar decision (fine-sweep vs P5-3) ===")
    print(f"  promote={decision['promote']} reason={decision['reason']}")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:+.4f} mae_delta={decision['mae_delta']:+.6f}")
    print(f"  horizon_mae_rel_worsen={decision['horizon_mae_rel_worsen']}")

    # 6. Backtest on h=1 (gate unchanged so should equal P5-3)
    gated = apply_consistency_and_magnitude_gate(
        h1_hard, h1_score, h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    print(
        f"\nh=1 gated backtest: ret={bt['total_return']:.2%} "
        f"hit={bt['hit_rate']:.2%} n={bt['n_trades']:.0f}"
    )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "baseline": "P5-3 (P4-9 per-horizon weights)",
        "p5_3_weights": p5_3_weights,
        "p5_3_summary": p5_3_summary,
        "sweep_grid_by_horizon": {str(k): list(v) for k, v in BLEND_GRID_BY_HORIZON.items()},
        "sweep_result": sweep,
        "fine_optimal_weights": optimal,
        "fine_summary": fine_summary,
        "decision": decision,
        "h1_gated_backtest": bt,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
