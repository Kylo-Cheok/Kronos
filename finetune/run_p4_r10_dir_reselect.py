"""Round P4-10: Per-horizon direction model re-selection under TTA + final.

P4-6 fixed the direction mapping as h=1,3,10=R10, h=5=R5 (chosen under
single-lookback inference in Phase-3). Under TTA, the optimal direction
model per horizon may differ. This script:

1. Computes nonflat accuracy for BOTH R10 and R5 under TTA for all horizons.
2. Picks the best model per horizon (auto-TTA mapping).
3. Evaluates the full config: auto-TTA direction + per-horizon blend (P4-9) + gate.
4. Compares against P4-9 (current promoted) and Phase-3 baseline.
5. Produces the final Phase-4 summary.

This is the last Phase-4 round.
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
    PHASE2_BASELINE_MODEL_BY_HORIZON,
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
)
from run_p4_r3_tta_sweep import CACHE_PATH, load_store
from run_p4_r9_blend_sweep import SL_CACHE_PATH, load_sl_store
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
R10 = "r10_joint_splitlr"
R5 = "r5_frozen_pool48"


def compute_tta_nonflat_per_model(
    tta_store: dict, models: list[str]
) -> dict[str, dict[int, float]]:
    """For each model and horizon, compute nonflat accuracy under TTA logits."""
    result: dict[str, dict[int, float]] = {}
    for name in models:
        result[name] = {}
        for h in HORIZONS:
            conf = direction_confidence_from_logits(tta_store[name][h]["logits"])
            td = tta_store[name][h]["td"]
            nf = nonflat_accuracy(conf["hard_pred"], td)
            result[name][h] = nf
    return result


def build_full_config(
    tta_store: dict,
    sl_store: dict,
    dir_map: dict[int, str],
    blend_weights: dict[int, float],
) -> dict:
    """Build full config: TTA direction + SL per-horizon blend + gate."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]

    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = dir_map[h]
        conf = direction_confidence_from_logits(tta_store[dn][h]["logits"])
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = sl_store[dn][h]["td"]
        t_ret[h] = sl_store[dn][h]["tr"]
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"],
            sl_store[secondary][h]["pret"],
            blend_weights[h],
        )

    # h=1 gate
    h1_dn = dir_map[1]
    conf1 = direction_confidence_from_logits(tta_store[h1_dn][1]["logits"])
    hard1 = conf1["hard_pred"]
    score1 = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = sl_store[primary][1]["pret"]

    gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.0,
        "require_sign_agree": False,
        "transaction_cost": 0.0005,
    }
    gated = apply_consistency_and_magnitude_gate(
        hard1, score1, gate_ret,
        confidence_threshold=gate["confidence_threshold"],
        min_abs_return=gate["min_abs_return"],
        require_sign_agree=False,
    )
    gate_m = gated_actionable_metrics(gated, t_dir[1])

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)
    summary["h1_gated_precision"] = gate_m["precision_on_calls"]
    summary["h1_gated_nonflat"] = gate_m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = gate_m["coverage"]
    summary["h1_gate_metrics"] = gate_m
    summary["gate"] = gate

    ungated_bt = absolute_direction_backtest(hard1, t_ret[1], transaction_cost=0.0005)
    gated_bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)

    return {
        "summary": summary,
        "ungated_bt": ungated_bt,
        "gated_bt": gated_bt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p4_r10_dir_reselect.json",
    )
    args = parser.parse_args()

    # 1. Load caches
    if not CACHE_PATH.exists():
        print(f"Cache not found: {CACHE_PATH}. Run run_p4_r3_tta_sweep.py first.")
        return 1
    tta_store = load_store(CACHE_PATH)
    n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded TTA store: {n_windows} windows")

    if not SL_CACHE_PATH.exists():
        print(f"Cache not found: {SL_CACHE_PATH}. Run run_p4_r9_blend_sweep.py first.")
        return 1
    sl_store = load_sl_store(SL_CACHE_PATH)
    print(f"Loaded SL store: {len(sl_store[next(iter(sl_store))][1]['td'])} windows")

    # 2. Compute nonflat per model × horizon under TTA
    print("\n=== TTA nonflat accuracy per model × horizon ===")
    models = [R10, R5]
    nf_per_model = compute_tta_nonflat_per_model(tta_store, models)

    print(f"{'Horizon':<10} {'R10':>10} {'R5':>10} {'Best':>10}")
    auto_map: dict[int, str] = {}
    for h in HORIZONS:
        r10_nf = nf_per_model[R10][h]
        r5_nf = nf_per_model[R5][h]
        best = R10 if r10_nf >= r5_nf else R5
        auto_map[h] = best
        print(f"h={h:<8} {r10_nf:>10.2%} {r5_nf:>10.2%} {best:>10}")

    # 3. Per-horizon blend weights (from P4-9)
    blend_weights = dict(PROMOTED_RETURN_BLEND["primary_weight_by_horizon"])
    print(f"\nPer-horizon blend weights: {blend_weights}")

    # 4. Evaluate configs
    configs = {
        "p4_9_current": (PROMOTED_MODEL_BY_HORIZON, blend_weights),
        "p4_10_auto_tta": (auto_map, blend_weights),
        "p4_10_auto_r10_all": ({h: R10 for h in HORIZONS}, blend_weights),
    }

    results = {}
    for name, (dir_map, bw) in configs.items():
        print(f"\n--- Config: {name} ---")
        print(f"  dir_map: {dir_map}")
        result = build_full_config(tta_store, sl_store, dir_map, bw)
        s = result["summary"]
        print(
            f"  nf={s['nonflat_accuracy_overall']:.2%} "
            f"mae={s['return_mae_overall']:.6f} "
            f"g_prec={s['h1_gated_precision']:.2%} "
            f"g_nf={s['h1_gated_nonflat']:.2%} "
            f"g_cov={s['h1_gated_coverage']:.2%}"
        )
        for h in HORIZONS:
            print(f"  h={h}: nf={s['by_horizon'][str(h)]['nonflat_accuracy']:.2%} mae={s['by_horizon'][str(h)]['return_mae']:.6f}")
        results[name] = result

    # 5. Decision: auto_tta vs current
    current = results["p4_9_current"]["summary"]
    auto = results["p4_10_auto_tta"]["summary"]
    decision = dual_bar_decision(auto, current)
    print(f"\n=== Decision: p4_10_auto_tta vs p4_9_current ===")
    print(f"  promote={decision['promote']} reason={decision['reason']}")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:.4f} mae_delta={decision['mae_delta']:.6f}")

    # Also check auto_r10_all
    r10_all = results["p4_10_auto_r10_all"]["summary"]
    decision_r10 = dual_bar_decision(r10_all, current)
    print(f"\n=== Decision: p4_10_auto_r10_all vs p4_9_current ===")
    print(f"  promote={decision_r10['promote']} reason={decision_r10['reason']}")

    # 6. Pick best
    best_name = "p4_10_auto_tta" if decision["promote"] else "p4_9_current"
    best = results[best_name]
    print(f"\nBest config: {best_name}")

    # 7. Final summary vs Phase-3 baseline
    phase3_baseline = {
        "nonflat_accuracy_overall": 0.70,
        "return_mae_overall": 0.03758,
        "by_horizon": {
            "1": {"nonflat_accuracy": 0.6484, "direction_accuracy": 0.4097, "return_mae": 0.01878},
            "3": {"nonflat_accuracy": 0.7059, "direction_accuracy": 0.4167, "return_mae": 0.02967},
            "5": {"nonflat_accuracy": 0.7386, "direction_accuracy": 0.4514, "return_mae": 0.04007},
            "10": {"nonflat_accuracy": 0.7075, "direction_accuracy": 0.5208, "return_mae": 0.06179},
        },
        "h1_gated_precision": 0.6364,
        "h1_gated_nonflat": 0.84,
        "h1_gated_coverage": 0.2292,
    }
    final_decision = dual_bar_decision(best["summary"], phase3_baseline)
    print(f"\n=== Final decision vs Phase-3 baseline ===")
    print(f"  promote={final_decision['promote']} reason={final_decision['reason']}")
    print(f"  direction_win={final_decision['direction_win']} mae_win={final_decision['mae_win']}")
    print(f"  nonflat_delta={final_decision['nonflat_delta']:.4f} mae_delta={final_decision['mae_delta']:.6f}")

    output = {
        "symbol": "688169",
        "n_windows": n_windows,
        "tta_nonflat_per_model": {
            name: {str(h): nf_per_model[name][h] for h in HORIZONS}
            for name in models
        },
        "auto_direction_mapping": {str(k): v for k, v in auto_map.items()},
        "blend_weights": {str(k): v for k, v in blend_weights.items()},
        "configs": {k: v["summary"] for k, v in results.items()},
        "backtests": {k: {"ungated": v["ungated_bt"], "gated": v["gated_bt"]} for k, v in results.items()},
        "decision_auto_vs_current": decision,
        "decision_r10_all_vs_current": decision_r10,
        "best_config": best_name,
        "best_summary": best["summary"],
        "final_decision_vs_phase3": final_decision,
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
