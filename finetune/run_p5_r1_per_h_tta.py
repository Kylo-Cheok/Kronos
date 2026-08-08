"""Round P5-1: Per-horizon TTA lookback configuration sweep.

Phase-4 uses the same TTA lookbacks [124,126,128,130,132] for all horizons.
Different horizons may prefer different lookback windows:
  - h=1 (short-term) might prefer tighter windows (closer to lb=128)
  - h=10 (long-term) might prefer wider windows (more context)

This script sweeps TTA lookback configs PER HORIZON, finding the optimal
lookback configuration for each horizon independently. Direction model
mapping stays as Phase-4 promoted.

Uses cached per_lb_store_cache.npz (7 lookbacks 122-134) and sl_store_cache.npz.
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
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)

# Candidate lookback configurations to sweep per horizon.
LOOKBACK_CONFIGS = [
    ("3lb_tight", (126, 128, 130)),
    ("4lb_left", (124, 126, 128, 130)),
    ("4lb_right", (126, 128, 130, 132)),
    ("5lb_current", (124, 126, 128, 130, 132)),  # Phase-4 default
    ("5lb_left", (122, 124, 126, 128, 130)),
    ("5lb_right", (126, 128, 130, 132, 134)),
    ("6lb", (124, 126, 128, 130, 132, 134)),
    ("7lb_full", (122, 124, 126, 128, 130, 132, 134)),
    ("single_128", (128,)),  # No TTA baseline
]


def eval_config_for_horizon(
    tta_store: dict,
    sl_store: dict,
    h: int,
    lookbacks: tuple[int, ...],
    model_name: str,
) -> dict:
    """Evaluate one lookback config for one horizon. Returns nonflat + targets."""
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [tta_store[model_name][h]["per_lb_logits"][idx] for idx in lb_indices]
    avg_logits = average_logits(per_lb, None)
    conf = direction_confidence_from_logits(avg_logits)
    td = tta_store[model_name][h]["td"]
    nf = nonflat_accuracy(conf["hard_pred"], td)
    return {
        "nonflat": nf,
        "hard_pred": conf["hard_pred"],
        "confidence": conf[PROMOTED_H1_GATE["confidence_key"]] if h == 1 else None,
        "avg_logits": avg_logits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "eval_p5_r1_per_h_tta.json"
    )
    args = parser.parse_args()

    print("=== Loading cached stores ===")
    tta_store = load_per_lb_store(PER_LB_CACHE, ALL_LOOKBACKS)
    sl_store = load_sl_store(SL_CACHE)
    n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded: {n_windows} windows, models={list(tta_store.keys())}")

    # 1. Sweep per-horizon lookback configs
    print("\n=== Per-horizon TTA lookback sweep ===")
    sweep: dict[str, list[dict]] = {}
    best_per_h: dict[int, dict] = {}
    for h in HORIZONS:
        model_name = PROMOTED_MODEL_BY_HORIZON[h]
        print(f"\nh={h} (model={model_name}):")
        rows = []
        for cfg_name, lookbacks in LOOKBACK_CONFIGS:
            r = eval_config_for_horizon(tta_store, sl_store, h, lookbacks, model_name)
            rows.append(
                {
                    "config": cfg_name,
                    "lookbacks": list(lookbacks),
                    "nonflat": r["nonflat"],
                }
            )
            print(
                f"  {cfg_name:16s} lb={lookbacks} -> nf={r['nonflat']:.2%}"
            )
        rows.sort(key=lambda r: r["nonflat"], reverse=True)
        sweep[str(h)] = rows
        best_per_h[h] = rows[0]
        print(f"  BEST: {rows[0]['config']} nf={rows[0]['nonflat']:.2%}")

    # 2. Build optimal config: use best lookback per horizon
    print("\n=== Building optimal per-horizon config ===")
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
        best_lookbacks = tuple(best_per_h[h]["lookbacks"])
        lb_indices = [ALL_LOOKBACKS.index(lb) for lb in best_lookbacks]
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
    # h=1 gate
    gate_ret = sl_store[primary][1]["pret"]
    gated = apply_consistency_and_magnitude_gate(
        h1_hard,
        h1_conf,
        gate_ret,
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

    print(
        f"\nOptimal per-horizon TTA: "
        f"nf={summary['nonflat_accuracy_overall']:.2%} "
        f"mae={summary['return_mae_overall']:.5f} "
        f"g_prec={summary['h1_gated_precision']:.2%} "
        f"g_nf={summary['h1_gated_nonflat']:.2%} "
        f"g_cov={summary['h1_gated_coverage']:.2%}"
    )
    print("Per-horizon best configs:")
    for h in HORIZONS:
        print(
            f"  h={h}: {best_per_h[h]['config']} lb={best_per_h[h]['lookbacks']} "
            f"nf={best_per_h[h]['nonflat']:.2%}"
        )

    # 3. Phase-4 baseline (uniform 5lb_current) for comparison
    p4_lookbacks = (124, 126, 128, 130, 132)
    p4_dir: dict[int, np.ndarray] = {}
    p4_ret: dict[int, np.ndarray] = {}
    p4_td: dict[int, np.ndarray] = {}
    p4_tr: dict[int, np.ndarray] = {}
    p4_h1_conf = None
    p4_h1_hard = None
    for h in HORIZONS:
        model_name = PROMOTED_MODEL_BY_HORIZON[h]
        lb_indices = [ALL_LOOKBACKS.index(lb) for lb in p4_lookbacks]
        per_lb = [tta_store[model_name][h]["per_lb_logits"][idx] for idx in lb_indices]
        avg_logits = average_logits(per_lb, None)
        conf = direction_confidence_from_logits(avg_logits)
        p4_dir[h] = conf["hard_pred"]
        p4_td[h] = tta_store[model_name][h]["td"]
        p4_tr[h] = tta_store[model_name][h]["tr"]
        w = float(per_h_weights[h])
        p4_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w
        )
        if h == 1:
            p4_h1_conf = conf[PROMOTED_H1_GATE["confidence_key"]]
            p4_h1_hard = conf["hard_pred"]
    p4_summary = summarize_horizons(p4_dir, p4_ret, p4_td, p4_tr, HORIZONS)
    p4_gated = apply_consistency_and_magnitude_gate(
        p4_h1_hard, p4_h1_conf, gate_ret,
        confidence_threshold=0.45, min_abs_return=0.0, require_sign_agree=False,
    )
    p4_m = gated_actionable_metrics(p4_gated, p4_td[1])
    p4_summary["h1_gated_precision"] = p4_m["precision_on_calls"]
    p4_summary["h1_gated_nonflat"] = p4_m["gated_nonflat_acc"]
    p4_summary["h1_gated_coverage"] = p4_m["coverage"]
    print(
        f"\nPhase-4 baseline (uniform 5lb): "
        f"nf={p4_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={p4_summary['return_mae_overall']:.5f} "
        f"g_prec={p4_summary['h1_gated_precision']:.2%} "
        f"g_nf={p4_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={p4_summary['h1_gated_coverage']:.2%}"
    )

    decision = dual_bar_decision(summary, p4_summary)
    print(f"\nDecision vs Phase-4: {decision['promote']} ({decision['reason']})")

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "sweep_per_horizon": sweep,
        "best_per_horizon": {str(h): best_per_h[h] for h in HORIZONS},
        "phase4_baseline": p4_summary,
        "optimal_per_h": summary,
        "decision_vs_phase4": decision,
        "h1_backtest": bt,
        "optimal_lookbacks_by_horizon": {
            str(h): best_per_h[h]["lookbacks"] for h in HORIZONS
        },
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
