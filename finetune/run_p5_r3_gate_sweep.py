"""Round P5-3: h=1 gate refinement — sweep thr × lookback interaction.

P5-2 found h=1=4lb_left gives g_prec=70.59% (+0.59pt) but g_nf=88.89%
(-2.41pt) at cov=23.6% with default thr=0.45. Hypothesis: raising thr on
4lb_left might reduce coverage back to ~20.8% while preserving the precision
gain and recovering g_nf.

Sweep: 6 h=1 lookbacks × 9 thresholds (0.42-0.50 step 0.01) × 3 mag options.
Fixed h=3/5/10 at P5-1 optimal. Find best in-band gate config.
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
from run_p5_r1_per_h_tta import LOOKBACK_CONFIGS
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

HORIZONS = (1, 3, 5, 10)
P5_1_OPTIMAL = {
    3: (124, 126, 128, 130),
    5: (124, 126, 128, 130, 132, 134),
    10: (126, 128, 130, 132),
}
THRESHOLDS = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
MAGS = [0.0, 0.001, 0.002]


def build_h1_logits(
    tta_store: dict, lookbacks: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (hard_pred, confidence, target_dir, target_ret) for h=1."""
    model = PROMOTED_MODEL_BY_HORIZON[1]
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [tta_store[model][1]["per_lb_logits"][idx] for idx in lb_indices]
    avg_logits = average_logits(per_lb, None)
    conf = direction_confidence_from_logits(avg_logits)
    return (
        conf["hard_pred"],
        conf[PROMOTED_H1_GATE["confidence_key"]],
        tta_store[model][1]["td"],
        tta_store[model][1]["tr"],
    )


def build_other_horizons(
    tta_store: dict, sl_store: dict
) -> tuple[dict, dict, dict, dict, np.ndarray]:
    """Build h=3/5/10 predictions with P5-1 optimal lookbacks; return h=1 gate_ret."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    per_h_weights = PROMOTED_RETURN_BLEND["primary_weight_by_horizon"]
    pred_dir, pred_ret, t_dir, t_ret = {}, {}, {}, {}
    for h in HORIZONS:
        if h == 1:
            continue
        model = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = P5_1_OPTIMAL[h]
        lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
        per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
        avg_logits = average_logits(per_lb, None)
        conf = direction_confidence_from_logits(avg_logits)
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = tta_store[model][h]["td"]
        t_ret[h] = tta_store[model][h]["tr"]
        w = float(per_h_weights[h])
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w
        )
    gate_ret = sl_store[primary][1]["pret"]
    return pred_dir, pred_ret, t_dir, t_ret, gate_ret


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "eval_p5_r3_gate_sweep.json"
    )
    args = parser.parse_args()

    print("=== Loading cached stores ===")
    tta_store = load_per_lb_store(PER_LB_CACHE, ALL_LOOKBACKS)
    sl_store = load_sl_store(SL_CACHE)
    n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded: {n_windows} windows")

    # Precompute h=3/5/10 (fixed at P5-1 optimal)
    other_dir, other_ret, other_td, other_tr, gate_ret = build_other_horizons(
        tta_store, sl_store
    )

    # Precompute h=1 logits for each lookback config
    h1_cache: dict[str, dict] = {}
    for cfg_name, lookbacks in LOOKBACK_CONFIGS:
        hard, conf, td, tr = build_h1_logits(tta_store, lookbacks)
        h1_cache[cfg_name] = {
            "hard": hard,
            "conf": conf,
            "td": td,
            "tr": tr,
            "lookbacks": lookbacks,
        }

    # Phase-5 baseline (P5-2 hybrid: h=1=5lb_current, thr=0.45, mag=0.0)
    p5_base = h1_cache["5lb_current"]
    p5_gated = apply_consistency_and_magnitude_gate(
        p5_base["hard"], p5_base["conf"], gate_ret,
        confidence_threshold=0.45, min_abs_return=0.0, require_sign_agree=False,
    )
    p5_m = gated_actionable_metrics(p5_gated, p5_base["td"])
    print(
        f"\nP5-2 baseline (h=1=5lb_current, thr=0.45, mag=0.0): "
        f"g_prec={p5_m['precision_on_calls']:.2%} "
        f"g_nf={p5_m['gated_nonflat_acc']:.2%} "
        f"g_cov={p5_m['coverage']:.2%}"
    )

    # Sweep: lookback × thr × mag
    print("\n=== Sweeping h=1 lookback × thr × mag ===")
    rows = []
    for cfg_name, lookbacks in LOOKBACK_CONFIGS:
        cache = h1_cache[cfg_name]
        for thr in THRESHOLDS:
            for mag in MAGS:
                gated = apply_consistency_and_magnitude_gate(
                    cache["hard"], cache["conf"], gate_ret,
                    confidence_threshold=thr, min_abs_return=mag,
                    require_sign_agree=False,
                )
                m = gated_actionable_metrics(gated, cache["td"])
                if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                    continue
                if m["n_calls"] < 8:
                    continue
                rows.append({
                    "lookback_config": cfg_name,
                    "lookbacks": list(lookbacks),
                    "thr": thr,
                    "mag": mag,
                    "coverage": m["coverage"],
                    "precision": m["precision_on_calls"],
                    "gated_nf": m["gated_nonflat_acc"],
                    "n_calls": m["n_calls"],
                    "d_prec_pt": (m["precision_on_calls"] - p5_m["precision_on_calls"]) * 100,
                    "d_gnf_pt": (m["gated_nonflat_acc"] - p5_m["gated_nonflat_acc"]) * 100,
                })

    # Sort by (precision, gated_nf) descending
    rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
    print(f"\n{len(rows)} in-band configs found")
    print("\nTop 10 by (precision, gated_nf):")
    for i, r in enumerate(rows[:10]):
        print(
            f"  {i+1}. lb={r['lookback_config']:12s} thr={r['thr']:.2f} mag={r['mag']:.3f}: "
            f"cov={r['coverage']:.2%} prec={r['precision']:.2%} gnf={r['gated_nf']:.2%} "
            f"(d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt)"
        )

    # Also find configs where BOTH prec and gnf are >= baseline (no regression + any gain)
    no_regression = [r for r in rows if r["d_prec_pt"] >= 0 and r["d_gnf_pt"] >= 0]
    print(f"\n{len(no_regression)} configs with NO regression on prec or gnf:")
    for r in no_regression[:5]:
        print(
            f"  lb={r['lookback_config']:12s} thr={r['thr']:.2f} mag={r['mag']:.3f}: "
            f"cov={r['coverage']:.2%} prec={r['precision']:.2%} gnf={r['gated_nf']:.2%} "
            f"(d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt)"
        )

    # Find configs with prec +≥1pt AND gnf +≥1pt (direction_win via gate)
    gate_win = [r for r in rows if r["d_prec_pt"] >= 1.0 and r["d_gnf_pt"] >= 1.0]
    print(f"\n{len(gate_win)} configs with gate direction_win (prec+≥1pt AND gnf+≥1pt):")
    for r in gate_win[:5]:
        print(
            f"  lb={r['lookback_config']:12s} thr={r['thr']:.2f} mag={r['mag']:.3f}: "
            f"cov={r['coverage']:.2%} prec={r['precision']:.2%} gnf={r['gated_nf']:.2%}"
        )

    # Find configs with prec +≥1pt OR gnf +≥1pt (one-sided gate win)
    one_sided = [r for r in rows if r["d_prec_pt"] >= 1.0 or r["d_gnf_pt"] >= 1.0]
    print(f"\n{len(one_sided)} configs with one-sided gate win (prec+≥1pt OR gnf+≥1pt):")
    for r in one_sided[:5]:
        print(
            f"  lb={r['lookback_config']:12s} thr={r['thr']:.2f} mag={r['mag']:.3f}: "
            f"cov={r['coverage']:.2%} prec={r['precision']:.2%} gnf={r['gated_nf']:.2%} "
            f"(d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt)"
        )

    # Build best config summary
    best = rows[0] if rows else None
    if best:
        # Build full summary with best config
        cache = h1_cache[best["lookback_config"]]
        pred_dir = {1: cache["hard"], **other_dir}
        pred_ret = {1: blend_returns(
            sl_store[PROMOTED_RETURN_BLEND["primary"]][1]["pret"],
            sl_store[PROMOTED_RETURN_BLEND["secondary"]][1]["pret"],
            float(PROMOTED_RETURN_BLEND["primary_weight_by_horizon"][1]),
        ), **other_ret}
        t_dir = {1: cache["td"], **other_td}
        t_ret = {1: cache["tr"], **other_tr}
        summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)
        gated = apply_consistency_and_magnitude_gate(
            cache["hard"], cache["conf"], gate_ret,
            confidence_threshold=best["thr"], min_abs_return=best["mag"],
            require_sign_agree=False,
        )
        m = gated_actionable_metrics(gated, cache["td"])
        bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
        summary["h1_gated_precision"] = m["precision_on_calls"]
        summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
        summary["h1_gated_coverage"] = m["coverage"]
        summary["h1_gate_metrics"] = m
        print(
            f"\nBest config full summary: "
            f"nf={summary['nonflat_accuracy_overall']:.2%} "
            f"mae={summary['return_mae_overall']:.5f} "
            f"g_prec={summary['h1_gated_precision']:.2%} "
            f"g_nf={summary['h1_gated_nonflat']:.2%} "
            f"g_cov={summary['h1_gated_coverage']:.2%}"
        )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "p5_2_baseline": {
            "lookback": "5lb_current",
            "thr": 0.45,
            "mag": 0.0,
            "precision": p5_m["precision_on_calls"],
            "gated_nf": p5_m["gated_nonflat_acc"],
            "coverage": p5_m["coverage"],
        },
        "sweep_rows": rows[:50],  # top 50
        "no_regression_configs": no_regression[:10],
        "gate_win_configs": gate_win[:10],
        "one_sided_win_configs": one_sided[:10],
        "best": best,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
