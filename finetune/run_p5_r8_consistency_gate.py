"""Round P5-8: Multi-horizon consistency gate for h=1.

P5-3 gate uses h=1 confidence + magnitude. Idea: also require that
longer-horizon predictions (h=3, h=5) agree with h=1 direction. This
filters out h=1 calls that contradict the longer-horizon trend, potentially
improving precision.

Variants:
  - "strict_h3": h=3 must predict same class as h=1 (FLAT = disagree)
  - "strict_h5": h=5 must predict same class as h=1
  - "strict_h3_or_h5": h=3 OR h=5 must agree (same class)
  - "strict_h3_and_h5": h=3 AND h=5 must agree
  - "soft_h3": h=3 must not predict opposite (FLAT is OK)
  - "soft_h5": h=5 must not predict opposite

For each variant, sweep h=1 confidence threshold to maintain 20-40% coverage.
The base gate (conf + mag) is applied first, then the consistency filter.
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
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)

# h=1 confidence thresholds to sweep (consistency filter reduces coverage,
# so lower thresholds may be needed to stay in 20-40% band)
THRESHOLDS = [0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
MAG = 0.002  # P5-3 optimal mag


def tta_logits_for(
    tta_store: dict, model: str, h: int, lookbacks: tuple[int, ...]
) -> np.ndarray:
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
    return average_logits(per_lb, None)


def apply_consistency_filter(
    h1_gated: np.ndarray,
    h1_pred: np.ndarray,  # ungated h=1 hard pred
    h_other_pred: np.ndarray,  # ungated h=other hard pred
    mode: str,
) -> np.ndarray:
    """Apply consistency filter on top of base-gated h=1 predictions.

    h1_gated: h=1 predictions after base gate (conf + mag). FLAT = abstain.
    h1_pred: original h=1 hard predictions (before gate).
    h_other_pred: h=3 or h=5 hard predictions.
    """
    out = h1_gated.copy()
    # Only check samples where h=1 made a call (non-FLAT after gate)
    called = h1_gated != FLAT_CLASS

    if mode == "strict":
        # h_other must predict same class as h=1 call
        agree = (h_other_pred == h1_gated) & called
        out[~agree] = FLAT_CLASS
    elif mode == "soft":
        # h_other must NOT predict opposite direction (FLAT is OK)
        # Opposite: h1=UP & h_other=DOWN, or h1=DOWN & h_other=UP
        h1_up = h1_gated == UP_CLASS
        h1_down = h1_gated == DOWN_CLASS
        opposite = (h1_up & (h_other_pred == DOWN_CLASS)) | \
                   (h1_down & (h_other_pred == UP_CLASS))
        filter_out = opposite & called
        out[filter_out] = FLAT_CLASS
    else:
        raise ValueError(f"unknown mode: {mode}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p5_r8_consistency_gate.json",
    )
    args = parser.parse_args()

    print("=== Loading cached stores ===")
    tta_store = load_per_lb_store(PER_LB_CACHE, ALL_LOOKBACKS)
    sl_store = load_sl_store(SL_CACHE)
    n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded: {n_windows} windows")

    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    per_h_weights = PROMOTED_RETURN_BLEND["primary_weight_by_horizon"]

    # 1. Precompute TTA logits per horizon
    print("\n=== Precomputing per-h TTA logits ===")
    logits_by_h: dict[int, np.ndarray] = {}
    hard_by_h: dict[int, np.ndarray] = {}
    score_by_h: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    base_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        model = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[h]
        logits_by_h[h] = tta_logits_for(tta_store, model, h, lookbacks)
        conf = direction_confidence_from_logits(logits_by_h[h])
        hard_by_h[h] = conf["hard_pred"]
        score_by_h[h] = conf[PROMOTED_H1_GATE["confidence_key"]]
        t_dir[h] = tta_store[model][h]["td"]
        t_ret[h] = tta_store[model][h]["tr"]
        base_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"],
            float(per_h_weights[h]),
        )

    h1_gate_ret = sl_store[primary][1]["pret"]

    # 2. P5-5 baseline (no consistency filter)
    print("\n=== P5-5 baseline (P5-3 gate, no consistency filter) ===")
    p5_5_dir = hard_by_h
    p5_5_summary = summarize_horizons(p5_5_dir, base_ret, t_dir, t_ret, HORIZONS)
    gated_base = apply_consistency_and_magnitude_gate(
        hard_by_h[1], score_by_h[1], h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    m_base = gated_actionable_metrics(gated_base, t_dir[1])
    p5_5_summary["h1_gated_precision"] = m_base["precision_on_calls"]
    p5_5_summary["h1_gated_nonflat"] = m_base["gated_nonflat_acc"]
    p5_5_summary["h1_gated_coverage"] = m_base["coverage"]
    print(
        f"  nf={p5_5_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={p5_5_summary['return_mae_overall']:.6f} "
        f"g_prec={p5_5_summary['h1_gated_precision']:.2%} "
        f"g_nf={p5_5_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={p5_5_summary['h1_gated_coverage']:.2%}"
    )

    # 3. Check raw agreement rates (informational)
    print("\n=== Raw h=1 vs h=3/h=5 agreement rates ===")
    for other_h in (3, 5):
        agree = (hard_by_h[other_h] == hard_by_h[1]).mean()
        same_nonflat = ((hard_by_h[1] != FLAT_CLASS) &
                        (hard_by_h[other_h] == hard_by_h[1])).mean()
        opposite = (((hard_by_h[1] == UP_CLASS) & (hard_by_h[other_h] == DOWN_CLASS)) |
                    ((hard_by_h[1] == DOWN_CLASS) & (hard_by_h[other_h] == UP_CLASS))).mean()
        print(f"  h=1 vs h={other_h}: agree={agree:.2%}, same_nonflat={same_nonflat:.2%}, opposite={opposite:.2%}")

    # 4. Sweep consistency filter variants × h=1 threshold
    print("\n=== Consistency gate sweep ===")
    variants = [
        ("strict_h3", 3, "strict"),
        ("strict_h5", 5, "strict"),
        ("strict_h3_or_h5", None, "strict"),  # special: OR of h=3 and h=5
        ("strict_h3_and_h5", None, "strict"),  # special: AND
        ("soft_h3", 3, "soft"),
        ("soft_h5", 5, "soft"),
        ("soft_h3_or_h5", None, "soft"),
    ]

    results = {}
    for variant_name, other_h, mode in variants:
        print(f"\n--- Variant: {variant_name} ---")
        rows = []
        for thr in THRESHOLDS:
            # Base gate
            base_gated = apply_consistency_and_magnitude_gate(
                hard_by_h[1], score_by_h[1], h1_gate_ret,
                confidence_threshold=thr, min_abs_return=MAG,
                require_sign_agree=False,
            )
            # Apply consistency filter
            if other_h is not None:
                # Single horizon filter
                gated = apply_consistency_filter(
                    base_gated, hard_by_h[1], hard_by_h[other_h], mode
                )
            elif variant_name.endswith("_or_h5"):
                # h=3 OR h=5: keep if either agrees
                g3 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[3], mode)
                g5 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[5], mode)
                called = base_gated != FLAT_CLASS
                gated = base_gated.copy()
                # Keep only if g3 OR g5 keeps it
                keep = (g3 != FLAT_CLASS) | (g5 != FLAT_CLASS)
                gated[~(keep & called)] = FLAT_CLASS
            elif variant_name.endswith("_and_h5"):
                # h=3 AND h=5: keep only if both agree
                g3 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[3], mode)
                g5 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[5], mode)
                called = base_gated != FLAT_CLASS
                gated = base_gated.copy()
                keep = (g3 != FLAT_CLASS) & (g5 != FLAT_CLASS)
                gated[~(keep & called)] = FLAT_CLASS

            m = gated_actionable_metrics(gated, t_dir[1])
            if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                continue
            if m["n_calls"] < 8:
                continue
            rows.append({
                "thr": thr,
                "cov": m["coverage"],
                "prec": m["precision_on_calls"],
                "gnf": m["gated_nonflat_acc"],
                "n_calls": m["n_calls"],
                "d_prec_pt": (m["precision_on_calls"] - m_base["precision_on_calls"]) * 100,
                "d_gnf_pt": (m["gated_nonflat_acc"] - m_base["gated_nonflat_acc"]) * 100,
                "gated": gated,  # keep for backtest
            })
        if not rows:
            print(f"  No in-band configs found")
            results[variant_name] = None
            continue
        # Pick best by (prec, gnf)
        rows.sort(key=lambda r: (r["prec"], r["gnf"]), reverse=True)
        best = rows[0]
        print(
            f"  Best: thr={best['thr']:.2f}: cov={best['cov']:.2%} "
            f"prec={best['prec']:.2%} gnf={best['gnf']:.2%} "
            f"(d_prec={best['d_prec_pt']:+.2f}pt d_gnf={best['d_gnf_pt']:+.2f}pt) "
            f"n={best['n_calls']:.0f}"
        )
        # Show top 3
        for i, r in enumerate(rows[:3]):
            print(
                f"    {i+1}. thr={r['thr']:.2f}: cov={r['cov']:.2%} "
                f"prec={r['prec']:.2%} gnf={r['gnf']:.2%} "
                f"(d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt)"
            )
        # Strip gated array for JSON (keep best only)
        best_clean = {k: v for k, v in best.items() if k != "gated"}
        results[variant_name] = {"best": best_clean, "all_in_band": [
            {k: v for k, v in r.items() if k != "gated"} for r in rows[:10]
        ]}

    # 5. Find best variant
    valid = [(name, r["best"]) for name, r in results.items() if r is not None]
    if not valid:
        print("\nNo valid consistency configs found.")
        return 0
    best_variant, best_config = max(valid, key=lambda x: (x[1]["prec"], x[1]["gnf"]))
    print(
        f"\n=== Best variant: {best_variant} ===\n"
        f"  thr={best_config['thr']:.2f} cov={best_config['cov']:.2%} "
        f"prec={best_config['prec']:.2%} gnf={best_config['gnf']:.2%} "
        f"(d_prec={best_config['d_prec_pt']:+.2f}pt d_gnf={best_config['d_gnf_pt']:+.2f}pt)"
    )

    # 6. Build full summary with best variant (direction unchanged, only gate changes)
    # Re-run the best variant to get gated array
    other_h = None
    mode = "strict"
    for vname, voh, vmode in variants:
        if vname == best_variant:
            other_h = voh
            mode = vmode
            break

    base_gated = apply_consistency_and_magnitude_gate(
        hard_by_h[1], score_by_h[1], h1_gate_ret,
        confidence_threshold=best_config["thr"], min_abs_return=MAG,
        require_sign_agree=False,
    )
    if other_h is not None:
        gated = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[other_h], mode)
    elif best_variant.endswith("_or_h5"):
        g3 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[3], mode)
        g5 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[5], mode)
        called = base_gated != FLAT_CLASS
        gated = base_gated.copy()
        keep = (g3 != FLAT_CLASS) | (g5 != FLAT_CLASS)
        gated[~(keep & called)] = FLAT_CLASS
    elif best_variant.endswith("_and_h5"):
        g3 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[3], mode)
        g5 = apply_consistency_filter(base_gated, hard_by_h[1], hard_by_h[5], mode)
        called = base_gated != FLAT_CLASS
        gated = base_gated.copy()
        keep = (g3 != FLAT_CLASS) & (g5 != FLAT_CLASS)
        gated[~(keep & called)] = FLAT_CLASS

    m_best = gated_actionable_metrics(gated, t_dir[1])
    best_summary = summarize_horizons(hard_by_h, base_ret, t_dir, t_ret, HORIZONS)
    best_summary["h1_gated_precision"] = m_best["precision_on_calls"]
    best_summary["h1_gated_nonflat"] = m_best["gated_nonflat_acc"]
    best_summary["h1_gated_coverage"] = m_best["coverage"]
    best_summary["h1_gate_metrics"] = m_best
    best_summary["h1_gate"] = {
        "thr": best_config["thr"], "mag": MAG,
        "consistency": best_variant,
    }

    # 7. Dual-bar decision vs P5-5
    decision = dual_bar_decision(best_summary, p5_5_summary)
    print(f"\n=== Dual-bar decision ({best_variant} vs P5-5) ===")
    print(f"  promote={decision['promote']} reason={decision['reason']}")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:+.4f} mae_delta={decision['mae_delta']:+.6f}")
    print(f"  gate_in_band={decision['gate_in_band']}")

    # 8. Backtest on h=1
    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    print(
        f"\nh=1 gated backtest ({best_variant}): ret={bt['total_return']:.2%} "
        f"hit={bt['hit_rate']:.2%} n={bt['n_trades']:.0f}"
    )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "baseline": "P5-5 (P5-3 gate, no consistency filter)",
        "p5_5_summary": p5_5_summary,
        "agreement_rates": {
            "h1_vs_h3": {
                "agree": float((hard_by_h[3] == hard_by_h[1]).mean()),
                "opposite": float((((hard_by_h[1] == UP_CLASS) & (hard_by_h[3] == DOWN_CLASS)) |
                                    ((hard_by_h[1] == DOWN_CLASS) & (hard_by_h[3] == UP_CLASS))).mean()),
            },
            "h1_vs_h5": {
                "agree": float((hard_by_h[5] == hard_by_h[1]).mean()),
                "opposite": float((((hard_by_h[1] == UP_CLASS) & (hard_by_h[5] == DOWN_CLASS)) |
                                    ((hard_by_h[1] == DOWN_CLASS) & (hard_by_h[5] == UP_CLASS))).mean()),
            },
        },
        "variant_results": results,
        "best_variant": best_variant,
        "best_summary": best_summary,
        "decision": decision,
        "h1_gated_backtest": bt,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
