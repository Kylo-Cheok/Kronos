"""Round P5-7: FLAT logit bias sweep — encourage non-FLAT predictions.

Subtract bias b from FLAT class logits before argmax/softmax. This makes
the model predict non-FLAT more often, potentially recovering true non-FLAT
samples that were incorrectly predicted as FLAT (boosting nonflat recall).

Trade-off: too much bias → false non-FLAT predictions on true FLAT samples
(doesn't affect nonflat_accuracy directly, but hurts direction_accuracy and
gate precision).

Sweep:
  - bias b in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
  - For each b, compute nonflat_accuracy per horizon
  - For h=1, re-sweep gate (thr × mag) at the new confidence distribution

Bias applied to TTA-averaged logits per horizon (P5-3 lookbacks).
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
    FLAT_CLASS,
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)

BIASES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# h=1 gate re-sweep at each bias level
THRESHOLDS = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
MAGS = [0.0, 0.002, 0.003]


def apply_flat_bias(logits: np.ndarray, bias: float) -> np.ndarray:
    """Subtract bias from FLAT class logits (index 1)."""
    if bias == 0.0:
        return logits
    out = logits.copy()
    out[..., FLAT_CLASS] -= float(bias)
    return out


def tta_logits_for(
    tta_store: dict, model: str, h: int, lookbacks: tuple[int, ...]
) -> np.ndarray:
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
    return average_logits(per_lb, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p5_r7_flat_bias.json",
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

    # 1. Precompute TTA-averaged logits per horizon (P5-3 lookbacks)
    print("\n=== Precomputing per-h TTA logits ===")
    base_logits_by_h: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    base_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        model = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[h]
        base_logits_by_h[h] = tta_logits_for(tta_store, model, h, lookbacks)
        t_dir[h] = tta_store[model][h]["td"]
        t_ret[h] = tta_store[model][h]["tr"]
        base_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"],
            float(per_h_weights[h]),
        )

    h1_gate_ret = sl_store[primary][1]["pret"]

    # 2. Sweep bias per horizon (direction-only effect)
    print("\n=== Per-horizon FLAT bias sweep (direction-only) ===")
    sweep: dict[str, list[dict]] = {}
    optimal_bias: dict[int, float] = {}
    for h in HORIZONS:
        td = t_dir[h]
        rows = []
        for b in BIASES:
            biased = apply_flat_bias(base_logits_by_h[h], b)
            conf = direction_confidence_from_logits(biased)
            nf = nonflat_accuracy(conf["hard_pred"], td)
            rows.append({
                "bias": b,
                "nonflat": nf,
                "delta_vs_base_pt": (nf - nonflat_accuracy(
                    direction_confidence_from_logits(base_logits_by_h[h])["hard_pred"], td
                )) * 100,
            })
        sweep[str(h)] = rows
        # Pick max nonflat; tie-break toward smaller bias (simpler)
        best = max(rows, key=lambda r: (r["nonflat"], -r["bias"]))
        optimal_bias[h] = best["bias"]
        print(f"  h={h}: best bias={best['bias']} nf={best['nonflat']:.2%} (d={best['delta_vs_base_pt']:+.2f}pt)")
        for r in rows:
            print(f"    bias={r['bias']:.2f}: nf={r['nonflat']:.2%} (d={r['delta_vs_base_pt']:+.2f}pt)")

    print(f"\nPer-h optimal biases: {optimal_bias}")

    # 3. Build combined config with per-h biases (gate fixed at P5-3)
    print("\n=== Combined bias config (gate fixed at P5-3) ===")
    pred_dir: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        biased = apply_flat_bias(base_logits_by_h[h], optimal_bias[h])
        conf = direction_confidence_from_logits(biased)
        pred_dir[h] = conf["hard_pred"]
    bias_summary = summarize_horizons(pred_dir, base_ret, t_dir, t_ret, HORIZONS)

    # h=1 gate: if h=1 bias changed, recompute gate inputs
    h1_biased = apply_flat_bias(base_logits_by_h[1], optimal_bias[1])
    conf1 = direction_confidence_from_logits(h1_biased)
    h1_hard = conf1["hard_pred"]
    h1_score = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gated = apply_consistency_and_magnitude_gate(
        h1_hard, h1_score, h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, t_dir[1])
    bias_summary["h1_gated_precision"] = m["precision_on_calls"]
    bias_summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    bias_summary["h1_gated_coverage"] = m["coverage"]
    print(
        f"  nf={bias_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={bias_summary['return_mae_overall']:.6f} "
        f"g_prec={bias_summary['h1_gated_precision']:.2%} "
        f"g_nf={bias_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={bias_summary['h1_gated_coverage']:.2%}"
    )

    # 4. Re-sweep h=1 gate at the new biased logits distribution
    gate_resweep: dict | None = None
    if optimal_bias[1] != 0.0:
        print("\n=== Re-sweeping h=1 gate (thr × mag) at biased logits ===")
        rows = []
        # Baseline gate metrics (bias=0, P5-3 gate) for delta comparison
        conf1_base = direction_confidence_from_logits(base_logits_by_h[1])
        gated_base = apply_consistency_and_magnitude_gate(
            conf1_base["hard_pred"],
            conf1_base[PROMOTED_H1_GATE["confidence_key"]],
            h1_gate_ret,
            confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
            min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
            require_sign_agree=False,
        )
        m_base = gated_actionable_metrics(gated_base, t_dir[1])

        for thr in THRESHOLDS:
            for mag in MAGS:
                g = apply_consistency_and_magnitude_gate(
                    h1_hard, h1_score, h1_gate_ret,
                    confidence_threshold=thr, min_abs_return=mag,
                    require_sign_agree=False,
                )
                mm = gated_actionable_metrics(g, t_dir[1])
                if not (0.20 - 1e-9 <= mm["coverage"] <= 0.40 + 1e-9):
                    continue
                if mm["n_calls"] < 8:
                    continue
                rows.append({
                    "thr": thr, "mag": mag,
                    "cov": mm["coverage"],
                    "prec": mm["precision_on_calls"],
                    "gnf": mm["gated_nonflat_acc"],
                    "n_calls": mm["n_calls"],
                    "d_prec_pt": (mm["precision_on_calls"] - m_base["precision_on_calls"]) * 100,
                    "d_gnf_pt": (mm["gated_nonflat_acc"] - m_base["gated_nonflat_acc"]) * 100,
                })
        rows.sort(key=lambda r: (r["prec"], r["gnf"]), reverse=True)
        gate_resweep = {"rows": rows[:20]}
        if rows:
            best = rows[0]
            print(
                f"  Best: thr={best['thr']:.2f} mag={best['mag']:.3f}: "
                f"cov={best['cov']:.2%} prec={best['prec']:.2%} gnf={best['gnf']:.2%} "
                f"(d_prec={best['d_prec_pt']:+.2f}pt d_gnf={best['d_gnf_pt']:+.2f}pt)"
            )
            # Rebuild summary with best gate
            g_best = apply_consistency_and_magnitude_gate(
                h1_hard, h1_score, h1_gate_ret,
                confidence_threshold=best["thr"], min_abs_return=best["mag"],
                require_sign_agree=False,
            )
            m_best = gated_actionable_metrics(g_best, t_dir[1])
            bias_summary["h1_gated_precision"] = m_best["precision_on_calls"]
            bias_summary["h1_gated_nonflat"] = m_best["gated_nonflat_acc"]
            bias_summary["h1_gated_coverage"] = m_best["coverage"]
            bias_summary["h1_gate_metrics"] = m_best
            bias_summary["h1_gate"] = {"thr": best["thr"], "mag": best["mag"]}
            print(
                f"\n  Updated bias summary: "
                f"nf={bias_summary['nonflat_accuracy_overall']:.2%} "
                f"mae={bias_summary['return_mae_overall']:.6f} "
                f"g_prec={bias_summary['h1_gated_precision']:.2%} "
                f"g_nf={bias_summary['h1_gated_nonflat']:.2%} "
                f"g_cov={bias_summary['h1_gated_coverage']:.2%}"
            )

    # 5. P5-5 baseline for dual-bar comparison
    base_dir: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        conf = direction_confidence_from_logits(base_logits_by_h[h])
        base_dir[h] = conf["hard_pred"]
    p5_5_summary = summarize_horizons(base_dir, base_ret, t_dir, t_ret, HORIZONS)
    conf1_base = direction_confidence_from_logits(base_logits_by_h[1])
    gated_base = apply_consistency_and_magnitude_gate(
        conf1_base["hard_pred"],
        conf1_base[PROMOTED_H1_GATE["confidence_key"]],
        h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    m_base = gated_actionable_metrics(gated_base, t_dir[1])
    p5_5_summary["h1_gated_precision"] = m_base["precision_on_calls"]
    p5_5_summary["h1_gated_nonflat"] = m_base["gated_nonflat_acc"]
    p5_5_summary["h1_gated_coverage"] = m_base["coverage"]

    # 6. Dual-bar decision vs P5-5
    decision = dual_bar_decision(bias_summary, p5_5_summary)
    print(f"\n=== Dual-bar decision (bias vs P5-5) ===")
    print(f"  promote={decision['promote']} reason={decision['reason']}")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:+.4f} mae_delta={decision['mae_delta']:+.6f}")
    print(f"  gate_in_band={decision['gate_in_band']}")

    # 7. Backtest on h=1
    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    print(
        f"\nh=1 gated backtest: ret={bt['total_return']:.2%} "
        f"hit={bt['hit_rate']:.2%} n={bt['n_trades']:.0f}"
    )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "baseline": "P5-5 (no bias)",
        "p5_5_summary": p5_5_summary,
        "sweep_per_h": sweep,
        "optimal_bias_per_h": optimal_bias,
        "bias_summary": bias_summary,
        "gate_resweep": gate_resweep,
        "decision": decision,
        "h1_gated_backtest": bt,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
