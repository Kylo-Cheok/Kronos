"""Round P5-6: Cross-model direction logits ensemble (R10 + R5 weighted avg).

P5-3 picks one model per horizon (h=1,10→R10; h=3,5→R5). But averaging
R10+R5 logits may reduce variance and improve direction accuracy / gate.

For each horizon, ensemble logits = w * R10_logits + (1-w) * R5_logits,
where both R10 and R5 use the same per-horizon TTA lookback from P5-3.

Sweep w_r10 in [0.0, 0.25, 0.5, 0.75, 1.0] per horizon. w=1.0 = pure R10
(baseline for h=1/10), w=0.0 = pure R5 (baseline for h=3/5).

Note: h=1 gate confidence is computed from the ensembled logits, so the
gate is affected. Re-sweep h=1 gate (thr × mag) at the new ensembled
distribution to find the best in-band config.
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
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
R10 = "r10_joint_splitlr"
R5 = "r5_frozen_pool48"

# Ensemble weight on R10 per horizon
ENSEMBLE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# h=1 gate re-sweep at ensembled distribution
THRESHOLDS = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
MAGS = [0.0, 0.002, 0.003]


def tta_logits_for(
    tta_store: dict, model: str, h: int, lookbacks: tuple[int, ...]
) -> np.ndarray:
    """TTA-averaged logits for given model/horizon/lookbacks."""
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
    return average_logits(per_lb, None)


def ensemble_logits(
    r10_logits: np.ndarray, r5_logits: np.ndarray, w_r10: float
) -> np.ndarray:
    """Weighted average of two logits arrays."""
    w = float(w_r10)
    return w * r10_logits + (1.0 - w) * r5_logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p5_r6_model_ensemble.json",
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

    # 1. Precompute TTA-averaged logits per (model, horizon) using P5-3 lookbacks
    print("\n=== Precomputing per-(model, h) TTA logits ===")
    r10_logits_by_h: dict[int, np.ndarray] = {}
    r5_logits_by_h: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        lookbacks = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[h]
        r10_logits_by_h[h] = tta_logits_for(tta_store, R10, h, lookbacks)
        r5_logits_by_h[h] = tta_logits_for(tta_store, R5, h, lookbacks)
        # Per-model nonflat for reference
        r10_nf = nonflat_accuracy(
            direction_confidence_from_logits(r10_logits_by_h[h])["hard_pred"],
            tta_store[R10][h]["td"],
        )
        r5_nf = nonflat_accuracy(
            direction_confidence_from_logits(r5_logits_by_h[h])["hard_pred"],
            tta_store[R5][h]["td"],
        )
        promoted = PROMOTED_MODEL_BY_HORIZON[h]
        print(
            f"  h={h} (lookbacks={lookbacks}): "
            f"R10 nf={r10_nf:.2%}, R5 nf={r5_nf:.2%}, promoted={promoted}"
        )

    # 2. P5-5 baseline (current promoted single-model per-h)
    print("\n=== P5-5 baseline (single-model per-h) ===")
    base_dir: dict[int, np.ndarray] = {}
    base_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = PROMOTED_MODEL_BY_HORIZON[h]
        logits = r10_logits_by_h[h] if dn == R10 else r5_logits_by_h[h]
        conf = direction_confidence_from_logits(logits)
        base_dir[h] = conf["hard_pred"]
        t_dir[h] = tta_store[dn][h]["td"]
        t_ret[h] = tta_store[dn][h]["tr"]
        base_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"],
            float(per_h_weights[h]),
        )
    base_summary = summarize_horizons(base_dir, base_ret, t_dir, t_ret, HORIZONS)
    # h=1 gate: P5-3 config
    h1_logits_base = r10_logits_by_h[1]  # h=1 promoted is R10
    conf1_base = direction_confidence_from_logits(h1_logits_base)
    h1_hard_base = conf1_base["hard_pred"]
    h1_score_base = conf1_base[PROMOTED_H1_GATE["confidence_key"]]
    h1_gate_ret = sl_store[primary][1]["pret"]
    h1_td_base = tta_store[R10][1]["td"]
    gated_base = apply_consistency_and_magnitude_gate(
        h1_hard_base, h1_score_base, h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    m_base = gated_actionable_metrics(gated_base, h1_td_base)
    base_summary["h1_gated_precision"] = m_base["precision_on_calls"]
    base_summary["h1_gated_nonflat"] = m_base["gated_nonflat_acc"]
    base_summary["h1_gated_coverage"] = m_base["coverage"]
    print(
        f"  nf={base_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={base_summary['return_mae_overall']:.6f} "
        f"g_prec={base_summary['h1_gated_precision']:.2%} "
        f"g_nf={base_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={base_summary['h1_gated_coverage']:.2%}"
    )

    # 3. Per-horizon ensemble sweep (direction-only effect, gate fixed at P5-3)
    print("\n=== Per-horizon ensemble weight sweep (gate fixed) ===")
    sweep: dict[str, list[dict]] = {}
    optimal: dict[int, float] = {}
    for h in HORIZONS:
        td = tta_store[R10][h]["td"]  # same target for both models (same windows)
        rows = []
        for w in ENSEMBLE_WEIGHTS:
            ens = ensemble_logits(r10_logits_by_h[h], r5_logits_by_h[h], w)
            conf = direction_confidence_from_logits(ens)
            nf = nonflat_accuracy(conf["hard_pred"], td)
            rows.append({
                "w_r10": w,
                "nonflat": nf,
                "delta_vs_base_pt": (nf - base_summary["by_horizon"][str(h)]["nonflat_accuracy"]) * 100,
            })
        sweep[str(h)] = rows
        # Pick max nonflat; tie-break toward simpler (extreme) weights
        best = max(rows, key=lambda r: (r["nonflat"], -abs(r["w_r10"] - 0.5)))
        optimal[h] = best["w_r10"]
        print(f"  h={h}: best w_r10={best['w_r10']} nf={best['nonflat']:.2%} (d={best['delta_vs_base_pt']:+.2f}pt)")
        for r in rows:
            print(f"    w_r10={r['w_r10']:.2f}: nf={r['nonflat']:.2%} (d={r['delta_vs_base_pt']:+.2f}pt)")

    print(f"\nPer-h ensemble optimal weights: {optimal}")

    # 4. Build combined ensemble config (gate fixed at P5-3 first)
    print("\n=== Combined ensemble (gate fixed at P5-3) ===")
    pred_dir: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        ens = ensemble_logits(r10_logits_by_h[h], r5_logits_by_h[h], optimal[h])
        conf = direction_confidence_from_logits(ens)
        pred_dir[h] = conf["hard_pred"]
    ens_summary = summarize_horizons(pred_dir, base_ret, t_dir, t_ret, HORIZONS)

    # h=1 gate: if h=1 weight changed, recompute gate inputs
    if optimal[1] != 1.0:
        h1_ens = ensemble_logits(r10_logits_by_h[1], r5_logits_by_h[1], optimal[1])
        conf1 = direction_confidence_from_logits(h1_ens)
        h1_hard = conf1["hard_pred"]
        h1_score = conf1[PROMOTED_H1_GATE["confidence_key"]]
    else:
        h1_hard = h1_hard_base
        h1_score = h1_score_base
    gated = apply_consistency_and_magnitude_gate(
        h1_hard, h1_score, h1_gate_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, h1_td_base)
    ens_summary["h1_gated_precision"] = m["precision_on_calls"]
    ens_summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    ens_summary["h1_gated_coverage"] = m["coverage"]
    print(
        f"  nf={ens_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={ens_summary['return_mae_overall']:.6f} "
        f"g_prec={ens_summary['h1_gated_precision']:.2%} "
        f"g_nf={ens_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={ens_summary['h1_gated_coverage']:.2%}"
    )

    # 5. If h=1 weight changed and gate is no longer in-band or worse,
    #    re-sweep h=1 gate (thr × mag) at the new ensembled distribution.
    gate_resweep: dict | None = None
    if optimal[1] != 1.0:
        print("\n=== Re-sweeping h=1 gate (thr × mag) at ensembled logits ===")
        rows = []
        for thr in THRESHOLDS:
            for mag in MAGS:
                g = apply_consistency_and_magnitude_gate(
                    h1_hard, h1_score, h1_gate_ret,
                    confidence_threshold=thr, min_abs_return=mag,
                    require_sign_agree=False,
                )
                mm = gated_actionable_metrics(g, h1_td_base)
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
            m_best = gated_actionable_metrics(g_best, h1_td_base)
            ens_summary["h1_gated_precision"] = m_best["precision_on_calls"]
            ens_summary["h1_gated_nonflat"] = m_best["gated_nonflat_acc"]
            ens_summary["h1_gated_coverage"] = m_best["coverage"]
            ens_summary["h1_gate_metrics"] = m_best
            ens_summary["h1_gate"] = {"thr": best["thr"], "mag": best["mag"]}
            print(
                f"\n  Updated ensemble summary: "
                f"nf={ens_summary['nonflat_accuracy_overall']:.2%} "
                f"mae={ens_summary['return_mae_overall']:.6f} "
                f"g_prec={ens_summary['h1_gated_precision']:.2%} "
                f"g_nf={ens_summary['h1_gated_nonflat']:.2%} "
                f"g_cov={ens_summary['h1_gated_coverage']:.2%}"
            )

    # 6. Dual-bar decision vs P5-5
    decision = dual_bar_decision(ens_summary, base_summary)
    print(f"\n=== Dual-bar decision (ensemble vs P5-5) ===")
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
        "baseline": "P5-5 (single-model per-h)",
        "p5_5_summary": base_summary,
        "sweep_per_h": sweep,
        "optimal_ensemble_weights": optimal,
        "ensemble_summary": ens_summary,
        "gate_resweep": gate_resweep,
        "decision": decision,
        "h1_gated_backtest": bt,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
