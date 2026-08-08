"""Round P5-9: h=1 gate magnitude source alternatives.

P5-8 added strict_h5 consistency filter (gate prec 75.00%, gnf 96.00%).
P5-3 uses SL R10 raw |ret| (lb=128) for the magnitude filter (mag≥0.002).

Test alternative magnitude sources, all combined with strict_h5:
  1. sl_r10: SL R10 raw |ret| at h=1 (current P5-8 baseline)
  2. sl_r5: SL R5 raw |ret| at h=1
  3. sl_blend: SL blended |ret| at h=1 (0.925*R10 + 0.075*R5)
  4. sl_h5_r5: SL R5 raw |ret| at h=5 (cross-horizon magnitude)
  5. sl_h3_r5: SL R5 raw |ret| at h=3 (cross-horizon magnitude)

For each source, sweep mag threshold in [0.0, 0.001, 0.002, 0.003, 0.005].
Find best (source, mag) combo that maximizes prec & gnf in 20-40% band.
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
    UP_CLASS,
    DOWN_CLASS,
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
R10 = "r10_joint_splitlr"
R5 = "r5_frozen_pool48"

MAGS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]


def tta_logits_for(
    tta_store: dict, model: str, h: int, lookbacks: tuple[int, ...]
) -> np.ndarray:
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
    return average_logits(per_lb, None)


def apply_strict_h5(
    h1_gated: np.ndarray, h5_pred: np.ndarray
) -> np.ndarray:
    """Filter h=1 calls where h=5 doesn't predict same direction."""
    out = h1_gated.copy()
    called = h1_gated != FLAT_CLASS
    agree = (h5_pred == h1_gated) & called
    out[~agree] = FLAT_CLASS
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p5_r9_mag_source.json",
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

    # 1. Precompute direction logits + h=5 hard pred (for strict_h5)
    print("\n=== Precomputing per-h TTA logits ===")
    hard_by_h: dict[int, np.ndarray] = {}
    score_by_h: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    base_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        model = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[h]
        logits = tta_logits_for(tta_store, model, h, lookbacks)
        conf = direction_confidence_from_logits(logits)
        hard_by_h[h] = conf["hard_pred"]
        score_by_h[h] = conf[PROMOTED_H1_GATE["confidence_key"]]
        t_dir[h] = tta_store[model][h]["td"]
        t_ret[h] = tta_store[model][h]["tr"]
        base_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"],
            float(per_h_weights[h]),
        )

    # 2. Magnitude source candidates
    mag_sources = {
        "sl_r10_h1": sl_store[primary][1]["pret"],
        "sl_r5_h1": sl_store[secondary][1]["pret"],
        "sl_blend_h1": base_ret[1],
        "sl_h5_r5": sl_store[secondary][5]["pret"],   # h=5 promoted is R5
        "sl_h3_r5": sl_store[secondary][3]["pret"],   # h=3 promoted is R5
        "sl_h5_r10": sl_store[primary][5]["pret"],
        "sl_h3_r10": sl_store[primary][3]["pret"],
    }

    # 3. P5-8 baseline (sl_r10_h1 + mag=0.002 + strict_h5)
    print("\n=== P5-8 baseline (sl_r10 + mag=0.002 + strict_h5) ===")
    base_gated = apply_consistency_and_magnitude_gate(
        hard_by_h[1], score_by_h[1], mag_sources["sl_r10_h1"],
        confidence_threshold=0.45, min_abs_return=0.002,
        require_sign_agree=False,
    )
    base_gated = apply_strict_h5(base_gated, hard_by_h[5])
    m_base = gated_actionable_metrics(base_gated, t_dir[1])
    print(
        f"  cov={m_base['coverage']:.2%} prec={m_base['precision_on_calls']:.2%} "
        f"gnf={m_base['gated_nonflat_acc']:.2%} n={m_base['n_calls']:.0f}"
    )

    p5_8_dir = hard_by_h
    p5_8_summary = summarize_horizons(p5_8_dir, base_ret, t_dir, t_ret, HORIZONS)
    p5_8_summary["h1_gated_precision"] = m_base["precision_on_calls"]
    p5_8_summary["h1_gated_nonflat"] = m_base["gated_nonflat_acc"]
    p5_8_summary["h1_gated_coverage"] = m_base["coverage"]

    # 4. Sweep mag source × mag threshold (all with strict_h5)
    print("\n=== Sweep mag_source × mag_threshold (with strict_h5) ===")
    rows = []
    for src_name, mag_arr in mag_sources.items():
        for mag in MAGS:
            gated = apply_consistency_and_magnitude_gate(
                hard_by_h[1], score_by_h[1], mag_arr,
                confidence_threshold=0.45, min_abs_return=mag,
                require_sign_agree=False,
            )
            gated = apply_strict_h5(gated, hard_by_h[5])
            m = gated_actionable_metrics(gated, t_dir[1])
            if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                continue
            if m["n_calls"] < 8:
                continue
            rows.append({
                "source": src_name,
                "mag": mag,
                "cov": m["coverage"],
                "prec": m["precision_on_calls"],
                "gnf": m["gated_nonflat_acc"],
                "n_calls": m["n_calls"],
                "d_prec_pt": (m["precision_on_calls"] - m_base["precision_on_calls"]) * 100,
                "d_gnf_pt": (m["gated_nonflat_acc"] - m_base["gated_nonflat_acc"]) * 100,
            })

    rows.sort(key=lambda r: (r["prec"], r["gnf"]), reverse=True)
    print(f"\n{len(rows)} in-band configs found")
    print("\nTop 15 by (prec, gnf):")
    for i, r in enumerate(rows[:15]):
        print(
            f"  {i+1}. src={r['source']:14s} mag={r['mag']:.3f}: "
            f"cov={r['cov']:.2%} prec={r['prec']:.2%} gnf={r['gnf']:.2%} "
            f"(d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt) n={r['n_calls']:.0f}"
        )

    # 5. Find configs with prec+≥1pt AND gnf+≥1pt (direction_win via gate)
    gate_win = [r for r in rows if r["d_prec_pt"] >= 1.0 and r["d_gnf_pt"] >= 1.0]
    print(f"\n{len(gate_win)} configs with gate direction_win:")
    for r in gate_win[:5]:
        print(
            f"  src={r['source']:14s} mag={r['mag']:.3f}: "
            f"cov={r['cov']:.2%} prec={r['prec']:.2%} gnf={r['gnf']:.2%}"
        )

    # 6. Build best config
    if not rows:
        print("\nNo in-band configs found.")
        return 0
    best = rows[0]
    print(f"\n=== Best: src={best['source']} mag={best['mag']} ===")
    print(
        f"  cov={best['cov']:.2%} prec={best['prec']:.2%} gnf={best['gnf']:.2%} "
        f"(d_prec={best['d_prec_pt']:+.2f}pt d_gnf={best['d_gnf_pt']:+.2f}pt)"
    )

    # Rebuild gated array for backtest
    mag_arr = mag_sources[best["source"]]
    gated = apply_consistency_and_magnitude_gate(
        hard_by_h[1], score_by_h[1], mag_arr,
        confidence_threshold=0.45, min_abs_return=best["mag"],
        require_sign_agree=False,
    )
    gated = apply_strict_h5(gated, hard_by_h[5])
    m_best = gated_actionable_metrics(gated, t_dir[1])

    best_summary = summarize_horizons(hard_by_h, base_ret, t_dir, t_ret, HORIZONS)
    best_summary["h1_gated_precision"] = m_best["precision_on_calls"]
    best_summary["h1_gated_nonflat"] = m_best["gated_nonflat_acc"]
    best_summary["h1_gated_coverage"] = m_best["coverage"]
    best_summary["h1_gate_metrics"] = m_best
    best_summary["h1_gate"] = {
        "thr": 0.45, "mag": best["mag"],
        "mag_source": best["source"],
        "consistency": "strict_h5",
    }

    # 7. Dual-bar decision vs P5-8
    decision = dual_bar_decision(best_summary, p5_8_summary)
    print(f"\n=== Dual-bar decision (best vs P5-8) ===")
    print(f"  promote={decision['promote']} reason={decision['reason']}")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:+.4f} mae_delta={decision['mae_delta']:+.6f}")
    print(f"  gate_in_band={decision['gate_in_band']}")

    # 8. Backtest
    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    print(
        f"\nh=1 gated backtest: ret={bt['total_return']:.2%} "
        f"hit={bt['hit_rate']:.2%} n={bt['n_trades']:.0f}"
    )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "baseline": "P5-8 (sl_r10 + mag=0.002 + strict_h5)",
        "p5_8_summary": p5_8_summary,
        "sweep_rows": rows[:30],
        "gate_win_configs": gate_win[:10],
        "best": best,
        "best_summary": best_summary,
        "decision": decision,
        "h1_gated_backtest": bt,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
