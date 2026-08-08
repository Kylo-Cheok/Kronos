"""Round P5-4: Logit temperature scaling sweep on h=1.

P5-3 achieved gate prec=72.73%, gnf=92.31% at cov=22.9% with h=1=4lb_left
+ thr=0.45 + mag=0.002. Temperature scaling (logits / T before softmax) can
adjust the confidence-coverage tradeoff:
  - T<1: sharpen logits -> higher confidence -> higher coverage, lower precision
  - T>1: soften logits -> lower confidence -> lower coverage, higher precision

Sweep T in [0.8, 1.2] on h=1 only (h=3/5/10 stay at P5-3 optimal).
For each T, sweep thr to find best in-band config.
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
P5_3_LOOKBACKS = {
    1: (124, 126, 128, 130),
    3: (124, 126, 128, 130),
    5: (124, 126, 128, 130, 132, 134),
    10: (126, 128, 130, 132),
}
TEMPERATURES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
THRESHOLDS = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55]


def confidence_with_temperature(logits: np.ndarray, temp: float) -> dict:
    """Compute confidence scores with temperature scaling (logits / temp)."""
    scaled = logits / float(temp)
    return direction_confidence_from_logits(scaled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "eval_p5_r4_temp.json"
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

    # Precompute h=3/5/10 (fixed at P5-3 optimal, T=1.0)
    other_dir, other_ret, other_td, other_tr = {}, {}, {}, {}
    for h in HORIZONS:
        if h == 1:
            continue
        model = PROMOTED_MODEL_BY_HORIZON[h]
        lookbacks = P5_3_LOOKBACKS[h]
        lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
        per_lb = [tta_store[model][h]["per_lb_logits"][idx] for idx in lb_indices]
        avg_logits = average_logits(per_lb, None)
        conf = direction_confidence_from_logits(avg_logits)
        other_dir[h] = conf["hard_pred"]
        other_td[h] = tta_store[model][h]["td"]
        other_tr[h] = tta_store[model][h]["tr"]
        w = float(per_h_weights[h])
        other_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w
        )

    # Precompute h=1 raw logits (4lb_left)
    h1_model = PROMOTED_MODEL_BY_HORIZON[1]
    h1_lookbacks = P5_3_LOOKBACKS[1]
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in h1_lookbacks]
    per_lb = [tta_store[h1_model][1]["per_lb_logits"][idx] for idx in lb_indices]
    h1_raw_logits = average_logits(per_lb, None)
    h1_td = tta_store[h1_model][1]["td"]
    h1_tr = tta_store[h1_model][1]["tr"]
    gate_ret = sl_store[primary][1]["pret"]
    h1_ret = blend_returns(
        sl_store[primary][1]["pret"],
        sl_store[secondary][1]["pret"],
        float(per_h_weights[1]),
    )

    # P5-3 baseline (T=1.0, thr=0.45, mag=0.002)
    p5_3_conf = direction_confidence_from_logits(h1_raw_logits)
    p5_3_gated = apply_consistency_and_magnitude_gate(
        p5_3_conf["hard_pred"], p5_3_conf["actionable_score"], gate_ret,
        confidence_threshold=0.45, min_abs_return=0.002, require_sign_agree=False,
    )
    p5_3_m = gated_actionable_metrics(p5_3_gated, h1_td)
    p5_3_nf = nonflat_accuracy(p5_3_conf["hard_pred"], h1_td)
    print(
        f"\nP5-3 baseline (T=1.0, thr=0.45, mag=0.002): "
        f"nf={p5_3_nf:.2%} g_prec={p5_3_m['precision_on_calls']:.2%} "
        f"g_nf={p5_3_m['gated_nonflat_acc']:.2%} g_cov={p5_3_m['coverage']:.2%}"
    )

    # Sweep temperature × threshold
    print("\n=== Sweeping temperature × threshold (mag=0.002) ===")
    rows = []
    for temp in TEMPERATURES:
        conf = confidence_with_temperature(h1_raw_logits, temp)
        hard = conf["hard_pred"]
        score = conf["actionable_score"]
        nf = nonflat_accuracy(hard, h1_td)
        for thr in THRESHOLDS:
            gated = apply_consistency_and_magnitude_gate(
                hard, score, gate_ret,
                confidence_threshold=thr, min_abs_return=0.002,
                require_sign_agree=False,
            )
            m = gated_actionable_metrics(gated, h1_td)
            if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                continue
            if m["n_calls"] < 8:
                continue
            rows.append({
                "temp": temp,
                "thr": thr,
                "mag": 0.002,
                "ungated_nf": nf,
                "coverage": m["coverage"],
                "precision": m["precision_on_calls"],
                "gated_nf": m["gated_nonflat_acc"],
                "n_calls": m["n_calls"],
                "d_prec_pt": (m["precision_on_calls"] - p5_3_m["precision_on_calls"]) * 100,
                "d_gnf_pt": (m["gated_nonflat_acc"] - p5_3_m["gated_nonflat_acc"]) * 100,
                "d_nf_pt": (nf - p5_3_nf) * 100,
            })

    rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
    print(f"\n{len(rows)} in-band configs found")
    print("\nTop 10 by (precision, gated_nf):")
    for i, r in enumerate(rows[:10]):
        print(
            f"  {i+1}. T={r['temp']:.2f} thr={r['thr']:.2f}: "
            f"nf={r['ungated_nf']:.2%} cov={r['coverage']:.2%} "
            f"prec={r['precision']:.2%} gnf={r['gated_nf']:.2%} "
            f"(d_nf={r['d_nf_pt']:+.2f}pt d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt)"
        )

    # Check for direction_win via gate (prec+≥1pt AND gnf+≥1pt)
    gate_win = [r for r in rows if r["d_prec_pt"] >= 1.0 and r["d_gnf_pt"] >= 1.0]
    print(f"\n{len(gate_win)} configs with gate direction_win (prec+≥1pt AND gnf+≥1pt):")
    for r in gate_win[:5]:
        print(
            f"  T={r['temp']:.2f} thr={r['thr']:.2f}: "
            f"prec={r['precision']:.2%} gnf={r['gated_nf']:.2%} "
            f"(d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt)"
        )

    # Check for ungated nonflat improvement
    nf_improved = [r for r in rows if r["d_nf_pt"] >= 0.5]
    print(f"\n{len(nf_improved)} configs with ungated nonflat +≥0.5pt:")
    for r in nf_improved[:5]:
        print(
            f"  T={r['temp']:.2f} thr={r['thr']:.2f}: "
            f"nf={r['ungated_nf']:.2%} (d_nf={r['d_nf_pt']:+.2f}pt) "
            f"prec={r['precision']:.2%} gnf={r['gated_nf']:.2%}"
        )

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "p5_3_baseline": {
            "temp": 1.0, "thr": 0.45, "mag": 0.002,
            "ungated_nf": p5_3_nf,
            "precision": p5_3_m["precision_on_calls"],
            "gated_nf": p5_3_m["gated_nonflat_acc"],
            "coverage": p5_3_m["coverage"],
        },
        "sweep_rows": rows[:50],
        "gate_win_configs": gate_win[:10],
        "nf_improved_configs": nf_improved[:10],
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
