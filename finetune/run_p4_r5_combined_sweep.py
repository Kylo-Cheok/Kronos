"""Round P4-5: Combined sweep — sign-agreement + confidence keys + 2-way logit ensemble.

Three orthogonal angles tested on the cached TTA store:
1. Sign-agreement gate: require direction and return sign to agree (UP→ret>0, DOWN→ret<0)
2. Confidence key: actionable_score vs margin vs nonflat_prob vs max_prob
3. 2-way logit ensemble for h=1: blend R5+R10 logits with various weights

Base: P4-3 best (T=1.0, thr=0.45, mag=0.0, cov=20.8%, prec=70.0%).
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
from run_p4_r3_tta_sweep import CACHE_PATH, confidence_with_temperature, load_store
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

HORIZONS = [1, 3, 5, 10]


def blend_logits(a: np.ndarray, b: np.ndarray, w_a: float) -> np.ndarray:
    """Convex blend of two logit arrays."""
    return float(w_a) * a + (1.0 - float(w_a)) * b


def sweep_combined(
    store: dict,
    *,
    thresholds: list[float],
    conf_keys: list[str],
    sign_agree_options: list[bool],
    ensemble_weights: list[float],
    baseline_prec: float,
) -> list[dict]:
    """Sweep h=1 gate configs across all dimensions."""
    primary = PROMOTED_RETURN_BLEND["primary"]       # r10
    secondary = PROMOTED_RETURN_BLEND["secondary"]    # r5

    h1_logits_r10 = store[primary][1]["logits"]
    h1_logits_r5 = store[secondary][1]["logits"]
    h1_pret_r10 = store[primary][1]["pret"]
    h1_td = store[primary][1]["td"]

    rows: list[dict] = []
    for ens_w in ensemble_weights:
        # Blend logits for h=1 direction
        if ens_w >= 1.0:
            h1_logits = h1_logits_r10.copy()
        else:
            h1_logits = blend_logits(h1_logits_r10, h1_logits_r5, ens_w)

        conf = direction_confidence_from_logits(h1_logits)
        hard = conf["hard_pred"]

        for conf_key in conf_keys:
            score = conf[conf_key]
            for sign_agree in sign_agree_options:
                for thr in thresholds:
                    gated = apply_consistency_and_magnitude_gate(
                        hard,
                        score,
                        h1_pret_r10,
                        confidence_threshold=thr,
                        min_abs_return=0.0,
                        require_sign_agree=sign_agree,
                    )
                    m = gated_actionable_metrics(gated, h1_td)
                    if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                        continue
                    if m["n_calls"] < 8:
                        continue
                    rows.append({
                        "ens_w_r10": float(ens_w),
                        "conf_key": conf_key,
                        "sign_agree": bool(sign_agree),
                        "thr": float(thr),
                        "coverage": m["coverage"],
                        "precision": m["precision_on_calls"],
                        "gated_nf": m["gated_nonflat_acc"],
                        "n_calls": m["n_calls"],
                        "d_prec_pt": (m["precision_on_calls"] - baseline_prec) * 100,
                    })
    rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_p4_r5_combined_sweep.json")
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        print(f"Cache not found: {CACHE_PATH}. Run run_p4_r3_tta_sweep.py first.")
        return 1
    store = load_store(CACHE_PATH)
    n_windows = len(store[next(iter(store))][1]["td"])
    print(f"Loaded TTA store: {n_windows} windows")

    baseline_prec = 0.6364  # Phase-3 promoted
    p4_3_prec = 0.70         # P4-3 best
    print(f"Phase-3 promoted prec: {baseline_prec:.1%}")
    print(f"P4-3 best prec: {p4_3_prec:.1%}")

    thresholds = [0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
    conf_keys = ["actionable_score", "margin", "nonflat_prob", "max_prob"]
    sign_agree_options = [False, True]
    ensemble_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    total = len(ensemble_weights) * len(conf_keys) * len(sign_agree_options) * len(thresholds)
    print(f"\n=== Sweep: {len(ensemble_weights)} ens_w x {len(conf_keys)} conf_key x {len(sign_agree_options)} sign_agree x {len(thresholds)} thr = {total} combos ===")

    rows = sweep_combined(
        store,
        thresholds=thresholds,
        conf_keys=conf_keys,
        sign_agree_options=sign_agree_options,
        ensemble_weights=ensemble_weights,
        baseline_prec=baseline_prec,
    )
    print(f"In-band rows: {len(rows)}")
    print("\nTop 20 in-band configs (by precision, then gated_nf):")
    for r in rows[:20]:
        print(
            f"  ens_w={r['ens_w_r10']:.1f} key={r['conf_key']:18s} sa={str(r['sign_agree']):5s} thr={r['thr']}: "
            f"cov={r['coverage']:.1%} prec={r['precision']:.1%} "
            f"gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f} "
            f"d_prec(vs P3)={r['d_prec_pt']:+.2f}pt"
        )

    # Winners vs P4-3 (must beat P4-3's 70.0% precision)
    winners_vs_p43 = [r for r in rows if r["precision"] > p4_3_prec + 1e-9]
    print(f"\nBeating P4-3 (prec > {p4_3_prec:.1%}): {len(winners_vs_p43)}")
    for r in winners_vs_p43[:10]:
        print(
            f"  ens_w={r['ens_w_r10']:.1f} key={r['conf_key']:18s} sa={str(r['sign_agree']):5s} thr={r['thr']}: "
            f"cov={r['coverage']:.1%} prec={r['precision']:.1%} gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f}"
        )

    # Best overall
    best = rows[0] if rows else None
    if best is None:
        print("No in-band config found!")
        return 1

    # Full eval with best config
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])

    # Build h=1 blended logits if ensemble weight != 1.0
    if best["ens_w_r10"] >= 1.0:
        h1_logits = store[primary][1]["logits"]
    else:
        h1_logits = blend_logits(store[primary][1]["logits"], store[secondary][1]["logits"], best["ens_w_r10"])

    h1_conf = direction_confidence_from_logits(h1_logits)
    h1_hard = h1_conf["hard_pred"]
    h1_score = h1_conf[best["conf_key"]]
    h1_pret = store[primary][1]["pret"]
    h1_td = store[primary][1]["td"]
    h1_tr = store[primary][1]["tr"]

    gated_h1 = apply_consistency_and_magnitude_gate(
        h1_hard, h1_score, h1_pret,
        confidence_threshold=best["thr"],
        min_abs_return=0.0,
        require_sign_agree=best["sign_agree"],
    )
    gate_m = gated_actionable_metrics(gated_h1, h1_td)

    # Full multi-horizon summary (ungated for nonflat; gated only for h1 gate metrics)
    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = PROMOTED_MODEL_BY_HORIZON[h]
        conf = direction_confidence_from_logits(store[dn][h]["logits"])
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = store[dn][h]["td"]
        t_ret[h] = store[dn][h]["tr"]
        pred_ret[h] = blend_returns(store[primary][h]["pret"], store[secondary][h]["pret"], w_blend)

    best_summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))
    best_summary["h1_gated_precision"] = gate_m["precision_on_calls"]
    best_summary["h1_gated_nonflat"] = gate_m["gated_nonflat_acc"]
    best_summary["h1_gated_coverage"] = gate_m["coverage"]
    best_summary["h1_gate_metrics"] = gate_m
    best_summary["gate"] = {
        "confidence_key": best["conf_key"],
        "confidence_threshold": best["thr"],
        "min_abs_return": 0.0,
        "sign_agree": best["sign_agree"],
        "ens_w_r10": best["ens_w_r10"],
    }

    bt = absolute_direction_backtest(gated_h1, h1_tr, transaction_cost=0.0005)

    # Phase-3 baseline for dual_bar
    phase3_baseline_summary = {
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
    decision = dual_bar_decision(best_summary, phase3_baseline_summary)

    print(f"\n=== Best: ens_w={best['ens_w_r10']:.1f} key={best['conf_key']} sa={best['sign_agree']} thr={best['thr']} ===")
    print(
        f"  nf={best_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={best_summary['return_mae_overall']:.5f} "
        f"g_prec={best_summary['h1_gated_precision']:.2%} "
        f"g_nf={best_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={best_summary['h1_gated_coverage']:.2%}"
    )
    print(f"  h1 bt: ret={bt['total_return']:.2%} hit={bt['hit_rate']:.2%} n={bt['n_trades']:.0f}")
    print(f"  decision vs Phase-3: {decision['promote']} ({decision['reason']})")

    result = {
        "symbol": "688169",
        "n_windows": n_windows,
        "sweep": {
            "thresholds": thresholds,
            "conf_keys": conf_keys,
            "sign_agree_options": sign_agree_options,
            "ensemble_weights": ensemble_weights,
            "in_band_rows": len(rows),
            "beating_p43": len(winners_vs_p43),
            "top20": rows[:20],
            "winners_vs_p43": winners_vs_p43[:10],
        },
        "best_config": best,
        "best_promoted_summary": best_summary,
        "best_h1_backtest": bt,
        "decision_vs_phase3": decision,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
