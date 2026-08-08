"""Round P4-4: Multi-horizon consistency gate on TTA logits.

Idea: filter h=1 trades when short and medium-horizon directions disagree.
A call is kept only if h=1 and h=3 (and optionally h=5) predict the same
non-FLAT direction. This should boost precision at the cost of coverage.

Base gate: P4-3 best (T=1.0, thr=0.45, mag=0.0, cov=20.8%, prec=70.0%).
Sweep: consistency modes x thresholds to find best in-band [0.20, 0.40].
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
from run_p4_r3_tta_sweep import (
    CACHE_PATH,
    confidence_with_temperature,
    evaluate_config,
    load_store,
)
from selective_prediction import (
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = [1, 3, 5, 10]


def apply_mh_consistency_gate(
    h1_hard: np.ndarray,
    h1_conf: np.ndarray,
    h1_ret: np.ndarray,
    h3_hard: np.ndarray,
    h5_hard: np.ndarray | None,
    *,
    thr: float,
    mag: float,
    mode: str,
) -> np.ndarray:
    """Gate h=1 calls with confidence + magnitude + multi-horizon agreement.

    Modes:
    - "none": base gate (conf + mag only)
    - "h1_h3_agree": keep only if h1 and h3 both non-FLAT and same direction
    - "h1_h3_any": keep only if h3 is non-FLAT (h3 confirms a move exists)
    - "h1_h3h5_agree": keep only if h1, h3, h5 all non-FLAT and same direction
    """
    gated = h1_hard.copy()
    keep = h1_conf >= float(thr)
    if mag > 0.0:
        keep = keep & (np.abs(h1_ret) >= float(mag))

    if mode == "h1_h3_agree":
        # Both must be non-FLAT and same direction
        both_nonflat = (h1_hard != FLAT_CLASS) & (h3_hard != FLAT_CLASS)
        same_dir = h1_hard == h3_hard
        keep = keep & both_nonflat & same_dir
    elif mode == "h1_h3_any":
        # h3 must be non-FLAT (confirms a directional move exists medium-term)
        keep = keep & (h3_hard != FLAT_CLASS)
    elif mode == "h1_h3h5_agree":
        if h5_hard is None:
            raise ValueError("h5_hard required for h1_h3h5_agree mode")
        all_nonflat = (h1_hard != FLAT_CLASS) & (h3_hard != FLAT_CLASS) & (h5_hard != FLAT_CLASS)
        same_dir = (h1_hard == h3_hard) & (h1_hard == h5_hard)
        keep = keep & all_nonflat & same_dir
    elif mode == "h1_h3_not_oppose":
        # Keep unless h3 explicitly disagrees (opposite direction)
        # h3 FLAT is OK; h3 same direction is OK; h3 opposite is not
        h3_opposite = (h3_hard != FLAT_CLASS) & (h3_hard != h1_hard) & (h1_hard != FLAT_CLASS)
        keep = keep & ~h3_opposite

    gated[~keep] = FLAT_CLASS
    return gated


def sweep_mh(
    store: dict,
    *,
    base_thr: float,
    base_mag: float,
    temperatures: list[float],
    thresholds: list[float],
    modes: list[str],
    baseline_metrics: dict,
) -> list[dict]:
    """Sweep multi-horizon consistency modes x thresholds."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    h1_logits = store[primary][1]["logits"]
    h1_pret = store[primary][1]["pret"]
    h1_td = store[primary][1]["td"]

    # h=3 direction from promoted mapping (R10 for h=3)
    h3_model = PROMOTED_MODEL_BY_HORIZON[3]
    h5_model = PROMOTED_MODEL_BY_HORIZON[5]

    rows: list[dict] = []
    for temp in temperatures:
        h1_conf_obj = confidence_with_temperature(h1_logits, temp)
        h1_hard = h1_conf_obj["hard_pred"]
        h1_score = h1_conf_obj[PROMOTED_H1_GATE["confidence_key"]]

        h3_conf_obj = confidence_with_temperature(store[h3_model][3]["logits"], temp)
        h3_hard = h3_conf_obj["hard_pred"]

        h5_conf_obj = confidence_with_temperature(store[h5_model][5]["logits"], temp)
        h5_hard = h5_conf_obj["hard_pred"]

        for mode in modes:
            for thr in thresholds:
                for mag in [0.0, 0.002, 0.003]:
                    gated = apply_mh_consistency_gate(
                        h1_hard, h1_score, h1_pret, h3_hard, h5_hard,
                        thr=thr, mag=mag, mode=mode,
                    )
                    m = gated_actionable_metrics(gated, h1_td)
                    if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                        continue
                    if m["n_calls"] < 8:
                        continue
                    rows.append({
                        "temperature": float(temp),
                        "mode": mode,
                        "thr": float(thr),
                        "mag": float(mag),
                        "coverage": m["coverage"],
                        "precision": m["precision_on_calls"],
                        "gated_nf": m["gated_nonflat_acc"],
                        "n_calls": m["n_calls"],
                        "d_prec_pt": (m["precision_on_calls"] - baseline_metrics["precision_on_calls"]) * 100,
                        "d_gnf_pt": (m["gated_nonflat_acc"] - baseline_metrics["gated_nonflat_acc"]) * 100,
                    })
    rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_p4_r4_mh_consistency.json")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args()

    if not args.cache.exists():
        print(f"Cache not found: {args.cache}. Run run_p4_r3_tta_sweep.py first.")
        return 1
    store = load_store(args.cache)
    n_windows = len(store[next(iter(store))][1]["td"])
    print(f"Loaded TTA store: {n_windows} windows")

    # P4-3 baseline: T=1.0, thr=0.45, mag=0.0 (best in-band from P4-3)
    p4_3_base_metrics = {"precision_on_calls": 0.70, "gated_nonflat_acc": 0.913}
    print(f"P4-3 baseline: prec={p4_3_base_metrics['precision_on_calls']:.1%} gnf={p4_3_base_metrics['gated_nonflat_acc']:.1%}")

    # Also compare against Phase-3 promoted baseline (the real promotion target)
    phase3_metrics = {"precision_on_calls": 0.6364, "gated_nonflat_acc": 0.84}
    print(f"Phase-3 promoted: prec={phase3_metrics['precision_on_calls']:.1%} gnf={phase3_metrics['gated_nonflat_acc']:.1%}")

    # Sweep
    temperatures = [0.8, 0.9, 1.0, 1.1]
    thresholds = [0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48]
    modes = ["none", "h1_h3_not_oppose", "h1_h3_any", "h1_h3_agree", "h1_h3h5_agree"]

    print(f"\n=== Sweep: {len(modes)} modes x {len(temperatures)} temp x {len(thresholds)} thr x 3 mag ===")
    rows = sweep_mh(
        store,
        base_thr=0.45,
        base_mag=0.0,
        temperatures=temperatures,
        thresholds=thresholds,
        modes=modes,
        baseline_metrics=p4_3_base_metrics,
    )
    print(f"In-band rows: {len(rows)}")
    print("\nTop 15 in-band configs (by precision, then gated_nf):")
    for r in rows[:15]:
        print(
            f"  {r['mode']:20s} T={r['temperature']} thr={r['thr']} mag={r['mag']}: "
            f"cov={r['coverage']:.1%} prec={r['precision']:.1%} "
            f"gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f} "
            f"d_prec(vs P43)={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt"
        )

    # Winners vs Phase-3 promoted baseline (the real target)
    winners_vs_phase3 = [
        r for r in rows
        if (r["precision"] - phase3_metrics["precision_on_calls"]) * 100 >= 1.0
        or (r["gated_nf"] - phase3_metrics["gated_nonflat_acc"]) * 100 >= 1.0
    ]
    print(f"\nIn-band +>=1pt winners vs Phase-3: {len(winners_vs_phase3)}")
    for r in winners_vs_phase3[:8]:
        d_p3_prec = (r["precision"] - phase3_metrics["precision_on_calls"]) * 100
        d_p3_gnf = (r["gated_nf"] - phase3_metrics["gated_nonflat_acc"]) * 100
        print(
            f"  {r['mode']:20s} T={r['temperature']} thr={r['thr']} mag={r['mag']}: "
            f"cov={r['coverage']:.1%} prec={r['precision']:.1%} gnf={r['gated_nf']:.1%} "
            f"d_prec(vs P3)={d_p3_prec:+.2f}pt d_gnf(vs P3)={d_p3_gnf:+.2f}pt"
        )

    # Best by precision (or best winner vs Phase-3)
    best = winners_vs_phase3[0] if winners_vs_phase3 else (rows[0] if rows else None)
    if best is None:
        print("No in-band config found!")
        return 1

    print(f"\n=== Best: {best['mode']} T={best['temperature']} thr={best['thr']} mag={best['mag']} ===")

    # Full eval: use promoted mapping + blend + TTA + MH consistency gate
    # We need to compute the full multi-horizon summary with the MH gate applied to h=1
    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])
    for h in HORIZONS:
        dn = PROMOTED_MODEL_BY_HORIZON[h]
        conf = confidence_with_temperature(store[dn][h]["logits"], best["temperature"])
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = store[dn][h]["td"]
        t_ret[h] = store[dn][h]["tr"]
        pred_ret[h] = blend_returns(store[primary][h]["pret"], store[secondary][h]["pret"], w_blend)

    # Apply MH consistency gate to h=1
    h1_conf_obj = confidence_with_temperature(store[primary][1]["logits"], best["temperature"])
    h3_conf_obj = confidence_with_temperature(store[PROMOTED_MODEL_BY_HORIZON[3]][3]["logits"], best["temperature"])
    h5_conf_obj = confidence_with_temperature(store[PROMOTED_MODEL_BY_HORIZON[5]][5]["logits"], best["temperature"])
    gated_h1 = apply_mh_consistency_gate(
        h1_conf_obj["hard_pred"],
        h1_conf_obj[PROMOTED_H1_GATE["confidence_key"]],
        store[primary][1]["pret"],
        h3_conf_obj["hard_pred"],
        h5_conf_obj["hard_pred"],
        thr=best["thr"], mag=best["mag"], mode=best["mode"],
    )
    pred_dir[1] = gated_h1  # override h=1 with gated version

    best_summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))
    m = gated_actionable_metrics(gated_h1, t_dir[1])
    best_summary["h1_gated_precision"] = m["precision_on_calls"]
    best_summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    best_summary["h1_gated_coverage"] = m["coverage"]
    best_summary["h1_gate_metrics"] = m
    best_summary["gate"] = {
        "confidence_key": PROMOTED_H1_GATE["confidence_key"],
        "confidence_threshold": best["thr"],
        "min_abs_return": best["mag"],
        "temperature": best["temperature"],
        "mh_mode": best["mode"],
        "require_sign_agree": False,
        "transaction_cost": PROMOTED_H1_GATE["transaction_cost"],
    }
    bt = absolute_direction_backtest(gated_h1, t_ret[1], transaction_cost=float(PROMOTED_H1_GATE["transaction_cost"]))

    # Build a simple Phase-3-like baseline summary for dual_bar
    # (without TTA, without MH gate — use the actual Phase-3 promoted numbers)
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

    print(
        f"\nBest promoted (MH={best['mode']}, T={best['temperature']}, thr={best['thr']}, mag={best['mag']}):"
    )
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
        "p4_3_baseline": p4_3_base_metrics,
        "phase3_promoted": phase3_metrics,
        "sweep": {
            "modes": modes,
            "temperatures": temperatures,
            "thresholds": thresholds,
            "in_band_rows": len(rows),
            "plus1pt_winners_vs_phase3": len(winners_vs_phase3),
            "top15": rows[:15],
            "winners_vs_phase3": winners_vs_phase3[:10],
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
