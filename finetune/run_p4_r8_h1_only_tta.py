"""Round P4-8: h=1-only TTA + threshold sweep on 7-lb.

Two experiments:
1. h=1-only TTA: TTA-averaged logits for h=1 direction only, SL logits for
   h=3/5/10 direction. Tests whether TTA's nonflat improvement comes from h=1
   or all horizons. If h=1-only is as good, we can simplify inference.
2. Threshold sweep on 7-lb config: 7-lb had prec=70.37% but cov=18.75% (below
   band). Sweep thr to find best in-band point.

Caches per-lookback store to disk for reuse by future rounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons
from evaluate_gated_ensemble import load_csv, load_model, make_windows, predict_window
from promoted_config import (
    PHASE2_BASELINE_MODEL_BY_HORIZON,
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
    model_dir,
)
from run_p4_r7_expanded_tta import (
    ALL_LOOKBACKS,
    average_logits,
    collect_per_lb_store,
    collect_single_lookback,
)
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

HORIZONS = (1, 3, 5, 10)
PER_LB_CACHE = ROOT / "outputs" / "per_lb_store_cache.npz"


def save_per_lb_store(store: dict, path: Path, lookbacks: tuple[int, ...]) -> None:
    flat: dict[str, np.ndarray] = {}
    for name in store:
        for h in HORIZONS:
            for lb_idx, lb in enumerate(lookbacks):
                flat[f"{name}__h{h}__lb{lb_idx}__logits"] = store[name][h]["per_lb_logits"][lb_idx]
            flat[f"{name}__h{h}__td"] = store[name][h]["td"]
            flat[f"{name}__h{h}__tr"] = store[name][h]["tr"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f"Cached per-lookback store -> {path}")


def load_per_lb_store(path: Path, lookbacks: tuple[int, ...]) -> dict:
    data = np.load(path, allow_pickle=False)
    names: set[str] = set()
    for k in data.files:
        names.add(k.split("__")[0])
    store: dict = {n: {h: {"per_lb_logits": [], "td": None, "tr": None} for h in HORIZONS} for n in names}
    for n in names:
        for h in HORIZONS:
            for lb_idx in range(len(lookbacks)):
                store[n][h]["per_lb_logits"].append(data[f"{n}__h{h}__lb{lb_idx}__logits"])
            store[n][h]["td"] = data[f"{n}__h{h}__td"]
            store[n][h]["tr"] = data[f"{n}__h{h}__tr"]
    return store


def eval_h1_only_tta(
    tta_store: dict,
    sl_store: dict,
    lookbacks: tuple[int, ...],
    weights: np.ndarray | None,
    mapping: dict[int, str],
) -> dict:
    """TTA for h=1 direction, SL for h=3/5/10 direction. All returns SL blend."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]

    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = mapping[h]
        if h == 1:
            # TTA for h=1 direction
            per_lb = [tta_store[dn][h]["per_lb_logits"][idx] for idx in lb_indices]
            avg_logits = average_logits(per_lb, weights)
            conf = direction_confidence_from_logits(avg_logits)
            pred_dir[h] = conf["hard_pred"]
            t_dir[h] = tta_store[dn][h]["td"]
            t_ret[h] = tta_store[dn][h]["tr"]
        else:
            # SL for h=3/5/10 direction
            conf = direction_confidence_from_logits(sl_store[dn][h]["logits"])
            pred_dir[h] = conf["hard_pred"]
            t_dir[h] = sl_store[dn][h]["td"]
            t_ret[h] = sl_store[dn][h]["tr"]
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w_blend
        )

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)

    # h=1 gate: TTA confidence + SL R10 return
    h1_dn = mapping[1]
    per_lb_h1 = [tta_store[h1_dn][1]["per_lb_logits"][idx] for idx in lb_indices]
    avg_logits_h1 = average_logits(per_lb_h1, weights)
    conf1 = direction_confidence_from_logits(avg_logits_h1)
    hard1 = conf1["hard_pred"]
    score1 = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = sl_store[primary][1]["pret"]

    gated = apply_consistency_and_magnitude_gate(
        hard1, score1, gate_ret,
        confidence_threshold=0.45, min_abs_return=0.0, require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, t_dir[1])
    summary["h1_gated_precision"] = m["precision_on_calls"]
    summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = m["coverage"]
    summary["h1_gate_metrics"] = m
    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    return summary, bt


def sweep_threshold(
    tta_store: dict,
    sl_store: dict,
    lookbacks: tuple[int, ...],
    weights: np.ndarray | None,
    mapping: dict[int, str],
    thresholds: list[float],
    coverage_band: tuple[float, float] = (0.20, 0.40),
) -> list[dict]:
    """Sweep gate threshold for a given TTA config; return in-band rows."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    h1_dn = mapping[1]
    lb_indices = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb_h1 = [tta_store[h1_dn][1]["per_lb_logits"][idx] for idx in lb_indices]
    avg_logits_h1 = average_logits(per_lb_h1, weights)
    conf1 = direction_confidence_from_logits(avg_logits_h1)
    hard1 = conf1["hard_pred"]
    score1 = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = sl_store[primary][1]["pret"]
    t_dir1 = tta_store[h1_dn][1]["td"]
    t_ret1 = tta_store[h1_dn][1]["tr"]

    rows: list[dict] = []
    for thr in thresholds:
        gated = apply_consistency_and_magnitude_gate(
            hard1, score1, gate_ret,
            confidence_threshold=float(thr), min_abs_return=0.0, require_sign_agree=False,
        )
        m = gated_actionable_metrics(gated, t_dir1)
        if not (coverage_band[0] - 1e-9 <= m["coverage"] <= coverage_band[1] + 1e-9):
            continue
        if m["n_calls"] < 8:
            continue
        bt = absolute_direction_backtest(gated, t_ret1, transaction_cost=0.0005)
        rows.append({
            "thr": float(thr),
            "coverage": m["coverage"],
            "precision": m["precision_on_calls"],
            "gated_nf": m["gated_nonflat_acc"],
            "n_calls": m["n_calls"],
            "bt_return": bt["total_return"],
            "bt_hit": bt["hit_rate"],
        })
    rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_p4_r8_h1_only_tta.json")
    args = parser.parse_args()

    # 1. Load or collect per-lookback store
    mapping = dict(PROMOTED_MODEL_BY_HORIZON)
    names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
    names |= {PROMOTED_RETURN_BLEND["primary"], PROMOTED_RETURN_BLEND["secondary"]}

    if PER_LB_CACHE.exists():
        print(f"Loading cached per-lookback store from {PER_LB_CACHE}")
        tta_store = load_per_lb_store(PER_LB_CACHE, ALL_LOOKBACKS)
        n_windows = len(tta_store[next(iter(tta_store))][1]["td"])
    else:
        print(f"Collecting per-lookback store (lookbacks={ALL_LOOKBACKS})")
        tta_store, n_windows = collect_per_lb_store(names, "688169", ALL_LOOKBACKS)
        save_per_lb_store(tta_store, PER_LB_CACHE, ALL_LOOKBACKS)

    # 2. Collect single-lookback store
    print(f"\nCollecting single-lookback store (lb=128)")
    sl_store, _ = collect_single_lookback(names, "688169")

    # 3. Phase-2 baseline
    primary = PROMOTED_RETURN_BLEND["primary"]
    b_dir, b_ret, b_td, b_tr = {}, {}, {}, {}
    for h in HORIZONS:
        dn = PHASE2_BASELINE_MODEL_BY_HORIZON[h]
        conf = direction_confidence_from_logits(sl_store[dn][h]["logits"])
        b_dir[h] = conf["hard_pred"]
        b_td[h] = sl_store[dn][h]["td"]
        b_tr[h] = sl_store[dn][h]["tr"]
        b_ret[h] = sl_store[dn][h]["pret"]
    baseline = summarize_horizons(b_dir, b_ret, b_td, b_tr, HORIZONS)
    b_conf1 = direction_confidence_from_logits(sl_store[PHASE2_BASELINE_MODEL_BY_HORIZON[1]][1]["logits"])
    b_gated = apply_consistency_and_magnitude_gate(
        b_conf1["hard_pred"], b_conf1["actionable_score"], sl_store[primary][1]["pret"],
        confidence_threshold=0.45, min_abs_return=0.003, require_sign_agree=False,
    )
    b_m = gated_actionable_metrics(b_gated, b_td[1])
    baseline["h1_gated_precision"] = b_m["precision_on_calls"]
    baseline["h1_gated_nonflat"] = b_m["gated_nonflat_acc"]
    baseline["h1_gated_coverage"] = b_m["coverage"]
    print(
        f"\nPhase-2 baseline: nf={baseline['nonflat_accuracy_overall']:.2%} "
        f"mae={baseline['return_mae_overall']:.5f} "
        f"g_prec={baseline['h1_gated_precision']:.2%} g_nf={baseline['h1_gated_nonflat']:.2%}"
    )

    # === Experiment 1: h=1-only TTA vs all-horizon TTA ===
    print(f"\n{'='*60}")
    print("Experiment 1: h=1-only TTA vs all-horizon TTA (5-lb)")
    print(f"{'='*60}")

    lb5 = (124, 126, 128, 130, 132)
    h1_only_summary, h1_only_bt = eval_h1_only_tta(
        tta_store, sl_store, lb5, None, mapping
    )
    h1_only_decision = dual_bar_decision(h1_only_summary, baseline)
    print(
        f"\nh=1-only TTA (5-lb): "
        f"nf={h1_only_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={h1_only_summary['return_mae_overall']:.5f} "
        f"g_prec={h1_only_summary['h1_gated_precision']:.2%} "
        f"g_nf={h1_only_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={h1_only_summary['h1_gated_coverage']:.2%}"
    )
    print(f"  bt: ret={h1_only_bt['total_return']:.2%} hit={h1_only_bt['hit_rate']:.2%}")
    print(f"  decision: {h1_only_decision['promote']} ({h1_only_decision['reason']})")

    # Per-horizon comparison
    print("\nPer-horizon nonflat (h=1-only TTA vs all-horizon TTA):")
    for h in HORIZONS:
        h1_nf = h1_only_summary["by_horizon"][str(h)]["nonflat_accuracy"]
        print(f"  h={h}: h1-only={h1_nf:.2%}", end="")
        if h == 1:
            print(" (TTA)", end="")
        else:
            print(" (SL)", end="")
        print()

    # === Experiment 2: Threshold sweep on 7-lb config ===
    print(f"\n{'='*60}")
    print("Experiment 2: Threshold sweep on 7-lb config")
    print(f"{'='*60}")

    lb7 = (122, 124, 126, 128, 130, 132, 134)
    thresholds = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48]
    sweep_rows = sweep_threshold(
        tta_store, sl_store, lb7, None, mapping, thresholds
    )
    print(f"\n7-lb in-band configs ({len(sweep_rows)} rows):")
    for r in sweep_rows[:10]:
        print(
            f"  thr={r['thr']}: cov={r['coverage']:.1%} prec={r['precision']:.1%} "
            f"gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f} "
            f"bt_ret={r['bt_return']:.1%} bt_hit={r['bt_hit']:.1%}"
        )

    # Also sweep 5-lb for comparison
    sweep_5lb = sweep_threshold(
        tta_store, sl_store, lb5, None, mapping, thresholds
    )
    print(f"\n5-lb in-band configs ({len(sweep_5lb)} rows):")
    for r in sweep_5lb[:5]:
        print(
            f"  thr={r['thr']}: cov={r['coverage']:.1%} prec={r['precision']:.1%} "
            f"gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f} "
            f"bt_ret={r['bt_return']:.1%} bt_hit={r['bt_hit']:.1%}"
        )

    # Find overall best in-band config
    all_sweep = [
        ("7lb", r) for r in sweep_rows
    ] + [
        ("5lb", r) for r in sweep_5lb
    ]
    if all_sweep:
        best_sweep = max(all_sweep, key=lambda x: (x[1]["precision"], x[1]["gated_nf"]))
        print(f"\nBest in-band sweep: {best_sweep[0]} thr={best_sweep[1]['thr']} "
              f"prec={best_sweep[1]['precision']:.1%} gnf={best_sweep[1]['gated_nf']:.1%} "
              f"cov={best_sweep[1]['coverage']:.1%}")
    else:
        best_sweep = None

    out = {
        "symbol": "688169",
        "n_windows": n_windows,
        "phase2_baseline": baseline,
        "experiment1_h1_only_tta": {
            "summary": h1_only_summary,
            "backtest": h1_only_bt,
            "decision": h1_only_decision,
        },
        "experiment2_7lb_threshold_sweep": sweep_rows[:15],
        "experiment2_5lb_threshold_sweep": sweep_5lb[:10],
        "best_sweep": {"config": best_sweep[0], **best_sweep[1]} if best_sweep else None,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
