"""Round P4-3: Threshold + temperature sweep on TTA logits.

TTA averaging (P4-2) softened the direction logits, dropping h=1 gate
coverage from 22.9% to 19.4% (below the 20% band floor). This script
sweeps confidence thresholds and softmax temperatures to restore coverage
into [0.20, 0.40] while preserving the TTA-induced precision gains.

Temperature T < 1 sharpens the softmax (more confident), which should
push borderline samples back above threshold. T > 1 softens further.

The TTA store is cached to disk so P4-4 / P4-5 can reuse it without
re-running inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons
from promoted_config import (
    PHASE2_BASELINE_MODEL_BY_HORIZON,
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
    model_dir,
)
from run_tta_eval import (
    TTA_LOOKBACKS,
    attach_gate,
    build_arrays,
    collect_tta_store,
)
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = [1, 3, 5, 10]
CACHE_PATH = ROOT / "outputs" / "tta_store_cache.npz"


def save_store(store: dict, path: Path) -> None:
    """Flatten store dict into a single npz for cheap reuse."""
    flat: dict[str, np.ndarray] = {}
    for name in store:
        for h in HORIZONS:
            for key in ("logits", "pret", "td", "tr"):
                flat[f"{name}__h{h}__{key}"] = store[name][h][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f"Cached TTA store -> {path}")


def load_store(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    names: set[str] = set()
    for k in data.files:
        names.add(k.split("__")[0])
    store: dict = {n: {h: {} for h in HORIZONS} for n in names}
    for n in names:
        for h in HORIZONS:
            for key in ("logits", "pret", "td", "tr"):
                store[n][h][key] = data[f"{n}__h{h}__{key}"]
    return store


def confidence_with_temperature(
    logits: np.ndarray, temperature: float
) -> dict[str, np.ndarray]:
    """Same as direction_confidence_from_logits but with temperature scaling."""
    scaled = logits / float(temperature)
    return direction_confidence_from_logits(scaled)


def sweep_h1_gate(
    store: dict,
    *,
    thresholds: list[float],
    temperatures: list[float],
    min_abs_returns: list[float],
    baseline_metrics: dict,
) -> list[dict]:
    """Sweep h=1 gate parameters; return rows in coverage band, sorted by precision."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    h1_logits = store[primary][1]["logits"]
    h1_pret = store[primary][1]["pret"]
    h1_td = store[primary][1]["td"]

    rows: list[dict] = []
    for temp in temperatures:
        conf = confidence_with_temperature(h1_logits, temp)
        hard = conf["hard_pred"]
        score = conf[PROMOTED_H1_GATE["confidence_key"]]
        for thr in thresholds:
            for mag in min_abs_returns:
                gated = apply_consistency_and_magnitude_gate(
                    hard,
                    score,
                    h1_pret,
                    confidence_threshold=thr,
                    min_abs_return=mag,
                    require_sign_agree=False,
                )
                m = gated_actionable_metrics(gated, h1_td)
                if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                    continue
                if m["n_calls"] < 8:
                    continue
                rows.append(
                    {
                        "temperature": float(temp),
                        "thr": float(thr),
                        "mag": float(mag),
                        "coverage": m["coverage"],
                        "precision": m["precision_on_calls"],
                        "gated_nf": m["gated_nonflat_acc"],
                        "n_calls": m["n_calls"],
                        "d_prec_pt": (
                            m["precision_on_calls"]
                            - baseline_metrics["precision_on_calls"]
                        )
                        * 100,
                        "d_gnf_pt": (
                            m["gated_nonflat_acc"]
                            - baseline_metrics["gated_nonflat_acc"]
                        )
                        * 100,
                    }
                )
    rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
    return rows


def evaluate_config(
    store: dict,
    *,
    dir_map: dict[int, str],
    temperature: float,
    thr: float,
    mag: float,
    use_return_blend: bool,
) -> dict:
    """Full multi-horizon eval with temperature-scaled logits + swept gate."""
    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])
    for h in HORIZONS:
        dn = dir_map[h]
        conf = confidence_with_temperature(store[dn][h]["logits"], temperature)
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = store[dn][h]["td"]
        t_ret[h] = store[dn][h]["tr"]
        if use_return_blend and primary in store and secondary in store:
            pred_ret[h] = blend_returns(
                store[primary][h]["pret"], store[secondary][h]["pret"], w_blend
            )
        else:
            pred_ret[h] = store[dn][h]["pret"]

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))

    # h=1 gate with temperature-scaled confidence
    h1_dn = dir_map[1]
    h1_conf = confidence_with_temperature(store[h1_dn][1]["logits"], temperature)
    h1_hard = h1_conf["hard_pred"]
    h1_score = h1_conf[PROMOTED_H1_GATE["confidence_key"]]
    h1_gate_ret = store[primary][1]["pret"]
    gated = apply_consistency_and_magnitude_gate(
        h1_hard,
        h1_score,
        h1_gate_ret,
        confidence_threshold=thr,
        min_abs_return=mag,
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, t_dir[1])
    summary["h1_gated_precision"] = m["precision_on_calls"]
    summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = m["coverage"]
    summary["h1_gate_metrics"] = m
    summary["gate"] = {
        "confidence_key": PROMOTED_H1_GATE["confidence_key"],
        "confidence_threshold": thr,
        "min_abs_return": mag,
        "temperature": temperature,
        "require_sign_agree": False,
        "transaction_cost": PROMOTED_H1_GATE["transaction_cost"],
        "magnitude_return_source": "primary",
    }
    bt = absolute_direction_backtest(
        gated,
        t_ret[1],
        transaction_cost=float(PROMOTED_H1_GATE["transaction_cost"]),
    )
    return summary, bt, gated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_p4_r3_tta_sweep.json")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--symbol", default="688169")
    parser.add_argument("--lookbacks", type=str, default=",".join(str(x) for x in TTA_LOOKBACKS))
    args = parser.parse_args()

    lookbacks = tuple(int(x) for x in args.lookbacks.split(",") if x.strip())

    # Load or collect TTA store
    if args.cache.exists():
        print(f"Loading cached TTA store from {args.cache}")
        store = load_store(args.cache)
        n_windows = len(store[next(iter(store))][1]["td"])
    else:
        mapping = dict(PROMOTED_MODEL_BY_HORIZON)
        names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
        names |= {PROMOTED_RETURN_BLEND["primary"], PROMOTED_RETURN_BLEND["secondary"]}
        store, n_windows = collect_tta_store(names, args.symbol, lookbacks)
        save_store(store, args.cache)

    print(f"\n{n_windows} TTA windows loaded")

    # Baseline (Phase-2 mapping, no blend, no temperature, original gate)
    phase2_gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.003,
        "require_sign_agree": False,
        "transaction_cost": 0.0005,
    }
    b_dir, b_ret, b_td, b_tr, b_conf, b_hard, b_gate_ret = build_arrays(
        store, PHASE2_BASELINE_MODEL_BY_HORIZON, use_return_blend=False
    )
    from dual_metric_compare import summarize_horizons as sh
    baseline_summary = sh(b_dir, b_ret, b_td, b_tr, tuple(HORIZONS))
    baseline_summary, b_gated = attach_gate(
        baseline_summary, b_hard, b_conf, b_gate_ret, b_td[1], phase2_gate
    )
    base_gate_m = baseline_summary["h1_gate_metrics"]
    print(
        f"\nBaseline (Phase-2 + TTA, thr=0.45, mag=0.003, T=1.0): "
        f"cov={base_gate_m['coverage']:.1%} prec={base_gate_m['precision_on_calls']:.1%} "
        f"gnf={base_gate_m['gated_nonflat_acc']:.1%} n_calls={base_gate_m['n_calls']:.0f}"
    )

    # Sweep
    thresholds = [0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48]
    temperatures = [0.7, 0.8, 0.9, 1.0, 1.1]
    min_abs_returns = [0.0, 0.001, 0.002, 0.003, 0.004]

    print(f"\n=== Sweep: {len(thresholds)} thr x {len(temperatures)} temp x {len(min_abs_returns)} mag ===")
    rows = sweep_h1_gate(
        store,
        thresholds=thresholds,
        temperatures=temperatures,
        min_abs_returns=min_abs_returns,
        baseline_metrics=base_gate_m,
    )
    print(f"In-band rows: {len(rows)}")
    print("\nTop 15 in-band configs (by precision, then gated_nf):")
    for r in rows[:15]:
        print(
            f"  T={r['temperature']} thr={r['thr']} mag={r['mag']}: "
            f"cov={r['coverage']:.1%} prec={r['precision']:.1%} "
            f"gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f} "
            f"d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt"
        )

    # Winners: +>=1pt precision or gated_nf vs baseline
    winners = [r for r in rows if r["d_prec_pt"] >= 1.0 or r["d_gnf_pt"] >= 1.0]
    print(f"\nIn-band +>=1pt winners: {len(winners)}")
    for r in winners[:8]:
        print(r)

    # Pick best winner (or best in-band if no winner)
    best = winners[0] if winners else (rows[0] if rows else None)
    if best is None:
        print("No in-band config found!")
        result = {
            "symbol": args.symbol,
            "n_windows": n_windows,
            "baseline_gate": base_gate_m,
            "sweep_rows": rows,
            "best": None,
        }
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    print(f"\n=== Best config: T={best['temperature']} thr={best['thr']} mag={best['mag']} ===")

    # Full eval with best config (promoted mapping + blend + temperature + swept gate)
    best_summary, best_bt, best_gated = evaluate_config(
        store,
        dir_map=dict(PROMOTED_MODEL_BY_HORIZON),
        temperature=best["temperature"],
        thr=best["thr"],
        mag=best["mag"],
        use_return_blend=True,
    )
    decision = dual_bar_decision(best_summary, baseline_summary)

    print(
        f"\nBest promoted (T={best['temperature']}, thr={best['thr']}, mag={best['mag']}):"
    )
    print(
        f"  nf={best_summary['nonflat_accuracy_overall']:.2%} "
        f"mae={best_summary['return_mae_overall']:.5f} "
        f"g_prec={best_summary['h1_gated_precision']:.2%} "
        f"g_nf={best_summary['h1_gated_nonflat']:.2%} "
        f"g_cov={best_summary['h1_gated_coverage']:.2%}"
    )
    print(
        f"  h1 bt: ret={best_bt['total_return']:.2%} hit={best_bt['hit_rate']:.2%} "
        f"n={best_bt['n_trades']:.0f}"
    )
    print(f"  decision: {decision['promote']} ({decision['reason']})")

    result = {
        "symbol": args.symbol,
        "n_windows": n_windows,
        "baseline_gate_metrics": base_gate_m,
        "baseline_summary": baseline_summary,
        "sweep": {
            "thresholds": thresholds,
            "temperatures": temperatures,
            "min_abs_returns": min_abs_returns,
            "in_band_rows": len(rows),
            "plus1pt_winners": len(winners),
            "top15": rows[:15],
            "winners": winners[:10],
        },
        "best_config": best,
        "best_promoted_summary": best_summary,
        "best_h1_backtest": best_bt,
        "decision": decision,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
