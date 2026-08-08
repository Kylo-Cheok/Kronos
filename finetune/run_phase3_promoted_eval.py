"""Evaluate Phase-3 promoted dual-metric ensemble vs Phase-2 baseline on 688169."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import (
    blend_returns,
    dual_bar_decision,
    h1_gate_metrics,
    summarize_horizons,
)
from evaluate_gated_ensemble import (
    HORIZONS,
    load_csv,
    load_model,
    make_windows,
    predict_window,
)
from promoted_config import (
    PHASE2_BASELINE_MODEL_BY_HORIZON,
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
    model_dir,
)
from selective_prediction import direction_confidence_from_logits


def collect(names: set[str], symbol: str = "688169") -> dict:
    loaded = {n: load_model(model_dir(n)) for n in names}
    windows = make_windows(load_csv(symbol))
    print(f"{len(windows)} windows, models={sorted(names)}")
    store = {
        n: {
            h: {"logits": [], "pret": [], "td": [], "tr": []}
            for h in HORIZONS
        }
        for n in names
    }
    for w in windows:
        for n, mod in loaded.items():
            pred = predict_window(mod, w)
            conf = direction_confidence_from_logits(pred["logits"])
            for hi, h in enumerate(HORIZONS):
                store[n][h]["logits"].append(pred["logits"][hi])
                store[n][h]["pret"].append(float(pred["pred_return"][hi]))
                store[n][h]["td"].append(int(pred["target_direction"][hi]))
                store[n][h]["tr"].append(float(pred["target_return"][hi]))
    for n in names:
        for h in HORIZONS:
            store[n][h]["logits"] = np.stack(store[n][h]["logits"])
            for k in ("pret", "td", "tr"):
                store[n][h][k] = np.asarray(store[n][h][k])
    return store


def build_summary(
    store: dict,
    dir_map: dict[int, str],
    *,
    blend: bool,
    gate: dict,
    gate_ret_primary: bool = False,
) -> dict:
    pred_dir = {}
    pred_ret = {}
    t_dir = {}
    t_ret = {}
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w = float(PROMOTED_RETURN_BLEND["primary_weight"])
    for h in HORIZONS:
        dn = dir_map[h]
        pred_dir[h] = direction_confidence_from_logits(store[dn][h]["logits"])[
            "hard_pred"
        ]
        t_dir[h] = store[dn][h]["td"]
        t_ret[h] = store[dn][h]["tr"]
        if blend:
            pred_ret[h] = blend_returns(store[primary][h]["pret"], store[secondary][h]["pret"], w)
        else:
            pred_ret[h] = store[dn][h]["pret"]
    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))
    h1_dir = dir_map[1]
    # Gate magnitude on primary raw return when requested (keeps 20-40% band)
    gate_pret = store[primary][1]["pret"] if gate_ret_primary else pred_ret[1]
    g = h1_gate_metrics(
        store[h1_dir][1]["logits"],
        gate_pret,
        t_dir[1],
        conf_thr=float(gate["confidence_threshold"]),
        min_abs_return=float(gate["min_abs_return"]),
        conf_key=str(gate["confidence_key"]),
    )
    summary["h1_gated_precision"] = g["precision_on_calls"]
    summary["h1_gated_nonflat"] = g["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = g["coverage"]
    summary["h1_gate"] = g
    summary["direction_mapping"] = {str(k): v for k, v in dir_map.items()}
    summary["return_blend"] = PROMOTED_RETURN_BLEND if blend else None
    summary["gate"] = gate
    summary["gate_magnitude_on_primary_return"] = bool(gate_ret_primary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="688169")
    args = parser.parse_args()

    names = set(PHASE2_BASELINE_MODEL_BY_HORIZON.values()) | set(
        PROMOTED_MODEL_BY_HORIZON.values()
    )
    store = collect(names, symbol=args.symbol)

    phase2_gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.003,
    }
    baseline = build_summary(
        store,
        PHASE2_BASELINE_MODEL_BY_HORIZON,
        blend=False,
        gate=phase2_gate,
        gate_ret_primary=True,
    )
    # intermediate: new mapping + blend, Phase-2 gate thresholds on primary ret
    mid = build_summary(
        store,
        PROMOTED_MODEL_BY_HORIZON,
        blend=True,
        gate=phase2_gate,
        gate_ret_primary=True,
    )
    promoted = build_summary(
        store,
        PROMOTED_MODEL_BY_HORIZON,
        blend=True,
        gate=PROMOTED_H1_GATE,
        gate_ret_primary=True,
    )

    d_mid = dual_bar_decision(mid, baseline)
    d_prom = dual_bar_decision(promoted, baseline)

    print("=== Phase-2 baseline ===")
    print(
        f"nf={baseline['nonflat_accuracy_overall']:.2%} mae={baseline['return_mae_overall']:.5f} "
        f"g_prec={baseline['h1_gated_precision']:.2%} g_nf={baseline['h1_gated_nonflat']:.2%} "
        f"cov={baseline['h1_gated_coverage']:.2%}"
    )
    for h in HORIZONS:
        r = baseline["by_horizon"][str(h)]
        print(f"  h={h}: nf={r['nonflat_accuracy']:.2%} mae={r['return_mae']:.5f}")

    print("=== Phase-3 mid (map+blend, old gate) ===")
    print(
        f"nf={mid['nonflat_accuracy_overall']:.2%} mae={mid['return_mae_overall']:.5f} "
        f"g_prec={mid['h1_gated_precision']:.2%} g_nf={mid['h1_gated_nonflat']:.2%} "
        f"cov={mid['h1_gated_coverage']:.2%} promote={d_mid['promote']} {d_mid['reason']}"
    )
    for h in HORIZONS:
        r = mid["by_horizon"][str(h)]
        print(f"  h={h}: nf={r['nonflat_accuracy']:.2%} mae={r['return_mae']:.5f}")

    print("=== Phase-3 promoted (map+blend+gate retune) ===")
    print(
        f"nf={promoted['nonflat_accuracy_overall']:.2%} mae={promoted['return_mae_overall']:.5f} "
        f"g_prec={promoted['h1_gated_precision']:.2%} g_nf={promoted['h1_gated_nonflat']:.2%} "
        f"cov={promoted['h1_gated_coverage']:.2%} promote={d_prom['promote']} {d_prom['reason']}"
    )
    for h in HORIZONS:
        r = promoted["by_horizon"][str(h)]
        print(f"  h={h}: nf={r['nonflat_accuracy']:.2%} mae={r['return_mae']:.5f}")

    out = {
        "symbol": args.symbol,
        "phase2_baseline": baseline,
        "phase3_mid_blend": mid,
        "phase3_promoted": promoted,
        "decision_mid": d_mid,
        "decision_promoted": d_prom,
        "promoted": bool(d_prom["promote"] or d_mid["promote"]),
        "promoted_config": {
            "direction_mapping": PROMOTED_MODEL_BY_HORIZON,
            "return_blend": PROMOTED_RETURN_BLEND,
            "h1_gate": PROMOTED_H1_GATE,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
