"""Evaluate the Phase-3 promoted ensemble + h=1 gate; write summary JSON.

Applies:
  - PROMOTED_MODEL_BY_HORIZON for direction hard labels
  - PROMOTED_RETURN_BLEND (0.85 R10 + 0.15 R5) for return endpoints / MAE
  - PROMOTED_H1_GATE for selective prediction (gate mag uses blended return)
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
from evaluate_gated_ensemble import load_csv, load_model, make_windows, predict_window
from promoted_config import (
    PHASE2_BASELINE_MODEL_BY_HORIZON,
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
    model_dir,
)
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)


def _collect(names: set[str], symbol: str) -> tuple[dict, int]:
    loaded = {}
    for name in sorted(names):
        path = model_dir(name)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading {name} from {path}")
        loaded[name] = load_model(path)
    windows = make_windows(load_csv(symbol))
    print(f"{len(windows)} windows for {symbol}")
    store = {
        n: {h: {"logits": [], "pret": [], "td": [], "tr": []} for h in HORIZONS}
        for n in loaded
    }
    for w in windows:
        cache = {n: predict_window(mod, w) for n, mod in loaded.items()}
        for n, pred in cache.items():
            conf = direction_confidence_from_logits(pred["logits"])
            for hi, h in enumerate(HORIZONS):
                store[n][h]["logits"].append(pred["logits"][hi])
                store[n][h]["pret"].append(float(pred["pred_return"][hi]))
                store[n][h]["td"].append(int(pred["target_direction"][hi]))
                store[n][h]["tr"].append(float(pred["target_return"][hi]))
    for n in loaded:
        for h in HORIZONS:
            store[n][h]["logits"] = np.stack(store[n][h]["logits"])
            for k in ("pret", "td", "tr"):
                store[n][h][k] = np.asarray(store[n][h][k])
    return store, len(windows)


def _build_arrays(
    store: dict,
    dir_map: dict[int, str],
    *,
    use_return_blend: bool,
) -> tuple[dict, dict, dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    per_h_weights = PROMOTED_RETURN_BLEND.get("primary_weight_by_horizon")
    default_w = float(PROMOTED_RETURN_BLEND["primary_weight"])
    for h in HORIZONS:
        dn = dir_map[h]
        conf = direction_confidence_from_logits(store[dn][h]["logits"])
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = store[dn][h]["td"]
        t_ret[h] = store[dn][h]["tr"]
        if use_return_blend and primary in store and secondary in store:
            w = float(per_h_weights[h]) if per_h_weights else default_w
            pred_ret[h] = blend_returns(
                store[primary][h]["pret"], store[secondary][h]["pret"], w
            )
        else:
            pred_ret[h] = store[dn][h]["pret"]
    # h1 conf score from direction model; gate magnitude uses primary raw return
    h1_dir = dir_map[1]
    conf1 = direction_confidence_from_logits(store[h1_dir][1]["logits"])
    conf_score = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = store[primary][1]["pret"] if primary in store else pred_ret[1]
    return pred_dir, pred_ret, t_dir, t_ret, conf_score, conf1["hard_pred"], gate_ret


def _attach_gate(
    summary: dict,
    hard: np.ndarray,
    conf: np.ndarray,
    pret_for_gate: np.ndarray,
    t_dir: np.ndarray,
    gate: dict,
) -> tuple[dict, np.ndarray]:
    gated = apply_consistency_and_magnitude_gate(
        hard,
        conf,
        pret_for_gate,
        confidence_threshold=float(gate["confidence_threshold"]),
        min_abs_return=float(gate["min_abs_return"]),
        require_sign_agree=bool(gate.get("require_sign_agree", False)),
    )
    m = gated_actionable_metrics(gated, t_dir)
    out = dict(summary)
    out["h1_gated_precision"] = m["precision_on_calls"]
    out["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    out["h1_gated_coverage"] = m["coverage"]
    out["h1_gate_metrics"] = m
    out["gate"] = gate
    return out, gated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="688169")
    parser.add_argument(
        "--override-h1-model",
        default=None,
        help="Optional exp name override for h=1 direction model",
    )
    args = parser.parse_args()

    mapping = dict(PROMOTED_MODEL_BY_HORIZON)
    if args.override_h1_model:
        mapping[1] = args.override_h1_model

    names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
    names |= {
        PROMOTED_RETURN_BLEND["primary"],
        PROMOTED_RETURN_BLEND["secondary"],
    }
    store, n_windows = _collect(names, args.symbol)

    # Phase-2 baseline (historical map, no blend, Phase-2 gate thr/mag)
    phase2_gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.003,
        "require_sign_agree": False,
        "transaction_cost": 0.0005,
    }
    b_dir, b_ret, b_td, b_tr, b_conf, b_hard, b_gate_ret = _build_arrays(
        store, PHASE2_BASELINE_MODEL_BY_HORIZON, use_return_blend=False
    )
    baseline = summarize_horizons(b_dir, b_ret, b_td, b_tr, HORIZONS)
    baseline, b_gated = _attach_gate(
        baseline, b_hard, b_conf, b_gate_ret, b_td[1], phase2_gate
    )
    baseline["direction_mapping"] = {
        str(k): v for k, v in PHASE2_BASELINE_MODEL_BY_HORIZON.items()
    }
    baseline["return_blend"] = None

    # Phase-3 promoted (new map + blend for MAE; primary raw ret for gate mag)
    p_dir, p_ret, p_td, p_tr, p_conf, p_hard, p_gate_ret = _build_arrays(
        store, mapping, use_return_blend=True
    )
    promoted = summarize_horizons(p_dir, p_ret, p_td, p_tr, HORIZONS)
    promoted, p_gated = _attach_gate(
        promoted, p_hard, p_conf, p_gate_ret, p_td[1], PROMOTED_H1_GATE
    )
    promoted["direction_mapping"] = {str(k): v for k, v in mapping.items()}
    promoted["return_blend"] = PROMOTED_RETURN_BLEND
    promoted["gate_magnitude_return_source"] = PROMOTED_H1_GATE.get(
        "magnitude_return_source", "primary"
    )

    decision = dual_bar_decision(promoted, baseline)

    # Backtests: positions from gate; PnL on blended h=1 realized targets
    ungated_bt = absolute_direction_backtest(
        p_hard, p_tr[1], transaction_cost=float(PROMOTED_H1_GATE["transaction_cost"])
    )
    gated_bt = absolute_direction_backtest(
        p_gated, p_tr[1], transaction_cost=float(PROMOTED_H1_GATE["transaction_cost"])
    )
    base_ungated_bt = absolute_direction_backtest(
        b_hard, b_tr[1], transaction_cost=float(phase2_gate["transaction_cost"])
    )
    base_gated_bt = absolute_direction_backtest(
        b_gated, b_tr[1], transaction_cost=float(phase2_gate["transaction_cost"])
    )

    result = {
        "symbol": args.symbol,
        "n_windows": n_windows,
        "phase2_baseline": baseline,
        "phase3_promoted": promoted,
        "decision": decision,
        "promoted_config": {
            "direction_mapping": {str(k): v for k, v in mapping.items()},
            "return_blend": PROMOTED_RETURN_BLEND,
            "h1_gate": PROMOTED_H1_GATE,
        },
        "h1_backtests": {
            "baseline_ungated": base_ungated_bt,
            "baseline_gated": base_gated_bt,
            "promoted_ungated": ungated_bt,
            "promoted_gated": gated_bt,
        },
        "acceptance": {
            "dual_bar_promote": decision["promote"],
            "dual_bar_reason": decision["reason"],
            "direction_win": decision["direction_win"],
            "mae_win": decision["mae_win"],
            "gate_in_band": decision["gate_in_band"],
            "ungated_nonflat_baseline": baseline["nonflat_accuracy_overall"],
            "ungated_nonflat_promoted": promoted["nonflat_accuracy_overall"],
            "nonflat_delta_pts": (
                promoted["nonflat_accuracy_overall"] - baseline["nonflat_accuracy_overall"]
            )
            * 100,
            "return_mae_baseline": baseline["return_mae_overall"],
            "return_mae_promoted": promoted["return_mae_overall"],
            "mae_rel_delta": (
                promoted["return_mae_overall"] - baseline["return_mae_overall"]
            )
            / baseline["return_mae_overall"],
            "h1_gated_coverage_promoted": promoted["h1_gated_coverage"],
            "h1_gated_precision_promoted": promoted["h1_gated_precision"],
            "h1_gated_nonflat_promoted": promoted["h1_gated_nonflat"],
            "coverage_band_required": [0.20, 0.40],
        },
    }

    print("=== Phase-2 baseline ===")
    print(
        f"nf={baseline['nonflat_accuracy_overall']:.2%} mae={baseline['return_mae_overall']:.5f} "
        f"g_cov={baseline['h1_gated_coverage']:.2%} g_prec={baseline['h1_gated_precision']:.2%} "
        f"g_nf={baseline['h1_gated_nonflat']:.2%}"
    )
    print("=== Phase-3 promoted (map+blend+gate) ===")
    print(
        f"nf={promoted['nonflat_accuracy_overall']:.2%} mae={promoted['return_mae_overall']:.5f} "
        f"g_cov={promoted['h1_gated_coverage']:.2%} g_prec={promoted['h1_gated_precision']:.2%} "
        f"g_nf={promoted['h1_gated_nonflat']:.2%}"
    )
    print("decision:", json.dumps(decision, indent=2))
    print("acceptance:", json.dumps(result["acceptance"], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
