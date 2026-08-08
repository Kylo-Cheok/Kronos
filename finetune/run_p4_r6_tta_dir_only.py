"""Round P4-6: TTA for direction only + single-lookback returns.

P4-3 found TTA improves h=1 gate precision (63.6%→70.0%) but slightly
worsens return MAE (0.03758→0.03770) because TTA averaging smooths return
predictions too. This script separates the two:

  - Direction (logits): TTA-averaged (from cached store)
  - Returns (MAE): single-lookback lb=128 (from standard inference)

This should preserve the direction gains while keeping MAE at Phase-3 level,
passing the dual_bar with direction_win via gate.
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
from run_p4_r3_tta_sweep import CACHE_PATH, load_store
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)


def collect_single_lookback(
    names: set[str], symbol: str
) -> tuple[dict, int]:
    """Run standard lb=128 inference; return store with logits + returns + targets."""
    loaded = {}
    for name in sorted(names):
        path = model_dir(name)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading {name} from {path}")
        loaded[name] = load_model(path)
    windows = make_windows(load_csv(symbol))
    print(f"{len(windows)} standard windows for {symbol}")
    store = {
        n: {h: {"logits": [], "pret": [], "td": [], "tr": []} for h in HORIZONS}
        for n in loaded
    }
    for i, w in enumerate(windows):
        if i % 40 == 0:
            print(f"  window {i}/{len(windows)} ({w['context_end_date']})")
        for n, mod in loaded.items():
            pred = predict_window(mod, w)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_p4_r6_tta_dir_only.json")
    args = parser.parse_args()

    # 1. Load TTA store (for direction logits)
    if not CACHE_PATH.exists():
        print(f"Cache not found: {CACHE_PATH}. Run run_p4_r3_tta_sweep.py first.")
        return 1
    tta_store = load_store(CACHE_PATH)
    n_windows_tta = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded TTA store: {n_windows_tta} windows")

    # 2. Run single-lookback inference (for returns)
    mapping = dict(PROMOTED_MODEL_BY_HORIZON)
    names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
    names |= {PROMOTED_RETURN_BLEND["primary"], PROMOTED_RETURN_BLEND["secondary"]}
    sl_store, n_windows_sl = collect_single_lookback(names, "688169")
    print(f"Single-lookback store: {n_windows_sl} windows")

    if n_windows_tta != n_windows_sl:
        print(f"WARNING: window count mismatch TTA={n_windows_tta} vs SL={n_windows_sl}")

    # 3. Build Phase-3 baseline (single-lookback, no TTA)
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])
    phase2_gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.003,
        "require_sign_agree": False,
        "transaction_cost": 0.0005,
    }

    b_dir: dict[int, np.ndarray] = {}
    b_ret: dict[int, np.ndarray] = {}
    b_td: dict[int, np.ndarray] = {}
    b_tr: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = PHASE2_BASELINE_MODEL_BY_HORIZON[h]
        conf = direction_confidence_from_logits(sl_store[dn][h]["logits"])
        b_dir[h] = conf["hard_pred"]
        b_td[h] = sl_store[dn][h]["td"]
        b_tr[h] = sl_store[dn][h]["tr"]
        b_ret[h] = sl_store[dn][h]["pret"]
    b_conf1 = direction_confidence_from_logits(sl_store[PHASE2_BASELINE_MODEL_BY_HORIZON[1]][1]["logits"])
    b_hard1 = b_conf1["hard_pred"]
    b_score1 = b_conf1[phase2_gate["confidence_key"]]
    b_gate_ret = sl_store[primary][1]["pret"]

    baseline = summarize_horizons(b_dir, b_ret, b_td, b_tr, HORIZONS)
    b_gated = apply_consistency_and_magnitude_gate(
        b_hard1, b_score1, b_gate_ret,
        confidence_threshold=phase2_gate["confidence_threshold"],
        min_abs_return=phase2_gate["min_abs_return"],
        require_sign_agree=False,
    )
    b_gate_m = gated_actionable_metrics(b_gated, b_td[1])
    baseline["h1_gated_precision"] = b_gate_m["precision_on_calls"]
    baseline["h1_gated_nonflat"] = b_gate_m["gated_nonflat_acc"]
    baseline["h1_gated_coverage"] = b_gate_m["coverage"]
    baseline["h1_gate_metrics"] = b_gate_m
    baseline["gate"] = phase2_gate

    print(
        f"\nPhase-2 baseline (single-lookback): "
        f"nf={baseline['nonflat_accuracy_overall']:.2%} "
        f"mae={baseline['return_mae_overall']:.5f} "
        f"g_prec={baseline['h1_gated_precision']:.2%} "
        f"g_nf={baseline['h1_gated_nonflat']:.2%} "
        f"g_cov={baseline['h1_gated_coverage']:.2%}"
    )

    # 4. Build TTA-direction + single-lookback-returns config
    # Direction: TTA logits from promoted mapping
    # Returns: single-lookback blended (0.85*R10 + 0.15*R5)
    # Gate: TTA logits for confidence, single-lookback R10 return for magnitude
    p_dir: dict[int, np.ndarray] = {}
    p_ret: dict[int, np.ndarray] = {}
    p_td: dict[int, np.ndarray] = {}
    p_tr: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = mapping[h]
        # TTA logits for direction
        conf = direction_confidence_from_logits(tta_store[dn][h]["logits"])
        p_dir[h] = conf["hard_pred"]
        # Single-lookback targets (should be same as TTA targets)
        p_td[h] = sl_store[dn][h]["td"]
        p_tr[h] = sl_store[dn][h]["tr"]
        # Single-lookback blended returns for MAE
        p_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w_blend
        )

    # h=1 gate: TTA confidence + single-lookback R10 return for magnitude
    h1_dn = mapping[1]
    p_conf1 = direction_confidence_from_logits(tta_store[h1_dn][1]["logits"])
    p_hard1 = p_conf1["hard_pred"]
    p_score1 = p_conf1[PROMOTED_H1_GATE["confidence_key"]]
    # Gate magnitude: use single-lookback R10 return (NOT TTA-averaged)
    p_gate_ret = sl_store[primary][1]["pret"]

    # P4-3 best gate: thr=0.45, mag=0.0 (no magnitude filter)
    p4_gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.0,
        "require_sign_agree": False,
        "transaction_cost": 0.0005,
        "magnitude_return_source": "single_lookback_primary",
        "direction_source": "tta_averaged",
        "return_source": "single_lookback_blend",
    }
    p_gated = apply_consistency_and_magnitude_gate(
        p_hard1, p_score1, p_gate_ret,
        confidence_threshold=p4_gate["confidence_threshold"],
        min_abs_return=p4_gate["min_abs_return"],
        require_sign_agree=False,
    )
    p_gate_m = gated_actionable_metrics(p_gated, p_td[1])

    promoted = summarize_horizons(p_dir, p_ret, p_td, p_tr, HORIZONS)
    promoted["h1_gated_precision"] = p_gate_m["precision_on_calls"]
    promoted["h1_gated_nonflat"] = p_gate_m["gated_nonflat_acc"]
    promoted["h1_gated_coverage"] = p_gate_m["coverage"]
    promoted["h1_gate_metrics"] = p_gate_m
    promoted["gate"] = p4_gate
    promoted["direction_mapping"] = {str(k): v for k, v in mapping.items()}
    promoted["return_blend"] = PROMOTED_RETURN_BLEND

    print(
        f"\nTTA-dir + SL-ret promoted: "
        f"nf={promoted['nonflat_accuracy_overall']:.2%} "
        f"mae={promoted['return_mae_overall']:.5f} "
        f"g_prec={promoted['h1_gated_precision']:.2%} "
        f"g_nf={promoted['h1_gated_nonflat']:.2%} "
        f"g_cov={promoted['h1_gated_coverage']:.2%}"
    )

    # 5. Decision vs Phase-2 baseline (same as run_promoted_eval)
    decision = dual_bar_decision(promoted, baseline)
    print(f"\nDecision vs Phase-2 baseline: {decision['promote']} ({decision['reason']})")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  dir_via_nonflat={decision['dir_via_nonflat']} dir_via_gate={decision['dir_via_gate']}")
    print(f"  gate_in_band={decision['gate_in_band']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:.4f} mae_delta={decision['mae_delta']:.6f}")

    # 6. Also compare vs Phase-3 promoted (the real promotion target)
    phase3_baseline = {
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
    decision_vs_p3 = dual_bar_decision(promoted, phase3_baseline)
    print(f"\nDecision vs Phase-3 promoted: {decision_vs_p3['promote']} ({decision_vs_p3['reason']})")
    print(f"  direction_win={decision_vs_p3['direction_win']} mae_win={decision_vs_p3['mae_win']}")
    print(f"  dir_via_nonflat={decision_vs_p3['dir_via_nonflat']} dir_via_gate={decision_vs_p3['dir_via_gate']}")
    print(f"  gate_in_band={decision_vs_p3['gate_in_band']}")
    print(f"  nonflat_delta={decision_vs_p3['nonflat_delta']:.4f} mae_delta={decision_vs_p3['mae_delta']:.6f}")

    # Backtests
    ungated_bt = absolute_direction_backtest(p_hard1, p_tr[1], transaction_cost=0.0005)
    gated_bt = absolute_direction_backtest(p_gated, p_tr[1], transaction_cost=0.0005)
    print(f"\nh=1 backtest (TTA-dir + SL-ret):")
    print(f"  ungated: ret={ungated_bt['total_return']:.2%} hit={ungated_bt['hit_rate']:.2%} n={ungated_bt['n_trades']:.0f}")
    print(f"  gated:   ret={gated_bt['total_return']:.2%} hit={gated_bt['hit_rate']:.2%} n={gated_bt['n_trades']:.0f}")

    result = {
        "symbol": "688169",
        "n_windows": n_windows_sl,
        "config": {
            "direction_source": "tta_averaged_logits",
            "return_source": "single_lookback_blend_0.85r10_0.15r5",
            "gate_magnitude_source": "single_lookback_r10",
            "gate": p4_gate,
            "direction_mapping": {str(k): v for k, v in mapping.items()},
        },
        "phase2_baseline": baseline,
        "tta_dir_sl_ret_promoted": promoted,
        "decision_vs_phase2": decision,
        "decision_vs_phase3": decision_vs_p3,
        "h1_backtests": {
            "ungated": ungated_bt,
            "gated": gated_bt,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
