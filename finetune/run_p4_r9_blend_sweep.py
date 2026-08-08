"""Round P4-9: Per-horizon return blend weight sweep.

P4-6 uses a fixed blend weight (0.85*R10 + 0.15*R5) for all horizons.
Different horizons may have different optimal blend weights — h=1 might
prefer pure R10 (lower MAE), while h=5/h=10 might benefit from more R5.

This script sweeps blend weights per horizon to minimize MAE while
preserving direction accuracy (direction comes from TTA logits, which
are blend-independent).

Sweep grid: [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0] per horizon.
For each horizon, find the weight that minimizes that horizon's MAE.
Then combine per-horizon optimal weights and evaluate the full config.

Also caches single-lookback store for reuse by future rounds.
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
)

HORIZONS = (1, 3, 5, 10)
SL_CACHE_PATH = ROOT / "outputs" / "sl_store_cache.npz"

# Blend weight grid: weight on R10 (primary)
BLEND_WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


def save_sl_store(store: dict, path: Path) -> None:
    """Cache single-lookback store to npz."""
    flat: dict[str, np.ndarray] = {}
    for name in store:
        for h in HORIZONS:
            for key in ("logits", "pret", "td", "tr"):
                flat[f"{name}__h{h}__{key}"] = store[name][h][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f"Cached single-lookback store -> {path}")


def load_sl_store(path: Path) -> dict:
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


def collect_single_lookback(names: set[str], symbol: str) -> tuple[dict, int]:
    """Standard lb=128 inference; return store with logits + returns + targets."""
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


def horizon_mae(pred_ret: np.ndarray, target_ret: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_ret - target_ret)))


def sweep_per_horizon(
    sl_store: dict,
    primary: str,
    secondary: str,
) -> dict:
    """For each horizon, sweep blend weight and find min-MAE point.

    Returns dict with per-horizon sweep results and optimal weights.
    """
    sweep: dict[str, list[dict]] = {}
    optimal: dict[int, float] = {}

    for h in HORIZONS:
        ret_r10 = sl_store[primary][h]["pret"]
        ret_r5 = sl_store[secondary][h]["pret"]
        target = sl_store[primary][h]["tr"]
        rows = []
        for w in BLEND_WEIGHTS:
            blended = blend_returns(ret_r10, ret_r5, w)
            mae = horizon_mae(blended, target)
            rows.append({
                "weight_r10": w,
                "mae": mae,
                "mae_rel_vs_pure_r10": (mae - horizon_mae(ret_r10, target)) / max(horizon_mae(ret_r10, target), 1e-12),
            })
        sweep[str(h)] = rows
        # Pick min MAE
        best = min(rows, key=lambda r: r["mae"])
        optimal[h] = best["weight_r10"]
        print(f"  h={h}: best w_r10={best['weight_r10']} mae={best['mae']:.6f} (pure_r10={horizon_mae(ret_r10, target):.6f}, pure_r5={horizon_mae(ret_r5, target):.6f})")

    return {"sweep": sweep, "optimal_weights": optimal}


def build_config(
    tta_store: dict,
    sl_store: dict,
    mapping: dict[int, str],
    blend_weights: dict[int, float],
    primary: str,
    secondary: str,
) -> dict:
    """Build full config with per-horizon blend weights."""
    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = mapping[h]
        conf = direction_confidence_from_logits(tta_store[dn][h]["logits"])
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = sl_store[dn][h]["td"]
        t_ret[h] = sl_store[dn][h]["tr"]
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"],
            sl_store[secondary][h]["pret"],
            blend_weights[h],
        )

    # h=1 gate: TTA confidence + SL R10 return for magnitude
    h1_dn = mapping[1]
    conf1 = direction_confidence_from_logits(tta_store[h1_dn][1]["logits"])
    hard1 = conf1["hard_pred"]
    score1 = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = sl_store[primary][1]["pret"]

    gate = {
        "confidence_key": "actionable_score",
        "confidence_threshold": 0.45,
        "min_abs_return": 0.0,
        "require_sign_agree": False,
        "transaction_cost": 0.0005,
    }
    gated = apply_consistency_and_magnitude_gate(
        hard1, score1, gate_ret,
        confidence_threshold=gate["confidence_threshold"],
        min_abs_return=gate["min_abs_return"],
        require_sign_agree=False,
    )
    gate_m = gated_actionable_metrics(gated, t_dir[1])

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)
    summary["h1_gated_precision"] = gate_m["precision_on_calls"]
    summary["h1_gated_nonflat"] = gate_m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = gate_m["coverage"]
    summary["h1_gate_metrics"] = gate_m
    summary["gate"] = gate
    summary["blend_weights"] = {str(k): v for k, v in blend_weights.items()}

    ungated_bt = absolute_direction_backtest(hard1, t_ret[1], transaction_cost=0.0005)
    gated_bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)

    return {
        "summary": summary,
        "ungated_bt": ungated_bt,
        "gated_bt": gated_bt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "eval_p4_r9_blend_sweep.json",
    )
    parser.add_argument("--symbol", default="688169")
    args = parser.parse_args()

    # 1. Load TTA store (direction logits)
    if not CACHE_PATH.exists():
        print(f"Cache not found: {CACHE_PATH}. Run run_p4_r3_tta_sweep.py first.")
        return 1
    tta_store = load_store(CACHE_PATH)
    n_windows_tta = len(tta_store[next(iter(tta_store))][1]["td"])
    print(f"Loaded TTA store: {n_windows_tta} windows")

    # 2. Load or collect single-lookback store (returns)
    if SL_CACHE_PATH.exists():
        print(f"Loading cached SL store from {SL_CACHE_PATH}")
        sl_store = load_sl_store(SL_CACHE_PATH)
        n_windows_sl = len(sl_store[next(iter(sl_store))][1]["td"])
    else:
        mapping = dict(PROMOTED_MODEL_BY_HORIZON)
        names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
        names |= {PROMOTED_RETURN_BLEND["primary"], PROMOTED_RETURN_BLEND["secondary"]}
        sl_store, n_windows_sl = collect_single_lookback(names, args.symbol)
        save_sl_store(sl_store, SL_CACHE_PATH)
    print(f"Single-lookback store: {n_windows_sl} windows")

    primary = PROMOTED_RETURN_BLEND["primary"]   # r10
    secondary = PROMOTED_RETURN_BLEND["secondary"]  # r5

    # 3. Per-horizon blend sweep
    print("\n=== Per-horizon blend weight sweep ===")
    sweep_result = sweep_per_horizon(sl_store, primary, secondary)
    optimal_weights = sweep_result["optimal_weights"]
    print(f"\nOptimal per-horizon weights: {optimal_weights}")

    # 4. Build configs: current (0.85 all), optimal (per-horizon), pure_r10, pure_r5
    mapping = dict(PROMOTED_MODEL_BY_HORIZON)

    configs = {
        "current_085": {h: 0.85 for h in HORIZONS},
        "optimal_per_h": optimal_weights,
        "pure_r10": {h: 1.0 for h in HORIZONS},
        "pure_r5": {h: 0.0 for h in HORIZONS},
    }

    results = {}
    for name, weights in configs.items():
        print(f"\n--- Config: {name} (weights={weights}) ---")
        result = build_config(tta_store, sl_store, mapping, weights, primary, secondary)
        s = result["summary"]
        print(
            f"  nf={s['nonflat_accuracy_overall']:.2%} "
            f"mae={s['return_mae_overall']:.6f} "
            f"g_prec={s['h1_gated_precision']:.2%} "
            f"g_nf={s['h1_gated_nonflat']:.2%} "
            f"g_cov={s['h1_gated_coverage']:.2%}"
        )
        for h in HORIZONS:
            print(f"  h={h}: nf={s['by_horizon'][str(h)]['nonflat_accuracy']:.2%} mae={s['by_horizon'][str(h)]['return_mae']:.6f}")
        results[name] = result

    # 5. Decision: optimal_per_h vs current_085 (P4-6 baseline)
    current = results["current_085"]["summary"]
    optimal = results["optimal_per_h"]["summary"]
    decision = dual_bar_decision(optimal, current)
    print(f"\n=== Decision: optimal_per_h vs current_085 ===")
    print(f"  promote={decision['promote']} reason={decision['reason']}")
    print(f"  direction_win={decision['direction_win']} mae_win={decision['mae_win']}")
    print(f"  nonflat_delta={decision['nonflat_delta']:.4f} mae_delta={decision['mae_delta']:.6f}")

    # 6. Pick best config
    # Prefer optimal_per_h if it wins; else keep current_085
    best_name = "optimal_per_h" if decision["promote"] else "current_085"
    best = results[best_name]
    print(f"\nBest config: {best_name}")

    output = {
        "symbol": args.symbol,
        "n_windows": n_windows_sl,
        "sweep_result": sweep_result,
        "configs": {k: v["summary"] for k, v in results.items()},
        "backtests": {k: {"ungated": v["ungated_bt"], "gated": v["gated_bt"]} for k, v in results.items()},
        "decision_optimal_vs_current": decision,
        "best_config": best_name,
        "best_summary": best["summary"],
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
