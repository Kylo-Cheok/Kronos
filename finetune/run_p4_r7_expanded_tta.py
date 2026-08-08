"""Round P4-7: Expanded TTA — test multiple lookback configurations.

Collects PER-LOOKBACK logits (not averaged) for 7 lookbacks (122-134),
then tests several TTA averaging configs from a single inference run:

  - 3-lb:  (126, 128, 130)              — tight, ±1 day
  - 5-lb:  (124, 126, 128, 130, 132)    — current P4-6 (±2 days)
  - 7-lb:  (122, 124, 126, 128, 130, 132, 134) — expanded (±3 days)
  - w5-lb: weighted 5-lb (lb=128 gets weight 2, others weight 1)

For each config, direction = TTA-averaged logits, returns = single-lookback
blend (0.85*R10 + 0.15*R5), gate = TTA confidence + SL R10 return magnitude.
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
from run_tta_eval import (
    CLIP,
    DEVICE,
    FEATURES,
    HORIZONS,
    LOOKBACK,
    PREDICT_WINDOW_LEN,
    derive_time_features,
    make_tta_windows,
    normalize_with_lookback,
)
from multihorizon_objective import make_multihorizon_targets
from selective_prediction import (
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

ALL_LOOKBACKS = (122, 124, 126, 128, 130, 132, 134)

# TTA configs to test: (name, lookbacks, weights)
TTA_CONFIGS = [
    ("3lb", (126, 128, 130), None),
    ("5lb_current", (124, 126, 128, 130, 132), None),
    ("7lb_expanded", (122, 124, 126, 128, 130, 132, 134), None),
    ("5lb_weighted", (124, 126, 128, 130, 132), (1.0, 1.0, 2.0, 1.0, 1.0)),
]


@torch.no_grad()
def predict_per_lookback(loaded: dict, w: dict, lookbacks: tuple[int, ...]) -> dict:
    """Run model with each lookback; return per-lookback logits list + targets."""
    max_lb = max(lookbacks)
    per_lb_logits = []  # list of [H, 3]
    target_dir = None
    target_ret = None
    for lb in lookbacks:
        drop = max_lb - lb
        x_full = w["features"][drop : drop + lb + PREDICT_WINDOW_LEN + 1]
        close_full = w["raw_close"][drop : drop + lb + PREDICT_WINDOW_LEN + 1]
        ts_full = w["timestamps"].iloc[drop : drop + lb + PREDICT_WINDOW_LEN + 1]

        x_norm = normalize_with_lookback(x_full, lb)
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
        stamp = derive_time_features(ts_full)
        stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
        raw_close_tensor = torch.from_numpy(close_full).unsqueeze(0).to(DEVICE)

        token_seq_0, token_seq_1 = loaded["tokenizer"].encode(x_tensor, half=True)
        _, _, hidden_states = loaded["model"](
            token_seq_0[:, :-1],
            token_seq_1[:, :-1],
            stamp_tensor[:, :-1, :],
            return_context=True,
        )
        outputs = loaded["head"](hidden_states, context_length=lb)
        logits = outputs["direction_logits"][0].cpu().numpy()
        per_lb_logits.append(logits.astype(np.float64))

        if target_dir is None:
            targets = make_multihorizon_targets(
                raw_close_tensor,
                context_length=lb,
                horizons=loaded["horizons"],
                min_deadzone=loaded["min_deadzone"],
                volatility_multiplier=loaded["vol_mult"],
            )
            target_dir = targets["direction"][0].cpu().numpy()
            target_ret = targets["returns"][0].cpu().numpy()

    return {
        "per_lb_logits": per_lb_logits,  # list[len(lookbacks)] of [H, 3]
        "target_direction": target_dir,
        "target_return": target_ret,
        "context_end_date": w["context_end_date"],
    }


def collect_per_lb_store(names: set[str], symbol: str, lookbacks: tuple[int, ...]) -> tuple[dict, int]:
    """Collect per-lookback logits for all models."""
    loaded = {}
    for name in sorted(names):
        path = model_dir(name)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading {name} from {path}")
        loaded[name] = load_model(path)
    windows = make_tta_windows(load_csv(symbol), lookbacks)
    print(f"{len(windows)} TTA windows for {symbol} (lookbacks={lookbacks})")
    store = {
        n: {h: {"per_lb_logits": [[] for _ in lookbacks], "td": [], "tr": []} for h in HORIZONS}
        for n in loaded
    }
    for i, w in enumerate(windows):
        if i % 30 == 0:
            print(f"  window {i}/{len(windows)} ({w['context_end_date']})")
        for n, mod in loaded.items():
            pred = predict_per_lookback(mod, w, lookbacks)
            for lb_idx, lb_logits in enumerate(pred["per_lb_logits"]):
                for hi, h in enumerate(HORIZONS):
                    store[n][h]["per_lb_logits"][lb_idx].append(lb_logits[hi])
            for hi, h in enumerate(HORIZONS):
                store[n][h]["td"].append(int(pred["target_direction"][hi]))
                store[n][h]["tr"].append(float(pred["target_return"][hi]))
    for n in loaded:
        for h in HORIZONS:
            for lb_idx in range(len(lookbacks)):
                store[n][h]["per_lb_logits"][lb_idx] = np.stack(
                    store[n][h]["per_lb_logits"][lb_idx]
                )
            for k in ("td", "tr"):
                store[n][h][k] = np.asarray(store[n][h][k])
    return store, len(windows)


def average_logits(per_lb_logits: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of per-lookback logits arrays."""
    stacked = np.stack(per_lb_logits, axis=0)  # [n_lb, N, 3]
    if weights is None:
        return stacked.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.tensordot(w, stacked, axes=([0], [0]))


def collect_single_lookback(names: set[str], symbol: str) -> tuple[dict, int]:
    """Standard lb=128 inference for returns + baseline."""
    loaded = {}
    for name in sorted(names):
        path = model_dir(name)
        if not path.exists():
            raise FileNotFoundError(path)
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


def evaluate_tta_config(
    tta_store: dict,
    sl_store: dict,
    all_lookbacks: tuple[int, ...],
    config_lookbacks: tuple[int, ...],
    weights: np.ndarray | None,
    mapping: dict[int, str],
) -> dict:
    """Evaluate one TTA config: TTA direction + SL returns + gate."""
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])

    # Map config lookbacks to indices in all_lookbacks
    lb_indices = [all_lookbacks.index(lb) for lb in config_lookbacks]

    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        dn = mapping[h]
        # TTA-averaged logits for direction
        per_lb = [tta_store[dn][h]["per_lb_logits"][idx] for idx in lb_indices]
        avg_logits = average_logits(per_lb, weights)
        conf = direction_confidence_from_logits(avg_logits)
        pred_dir[h] = conf["hard_pred"]
        # SL targets + blended returns
        t_dir[h] = tta_store[dn][h]["td"]
        t_ret[h] = tta_store[dn][h]["tr"]
        pred_ret[h] = blend_returns(
            sl_store[primary][h]["pret"], sl_store[secondary][h]["pret"], w_blend
        )

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)

    # h=1 gate: TTA confidence + SL R10 return for magnitude
    h1_dn = mapping[1]
    per_lb_h1 = [tta_store[h1_dn][1]["per_lb_logits"][idx] for idx in lb_indices]
    avg_logits_h1 = average_logits(per_lb_h1, weights)
    conf1 = direction_confidence_from_logits(avg_logits_h1)
    hard1 = conf1["hard_pred"]
    score1 = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = sl_store[primary][1]["pret"]

    gated = apply_consistency_and_magnitude_gate(
        hard1, score1, gate_ret,
        confidence_threshold=0.45,
        min_abs_return=0.0,
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, t_dir[1])
    summary["h1_gated_precision"] = m["precision_on_calls"]
    summary["h1_gated_nonflat"] = m["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = m["coverage"]
    summary["h1_gate_metrics"] = m

    bt = absolute_direction_backtest(gated, t_ret[1], transaction_cost=0.0005)
    return summary, bt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_p4_r7_expanded_tta.json")
    args = parser.parse_args()

    # 1. Collect per-lookback TTA store (7 lookbacks)
    mapping = dict(PROMOTED_MODEL_BY_HORIZON)
    names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
    names |= {PROMOTED_RETURN_BLEND["primary"], PROMOTED_RETURN_BLEND["secondary"]}
    print(f"=== Collecting per-lookback TTA store (lookbacks={ALL_LOOKBACKS}) ===")
    tta_store, n_windows = collect_per_lb_store(names, "688169", ALL_LOOKBACKS)

    # 2. Collect single-lookback store for returns
    print(f"\n=== Collecting single-lookback store (lb=128) ===")
    sl_store, n_windows_sl = collect_single_lookback(names, "688169")
    print(f"TTA windows: {n_windows}, SL windows: {n_windows_sl}")

    # 3. Phase-3 baseline (single-lookback, no TTA) for comparison
    primary = PROMOTED_RETURN_BLEND["primary"]
    secondary = PROMOTED_RETURN_BLEND["secondary"]
    w_blend = float(PROMOTED_RETURN_BLEND["primary_weight"])
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
    b_hard1 = b_conf1["hard_pred"]
    b_score1 = b_conf1["actionable_score"]
    b_gate_ret = sl_store[primary][1]["pret"]
    b_gated = apply_consistency_and_magnitude_gate(
        b_hard1, b_score1, b_gate_ret,
        confidence_threshold=0.45, min_abs_return=0.003, require_sign_agree=False,
    )
    b_m = gated_actionable_metrics(b_gated, b_td[1])
    baseline["h1_gated_precision"] = b_m["precision_on_calls"]
    baseline["h1_gated_nonflat"] = b_m["gated_nonflat_acc"]
    baseline["h1_gated_coverage"] = b_m["coverage"]
    print(
        f"\nPhase-2 baseline (SL): nf={baseline['nonflat_accuracy_overall']:.2%} "
        f"mae={baseline['return_mae_overall']:.5f} "
        f"g_prec={baseline['h1_gated_precision']:.2%} "
        f"g_nf={baseline['h1_gated_nonflat']:.2%} "
        f"g_cov={baseline['h1_gated_coverage']:.2%}"
    )

    # 4. Test each TTA config
    results = {}
    for name, lookbacks, weights in TTA_CONFIGS:
        print(f"\n=== TTA config: {name} (lookbacks={lookbacks}, weights={weights}) ===")
        summary, bt = evaluate_tta_config(
            tta_store, sl_store, ALL_LOOKBACKS, lookbacks, weights, mapping
        )
        decision = dual_bar_decision(summary, baseline)
        results[name] = {
            "lookbacks": list(lookbacks),
            "weights": list(weights) if weights is not None else None,
            "summary": summary,
            "backtest": bt,
            "decision": decision,
        }
        print(
            f"  nf={summary['nonflat_accuracy_overall']:.2%} "
            f"mae={summary['return_mae_overall']:.5f} "
            f"g_prec={summary['h1_gated_precision']:.2%} "
            f"g_nf={summary['h1_gated_nonflat']:.2%} "
            f"g_cov={summary['h1_gated_coverage']:.2%}"
        )
        print(
            f"  bt: ret={bt['total_return']:.2%} hit={bt['hit_rate']:.2%} n={bt['n_trades']:.0f}"
        )
        print(f"  decision: {decision['promote']} ({decision['reason']})")

    # 5. Find best config
    best_name = max(
        results.keys(),
        key=lambda k: (
            results[k]["summary"]["h1_gated_precision"],
            results[k]["summary"]["h1_gated_nonflat"],
            results[k]["summary"]["nonflat_accuracy_overall"],
        ),
    )
    print(f"\n=== Best TTA config: {best_name} ===")
    best = results[best_name]
    print(
        f"  nf={best['summary']['nonflat_accuracy_overall']:.2%} "
        f"mae={best['summary']['return_mae_overall']:.5f} "
        f"g_prec={best['summary']['h1_gated_precision']:.2%} "
        f"g_nf={best['summary']['h1_gated_nonflat']:.2%} "
        f"g_cov={best['summary']['h1_gated_coverage']:.2%}"
    )

    out = {
        "symbol": "688169",
        "n_windows": n_windows,
        "all_lookbacks": list(ALL_LOOKBACKS),
        "tta_configs": [c[0] for c in TTA_CONFIGS],
        "phase2_baseline": baseline,
        "results": results,
        "best_config": best_name,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
