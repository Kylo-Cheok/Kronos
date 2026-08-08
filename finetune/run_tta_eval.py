"""Round P4-2: Test-Time Augmentation (TTA) via multi-lookback inference.

For each 688169 test window, run inference with lookback lengths in
TTA_LOOKBACKS (default [124, 126, 128, 130, 132]) and average direction
logits + return predictions across augmentations. The prediction target
(close[ctx_end+h] / close[ctx_end]) is identical for every augmentation,
only the context length feeding the head varies.

Compares TTA-augmented ensemble vs Phase-3 promoted baseline.
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
from evaluate_gated_ensemble import (
    CLIP,
    CSV_PATH,
    DEVICE,
    FEATURES,
    HORIZONS,
    LOOKBACK,
    TOKENIZER_PATH,
    VAL_END,
    WINDOW,
    derive_time_features,
    load_csv,
    load_model,
)
from multihorizon_objective import (
    DEFAULT_HORIZONS,
    MultiHorizonForecastHead,
    make_multihorizon_targets,
)
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

# TTA lookback lengths. All must be <= available context history.
# 128 is the trained lookback; ±4 covers a 9-day neighbourhood.
TTA_LOOKBACKS = (124, 126, 128, 130, 132)
PREDICT_WINDOW_LEN = 10  # predict_window in config


def make_tta_windows(df: pd.DataFrame, lookbacks: tuple[int, ...]) -> list[dict]:
    """Build windows whose target day is fixed; context length varies.

    Each window carries the longest slice needed for max(lookbacks) so the
    per-augmentation slicing can happen in predict_tta.
    """
    val_cut = pd.Timestamp(VAL_END)
    max_lb = max(lookbacks)
    needed = max_lb + PREDICT_WINDOW_LEN + 1
    windows = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        context_end_date = df["timestamps"].iloc[start + LOOKBACK - 1]
        if context_end_date <= val_cut:
            continue
        # Need (max_lb - LOOKBACK) extra days before `start`.
        extra = max_lb - LOOKBACK
        if start - extra < 0:
            continue
        big_start = start - extra
        window = df.iloc[big_start : big_start + needed].copy()
        windows.append(
            {
                "start": start,
                "context_end_date": str(context_end_date.date()),
                "features": window[FEATURES].to_numpy(dtype=np.float32),
                "raw_close": window["close"].to_numpy(dtype=np.float32),
                "timestamps": window["timestamps"],
            }
        )
    return windows


def normalize_with_lookback(features: np.ndarray, lookback: int) -> np.ndarray:
    ctx = features[:lookback]
    mean = ctx.mean(axis=0, keepdims=True)
    std = ctx.std(axis=0, keepdims=True)
    normalized = (features - mean) / (std + 1e-5)
    return np.clip(normalized, -CLIP, CLIP)


@torch.no_grad()
def predict_tta(loaded: dict, w: dict, lookbacks: tuple[int, ...]) -> dict:
    """Run model with multiple lookback lengths; average logits + return."""
    logits_accum = None
    ret_accum = None
    n = 0
    target_dir = None
    target_ret = None
    for lb in lookbacks:
        # Slice the pre-extracted window to (lb + PREDICT_WINDOW_LEN + 1).
        # The pre-extracted window starts at big_start = start - (max_lb - LOOKBACK).
        # We need the slice ending at the same context_end day, so we drop
        # (max_lb - lb) leading days.
        drop = max(lookbacks) - lb
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
        logits = outputs["direction_logits"][0].cpu().numpy()  # [H, 3]
        ret_pred = outputs["return_prediction"][0].cpu().numpy()  # [H]

        if logits_accum is None:
            logits_accum = logits.astype(np.float64).copy()
            ret_accum = ret_pred.astype(np.float64).copy()
        else:
            logits_accum += logits
            ret_accum += ret_pred
        n += 1

        # Targets are identical for every lookback (only context varies).
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

    logits_avg = logits_accum / n
    ret_avg = ret_accum / n
    return {
        "logits": logits_avg,
        "pred_return": ret_avg,
        "target_direction": target_dir,
        "target_return": target_ret,
        "context_end_date": w["context_end_date"],
        "n_augmentations": n,
    }


def collect_tta_store(
    names: set[str], symbol: str, lookbacks: tuple[int, ...]
) -> tuple[dict, int]:
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
        n: {h: {"logits": [], "pret": [], "td": [], "tr": []} for h in HORIZONS}
        for n in loaded
    }
    for i, w in enumerate(windows):
        if i % 30 == 0:
            print(f"  window {i}/{len(windows)} ({w['context_end_date']})")
        for n, mod in loaded.items():
            pred = predict_tta(mod, w, lookbacks)
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


def build_arrays(
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
    w = float(PROMOTED_RETURN_BLEND["primary_weight"])
    for h in HORIZONS:
        dn = dir_map[h]
        conf = direction_confidence_from_logits(store[dn][h]["logits"])
        pred_dir[h] = conf["hard_pred"]
        t_dir[h] = store[dn][h]["td"]
        t_ret[h] = store[dn][h]["tr"]
        if use_return_blend and primary in store and secondary in store:
            pred_ret[h] = blend_returns(
                store[primary][h]["pret"], store[secondary][h]["pret"], w
            )
        else:
            pred_ret[h] = store[dn][h]["pret"]
    h1_dir = dir_map[1]
    conf1 = direction_confidence_from_logits(store[h1_dir][1]["logits"])
    conf_score = conf1[PROMOTED_H1_GATE["confidence_key"]]
    gate_ret = store[primary][1]["pret"] if primary in store else pred_ret[1]
    return pred_dir, pred_ret, t_dir, t_ret, conf_score, conf1["hard_pred"], gate_ret


def attach_gate(summary, hard, conf, pret_for_gate, t_dir, gate):
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
        "--lookbacks",
        type=str,
        default=",".join(str(x) for x in TTA_LOOKBACKS),
        help="Comma-separated lookback lengths",
    )
    args = parser.parse_args()

    lookbacks = tuple(int(x) for x in args.lookbacks.split(",") if x.strip())
    mapping = dict(PROMOTED_MODEL_BY_HORIZON)
    names = set(mapping.values()) | set(PHASE2_BASELINE_MODEL_BY_HORIZON.values())
    names |= {PROMOTED_RETURN_BLEND["primary"], PROMOTED_RETURN_BLEND["secondary"]}
    store, n_windows = collect_tta_store(names, args.symbol, lookbacks)

    # Baseline (no blend, Phase-2 gate)
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
    baseline = summarize_horizons(b_dir, b_ret, b_td, b_tr, HORIZONS)
    baseline, b_gated = attach_gate(
        baseline, b_hard, b_conf, b_gate_ret, b_td[1], phase2_gate
    )

    # TTA promoted
    p_dir, p_ret, p_td, p_tr, p_conf, p_hard, p_gate_ret = build_arrays(
        store, mapping, use_return_blend=True
    )
    promoted = summarize_horizons(p_dir, p_ret, p_td, p_tr, HORIZONS)
    promoted, p_gated = attach_gate(
        promoted, p_hard, p_conf, p_gate_ret, p_td[1], PROMOTED_H1_GATE
    )

    decision = dual_bar_decision(promoted, baseline)
    ungated_bt = absolute_direction_backtest(
        p_hard, p_tr[1], transaction_cost=float(PROMOTED_H1_GATE["transaction_cost"])
    )
    gated_bt = absolute_direction_backtest(
        p_gated, p_tr[1], transaction_cost=float(PROMOTED_H1_GATE["transaction_cost"])
    )

    result = {
        "symbol": args.symbol,
        "n_windows": n_windows,
        "tta_lookbacks": list(lookbacks),
        "phase2_baseline": baseline,
        "tta_promoted": promoted,
        "decision": decision,
        "h1_backtests": {
            "tta_ungated": ungated_bt,
            "tta_gated": gated_bt,
        },
    }

    print("\n=== TTA Round P4-2 ===")
    print(
        f"baseline: nf={baseline['nonflat_accuracy_overall']:.2%} "
        f"mae={baseline['return_mae_overall']:.5f} "
        f"g_prec={baseline['h1_gated_precision']:.2%} "
        f"g_nf={baseline['h1_gated_nonflat']:.2%}"
    )
    print(
        f"tta:       nf={promoted['nonflat_accuracy_overall']:.2%} "
        f"mae={promoted['return_mae_overall']:.5f} "
        f"g_prec={promoted['h1_gated_precision']:.2%} "
        f"g_nf={promoted['h1_gated_nonflat']:.2%}"
    )
    print(f"tta ungated h1 bt: ret={ungated_bt['total_return']:.2%} hit={ungated_bt['hit_rate']:.2%}")
    print(f"tta gated   h1 bt: ret={gated_bt['total_return']:.2%} hit={gated_bt['hit_rate']:.2%} n={gated_bt['n_trades']:.0f}")
    print(f"decision: {decision['promote']} ({decision['reason']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
