"""Evaluate a trained multihorizon predictor on the 688169 test set.

Loads a Kronos backbone + MultiHorizonForecastHead checkpoint and runs a
deterministic, exhaustive evaluation on every valid test window for 688169
(context_end_date > 2025-12-15).  Reports per-horizon direction accuracy,
return MAE, and class-wise confusion.  Results are saved as JSON.

Usage::

    python finetune/evaluate_multihorizon.py --exp-name joint_d005
    python finetune/evaluate_multihorizon.py --model-dir outputs/models/a_share_multihorizon_predictor/checkpoints/best_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.kronos import Kronos, KronosTokenizer
from multihorizon_objective import (
    DEFAULT_HORIZONS,
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    MultiHorizonForecastHead,
    make_direction_deadzones,
    make_multihorizon_targets,
)


SYMBOL = "688169"
CSV_PATH = ROOT / "data" / "a_share_finetune_multiboard" / "csv" / f"{SYMBOL}.csv"
TOKENIZER_PATH = ROOT / "outputs" / "models" / "a_share_multi_tokenizer" / "checkpoints" / "best_model"
DEFAULT_MODEL_DIR = ROOT / "outputs" / "models" / "a_share_multihorizon_predictor" / "checkpoints" / "best_model"
LOOKBACK = 128
PREDICT_WINDOW = 10
WINDOW = LOOKBACK + PREDICT_WINDOW + 1  # 139
VAL_END = "2025-12-15"
FEATURES = ["open", "high", "low", "close", "vol", "amt"]
TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]
CLIP = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.normalize()
    df = df.sort_values("timestamps").reset_index(drop=True)
    # Map volume/amount to vol/amt to match the pickle feature columns
    df["vol"] = df["volume"]
    df["amt"] = df["amount"]
    return df


def derive_time_features(dates: pd.Series) -> np.ndarray:
    """Return [T, 5] time-stamp features matching Kronos TemporalEmbedding."""
    stamps = pd.DatetimeIndex(dates)
    return np.stack([
        np.zeros(len(stamps)),  # minute (0 for daily)
        np.zeros(len(stamps)),  # hour (0 for daily)
        stamps.weekday.to_numpy(),
        stamps.day.to_numpy(),
        stamps.month.to_numpy(),
    ], axis=1).astype(np.float32)


def make_windows(df: pd.DataFrame) -> list[dict]:
    """Build all valid test windows where context_end > VAL_END."""
    val_cut = pd.Timestamp(VAL_END)
    windows = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        context_end_date = df["timestamps"].iloc[start + LOOKBACK - 1]
        if context_end_date <= val_cut:
            continue
        window = df.iloc[start : start + WINDOW].copy()
        windows.append({
            "start": start,
            "context_end_date": str(context_end_date.date()),
            "features": window[FEATURES].to_numpy(dtype=np.float32),
            "raw_close": window["close"].to_numpy(dtype=np.float32),
            "timestamps": window["timestamps"],
        })
    return windows


def normalize_window(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Past-only normalization: mean/std over lookback, applied to full window."""
    lookback = features[:LOOKBACK]
    mean = lookback.mean(axis=0, keepdims=True)
    std = lookback.std(axis=0, keepdims=True)
    normalized = (features - mean) / (std + 1e-5)
    normalized = np.clip(normalized, -CLIP, CLIP)
    return normalized, mean, std


@torch.no_grad()
def evaluate(
    model_dir: Path,
    tokenizer_path: Path,
    output_json: Path | None = None,
) -> dict:
    df = load_csv()
    windows = make_windows(df)
    print(f"Loaded {len(windows)} test windows for {SYMBOL} (context_end > {VAL_END})")

    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.eval().to(DEVICE)

    model = Kronos.from_pretrained(str(model_dir))
    model.eval().to(DEVICE)

    head_path = model_dir / "multihorizon_head.pt"
    head_ckpt = torch.load(head_path, map_location=DEVICE, weights_only=False)
    d_model = head_ckpt["d_model"]
    horizons = tuple(head_ckpt.get("horizons", list(DEFAULT_HORIZONS)))
    pool_size = head_ckpt.get("pool_size", 16)
    min_deadzone = head_ckpt.get("direction_min_deadzone", 0.003)
    vol_mult = head_ckpt.get("direction_volatility_multiplier", 0.5)
    head = MultiHorizonForecastHead(d_model, horizons=horizons, pool_size=pool_size, dropout=0.0).to(DEVICE)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()
    print(f"Loaded head: horizons={horizons}, pool_size={pool_size}, d_model={d_model}")

    class_names = ["down", "flat", "up"]
    n_classes = 3
    confusion = np.zeros((len(horizons), n_classes, n_classes), dtype=np.int64)
    return_errors = {h: [] for h in horizons}
    direction_correct = {h: 0 for h in horizons}
    direction_total = {h: 0 for h in horizons}
    # Non-flat accuracy (exclude FLAT labels, measure up/down calls)
    nonflat_correct = {h: 0 for h in horizons}
    nonflat_total = {h: 0 for h in horizons}

    for wi, w in enumerate(windows):
        features = w["features"]
        raw_close = w["raw_close"]
        x_norm, _, _ = normalize_window(features)
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)  # [1, 139, 6]
        stamp = derive_time_features(w["timestamps"])
        stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)  # [1, 139, 5]
        raw_close_tensor = torch.from_numpy(raw_close).unsqueeze(0).to(DEVICE)  # [1, 139]

        token_seq_0, token_seq_1 = tokenizer.encode(x_tensor, half=True)
        token_in_0 = token_seq_0[:, :-1]
        token_in_1 = token_seq_1[:, :-1]
        _, _, hidden_states = model(
            token_in_0, token_in_1, stamp_tensor[:, :-1, :],
            return_context=True,
        )
        outputs = head(hidden_states, context_length=LOOKBACK)
        direction_logits = outputs["direction_logits"]  # [1, H, 3]
        return_pred = outputs["return_prediction"]  # [1, H]

        targets = make_multihorizon_targets(
            raw_close_tensor, context_length=LOOKBACK, horizons=horizons,
            min_deadzone=min_deadzone, volatility_multiplier=vol_mult,
        )
        target_direction = targets["direction"][0]  # [H]
        target_returns = targets["returns"][0]  # [H]

        pred_direction = direction_logits.argmax(dim=-1)[0]  # [H]

        for hi, h in enumerate(horizons):
            t_dir = int(target_direction[hi].item())
            p_dir = int(pred_direction[hi].item())
            confusion[hi, t_dir, p_dir] += 1
            if t_dir == p_dir:
                direction_correct[h] += 1
            direction_total[h] += 1
            # Non-flat: when true label is up/down, did we call it correctly?
            if t_dir != FLAT_CLASS:
                nonflat_total[h] += 1
                if p_dir == t_dir:
                    nonflat_correct[h] += 1
            return_errors[h].append(float(return_pred[0, hi].item() - target_returns[hi].item()))

    metrics = {
        "symbol": SYMBOL,
        "model_dir": str(model_dir),
        "n_test_windows": len(windows),
        "horizons": list(horizons),
        "direction_accuracy_by_horizon": {
            str(h): direction_correct[h] / direction_total[h] if direction_total[h] else 0.0
            for h in horizons
        },
        "direction_accuracy_overall": sum(direction_correct.values()) / sum(direction_total.values()),
        "nonflat_accuracy_by_horizon": {
            str(h): nonflat_correct[h] / nonflat_total[h] if nonflat_total[h] else 0.0
            for h in horizons
        },
        "nonflat_accuracy_overall": sum(nonflat_correct.values()) / sum(nonflat_total.values()) if sum(nonflat_total.values()) else 0.0,
        "return_mae_by_horizon": {
            str(h): float(np.mean(np.abs(return_errors[h]))) if return_errors[h] else None
            for h in horizons
        },
        "return_rmse_by_horizon": {
            str(h): float(np.sqrt(np.mean(np.square(return_errors[h])))) if return_errors[h] else None
            for h in horizons
        },
        "confusion_by_horizon": {
            str(h): confusion[hi].tolist() for hi, h in enumerate(horizons)
        },
        "class_distribution_by_horizon": {
            str(h): {
                "down": int(confusion[hi, DOWN_CLASS].sum()),
                "flat": int(confusion[hi, FLAT_CLASS].sum()),
                "up": int(confusion[hi, UP_CLASS].sum()),
            }
            for hi, h in enumerate(horizons)
        },
    }

    print("\n=== Evaluation Results ===")
    print(f"Test windows: {len(windows)}")
    print(f"Direction accuracy (overall): {metrics['direction_accuracy_overall']:.2%}")
    for h in horizons:
        acc = metrics["direction_accuracy_by_horizon"][str(h)]
        nf = metrics["nonflat_accuracy_by_horizon"][str(h)]
        mae = metrics["return_mae_by_horizon"][str(h)]
        dist = metrics["class_distribution_by_horizon"][str(h)]
        print(f"  h={h}: dir_acc={acc:.2%}, nonflat_acc={nf:.2%}, return_MAE={mae:.4f}, "
              f"classes(D/F/U)={dist['down']}/{dist['flat']}/{dist['up']}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {output_json}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=None, help="Experiment tag (e.g. joint_d005)")
    parser.add_argument("--model-dir", type=Path, default=None, help="Direct path to best_model dir")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = args.model_dir
    elif args.exp_name:
        model_dir = ROOT / "outputs" / "models" / f"a_share_multihorizon_predictor_{args.exp_name}" / "checkpoints" / "best_model"
    else:
        model_dir = DEFAULT_MODEL_DIR

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.output:
        output_json = args.output
    elif args.exp_name:
        output_json = ROOT / "outputs" / f"eval_{SYMBOL}_{args.exp_name}.json"
    else:
        output_json = ROOT / "outputs" / f"eval_{SYMBOL}_baseline.json"

    evaluate(model_dir, TOKENIZER_PATH, output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
