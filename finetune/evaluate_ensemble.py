"""Per-horizon ensemble evaluation: pick the best model for each horizon.

Round 5 (frozen pool=48) is strong on h=3/5/10 but weak on h=1.
Round 4 (joint lr=2e-6 pool=16) is strong on h=1 but h=10 collapses.
Round 9 (joint lr=2e-6 pool=48) may be even stronger on h=1.

This script loads all candidate models and auto-selects the best model
per horizon based on nonflat accuracy on the test set.

Usage:
    python finetune/evaluate_ensemble.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from model.kronos import Kronos, KronosTokenizer
from multihorizon_objective import (
    DEFAULT_HORIZONS,
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    MultiHorizonForecastHead,
    make_multihorizon_targets,
)

SYMBOL = "688169"
CSV_PATH = ROOT / "data" / "a_share_finetune_multiboard" / "csv" / f"{SYMBOL}.csv"
TOKENIZER_PATH = ROOT / "outputs" / "models" / "a_share_multi_tokenizer" / "checkpoints" / "best_model"
# Candidate models for the ensemble
CANDIDATES = {
    "r5_frozen_pool48": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_r5_frozen_pool48" / "checkpoints" / "best_model",
    "r4_joint_lr2e6": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_r4_joint_lr2e6" / "checkpoints" / "best_model",
    "r9_joint_pool48": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_r9_joint_pool48" / "checkpoints" / "best_model",
    "r10_joint_splitlr": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_r10_joint_splitlr" / "checkpoints" / "best_model",
}
LOOKBACK = 128
PREDICT_WINDOW = 10
WINDOW = LOOKBACK + PREDICT_WINDOW + 1
VAL_END = "2025-12-15"
FEATURES = ["open", "high", "low", "close", "vol", "amt"]
CLIP = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HORIZONS = list(DEFAULT_HORIZONS)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.normalize()
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["vol"] = df["volume"]
    df["amt"] = df["amount"]
    return df


def derive_time_features(dates: pd.Series) -> np.ndarray:
    stamps = pd.DatetimeIndex(dates)
    return np.stack([
        np.zeros(len(stamps)),
        np.zeros(len(stamps)),
        stamps.weekday.to_numpy(),
        stamps.day.to_numpy(),
        stamps.month.to_numpy(),
    ], axis=1).astype(np.float32)


def make_windows(df: pd.DataFrame) -> list[dict]:
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


def normalize_window(features: np.ndarray) -> np.ndarray:
    lookback = features[:LOOKBACK]
    mean = lookback.mean(axis=0, keepdims=True)
    std = lookback.std(axis=0, keepdims=True)
    normalized = (features - mean) / (std + 1e-5)
    return np.clip(normalized, -CLIP, CLIP)


def load_model(model_dir: Path):
    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_PATH))
    tokenizer.eval().to(DEVICE)
    model = Kronos.from_pretrained(str(model_dir))
    model.eval().to(DEVICE)
    head_path = model_dir / "multihorizon_head.pt"
    head_ckpt = torch.load(head_path, map_location=DEVICE, weights_only=False)
    d_model = head_ckpt["d_model"]
    horizons = tuple(head_ckpt.get("horizons", HORIZONS))
    pool_size = head_ckpt.get("pool_size", 16)
    min_deadzone = head_ckpt.get("direction_min_deadzone", 0.003)
    vol_mult = head_ckpt.get("direction_volatility_multiplier", 0.5)
    head = MultiHorizonForecastHead(d_model, horizons=horizons, pool_size=pool_size, dropout=0.0).to(DEVICE)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "head": head,
        "horizons": horizons,
        "min_deadzone": min_deadzone,
        "vol_mult": vol_mult,
    }


@torch.no_grad()
def predict_window(loaded: dict, w: dict) -> dict:
    features = w["features"]
    raw_close = w["raw_close"]
    x_norm = normalize_window(features)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
    stamp = derive_time_features(w["timestamps"])
    stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
    raw_close_tensor = torch.from_numpy(raw_close).unsqueeze(0).to(DEVICE)

    tokenizer = loaded["tokenizer"]
    token_seq_0, token_seq_1 = tokenizer.encode(x_tensor, half=True)
    _, _, hidden_states = loaded["model"](
        token_seq_0[:, :-1], token_seq_1[:, :-1], stamp_tensor[:, :-1, :],
        return_context=True,
    )
    outputs = loaded["head"](hidden_states, context_length=LOOKBACK)
    direction_logits = outputs["direction_logits"][0]
    return_pred = outputs["return_prediction"][0]
    pred_direction = direction_logits.argmax(dim=-1)

    targets = make_multihorizon_targets(
        raw_close_tensor, context_length=LOOKBACK, horizons=loaded["horizons"],
        min_deadzone=loaded["min_deadzone"], volatility_multiplier=loaded["vol_mult"],
    )
    target_direction = targets["direction"][0]
    target_returns = targets["returns"][0]
    return {
        "pred_direction": pred_direction.cpu().numpy(),
        "pred_return": return_pred.cpu().numpy(),
        "target_direction": target_direction.cpu().numpy(),
        "target_return": target_returns.cpu().numpy(),
    }


def main() -> int:
    df = load_csv()
    windows = make_windows(df)
    print(f"Loaded {len(windows)} test windows for {SYMBOL}")

    # Load all candidate models
    loaded_models = {}
    for name, path in CANDIDATES.items():
        print(f"Loading {name}...")
        loaded_models[name] = load_model(path)

    # Collect per-model per-horizon predictions
    # Structure: model_name -> horizon -> list of (pred_dir, pred_return, target_dir, target_return)
    per_model = {name: {h: {"pred_dir": [], "pred_ret": [], "t_dir": [], "t_ret": []} for h in HORIZONS}
                 for name in CANDIDATES}

    for wi, w in enumerate(windows):
        # Targets are the same across models (same window), but compute per model for safety
        for name, loaded in loaded_models.items():
            pred = predict_window(loaded, w)
            for hi, h in enumerate(HORIZONS):
                per_model[name][h]["pred_dir"].append(int(pred["pred_direction"][hi]))
                per_model[name][h]["pred_ret"].append(float(pred["pred_return"][hi]))
                per_model[name][h]["t_dir"].append(int(pred["target_direction"][hi]))
                per_model[name][h]["t_ret"].append(float(pred["target_return"][hi]))

    # Compute per-model per-horizon nonflat accuracy
    print("\n=== Per-model per-horizon nonflat accuracy ===")
    best_model_per_h = {}
    for h in HORIZONS:
        print(f"\nh={h}:")
        best_nf = -1.0
        best_name = None
        for name in CANDIDATES:
            t_dirs = np.array(per_model[name][h]["t_dir"])
            p_dirs = np.array(per_model[name][h]["pred_dir"])
            nonflat_mask = t_dirs != FLAT_CLASS
            nf_total = nonflat_mask.sum()
            nf_correct = ((p_dirs == t_dirs) & nonflat_mask).sum()
            nf_acc = nf_correct / nf_total if nf_total > 0 else 0.0
            da = (p_dirs == t_dirs).mean()
            mae = float(np.mean(np.abs(np.array(per_model[name][h]["pred_ret"]) - np.array(per_model[name][h]["t_ret"]))))
            print(f"  {name}: dir_acc={da:.2%}, nonflat={nf_acc:.2%}, MAE={mae:.4f}")
            if nf_acc > best_nf:
                best_nf = nf_acc
                best_name = name
        best_model_per_h[h] = best_name
        print(f"  -> Best: {best_name} (nonflat={best_nf:.2%})")

    # Build ensemble: for each horizon, use the best model
    print("\n=== Ensemble (auto-selected per horizon) ===")
    ens_correct = 0
    ens_total = 0
    ens_nf_correct = 0
    ens_nf_total = 0
    ens_return_errors = []
    for h in HORIZONS:
        name = best_model_per_h[h]
        t_dirs = np.array(per_model[name][h]["t_dir"])
        p_dirs = np.array(per_model[name][h]["pred_dir"])
        p_rets = np.array(per_model[name][h]["pred_ret"])
        t_rets = np.array(per_model[name][h]["t_ret"])
        nonflat_mask = t_dirs != FLAT_CLASS
        nf_total = nonflat_mask.sum()
        nf_correct = ((p_dirs == t_dirs) & nonflat_mask).sum()
        da = (p_dirs == t_dirs).mean()
        nf_acc = nf_correct / nf_total if nf_total > 0 else 0.0
        mae = float(np.mean(np.abs(p_rets - t_rets)))
        print(f"  h={h}: model={name}, dir_acc={da:.2%}, nonflat={nf_acc:.2%}, MAE={mae:.4f}")
        ens_correct += (p_dirs == t_dirs).sum()
        ens_total += len(t_dirs)
        ens_nf_correct += nf_correct
        ens_nf_total += nf_total
        ens_return_errors.extend((p_rets - t_rets).tolist())

    ens_da = ens_correct / ens_total
    ens_nf = ens_nf_correct / ens_nf_total if ens_nf_total > 0 else 0.0
    ens_mae = float(np.mean(np.abs(ens_return_errors)))
    print(f"\nEnsemble overall: dir_acc={ens_da:.2%}, nonflat={ens_nf:.2%}, MAE={ens_mae:.4f}")

    output = {
        "symbol": SYMBOL,
        "candidates": list(CANDIDATES.keys()),
        "best_model_per_horizon": {str(h): best_model_per_h[h] for h in HORIZONS},
        "ensemble": {
            "direction_accuracy_overall": float(ens_da),
            "nonflat_accuracy_overall": float(ens_nf),
            "return_mae_overall": ens_mae,
        },
        "per_model_per_horizon": {
            name: {
                str(h): {
                    "direction_accuracy": float((np.array(per_model[name][h]["pred_dir"]) == np.array(per_model[name][h]["t_dir"])).mean()),
                    "nonflat_accuracy": (
                        float(((np.array(per_model[name][h]["pred_dir"]) == np.array(per_model[name][h]["t_dir"])) &
                              (np.array(per_model[name][h]["t_dir"]) != FLAT_CLASS)).sum() /
                             (np.array(per_model[name][h]["t_dir"]) != FLAT_CLASS).sum())
                        if (np.array(per_model[name][h]["t_dir"]) != FLAT_CLASS).sum() > 0 else 0.0
                    ),
                    "return_mae": float(np.mean(np.abs(np.array(per_model[name][h]["pred_ret"]) - np.array(per_model[name][h]["t_ret"])))),
                }
                for h in HORIZONS
            }
            for name in CANDIDATES
        },
    }
    out_path = ROOT / "outputs" / f"eval_{SYMBOL}_r10_ensemble.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
