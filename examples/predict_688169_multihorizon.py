"""Predict 688169 future direction/returns: compare three models side by side.

Models:
  1. Exp F (frozen + cw): frozen backbone + class-weighted direction head, no consistency loss
  2. Exp F-consist: same as Exp F plus sign-consistency loss between direction/return heads
  3. Public Kronos-base: public weights, KronosPredictor forecasts OHLCV

Output:
  - outputs/pred_688169_multihorizon.json  (machine-readable)
  - stdout three-way comparison table with consistency check
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from model import Kronos, KronosPredictor, KronosTokenizer, load_model, load_tokenizer
from multihorizon_objective import (
    DEFAULT_HORIZONS,
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    MultiHorizonForecastHead,
    make_direction_deadzones,
)

SYMBOL = "688169"
CSV_PATH = ROOT / "data" / "a_share_finetune_multiboard" / "csv" / f"{SYMBOL}.csv"
TOKENIZER_FT_PATH = ROOT / "outputs" / "models" / "a_share_multi_tokenizer" / "checkpoints" / "best_model"
# Two Exp F variants: without and with consistency loss.
EXP_F_NO_CONSIST_DIR = ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_frozen_d025_cw" / "checkpoints" / "best_model"
EXP_F_CONSIST_DIR = ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_frozen_cw_consist" / "checkpoints" / "best_model"
PUBLIC_TOKENIZER = "Kronos-Tokenizer-base"
PUBLIC_MODEL = "Kronos-base"

LOOKBACK = 128
HORIZONS = list(DEFAULT_HORIZONS)  # [1, 3, 5, 10]
CLIP = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LIMIT_RATE = 0.20  # 688169 is a Ke Chuang Ban stock: ±20% daily limit

CLASS_NAMES = {DOWN_CLASS: "DOWN", FLAT_CLASS: "FLAT", UP_CLASS: "UP"}


def load_multiboard_csv() -> pd.DataFrame:
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


def normalize_window(features: np.ndarray) -> np.ndarray:
    lookback = features[:LOOKBACK]
    mean = lookback.mean(axis=0, keepdims=True)
    std = lookback.std(axis=0, keepdims=True)
    normalized = (features - mean) / (std + 1e-5)
    return np.clip(normalized, -CLIP, CLIP)


@torch.no_grad()
def predict_exp_f_variant(df: pd.DataFrame, model_dir: Path, label: str) -> dict:
    """Run a frozen-backbone + multihorizon head variant on the latest window."""
    window = df.iloc[-LOOKBACK:].copy()
    features = window[["open", "high", "low", "close", "vol", "amt"]].to_numpy(dtype=np.float32)
    raw_close = window["close"].to_numpy(dtype=np.float32)
    x_norm = normalize_window(features)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
    stamp = derive_time_features(window["timestamps"])
    stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
    raw_close_tensor = torch.from_numpy(raw_close).unsqueeze(0).to(DEVICE)

    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_FT_PATH))
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

    token_seq_0, token_seq_1 = tokenizer.encode(x_tensor, half=True)
    # No :-1 slicing: we only need hidden states for the head, and keeping all
    # tokens keeps hidden_states length == LOOKBACK so context_length is valid.
    _, _, hidden_states = model(token_seq_0, token_seq_1, stamp_tensor, return_context=True)
    outputs = head(hidden_states, context_length=LOOKBACK)
    direction_logits = outputs["direction_logits"][0]  # [H, 3]
    return_pred = outputs["return_prediction"][0]  # [H]
    direction_probs = torch.softmax(direction_logits, dim=-1)
    pred_direction = direction_logits.argmax(dim=-1)

    deadzones = make_direction_deadzones(
        raw_close_tensor, context_length=LOOKBACK, horizons=horizons,
        min_deadzone=min_deadzone, volatility_multiplier=vol_mult,
    )[0]

    current_close = float(raw_close[-1])
    results = {
        "model": label,
        "model_dir": str(model_dir),
        "context_end_date": str(window["timestamps"].iloc[-1].date()),
        "current_close": current_close,
        "horizons": {},
    }
    for hi, h in enumerate(horizons):
        log_ret = float(return_pred[hi].item())
        pred_close = current_close * float(np.exp(log_ret))
        dir_class = int(pred_direction[hi].item())
        probs = direction_probs[hi].cpu().tolist()
        # Consistency check: direction sign vs return sign.
        dir_sign = 1 if dir_class == UP_CLASS else (-1 if dir_class == DOWN_CLASS else 0)
        ret_sign = 1 if log_ret > 0 else (-1 if log_ret < 0 else 0)
        consistent = (dir_sign == 0) or (dir_sign == ret_sign)
        results["horizons"][str(h)] = {
            "direction": CLASS_NAMES[dir_class],
            "direction_probs": {"down": probs[0], "flat": probs[1], "up": probs[2]},
            "predicted_log_return": log_ret,
            "predicted_return_pct": log_ret * 100,
            "predicted_close": pred_close,
            "deadzone_log_return": float(deadzones[hi].item()),
            "direction_return_consistent": consistent,
        }
    return results


@torch.no_grad()
def predict_public_kronos(df: pd.DataFrame) -> dict:
    """Run public Kronos KronosPredictor and derive direction/return at horizons."""
    history = df.iloc[-LOOKBACK:].copy()
    x_df = history[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = history["timestamps"].reset_index(drop=True)
    last_date = history["timestamps"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max(HORIZONS))
    y_timestamp = pd.Series(future_dates)

    tokenizer = load_tokenizer(PUBLIC_TOKENIZER)
    model = load_model(PUBLIC_MODEL)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=max(HORIZONS), T=1.0, top_p=0.9, sample_count=1,
        verbose=False, deterministic=True,
    )
    pred_close = pred_df["close"].to_numpy(dtype=float)
    current_close = float(history["close"].iloc[-1])

    # Apply ±20% daily price limit sequentially (Ke Chuang Ban rule).
    prev = current_close
    limited_close = []
    for c in pred_close:
        upper = prev * (1 + LIMIT_RATE)
        lower = prev * (1 - LIMIT_RATE)
        c_limited = float(np.clip(c, lower, upper))
        limited_close.append(c_limited)
        prev = c_limited
    limited_close = np.array(limited_close)

    hist_close = history["close"].to_numpy(dtype=float)
    past_log_returns = np.log(hist_close[1:] / hist_close[:-1])
    daily_vol = float(np.std(past_log_returns))
    results = {
        "model": "Public Kronos-base",
        "context_end_date": str(history["timestamps"].iloc[-1].date()),
        "current_close": current_close,
        "horizons": {},
    }
    for h in HORIZONS:
        idx = h - 1
        future_c = float(limited_close[idx])
        log_ret = float(np.log(future_c / current_close))
        deadzone = max(daily_vol * np.sqrt(h) * 0.5, 0.003)
        if log_ret > deadzone:
            dir_name = "UP"
        elif log_ret < -deadzone:
            dir_name = "DOWN"
        else:
            dir_name = "FLAT"
        dir_sign = 1 if dir_name == "UP" else (-1 if dir_name == "DOWN" else 0)
        ret_sign = 1 if log_ret > 0 else (-1 if log_ret < 0 else 0)
        consistent = (dir_sign == 0) or (dir_sign == ret_sign)
        results["horizons"][str(h)] = {
            "direction": dir_name,
            "predicted_log_return": log_ret,
            "predicted_return_pct": log_ret * 100,
            "predicted_close": future_c,
            "deadzone_log_return": deadzone,
            "direction_return_consistent": consistent,
        }
    return results


def print_three_way_comparison(no_consist: dict, consist: dict, public: dict) -> None:
    print("=" * 110)
    print(f"688169 three-way forecast  (context end: {no_consist['context_end_date']}, last close: {no_consist['current_close']:.2f})")
    print("=" * 110)
    header = f"{'Horizon':<8}{'Model':<26}{'Direction':<10}{'LogRet':>10}{'Ret%':>9}{'PredClose':>11}{'Consistent?':>13}"
    print(header)
    print("-" * 110)
    for h in HORIZONS:
        for label, res in [
            ("Exp F (no consist)", no_consist),
            ("Exp F (consist)", consist),
            ("Public Kronos", public),
        ]:
            r = res["horizons"][str(h)]
            ok = "YES" if r["direction_return_consistent"] else "** NO **"
            print(f"h={h:<6}{label:<26}{r['direction']:<10}{r['predicted_log_return']:>10.4f}{r['predicted_return_pct']:>8.2f}%{r['predicted_close']:>11.2f}{ok:>13}")
            if "direction_probs" in r:
                probs = r["direction_probs"]
                print(f"{'':<8}{'  probs':<26}D={probs['down']:.2f} F={probs['flat']:.2f} U={probs['up']:.2f}")
        print("-" * 110)

    # Consistency summary.
    nc_ok = sum(no_consist["horizons"][str(h)]["direction_return_consistent"] for h in HORIZONS)
    c_ok = sum(consist["horizons"][str(h)]["direction_return_consistent"] for h in HORIZONS)
    pb_ok = sum(public["horizons"][str(h)]["direction_return_consistent"] for h in HORIZONS)
    print(f"Consistency: Exp F (no consist) {nc_ok}/4   Exp F (consist) {c_ok}/4   Public {pb_ok}/4")


def main() -> int:
    df = load_multiboard_csv()
    print(f"Loaded {SYMBOL}: {df['timestamps'].iloc[0].date()} -> {df['timestamps'].iloc[-1].date()} ({len(df)} bars)")

    no_consist = predict_exp_f_variant(df, EXP_F_NO_CONSIST_DIR, "Exp F (frozen + cw, no consist)")
    consist = predict_exp_f_variant(df, EXP_F_CONSIST_DIR, "Exp F (frozen + cw + consist)")
    public = predict_public_kronos(df)

    print_three_way_comparison(no_consist, consist, public)

    output = {
        "symbol": SYMBOL,
        "exp_f_no_consist": no_consist,
        "exp_f_consist": consist,
        "public": public,
    }
    out_path = ROOT / "outputs" / f"pred_{SYMBOL}_multihorizon.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
