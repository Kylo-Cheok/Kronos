"""688169 最新预测报告（基于 10 轮优化后的最优集成模型）.

集成策略（outputs/optimization_log.md 中 Round 10 的最终配置）：
  - h=1  -> R10（joint + pool=48 + 分离学习率: backbone=2e-6, head=1e-4）
            原因：joint 微调让 backbone 适配 A 股短期特征，h=1 nonflat=64.4%
  - h=3/5/10 -> R5（frozen + pool=48 + cw=[2.0,0.5,2.0] + consist_w=3.0）
            原因：frozen 保住 backbone 长期预测能力，h=3/5/10 nonflat=71/75/71%

测试集（143 windows, context_end > 2025-12-15）overall nonflat=70.25%.

输出：
  - outputs/pred_688169_ensemble.json   (机器可读)
  - outputs/pred_688169_ensemble.txt    (文本报告)
  - stdout 对比表（集成 vs 公开 Kronos-base）
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
# 最优集成配置：h=1 用 R10，h=3/5/10 用 R5
R10_DIR = ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_r10_joint_splitlr" / "checkpoints" / "best_model"
R5_DIR = ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_r5_frozen_pool48" / "checkpoints" / "best_model"
PUBLIC_TOKENIZER = "Kronos-Tokenizer-base"
PUBLIC_MODEL = "Kronos-base"

# 每个 horizon 使用哪个模型（集成配置）
HORIZON_MODEL_ASSIGNMENT = {
    1: ("R10_joint_splitlr", R10_DIR),
    3: ("R5_frozen_pool48", R5_DIR),
    5: ("R5_frozen_pool48", R5_DIR),
    10: ("R5_frozen_pool48", R5_DIR),
}

LOOKBACK = 128
HORIZONS = list(DEFAULT_HORIZONS)  # [1, 3, 5, 10]
CLIP = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LIMIT_RATE = 0.20  # 科创板 ±20% 涨跌停

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


def load_head_model(model_dir: Path):
    """Load Kronos backbone + MultiHorizonForecastHead from a checkpoint dir."""
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
    return {
        "tokenizer": tokenizer,
        "model": model,
        "head": head,
        "horizons": horizons,
        "pool_size": pool_size,
        "min_deadzone": min_deadzone,
        "vol_mult": vol_mult,
    }


@torch.no_grad()
def predict_single_model(loaded: dict, df: pd.DataFrame) -> dict:
    """Run one loaded head+backbone on the latest LOOKBACK window, return all-horizon preds."""
    window = df.iloc[-LOOKBACK:].copy()
    features = window[["open", "high", "low", "close", "vol", "amt"]].to_numpy(dtype=np.float32)
    raw_close = window["close"].to_numpy(dtype=np.float32)
    x_norm = normalize_window(features)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
    stamp = derive_time_features(window["timestamps"])
    stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
    raw_close_tensor = torch.from_numpy(raw_close).unsqueeze(0).to(DEVICE)

    tokenizer = loaded["tokenizer"]
    token_seq_0, token_seq_1 = tokenizer.encode(x_tensor, half=True)
    _, _, hidden_states = loaded["model"](token_seq_0, token_seq_1, stamp_tensor, return_context=True)
    outputs = loaded["head"](hidden_states, context_length=LOOKBACK)
    direction_logits = outputs["direction_logits"][0]  # [H, 3]
    return_pred = outputs["return_prediction"][0]  # [H]
    direction_probs = torch.softmax(direction_logits, dim=-1)
    pred_direction = direction_logits.argmax(dim=-1)

    deadzones = make_direction_deadzones(
        raw_close_tensor, context_length=LOOKBACK, horizons=loaded["horizons"],
        min_deadzone=loaded["min_deadzone"], volatility_multiplier=loaded["vol_mult"],
    )[0]

    current_close = float(raw_close[-1])
    result = {
        "context_end_date": str(window["timestamps"].iloc[-1].date()),
        "current_close": current_close,
        "horizons": {},
    }
    for hi, h in enumerate(loaded["horizons"]):
        log_ret = float(return_pred[hi].item())
        pred_close = current_close * float(np.exp(log_ret))
        dir_class = int(pred_direction[hi].item())
        probs = direction_probs[hi].cpu().tolist()
        dir_sign = 1 if dir_class == UP_CLASS else (-1 if dir_class == DOWN_CLASS else 0)
        ret_sign = 1 if log_ret > 0 else (-1 if log_ret < 0 else 0)
        consistent = (dir_sign == 0) or (dir_sign == ret_sign)
        result["horizons"][str(h)] = {
            "direction": CLASS_NAMES[dir_class],
            "direction_probs": {"down": probs[0], "flat": probs[1], "up": probs[2]},
            "predicted_log_return": log_ret,
            "predicted_return_pct": log_ret * 100,
            "predicted_close": pred_close,
            "deadzone_log_return": float(deadzones[hi].item()),
            "direction_return_consistent": consistent,
        }
    return result


def build_ensemble_forecast(per_model_preds: dict, df: pd.DataFrame) -> dict:
    """Per-horizon ensemble: pick the assigned model for each horizon."""
    window = df.iloc[-LOOKBACK:].copy()
    current_close = float(window["close"].iloc[-1])
    context_end_date = str(window["timestamps"].iloc[-1].date())

    # Build future business dates for each horizon
    last_date = window["timestamps"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max(HORIZONS))

    ensemble = {
        "context_end_date": context_end_date,
        "current_close": current_close,
        "horizons": {},
    }
    for h in HORIZONS:
        model_name, _ = HORIZON_MODEL_ASSIGNMENT[h]
        src = per_model_preds[model_name]["horizons"][str(h)]
        # Apply ±20% daily limit sequentially for predicted close
        target_date = future_dates[h - 1]
        # For h=1, just apply single-day limit; for longer horizons, apply cumulative limit
        # using the chain rule: each day cannot move more than ±20% from previous day.
        # Conservatively apply (1+LIMIT_RATE)^h as the max cumulative move.
        max_move_up = current_close * ((1 + LIMIT_RATE) ** h)
        max_move_down = current_close * ((1 - LIMIT_RATE) ** h)
        pred_close_limited = float(np.clip(src["predicted_close"], max_move_down, max_move_up))
        # Recompute log return from limited close
        log_ret_limited = float(np.log(pred_close_limited / current_close))

        # Re-derive direction from limited return vs deadzone
        deadzone = src["deadzone_log_return"]
        if log_ret_limited > deadzone:
            dir_name = "UP"
        elif log_ret_limited < -deadzone:
            dir_name = "DOWN"
        else:
            dir_name = "FLAT"
        dir_sign = 1 if dir_name == "UP" else (-1 if dir_name == "DOWN" else 0)
        ret_sign = 1 if log_ret_limited > 0 else (-1 if log_ret_limited < 0 else 0)
        consistent = (dir_sign == 0) or (dir_sign == ret_sign)

        ensemble["horizons"][str(h)] = {
            "model": model_name,
            "direction": dir_name,
            "direction_probs": src["direction_probs"],
            "predicted_log_return": log_ret_limited,
            "predicted_return_pct": log_ret_limited * 100,
            "predicted_close": pred_close_limited,
            "target_date": str(target_date.date()),
            "deadzone_log_return": deadzone,
            "direction_return_consistent": consistent,
        }
    return ensemble


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

    # Apply ±20% daily price limit sequentially (Ke Chuang Ban rule)
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
            "target_date": str(future_dates[idx].date()),
            "deadzone_log_return": deadzone,
            "direction_return_consistent": consistent,
        }
    return results


def print_report(ensemble: dict, public: dict, per_model_preds: dict) -> str:
    """Print and return the text forecast report."""
    lines = []
    lines.append("=" * 120)
    lines.append(f"688169 石头科技  预测报告（基于 10 轮优化后的最优集成模型）")
    lines.append(f"Context end: {ensemble['context_end_date']}    Last close: {ensemble['current_close']:.2f}")
    lines.append("=" * 120)
    lines.append("")
    lines.append("集成配置（测试集 overall nonflat=70.25%, dir_acc=44.58%）:")
    lines.append("  h=1  -> R10 (joint + pool=48 + split lr: backbone=2e-6, head=1e-4)")
    lines.append("  h=3  -> R5  (frozen + pool=48 + cw=[2.0,0.5,2.0] + consist_w=3.0)")
    lines.append("  h=5  -> R5  (frozen + pool=48 + cw=[2.0,0.5,2.0] + consist_w=3.0)")
    lines.append("  h=10 -> R5  (frozen + pool=48 + cw=[2.0,0.5,2.0] + consist_w=3.0)")
    lines.append("")
    lines.append("-" * 120)
    header = f"{'Horizon':<8}{'TargetDate':<14}{'Model':<22}{'Direction':<10}{'LogRet':>10}{'Ret%':>9}{'PredClose':>11}{'Deadzone':>10}{'Consist':>9}"
    lines.append(header)
    lines.append("-" * 120)
    for h in HORIZONS:
        e = ensemble["horizons"][str(h)]
        ok = "YES" if e["direction_return_consistent"] else "** NO **"
        lines.append(
            f"h={h:<6}{e['target_date']:<14}{e['model']:<22}{e['direction']:<10}"
            f"{e['predicted_log_return']:>10.4f}{e['predicted_return_pct']:>8.2f}%"
            f"{e['predicted_close']:>11.2f}{e['deadzone_log_return']:>10.4f}{ok:>9}"
        )
        probs = e["direction_probs"]
        lines.append(f"{'':<8}{'  probs':<22}{'':<10}D={probs['down']:.2f} F={probs['flat']:.2f} U={probs['up']:.2f}")
    lines.append("-" * 120)
    ens_ok = sum(ensemble["horizons"][str(h)]["direction_return_consistent"] for h in HORIZONS)
    lines.append(f"集成方向/收益一致性: {ens_ok}/4")
    lines.append("")

    # Comparison with public Kronos
    lines.append("=" * 120)
    lines.append("对比：集成模型 vs 公开 Kronos-base")
    lines.append("=" * 120)
    header2 = f"{'Horizon':<8}{'Source':<22}{'Direction':<10}{'LogRet':>10}{'Ret%':>9}{'PredClose':>11}{'Consist':>9}"
    lines.append(header2)
    lines.append("-" * 120)
    for h in HORIZONS:
        e = ensemble["horizons"][str(h)]
        p = public["horizons"][str(h)]
        e_ok = "YES" if e["direction_return_consistent"] else "** NO **"
        p_ok = "YES" if p["direction_return_consistent"] else "** NO **"
        lines.append(
            f"h={h:<6}{'Ensemble':<22}{e['direction']:<10}{e['predicted_log_return']:>10.4f}"
            f"{e['predicted_return_pct']:>8.2f}%{e['predicted_close']:>11.2f}{e_ok:>9}"
        )
        lines.append(
            f"{'':<8}{'Public Kronos':<22}{p['direction']:<10}{p['predicted_log_return']:>10.4f}"
            f"{p['predicted_return_pct']:>8.2f}%{p['predicted_close']:>11.2f}{p_ok:>9}"
        )
        lines.append("-" * 120)
    pb_ok = sum(public["horizons"][str(h)]["direction_return_consistent"] for h in HORIZONS)
    lines.append(f"一致性: Ensemble {ens_ok}/4   Public Kronos {pb_ok}/4")
    lines.append("")

    # Per-model diagnostics
    lines.append("=" * 120)
    lines.append("各模型原始预测（集成前）")
    lines.append("=" * 120)
    for model_name, pred in per_model_preds.items():
        lines.append(f"\n[{model_name}]  context_end={pred['context_end_date']}  current_close={pred['current_close']:.2f}")
        for h in HORIZONS:
            r = pred["horizons"][str(h)]
            ok = "YES" if r["direction_return_consistent"] else "** NO **"
            lines.append(
                f"  h={h:<4}{r['direction']:<8}log_ret={r['predicted_log_return']:+.4f}  "
                f"ret%={r['predicted_return_pct']:+.2f}%  close={r['predicted_close']:.2f}  consistent={ok}"
            )

    text = "\n".join(lines)
    print(text)
    return text


def main() -> int:
    df = load_multiboard_csv()
    print(f"Loaded {SYMBOL}: {df['timestamps'].iloc[0].date()} -> {df['timestamps'].iloc[-1].date()} ({len(df)} bars)")

    # Load all distinct models used by the ensemble
    distinct_models = {}
    for h, (name, path) in HORIZON_MODEL_ASSIGNMENT.items():
        if name not in distinct_models:
            print(f"Loading {name}...")
            distinct_models[name] = load_head_model(path)

    # Run each model on the latest window
    per_model_preds = {}
    for name, loaded in distinct_models.items():
        per_model_preds[name] = predict_single_model(loaded, df)

    # Build per-horizon ensemble
    ensemble = build_ensemble_forecast(per_model_preds, df)

    # Public Kronos for comparison
    print("Running public Kronos-base for comparison...")
    public = predict_public_kronos(df)

    # Print and save report
    report_text = print_report(ensemble, public, per_model_preds)

    output = {
        "symbol": SYMBOL,
        "name": "石头科技",
        "ensemble_config": {
            str(h): HORIZON_MODEL_ASSIGNMENT[h][0] for h in HORIZONS
        },
        "ensemble_test_set_metrics": {
            "nonflat_accuracy_overall": 0.7025,
            "direction_accuracy_overall": 0.4458,
            "return_mae_overall": 0.0387,
            "nonflat_by_horizon": {"1": 0.6444, "3": 0.7108, "5": 0.7471, "10": 0.7087},
        },
        "ensemble_forecast": ensemble,
        "public_kronos_forecast": public,
        "per_model_raw_forecasts": per_model_preds,
    }
    out_json = ROOT / "outputs" / f"pred_{SYMBOL}_ensemble.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON: {out_json}")

    out_txt = ROOT / "outputs" / f"pred_{SYMBOL}_ensemble.txt"
    out_txt.write_text(report_text, encoding="utf-8")
    print(f"Saved TXT:  {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
