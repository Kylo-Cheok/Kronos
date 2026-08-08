"""Compare the public and 688169-finetuned Kronos models.

The script preserves the existing public-weight forecast, generates a new
same-horizon forecast with the fine-tuned checkpoints, and runs a deterministic
60-trading-day historical comparison on the latest known data.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import Kronos, KronosPredictor, KronosTokenizer, load_model, load_tokenizer


SYMBOL = "688169"
CSV_PATH = os.path.join(ROOT, "data", f"{SYMBOL}_daily.csv")
PUBLIC_PRED_PATH = os.path.join(ROOT, "outputs", f"pred_{SYMBOL}_data.csv")
FINETUNED_TOKENIZER_PATH = os.path.join(
    ROOT, "outputs", "finetuned", f"{SYMBOL}_daily_finetune", "tokenizer", "best_model"
)
FINETUNED_MODEL_PATH = os.path.join(
    ROOT, "outputs", "finetuned", f"{SYMBOL}_daily_finetune", "basemodel", "best_model"
)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

DEVICE = "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 400
PRED_LEN = 60
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1
LIMIT_RATE = 0.20


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isna().any().any():
        raise ValueError("688169 input data contains NaN values")
    return df


def prepare_inputs(df: pd.DataFrame, timestamps: pd.Series | None = None):
    history = df.iloc[-LOOKBACK:]
    x_df = history[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = history["date"].reset_index(drop=True)
    if timestamps is None:
        timestamps = pd.Series(
            pd.bdate_range(
                start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN
            )
        )
    return x_df, x_timestamp, pd.Series(timestamps).reset_index(drop=True)


def apply_price_limits(pred_df: pd.DataFrame, last_close: float) -> pd.DataFrame:
    pred_df = pred_df.reset_index(drop=True).copy()
    price_cols = ["open", "high", "low", "close"]
    prev_close = float(last_close)
    for i in range(len(pred_df)):
        upper = prev_close * (1 + LIMIT_RATE)
        lower = prev_close * (1 - LIMIT_RATE)
        for col in price_cols:
            pred_df.at[i, col] = float(np.clip(pred_df.at[i, col], lower, upper))
        prev_close = float(pred_df.at[i, "close"])
    return pred_df


def run_predictor(predictor: KronosPredictor, history: pd.DataFrame, y_timestamps, deterministic: bool):
    x_df, x_timestamp, y_timestamp = prepare_inputs(history, y_timestamps)
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=len(y_timestamp),
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
        deterministic=deterministic,
    )
    pred_df["date"] = y_timestamp.values
    return apply_price_limits(pred_df, float(history["close"].iloc[-1]))


def forecast_summary(pred_df: pd.DataFrame, last_close: float) -> dict:
    first_close = float(pred_df["close"].iloc[0])
    last_pred_close = float(pred_df["close"].iloc[-1])
    return {
        "first_close": first_close,
        "last_close": last_pred_close,
        "return_pct": (last_pred_close / last_close - 1) * 100,
        "min_close": float(pred_df["close"].min()),
        "max_close": float(pred_df["close"].max()),
    }


def backtest_metrics(pred_df: pd.DataFrame, actual_df: pd.DataFrame, last_history_close: float) -> dict:
    actual_close = actual_df["close"].to_numpy(dtype=float)
    predicted_close = pred_df["close"].to_numpy(dtype=float)
    errors = predicted_close - actual_close

    actual_previous = np.concatenate(([last_history_close], actual_close[:-1]))
    predicted_previous = np.concatenate(([last_history_close], predicted_close[:-1]))
    actual_direction = np.sign(actual_close - actual_previous)
    predicted_direction = np.sign(predicted_close - predicted_previous)

    return {
        "mae_close": float(np.mean(np.abs(errors))),
        "rmse_close": float(np.sqrt(np.mean(errors ** 2))),
        "mape_close_pct": float(np.mean(np.abs(errors / actual_close)) * 100),
        "direction_accuracy_pct": float(np.mean(actual_direction == predicted_direction) * 100),
        "actual_return_pct": float((actual_close[-1] / last_history_close - 1) * 100),
        "predicted_return_pct": float((predicted_close[-1] / last_history_close - 1) * 100),
    }


def save_forecast_csv(history: pd.DataFrame, pred_df: pd.DataFrame, path: str):
    output = pd.concat(
        [history[["date", "open", "high", "low", "close", "volume", "amount"]], pred_df[
            ["date", "open", "high", "low", "close", "volume", "amount"]
        ]],
        ignore_index=True,
    )
    output.to_csv(path, index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(CSV_PATH)

    public_output = load_data(PUBLIC_PRED_PATH)
    public_future = public_output[public_output["date"] > df["date"].iloc[-1]].copy()
    if len(public_future) != PRED_LEN:
        raise ValueError(f"Expected {PRED_LEN} public forecast rows, got {len(public_future)}")

    tokenizer = KronosTokenizer.from_pretrained(FINETUNED_TOKENIZER_PATH)
    model = Kronos.from_pretrained(FINETUNED_MODEL_PATH)
    finetuned_predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    future_timestamps = public_future["date"].reset_index(drop=True)
    finetuned_future = run_predictor(finetuned_predictor, df, future_timestamps, deterministic=False)
    finetuned_output_path = os.path.join(OUTPUT_DIR, f"pred_{SYMBOL}_finetuned_data.csv")
    save_forecast_csv(df, finetuned_future, finetuned_output_path)

    # Use the latest 60 known trading days as a fair historical holdout.
    cutoff = len(df) - PRED_LEN
    backtest_history = df.iloc[:cutoff].copy()
    actual_future = df.iloc[cutoff:].copy().reset_index(drop=True)
    actual_timestamps = actual_future["date"]

    public_tokenizer = load_tokenizer("Kronos-Tokenizer-base")
    public_model = load_model("Kronos-base")
    public_predictor = KronosPredictor(public_model, public_tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    public_backtest = run_predictor(public_predictor, backtest_history, actual_timestamps, deterministic=True)
    finetuned_backtest = run_predictor(finetuned_predictor, backtest_history, actual_timestamps, deterministic=True)

    backtest_output = pd.DataFrame(
        {
            "date": actual_future["date"],
            "actual_close": actual_future["close"],
            "public_close": public_backtest["close"],
            "finetuned_close": finetuned_backtest["close"],
        }
    )
    backtest_output_path = os.path.join(OUTPUT_DIR, f"backtest_{SYMBOL}_model_comparison.csv")
    backtest_output.to_csv(backtest_output_path, index=False)

    metrics = {
        "symbol": SYMBOL,
        "forecast_cutoff": str(df["date"].iloc[-1].date()),
        "forecast_horizon": PRED_LEN,
        "public_forecast_file": PUBLIC_PRED_PATH,
        "finetuned_forecast_file": finetuned_output_path,
        "forecast_summary": {
            "public": forecast_summary(public_future, float(df["close"].iloc[-1])),
            "finetuned": forecast_summary(finetuned_future, float(df["close"].iloc[-1])),
        },
        "historical_backtest": {
            "cutoff": str(backtest_history["date"].iloc[-1].date()),
            "actual_start": str(actual_future["date"].iloc[0].date()),
            "actual_end": str(actual_future["date"].iloc[-1].date()),
            "public": backtest_metrics(public_backtest, actual_future, float(backtest_history["close"].iloc[-1])),
            "finetuned": backtest_metrics(finetuned_backtest, actual_future, float(backtest_history["close"].iloc[-1])),
        },
    }
    metrics_path = os.path.join(OUTPUT_DIR, f"compare_{SYMBOL}_models.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    chart_path = os.path.join(OUTPUT_DIR, f"pred_{SYMBOL}_model_compare.png")
    plt.figure(figsize=(14, 6))
    chart_history = df.iloc[-LOOKBACK:]
    plt.plot(chart_history["date"], chart_history["close"], label="Actual history", color="steelblue")
    plt.plot(public_future["date"], public_future["close"], label="Public weights", color="tomato", linestyle="--")
    plt.plot(finetuned_future["date"], finetuned_future["close"], label="Fine-tuned", color="seagreen", linestyle="--")
    plt.axvline(df["date"].iloc[-1], color="gray", linestyle=":", label="Forecast cutoff")
    plt.title(f"{SYMBOL}: public vs fine-tuned Kronos forecast")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved fine-tuned forecast: {finetuned_output_path}")
    print(f"Saved backtest comparison: {backtest_output_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved chart: {chart_path}")


if __name__ == "__main__":
    main()
