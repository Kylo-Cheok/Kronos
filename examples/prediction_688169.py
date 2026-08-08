# -*- coding: utf-8 -*-
"""
prediction_688169.py

Description:
    Predicts future daily K-line (1D) data for stock 688169 (石头科技) using the
    Kronos foundation model. Uses the locally cached historical data in
    `data/688169_daily.csv` (no network calls) and the local Kronos-base weights.

Outputs:
    - outputs/pred_688169_data.csv       (merged history + prediction)
    - outputs/pred_688169_chart.png      (history vs prediction chart)
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Make the project root importable when run from examples/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import KronosPredictor, load_tokenizer, load_model

# ----------------------------- configuration -------------------------------- #
SYMBOL = "688169"
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    f"{SYMBOL}_daily.csv",
)
SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)

TOKENIZER_NAME = "Kronos-Tokenizer-base"
MODEL_NAME = "Kronos-base"
DEVICE = "cuda:0"  # set to "cpu" if no GPU available
MAX_CONTEXT = 512
LOOKBACK = 400        # historical context length
PRED_LEN = 60         # forecast horizon (trading days)
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1
LIMIT_RATE = 0.20     # 688169 is a STAR market stock → ±20% daily price limit


def load_local_data(path: str) -> pd.DataFrame:
    print(f"📥 Loading local data from: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns are floats
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix invalid open values
    bad_open = (df["open"].isna()) | (df["open"] == 0)
    if bad_open.any():
        df.loc[bad_open, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # Fix missing amount
    if "amount" not in df.columns or df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    print(
        f"✅ Loaded {len(df)} rows, range: {df['date'].min().date()} ~ {df['date'].max().date()}"
    )
    print(f"   Latest close: {df['close'].iloc[-1]:.2f}")
    return df


def prepare_inputs(df: pd.DataFrame):
    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = df.iloc[-LOOKBACK:]["date"].reset_index(drop=True)
    y_timestamp = pd.Series(
        pd.bdate_range(
            start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN
        )
    )
    return x_df, x_timestamp, y_timestamp


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float):
    """Clamp each forecast day to the daily ±limit_rate band relative to the
    previous day's close (so it respects the A-share STAR market ±20% rule)."""
    print(f"🔒 Applying ±{limit_rate * 100:.0f}% daily price limit ...")
    pred_df = pred_df.reset_index(drop=True).copy()
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    prev_close = float(last_close)
    for i in range(len(pred_df)):
        up = prev_close * (1 + limit_rate)
        down = prev_close * (1 - limit_rate)
        for col in cols:
            v = pred_df.at[i, col]
            if pd.notna(v):
                pred_df.at[i, col] = float(max(min(v, up), down))
        prev_close = float(pred_df.at[i, "close"])
    return pred_df


def plot_result(df_hist: pd.DataFrame, df_pred: pd.DataFrame, symbol: str):
    plt.figure(figsize=(13, 6))
    plt.plot(df_hist["date"], df_hist["close"], label="Historical", color="steelblue")
    plt.plot(
        df_pred["date"],
        df_pred["close"],
        label="Predicted",
        color="tomato",
        linestyle="--",
    )
    plt.title(f"Kronos Prediction for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Close Price (CNY)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"pred_{symbol}_chart.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Chart saved: {plot_path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"🚀 Loading Kronos tokenizer:{TOKENIZER_NAME} model:{MODEL_NAME} ...")
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    model = load_model(MODEL_NAME)
    predictor = KronosPredictor(
        model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT
    )

    df = load_local_data(CSV_PATH)
    x_df, x_timestamp, y_timestamp = prepare_inputs(df)

    print(
        f"🔮 Generating predictions | lookback={LOOKBACK} pred_len={PRED_LEN} device={DEVICE}"
    )
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=True,
    )
    pred_df["date"] = y_timestamp.values

    # Apply A-share STAR market ±20% daily price limit
    last_close = float(df["close"].iloc[-1])
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=LIMIT_RATE)

    # Merge historical + predicted for the saved CSV
    df_out = pd.concat(
        [
            df[["date", "open", "high", "low", "close", "volume", "amount"]],
            pred_df[["date", "open", "high", "low", "close", "volume", "amount"]],
        ]
    ).reset_index(drop=True)

    out_file = os.path.join(SAVE_DIR, f"pred_{SYMBOL}_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction completed and saved: {out_file}")

    plot_result(df, pred_df, SYMBOL)

    # Brief summary
    pred_first = float(pred_df["close"].iloc[0])
    pred_last = float(pred_df["close"].iloc[-1])
    print("-" * 60)
    print(f"Last historical close ({df['date'].iloc[-1].date()}): {last_close:.2f}")
    print(
        f"Predicted first day ({pred_df['date'].iloc[0].date()}): {pred_first:.2f} "
        f"({(pred_first / last_close - 1) * 100:+.2f}%)"
    )
    print(
        f"Predicted last day  ({pred_df['date'].iloc[-1].date()}): {pred_last:.2f} "
        f"({(pred_last / last_close - 1) * 100:+.2f}%)"
    )


if __name__ == "__main__":
    main()
