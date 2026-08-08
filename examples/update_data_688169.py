# -*- coding: utf-8 -*-
"""
update_data_688169.py

Description:
    Merge the latest TDX K-line data (saved by the agent as
    data/688169_tdx_latest.json) into data/688169_daily.csv.
    Only appends rows whose date is newer than the last date in the CSV.

Workflow (run by the agent):
    1. Agent calls the TDX MCP `tdx_kline` tool for 688169 (daily, qfq).
    2. Agent saves the parsed TDX response to data/688169_tdx_latest.json.
    3. This script reads that JSON and merges new rows into the daily CSV.
    4. examples/prediction_688169.py runs next on the refreshed CSV.

Columns (English, matching the existing CSV):
    date, open, high, low, close, volume, amount
"""

import os
import json
import sys

import pandas as pd

SYMBOL = "688169"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", f"{SYMBOL}_daily.csv")
TDX_JSON_PATH = os.path.join(PROJECT_ROOT, "data", f"{SYMBOL}_tdx_latest.json")


def parse_tdx_rows(obj: dict) -> pd.DataFrame:
    """Convert TDX kline Rows into the daily-CSV schema."""
    rows = []
    for r in obj.get("Rows", []):
        # TDX "Data" is "YYYYMMDD"; Open/High/Low/Close are strings;
        # Volume is the final usable volume (手); Amount is the final amount (元).
        rows.append({
            "date": pd.to_datetime(r["Data"], format="%Y%m%d"),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": float(r["Volume"]),
            "amount": float(r["Amount"]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def main():
    if not os.path.exists(TDX_JSON_PATH):
        print(f"❌ TDX JSON not found: {TDX_JSON_PATH}")
        print("   The agent must call the TDX MCP and save the response first.")
        sys.exit(1)

    if not os.path.exists(CSV_PATH):
        print(f"❌ Daily CSV not found: {CSV_PATH}")
        sys.exit(1)

    with open(TDX_JSON_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)
    new = parse_tdx_rows(obj)
    print(f"🌐 TDX rows parsed: {len(new)}, "
          f"range {new['date'].min().date()} ~ {new['date'].max().date()}")

    old = pd.read_csv(CSV_PATH)
    old["date"] = pd.to_datetime(old["date"])
    last_date = old["date"].max()
    print(f"📦 Existing CSV: {len(old)} rows, last date = {last_date.date()}")

    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    added = int((merged["date"] > last_date).sum())
    print(f"➕ New rows appended: {added}")

    if added == 0:
        print("ℹ️  No new trading data to add.")
        print(f"   Latest date remains: {merged['date'].max().date()}")
        return False

    merged.to_csv(CSV_PATH, index=False)
    print(f"✅ Updated CSV: {CSV_PATH} (total {len(merged)} rows)")
    print("Latest 3 rows:")
    print(merged.tail(3).to_string(index=False))
    return True


if __name__ == "__main__":
    main()
