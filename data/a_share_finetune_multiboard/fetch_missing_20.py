# -*- coding: utf-8 -*-
"""Fetch full qfq daily history for the 20 expansion symbols and write standard CSVs.

The 20 symbols (banks / brokers / real-estate / construction) were targeted by
``build_20_stocks.py`` via the TDX MCP path, but no TDX data ever arrived, so
their CSVs are header-only stubs.  The TDX MCP HTTP channel is now dead (JWT
401) and the eastmoney akshare endpoint is blocked on this network, so we fetch
from akshare's sina source (``stock_zh_a_daily``), which returns qfq daily bars
with volume in shares (matching the TDX ``RawVolume`` convention of existing
CSVs).

Output format matches the canonical CSV contract used by
``data/extend_multiboard_from_json.py``:

    timestamps, open, high, low, close, volume, amount

Run from the repository root::

    python data/a_share_finetune_multiboard/fetch_missing_20.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_DIR = BASE / "csv"
MIN_DATE = "2010-01-01"
CANONICAL_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
MIN_ACCEPTABLE_ROWS = 500  # Safety floor: refuse to overwrite a CSV with junk

# (code, sina_prefix, name)
STOCKS: list[tuple[str, str, str]] = [
    ("601328", "sh", "交通银行"),
    ("600016", "sh", "华夏银行"),
    ("601009", "sh", "南京银行"),
    ("002142", "sz", "宁波银行"),
    ("600926", "sh", "杭州银行"),
    ("601169", "sh", "北京银行"),
    ("601788", "sh", "光大证券"),
    ("601555", "sh", "东吴证券"),
    ("601198", "sh", "东兴证券"),
    ("600340", "sh", "华夏幸福"),
    ("001979", "sz", "招商蛇口"),
    ("600606", "sh", "绿地控股"),
    ("600383", "sh", "金地集团"),
    ("601800", "sh", "中国交建"),
    ("601669", "sh", "中国电建"),
    ("601186", "sh", "中国铁建"),
    ("601390", "sh", "中国中铁"),
    ("600528", "sh", "中铁工业"),
    ("601117", "sh", "中国化学"),
    ("600170", "sh", "上海建工"),
]


def fetch_sina(code: str, prefix: str) -> Optional[pd.DataFrame]:
    """Fetch qfq daily bars from akshare sina source with retry."""
    import akshare as ak

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # network hiccups are common; retry
            last_error = exc
        time.sleep(2 * (attempt + 1))
    print(f"    ERROR after 3 attempts: {last_error}")
    return None


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a sina frame to the canonical CSV contract and validate it."""
    out = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float).round().astype(np.int64),
            "amount": df["amount"].astype(float),
        }
    )
    # Range filter and sanity checks (mirror the merge-script conventions).
    out = out[out["timestamps"] >= MIN_DATE]
    out = out[out["close"] > 0]
    out = out[out["volume"] > 0]
    out = out[
        (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-6)
        & (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-6)
        & (out["high"] >= out["low"] - 1e-6)
    ]
    out = out.dropna(subset=CANONICAL_COLUMNS)
    out = out.drop_duplicates(subset="timestamps", keep="last")
    out = out.sort_values("timestamps").reset_index(drop=True)
    if not np.isfinite(out[CANONICAL_COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite values survived filtering")
    return out[CANONICAL_COLUMNS]


def save_safely(frame: pd.DataFrame, code: str) -> bool:
    """Write the CSV via a temp file only when the result is substantial."""
    if len(frame) < MIN_ACCEPTABLE_ROWS:
        print(f"    SKIP: only {len(frame)} rows (< {MIN_ACCEPTABLE_ROWS})")
        return False
    out_path = CSV_DIR / f"{code}.csv"
    tmp_path = out_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return True


def main() -> int:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed, skipped = [], [], []
    for code, prefix, name in STOCKS:
        print(f"{code} {name} ...", flush=True)
        df = fetch_sina(code, prefix)
        if df is None:
            failed.append((code, name, "fetch failed"))
            continue
        try:
            frame = to_canonical(df)
        except ValueError as exc:
            failed.append((code, name, str(exc)))
            continue
        if save_safely(frame, code):
            ok.append((code, name, len(frame), frame["timestamps"].iloc[0], frame["timestamps"].iloc[-1]))
        else:
            skipped.append((code, name, len(frame)))
        time.sleep(1)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for code, name, n, first, last in ok:
        print(f"  OK   {code} {name:<6} {n:>5} rows  {first} -> {last}")
    for code, name, n in skipped:
        print(f"  SKIP {code} {name:<6} {n} rows (too few)")
    for code, name, reason in failed:
        print(f"  FAIL {code} {name:<6} {reason}")
    print(f"\n{len(ok)} OK, {len(skipped)} skipped, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
