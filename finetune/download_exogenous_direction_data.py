"""Download and cache index/sector inputs for direction experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from finetune.exogenous_direction import sanitize_exogenous_frame


INDEX_ASSETS = {
    "sse": "sh000001",
    "csi300": "sh000300",
    "csi500": "sh000905",
    "star50": "sh000688",
    "szse": "sz399001",
    "chinext": "sz399006",
}

SECTOR_ASSETS = {
    "ecovacs": "sh603486",
    "midea": "sz000333",
    "gree": "sz000651",
    "supor": "sz002032",
    "flyco": "sh603868",
    "bear": "sz002959",
    "haier": "sh600690",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/exogenous"))
    parser.add_argument("--start-date", default="20190101")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def _download_with_retry(kind: str, symbol: str, start_date: str, end_date: str):
    import akshare as ak

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if kind == "index":
                return ak.stock_zh_index_daily_tx(symbol=symbol)
            return ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to download {symbol}") from last_error


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = [("index", name, symbol) for name, symbol in INDEX_ASSETS.items()]
    assets += [("sector", name, symbol) for name, symbol in SECTOR_ASSETS.items()]
    for position, (kind, name, symbol) in enumerate(assets, start=1):
        path = args.output_dir / f"{name}.csv"
        if path.exists() and not args.refresh:
            frame = sanitize_exogenous_frame(pd.read_csv(path), name)
            source = "cache"
        else:
            raw = _download_with_retry(kind, symbol, args.start_date, args.end_date)
            frame = sanitize_exogenous_frame(raw, name)
            frame = frame[
                (frame["date"] >= pd.Timestamp(args.start_date))
                & (frame["date"] <= pd.Timestamp(args.end_date))
            ].reset_index(drop=True)
            frame.to_csv(path, index=False)
            source = "download"
        print(
            f"[{position}/{len(assets)}] {name:8s} {len(frame):4d} rows "
            f"{frame['date'].min().date()}..{frame['date'].max().date()} ({source})"
        )


if __name__ == "__main__":
    main()
