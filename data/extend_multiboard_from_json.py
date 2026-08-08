"""Extend the multiboard fine-tuning CSVs with new bars from a JSON file and
rebuild the pickle dataset + manifest.

The JSON file (``data/tdx_latest_bars.json``) maps each six-digit symbol to a
list of daily bar dicts with keys: date (YYYY-MM-DD), open, high, low, close,
volume (shares), amount (yuan).  Only bars whose date is strictly later than the
last existing timestamp are appended; bars colliding with existing dates are
validated against the existing row (qfq price continuity check).

After extending the CSVs the script rebuilds ``train_data.pkl`` / ``val_data.pkl``
/ ``test_data.pkl`` and ``manifest.json`` in the target output directory using
the same split logic and contract as ``build_local_finetune_dataset.py``.

Run from the repository root::

    python data/extend_multiboard_from_json.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = ROOT / "data" / "a_share_finetune_multiboard" / "csv"
DEFAULT_NEW_BARS = ROOT / "data" / "tdx_latest_bars.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "a_share_finetune_multiboard"
CANONICAL_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
PICKLE_FEATURE_COLUMNS = ["open", "high", "low", "close", "vol", "amt"]
DEFAULT_LOOKBACK = 128
DEFAULT_PREDICT_WINDOW = 10
DEFAULT_TRAIN_END = "2024-12-31"
DEFAULT_VAL_END = "2025-12-15"


def load_existing_csvs(csv_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        symbol = path.stem
        df = pd.read_csv(path)
        df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.normalize()
        frames[symbol] = df
    return frames


def extend_frame(frame: pd.DataFrame, new_bars: List[dict], symbol: str) -> pd.DataFrame:
    """Append strictly-new bars and validate overlap bars for qfq continuity."""
    if not new_bars:
        return frame
    last_ts = frame["timestamps"].iloc[-1]
    appended = 0
    validated = 0
    for bar in new_bars:
        bar_ts = pd.Timestamp(bar["date"]).normalize()
        row = {
            "timestamps": bar_ts,
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
            "amount": float(bar["amount"]),
        }
        # OHLC sanity
        if row["high"] < max(row["open"], row["close"]) or row["low"] > min(row["open"], row["close"]) or row["high"] < row["low"]:
            raise ValueError(f"{symbol} {bar['date']}: invalid OHLC relationships")
        if row["open"] <= 0 or row["close"] <= 0:
            raise ValueError(f"{symbol} {bar['date']}: non-positive price")
        if bar_ts in set(frame["timestamps"]):
            # Overlap: validate qfq continuity
            existing = frame.loc[frame["timestamps"] == bar_ts].iloc[0]
            for col in ("open", "high", "low", "close"):
                if abs(existing[col] - row[col]) > 1e-2:
                    raise ValueError(
                        f"{symbol} {bar['date']}: qfq mismatch on {col}: "
                        f"existing={existing[col]} tdx={row[col]}"
                    )
            validated += 1
        elif bar_ts > last_ts:
            # New bar: append
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            appended += 1
            last_ts = bar_ts
        # bar_ts < last_ts and not in frame: ignore (gap fill not supported here)
    print(f"  {symbol}: appended {appended} new bar(s), validated {validated} overlap bar(s)")
    return frame


def as_pickle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["vol"] = result["volume"]
    result["amt"] = result["amount"]
    result = result.set_index("timestamps")
    result.index.name = "datetime"
    return result[PICKLE_FEATURE_COLUMNS]


def split_frames(
    frames: Mapping[str, pd.DataFrame],
    train_end: str,
    val_end: str,
    window: int,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    train_cut = pd.Timestamp(train_end)
    val_cut = pd.Timestamp(val_end)
    splits: Dict[str, Dict[str, pd.DataFrame]] = {"train": {}, "val": {}, "test": {}}
    for symbol, frame in frames.items():
        indexed = as_pickle_frame(frame)
        train = indexed[indexed.index <= train_cut]
        val = indexed[(indexed.index > train_cut) & (indexed.index <= val_cut)]
        test = indexed[indexed.index > val_cut]
        for name, part in (("train", train), ("val", val), ("test", test)):
            if len(part) < window:
                raise ValueError(
                    f"{symbol}: {name} split has {len(part)} rows; need at least {window}."
                )
            splits[name][symbol] = part
    return splits


def write_dataset(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    splits: Mapping[str, Mapping[str, pd.DataFrame]],
    train_end: str,
    val_end: str,
    lookback_window: int,
    predict_window: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    file_records = []
    for symbol, frame in frames.items():
        output_path = csv_dir / f"{symbol}.csv"
        frame.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        file_records.append({
            "symbol": symbol,
            "file": str(Path("csv") / f"{symbol}.csv"),
            "rows": len(frame),
            "start": str(frame["timestamps"].iloc[0].date()),
            "end": str(frame["timestamps"].iloc[-1].date()),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        })
    for split_name, split in splits.items():
        with (output_dir / f"{split_name}_data.pkl").open("wb") as handle:
            pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema_version": 1,
        "source_dir": str(output_dir),
        "symbols": sorted(frames),
        "symbol_count": len(frames),
        "board_scope": "Live multi-board source; every symbol passed current CSI300, ChiNext, or STAR eligibility.",
        "columns": CANONICAL_COLUMNS,
        "pickle_columns": PICKLE_FEATURE_COLUMNS,
        "lookback_window": lookback_window,
        "predict_window": predict_window,
        "window": lookback_window + predict_window + 1,
        "row_count": sum(len(frame) for frame in frames.values()),
        "records": sorted(file_records, key=lambda r: r["symbol"]),
        "splits": {"train_end": train_end, "val_end": val_end},
        "rows_by_symbol": {
            symbol: {
                "total": len(frame),
                "start": str(frame["timestamps"].iloc[0].date()),
                "end": str(frame["timestamps"].iloc[-1].date()),
                "train": len(splits["train"][symbol]),
                "val": len(splits["val"][symbol]),
                "test": len(splits["test"][symbol]),
            }
            for symbol, frame in sorted(frames.items())
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--new-bars", type=Path, default=DEFAULT_NEW_BARS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--val-end", default=DEFAULT_VAL_END)
    parser.add_argument("--lookback-window", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--predict-window", type=int, default=DEFAULT_PREDICT_WINDOW)
    args = parser.parse_args()

    window = args.lookback_window + args.predict_window + 1
    frames = load_existing_csvs(args.csv_dir)
    print(f"Loaded {len(frames)} symbols from {args.csv_dir}")

    if not args.new_bars.exists():
        print(f"WARNING: {args.new_bars} not found; rebuilding pickles only")
    else:
        new_bars_map = json.loads(args.new_bars.read_text(encoding="utf-8"))
        print(f"Extending {len(new_bars_map)} symbols from {args.new_bars}")
        for symbol, bars in new_bars_map.items():
            if symbol not in frames:
                print(f"  WARNING: symbol {symbol} not in existing CSVs; skipping")
                continue
            frames[symbol] = extend_frame(frames[symbol], bars, symbol)

    # Final validation pass
    for symbol, frame in frames.items():
        if frame["timestamps"].duplicated().any():
            raise ValueError(f"{symbol}: duplicate timestamps after extension")
        if not frame["timestamps"].is_monotonic_increasing:
            frames[symbol] = frame.sort_values("timestamps").reset_index(drop=True)
        if (frame[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"{symbol}: non-positive OHLC after extension")
        if not np.isfinite(frame[CANONICAL_COLUMNS[1:]].to_numpy(dtype=float)).all():
            raise ValueError(f"{symbol}: non-finite values after extension")

    splits = split_frames(frames, args.train_end, args.val_end, window)
    write_dataset(
        args.output_dir, frames, splits, args.train_end, args.val_end,
        args.lookback_window, args.predict_window,
    )
    total = sum(len(f) for f in frames.values())
    print(f"\nRebuilt {len(frames)} symbols / {total} rows -> {args.output_dir}")
    for symbol in sorted(frames):
        f = frames[symbol]
        print(f"  {symbol}: {len(f)} rows, {f['timestamps'].iloc[0].date()} -> {f['timestamps'].iloc[-1].date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
