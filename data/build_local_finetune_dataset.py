"""Build a multi-symbol fine-tuning dataset from local A-share CSV files.

This is an offline companion to ``prepare_a_share_finetune.py``. It uses the
28 locally cached STAR Market files under ``data/direction_universe`` and
produces the pickle layout expected by ``finetune/dataset.py``. It can also
consume a directory of canonical per-symbol CSVs produced by the online
preparation pipeline.

The default local source is explicitly limited to STAR Market codes (688/689),
so it does not claim current CSI 300 membership for cached files. When the
online pipeline succeeds, point ``--source-dir`` at its output directory to
consume the live-eligibility manifest instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "direction_universe"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "a_share_finetune_local"
CANONICAL_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
PICKLE_FEATURE_COLUMNS = ["open", "high", "low", "close", "vol", "amt"]
DEFAULT_LOOKBACK = 128
DEFAULT_PREDICT_WINDOW = 10


def symbol_from_path(path: Path) -> str:
    match = re.fullmatch(r"(\d{6})(?:_qfq)?", path.stem)
    if not match:
        raise ValueError(f"Cannot infer a six-digit symbol from {path.name}")
    return match.group(1)


def canonicalize(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    aliases = {
        "timestamps": ("timestamps", "timestamp", "date"),
        "open": ("open",),
        "high": ("high",),
        "low": ("low",),
        "close": ("close",),
        "volume": ("volume", "vol"),
        "amount": ("amount", "amt"),
    }
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    selected = {}
    for field, candidates in aliases.items():
        source = next((columns.get(candidate.casefold()) for candidate in candidates if columns.get(candidate.casefold()) is not None), None)
        if source is None:
            raise ValueError(f"{symbol}: missing {field}; columns={list(frame.columns)}")
        selected[field] = frame[source]

    normalized = pd.DataFrame(selected)
    normalized["timestamps"] = pd.to_datetime(normalized["timestamps"], errors="coerce").dt.normalize()
    if normalized["timestamps"].isna().any():
        raise ValueError(f"{symbol}: invalid timestamp values")
    for column in CANONICAL_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[CANONICAL_COLUMNS[1:]].isna().any().any():
        raise ValueError(f"{symbol}: missing or non-numeric OHLCV values")
    if normalized["timestamps"].duplicated().any():
        duplicate_count = int(normalized["timestamps"].duplicated().sum())
        raise ValueError(f"{symbol}: {duplicate_count} duplicate timestamps found")
    normalized = normalized.sort_values("timestamps").reset_index(drop=True)
    if not np.isfinite(normalized[CANONICAL_COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol}: non-finite numeric values")
    if (normalized[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive OHLC values")
    if (
        (normalized["high"] < normalized[["open", "close"]].max(axis=1))
        | (normalized["low"] > normalized[["open", "close"]].min(axis=1))
        | (normalized["high"] < normalized["low"])
    ).any():
        raise ValueError(f"{symbol}: invalid OHLC high/low relationships")
    if (normalized[["volume", "amount"]] < 0).any().any():
        raise ValueError(f"{symbol}: negative volume or amount")
    return normalized[CANONICAL_COLUMNS]


def load_frames(source_dir: Path, symbols: Sequence[str] | None, window: int) -> Dict[str, pd.DataFrame]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    requested = set(symbols or [])
    manifest_path = source_dir / "manifest.json"
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = source_manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{manifest_path}: expected a non-empty records list")
        records_by_symbol = {}
        for record in records:
            symbol = str(record.get("symbol", ""))
            if not re.fullmatch(r"\d{6}", symbol) or symbol in records_by_symbol:
                raise ValueError(f"{manifest_path}: invalid or duplicate symbol record {symbol!r}")
            records_by_symbol[symbol] = record
        selected_symbols = sorted(requested or records_by_symbol)
        missing = [symbol for symbol in selected_symbols if symbol not in records_by_symbol]
        if missing:
            raise ValueError(f"{manifest_path}: requested symbols absent from manifest: {missing}")
        paths_and_records = []
        for symbol in selected_symbols:
            record = records_by_symbol[symbol]
            file_name = str(record.get("file", f"{symbol}.csv"))
            path = source_dir / file_name
            source_root = source_dir.resolve()
            resolved_path = path.resolve()
            if source_root not in resolved_path.parents or path.suffix.casefold() != ".csv":
                raise ValueError(f"{manifest_path}: invalid file for {symbol}: {file_name!r}")
            if not path.exists():
                raise FileNotFoundError(f"{manifest_path}: missing data file {path.name}")
            expected_hash = str(record.get("sha256", ""))
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected_hash != actual_hash:
                raise ValueError(
                    f"{path.name}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}"
                )
            paths_and_records.append((path, record))
    else:
        paths_and_records = [(path, None) for path in sorted(source_dir.glob("*.csv"))]

    frames: Dict[str, pd.DataFrame] = {}
    for path, record in paths_and_records:
        try:
            symbol = symbol_from_path(path)
        except ValueError:
            continue
        if requested and symbol not in requested:
            continue
        # Offline cache is explicitly STAR-only; online pipeline outputs may
        # include CSI300/ChiNext symbols and are accepted when passed directly.
        if source_dir == DEFAULT_SOURCE_DIR and not symbol.startswith(("688", "689")):
            continue
        frame = canonicalize(pd.read_csv(path), symbol)
        if record is not None and int(record.get("rows", len(frame))) != len(frame):
            raise ValueError(f"{symbol}: row count differs from source manifest")
        if len(frame) < window:
            raise ValueError(f"{symbol}: only {len(frame)} rows, need at least {window}")
        frames[symbol] = frame
    if not frames:
        raise RuntimeError(f"No eligible CSV files found under {source_dir}")
    return frames


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
                    f"{symbol}: {name} split has {len(part)} rows; need at least {window}. "
                    "Choose earlier split dates or add more history."
                )
            splits[name][symbol] = part
    return splits


def write_dataset(
    source_dir: Path,
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
        file_records.append(
            {
                "symbol": symbol,
                "file": str(Path("csv") / f"{symbol}.csv"),
                "rows": len(frame),
                "start": str(frame["timestamps"].iloc[0].date()),
                "end": str(frame["timestamps"].iloc[-1].date()),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
        )
    for split_name, split in splits.items():
        with (output_dir / f"{split_name}_data.pkl").open("wb") as handle:
            pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)

    source_manifest_path = source_dir / "manifest.json"
    board_scope = (
        "Live multi-board source; every symbol passed current CSI300, ChiNext, or STAR eligibility."
        if source_manifest_path.exists()
        else "STAR Market cached source only (688/689 prefixes)"
    )
    manifest = {
        "schema_version": 1,
        "source_dir": str(source_dir),
        "symbols": sorted(frames),
        "symbol_count": len(frames),
        "board_scope": board_scope,
        "columns": CANONICAL_COLUMNS,
        "pickle_columns": PICKLE_FEATURE_COLUMNS,
        "lookback_window": lookback_window,
        "predict_window": predict_window,
        "window": lookback_window + predict_window + 1,
        "row_count": sum(len(frame) for frame in frames.values()),
        "records": sorted(file_records, key=lambda record: record["symbol"]),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", help="Optional comma-separated six-digit symbols")
    parser.add_argument("--train-end", default="2024-12-31")
    # Keep at least 139 rows in the shortest local STAR test split for 128->10.
    parser.add_argument("--val-end", default="2025-12-15")
    parser.add_argument("--lookback-window", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--predict-window", type=int, default=DEFAULT_PREDICT_WINDOW)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lookback_window <= 0 or args.predict_window <= 0:
        raise ValueError("lookback-window and predict-window must be positive")
    window = args.lookback_window + args.predict_window + 1
    symbols = [value.strip() for value in args.symbols.split(",")] if args.symbols else None
    frames = load_frames(args.source_dir.resolve(), symbols, window)
    splits = split_frames(frames, args.train_end, args.val_end, window)
    write_dataset(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        frames,
        splits,
        args.train_end,
        args.val_end,
        args.lookback_window,
        args.predict_window,
    )
    total = sum(len(frame) for frame in frames.values())
    print(f"Built {len(frames)} symbols / {total} rows from {args.source_dir}")
    print(f"Wrote {args.output_dir / 'train_data.pkl'}, val_data.pkl, test_data.pkl and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
