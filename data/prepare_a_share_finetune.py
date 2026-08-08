"""Prepare a small, eligibility-filtered A-share dataset for Kronos fine-tuning.

The script uses live eligibility tables instead of treating a hard-coded list as
authoritative.  It selects a small, diversified preference list, downloads daily
adjusted OHLCV data through a provider chain, validates it, and only then writes
per-symbol CSV files and a JSON manifest.

Run from the repository root, for example::

    python data/prepare_a_share_finetune.py

AkShare, adata, and the network are optional at install time, but at least one
provider is required when this script is used to produce data. There is
intentionally no stale/local-symbol fallback because the selection constraint is
about current eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    "timestamps",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)
BOARD_PREFIXES = {
    "ChiNext": ("300", "301"),
    "STAR": ("688", "689"),
}
CSI300_SYMBOL = "000300"
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = date.today().strftime("%Y%m%d")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "a_share_finetune"


@dataclass(frozen=True)
class Candidate:
    """An ordered sector-leader preference, not an eligibility bypass."""

    sector: str
    symbol: str
    rationale: str


@dataclass(frozen=True)
class SelectedSymbol:
    candidate: Candidate
    name: str
    eligible_via: Tuple[str, ...]


@dataclass(frozen=True)
class EligibilityUniverse:
    csi300_codes: Set[str]
    listed_names: Mapping[str, str]
    csi300_rows: int
    csi300_unique_codes: int
    listed_rows: int


class PipelineError(RuntimeError):
    """A user-actionable pipeline failure."""


class ProviderError(PipelineError):
    """AkShare is unavailable or a provider request failed."""


class DataValidationError(PipelineError):
    """Downloaded data did not meet the canonical daily K-line contract."""


class SelectionError(PipelineError):
    """The requested diversified, currently eligible selection cannot be built."""


# The order is intentional: one preferred name per broad sector is selected
# first, and later entries act as fallbacks.  Every symbol still has to pass the
# live CSI 300/current-board eligibility checks before it can be downloaded.
PREFERRED_CANDIDATES: Tuple[Candidate, ...] = (
    Candidate(
        "consumer",
        "600519",
        "Large-cap consumer staple leader preference (Kweichow Moutai).",
    ),
    Candidate(
        "financials",
        "600036",
        "Large-cap commercial bank leader preference (China Merchants Bank).",
    ),
    Candidate(
        "healthcare",
        "600276",
        "Large-cap innovative pharmaceutical leader preference (Hengrui).",
    ),
    Candidate(
        "energy",
        "601857",
        "Large-cap integrated energy leader preference (PetroChina).",
    ),
    Candidate(
        "semiconductors",
        "688981",
        "STAR Market semiconductor manufacturing leader preference (SMIC).",
    ),
    Candidate(
        "software",
        "688111",
        "STAR Market office-software leader preference (Kingsoft Office).",
    ),
    Candidate(
        "new_energy",
        "300750",
        "ChiNext battery and EV-supply-chain leader preference (CATL).",
    ),
    Candidate(
        "materials",
        "601899",
        "Mining and non-ferrous materials leader preference (Zijin Mining).",
    ),
    Candidate(
        "electronics",
        "002415",
        "Large-cap electronic equipment leader preference (Hikvision).",
    ),
    Candidate(
        "fintech",
        "300059",
        "ChiNext financial-information platform leader preference (East Money).",
    ),
    Candidate(
        "consumer",
        "000333",
        "Consumer durable and appliance leader fallback (Midea Group).",
    ),
    Candidate(
        "consumer",
        "000858",
        "Consumer staple leader fallback (Wuliangye).",
    ),
    Candidate(
        "healthcare",
        "300760",
        "ChiNext medical-device leader fallback (Mindray).",
    ),
    Candidate(
        "new_energy",
        "601012",
        "Renewable-energy equipment leader fallback (LONGi).",
    ),
    Candidate(
        "semiconductors",
        "688041",
        "STAR Market computing-chip leader fallback (Hygon).",
    ),
    Candidate(
        "semiconductors",
        "300308",
        "ChiNext optical and computing interconnect leader fallback (Zhongji).",
    ),
    Candidate(
        "software",
        "300059",
        "ChiNext financial software fallback (East Money).",
    ),
    Candidate(
        "consumer_durables",
        "688169",
        "STAR Market smart-appliance leader preference (Roborock).",
    ),
)


COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "timestamps": (
        "timestamps",
        "timestamp",
        "date",
        "trade_time",
        "trade_date",
        "日期",
        "交易日期",
    ),
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价"),
    "volume": ("volume", "成交量", "交易量"),
    "amount": ("amount", "成交额", "成交金额", "交易金额"),
}


def normalize_symbol(value: Any) -> Optional[str]:
    """Return a six-digit A-share code or ``None`` for a malformed value."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", text)
    if 1 <= len(digits) <= 6:
        return digits.zfill(6)
    return None


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> Optional[Any]:
    """Find a provider column by exact, case-insensitive name."""

    columns = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in aliases:
        found = columns.get(alias.casefold())
        if found is not None:
            return found
    return None


def _find_symbol_column(frame: pd.DataFrame, aliases: Iterable[str]) -> Optional[Any]:
    """Prefer known aliases, then infer the column containing most A-share codes."""

    direct = _find_column(frame, aliases)
    if direct is not None:
        return direct
    best_column = None
    best_count = 0
    for column in frame.columns:
        valid_count = sum(normalize_symbol(value) is not None for value in frame[column])
        if valid_count > best_count:
            best_column = column
            best_count = valid_count
    return best_column if best_count else None


def _required_column(
    frame: pd.DataFrame, field: str, *, allow_first_column: bool = False
) -> Any:
    column = _find_column(frame, COLUMN_ALIASES[field])
    if column is not None:
        return column
    if allow_first_column and len(frame.columns) > 0:
        return frame.columns[0]
    raise ProviderError(
        f"AkShare returned no column for {field!r}; received columns "
        f"{list(frame.columns)!r}."
    )


def _provider_call(label: str, function: Any) -> pd.DataFrame:
    try:
        result = function()
    except Exception as exc:  # third-party providers expose varied exception types
        raise ProviderError(
            f"Provider request failed while {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(result, pd.DataFrame):
        raise ProviderError(
            f"AkShare returned {type(result).__name__} instead of a DataFrame while "
            f"{label}."
        )
    return result


def load_provider(provider_name: str = "akshare") -> Any:
    """Import one optional provider only when the pipeline is actually run."""

    if provider_name == "adata":
        module_name, display_name = "adata", "adata"
    elif provider_name in {"akshare", "akshare_tencent"}:
        module_name, display_name = "akshare", "AkShare"
    else:
        raise ProviderError(f"Unknown provider: {provider_name}")

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ProviderError(
            f"{display_name} is not installed. Install it with "
            f"`python -m pip install {module_name}`, then rerun the command."
        ) from exc


def fetch_current_eligibility(
    provider: Any, provider_name: str = "akshare"
) -> EligibilityUniverse:
    """Fetch current CSI 300 membership and current listed A-share codes."""

    if provider_name == "adata":
        csi300 = _provider_call(
            "adata CSI 300 constituents (000300)",
            lambda: provider.stock.info.index_constituent(index_code=CSI300_SYMBOL),
        )
        listed = _provider_call(
            "adata current A-share code table",
            provider.stock.info.all_code,
        )
        csi_code_aliases = ("stock_code", "code", "代码", "证券代码")
        listed_code_aliases = ("stock_code", "code", "代码", "证券代码")
        listed_name_aliases = ("short_name", "name", "名称", "证券简称")
    elif provider_name == "akshare":
        # The Sina route is frequently rate-limited. Prefer CSIndex's current
        # constituent endpoint and keep the old route as a compatibility
        # fallback for older AkShare installations.
        try:
            csi300 = _provider_call(
                f"fetching current CSI 300 constituents from CSIndex ({CSI300_SYMBOL})",
                lambda: provider.index_stock_cons_csindex(symbol=CSI300_SYMBOL),
            )
        except ProviderError:
            csi300 = _provider_call(
                f"fetching current CSI 300 constituents from Sina ({CSI300_SYMBOL})",
                lambda: provider.index_stock_cons(symbol=CSI300_SYMBOL),
            )
        listed = _provider_call(
            "fetching the current A-share code/name table",
            provider.stock_info_a_code_name,
        )
        csi_code_aliases = (
            "code",
            "代码",
            "品种代码",
            "成分券代码",
            "成份券代码",
            "证券代码",
        )
        listed_code_aliases = ("code", "代码", "证券代码")
        listed_name_aliases = ("name", "名称", "证券简称")
    else:
        raise ProviderError(
            f"{provider_name} cannot provide the live eligibility contract; "
            "use adata or akshare for current membership checks."
        )

    csi_code_column = _find_symbol_column(csi300, csi_code_aliases)
    if csi_code_column is None:
        raise ProviderError("The current CSI 300 response has no usable symbol column.")
    csi_codes = {
        code
        for value in csi300[csi_code_column]
        if (code := normalize_symbol(value)) is not None
    }

    listed_code_column = _find_column(listed, listed_code_aliases)
    listed_name_column = _find_column(listed, listed_name_aliases)
    if listed_code_column is None or listed_name_column is None:
        raise ProviderError(
            "AkShare's current A-share code table is missing code/name columns; "
            f"received columns {list(listed.columns)!r}."
        )

    listed_names: Dict[str, str] = {}
    for raw_code, raw_name in zip(
        listed[listed_code_column], listed[listed_name_column]
    ):
        code = normalize_symbol(raw_code)
        if code is not None:
            listed_names[code] = str(raw_name).strip()

    if not csi_codes:
        raise ProviderError("The current CSI 300 response contained no valid symbols.")
    if not listed_names:
        raise ProviderError("The current A-share code table contained no valid symbols.")

    # Membership is the intersection of the live listing table and either the
    # live CSI 300 list or a live board prefix.  This avoids downloading a symbol
    # merely because it appears in a stale hard-coded preference list.
    current_codes = set(listed_names)
    csi300_codes = csi_codes & current_codes
    board_codes = {
        code
        for code in current_codes
        if any(
            code.startswith(prefix)
            for prefixes in BOARD_PREFIXES.values()
            for prefix in prefixes
        )
    }
    if not csi300_codes and not board_codes:
        raise ProviderError(
            "The live provider responses had no overlap between CSI 300/listed "
            "symbols and current board prefixes."
        )

    return EligibilityUniverse(
        csi300_codes=csi300_codes,
        listed_names=listed_names,
        csi300_rows=len(csi300),
        csi300_unique_codes=len(csi_codes),
        listed_rows=len(listed),
    )


def eligibility_labels(symbol: str, universe: EligibilityUniverse) -> Tuple[str, ...]:
    """Return the live eligibility sources for a listed symbol."""

    if symbol not in universe.listed_names:
        return ()
    labels: List[str] = []
    if symbol in universe.csi300_codes:
        labels.append("CSI300")
    for board, prefixes in BOARD_PREFIXES.items():
        if any(symbol.startswith(prefix) for prefix in prefixes):
            labels.append(board)
    return tuple(labels)


def _parse_requested_symbols(value: Optional[str]) -> List[str]:
    if not value:
        return []
    symbols = [normalize_symbol(item) for item in value.split(",")]
    if any(symbol is None for symbol in symbols):
        raise SelectionError(
            "--symbols contains a malformed value; provide comma-separated six-digit codes."
        )
    result = [symbol for symbol in symbols if symbol is not None]
    if len(set(result)) != len(result):
        raise SelectionError("--symbols contains duplicate stock codes.")
    return result


def select_symbols(
    universe: EligibilityUniverse,
    count: int,
    requested_symbols: Optional[str] = None,
) -> List[SelectedSymbol]:
    """Select requested or preferred symbols only after live eligibility checks."""

    requested = _parse_requested_symbols(requested_symbols)
    if requested:
        selected: List[SelectedSymbol] = []
        rejected = []
        for symbol in requested:
            labels = eligibility_labels(symbol, universe)
            if symbol not in universe.listed_names or not labels:
                rejected.append(symbol)
                continue
            selected.append(
                SelectedSymbol(
                    candidate=Candidate(
                        "requested",
                        symbol,
                        "Explicitly requested; retained only after live eligibility validation.",
                    ),
                    name=universe.listed_names[symbol],
                    eligible_via=labels,
                )
            )
        if rejected:
            raise SelectionError(
                "These requested symbols are not currently eligible in CSI 300, "
                f"ChiNext, or STAR Market: {', '.join(rejected)}."
            )
        return selected

    if count <= 0:
        raise SelectionError("--count must be a positive integer.")

    selected = []
    selected_codes: Set[str] = set()
    selected_sectors: Set[str] = set()

    # First pass: one candidate per broad sector, which keeps the default set
    # diversified even if the preferred list has several candidates in one area.
    for candidate in PREFERRED_CANDIDATES:
        labels = eligibility_labels(candidate.symbol, universe)
        if (
            candidate.symbol in selected_codes
            or candidate.sector in selected_sectors
            or not labels
        ):
            continue
        selected.append(
            SelectedSymbol(
                candidate=candidate,
                name=universe.listed_names[candidate.symbol],
                eligible_via=labels,
            )
        )
        selected_codes.add(candidate.symbol)
        selected_sectors.add(candidate.sector)
        if len(selected) >= count:
            return selected

    # Second pass: use additional eligible leaders as needed, preserving the
    # preference order while making --count useful when a candidate exits.
    for candidate in PREFERRED_CANDIDATES:
        labels = eligibility_labels(candidate.symbol, universe)
        if candidate.symbol in selected_codes or not labels:
            continue
        selected.append(
            SelectedSymbol(
                candidate=candidate,
                name=universe.listed_names[candidate.symbol],
                eligible_via=labels,
            )
        )
        selected_codes.add(candidate.symbol)
        if len(selected) >= count:
            return selected

    raise SelectionError(
        f"Only {len(selected)} preferred symbols passed current eligibility, but "
        f"--count={count} was requested. Use a smaller --count or add eligible "
        "codes to the ordered preference list in this script."
    )


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False), errors="coerce"
    )


def normalize_and_validate_history(
    raw: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
    min_rows: int,
) -> pd.DataFrame:
    """Normalize an AkShare daily frame and enforce the Kronos CSV contract."""

    if raw.empty:
        raise DataValidationError(f"{symbol}: provider returned no daily rows.")

    normalized = pd.DataFrame(
        {
            field: raw[_required_column(raw, field)]
            for field in CANONICAL_COLUMNS
        }
    )
    timestamps = pd.to_datetime(normalized["timestamps"], errors="coerce")
    if timestamps.isna().any():
        raise DataValidationError(
            f"{symbol}: {int(timestamps.isna().sum())} invalid or missing dates."
        )
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)

    for field in CANONICAL_COLUMNS[1:]:
        normalized[field] = _clean_numeric(normalized[field])
    numeric_values = normalized[list(CANONICAL_COLUMNS[1:])]
    invalid_numeric = numeric_values.isna().any(axis=1) | ~np.isfinite(
        numeric_values.to_numpy(dtype=float)
    ).all(axis=1)
    if invalid_numeric.any():
        raise DataValidationError(
            f"{symbol}: {int(invalid_numeric.sum())} rows contain missing/non-numeric "
            "OHLCV or amount values."
        )

    # A daily provider should return one observation per calendar day. Normalize
    # any accidental time component before checking uniqueness so it cannot
    # become a duplicate date after CSV serialization.
    normalized["timestamps"] = timestamps.dt.normalize()
    normalized = normalized.sort_values("timestamps").reset_index(drop=True)
    if normalized["timestamps"].duplicated().any():
        duplicate_count = int(normalized["timestamps"].duplicated().sum())
        raise DataValidationError(f"{symbol}: {duplicate_count} duplicate dates found.")

    lower_bound = pd.Timestamp(start_date)
    upper_bound = pd.Timestamp(end_date)
    if normalized["timestamps"].min() < lower_bound:
        raise DataValidationError(f"{symbol}: data contains rows before {start_date}.")
    if normalized["timestamps"].max() > upper_bound:
        raise DataValidationError(f"{symbol}: data contains rows after {end_date}.")
    if (normalized["timestamps"].dt.weekday >= 5).any():
        raise DataValidationError(f"{symbol}: daily data contains a weekend date.")

    open_price = normalized["open"]
    high_price = normalized["high"]
    low_price = normalized["low"]
    close_price = normalized["close"]
    invalid_ohlc = (
        (open_price <= 0)
        | (high_price <= 0)
        | (low_price <= 0)
        | (close_price <= 0)
        | (high_price < open_price)
        | (high_price < close_price)
        | (low_price > open_price)
        | (low_price > close_price)
        | (high_price < low_price)
    )
    if invalid_ohlc.any():
        raise DataValidationError(
            f"{symbol}: {int(invalid_ohlc.sum())} rows violate positive OHLC/high-low invariants."
        )
    if (normalized["volume"] < 0).any() or (normalized["amount"] < 0).any():
        raise DataValidationError(f"{symbol}: volume and amount must be non-negative.")
    if len(normalized) < min_rows:
        raise DataValidationError(
            f"{symbol}: only {len(normalized)} rows available; at least {min_rows} "
            "validated daily rows are required for the default fine-tuning context."
        )

    normalized["timestamps"] = normalized["timestamps"].dt.strftime("%Y-%m-%d")
    return normalized.loc[:, CANONICAL_COLUMNS]


def download_history(
    provider: Any,
    selected: SelectedSymbol,
    start_date: str,
    end_date: str,
    adjust: str,
    timeout: float,
    min_rows: int,
    provider_name: str = "akshare",
    retries: int = 2,
    request_delay: float = 1.0,
) -> pd.DataFrame:
    """Download and validate one symbol's daily history."""

    symbol = selected.candidate.symbol
    last_error: Optional[PipelineError] = None
    for attempt in range(retries + 1):
        try:
            if provider_name == "adata":
                if adjust not in {"qfq", ""}:
                    raise ProviderError(
                        f"adata history currently supports qfq/raw-compatible mode only; "
                        f"got adjust={adjust!r} for {symbol}."
                    )
                start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
                end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
                raw = _provider_call(
                    f"adata daily history for {symbol}",
                    lambda: provider.stock.market.get_market(
                        stock_code=symbol,
                        start_date=start,
                        end_date=end,
                        k_type=1,
                        adjust_type=1,
                    ),
                )
            elif provider_name == "akshare":
                raw = _provider_call(
                    f"AkShare daily history for {symbol}",
                    lambda: provider.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                        timeout=timeout,
                    ),
                )
            elif provider_name == "akshare_tencent":
                exchange_symbol = (
                    f"sh{symbol}" if symbol.startswith(("6", "688", "689")) else f"sz{symbol}"
                )
                raw = _provider_call(
                    f"AkShare Tencent daily history for {symbol}",
                    lambda: provider.stock_zh_a_hist_tx(
                        symbol=exchange_symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                        timeout=timeout,
                    ),
                )
            else:
                raise ProviderError(f"Unknown history provider: {provider_name}")
            return normalize_and_validate_history(
                raw,
                symbol,
                start_date,
                end_date,
                min_rows,
            )
        except PipelineError as exc:
            last_error = exc
            if attempt < retries and request_delay > 0:
                time.sleep(request_delay * (attempt + 1))
    assert last_error is not None
    raise last_error


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize exactly the bytes that will be hashed and written."""

    buffer = StringIO()
    frame.to_csv(buffer, index=False, float_format="%.10g", lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def build_manifest(
    provider: Any,
    universe: EligibilityUniverse,
    selected: Sequence[SelectedSymbol],
    frames: Mapping[str, pd.DataFrame],
    file_payloads: Mapping[str, bytes],
    start_date: str,
    end_date: str,
    adjust: str,
    min_rows: int,
    provider_name: str = "akshare",
    providers_used: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    providers_used = providers_used or {}
    records = []
    for item in selected:
        symbol = item.candidate.symbol
        frame = frames[symbol]
        records.append(
            {
                "symbol": symbol,
                "name": item.name,
                "sector": item.candidate.sector,
                "eligible_via": list(item.eligible_via),
                "selection_rationale": item.candidate.rationale,
                "file": f"{symbol}.csv",
                "rows": len(frame),
                "start": frame["timestamps"].iloc[0],
                "end": frame["timestamps"].iloc[-1],
                "sha256": hashlib.sha256(file_payloads[symbol]).hexdigest(),
                "provider": providers_used.get(symbol, provider_name),
                "provider_endpoint": {
                    "adata": "adata.stock.market.get_market",
                    "akshare": "ak.stock_zh_a_hist",
                    "akshare_tencent": "ak.stock_zh_a_hist_tx",
                }[providers_used.get(symbol, provider_name)],
            }
        )

    provider_version = getattr(provider, "__version__", "unknown")
    provider_set = sorted(set(providers_used.values()) or {provider_name})
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider_set[0] if len(provider_set) == 1 else "multi",
        "providers": provider_set,
        "provider_version": str(provider_version),
        "eligibility_provider": provider_name,
        "frequency": "daily",
        "adjustment": adjust or "raw",
        "requested_range": {"start": start_date, "end": end_date},
        "minimum_rows": min_rows,
        "columns": list(CANONICAL_COLUMNS),
        "eligibility": {
            "csi300_symbol": CSI300_SYMBOL,
            "csi300_rows_from_provider": universe.csi300_rows,
            "csi300_unique_codes_from_provider": universe.csi300_unique_codes,
            "current_listed_code_rows": universe.listed_rows,
            "board_prefixes": BOARD_PREFIXES,
            "rule": "listed and (current CSI 300 member or current ChiNext/STAR prefix)",
        },
        "selection_policy": {
            "default_count": len(selected),
            "method": (
                "ordered sector-leader preferences; one per broad sector first, "
                "then fallbacks"
            ),
            "current_eligibility_is_required": True,
            "fixed_symbols_are_not_an_eligibility_source": True,
        },
        "records": records,
    }


def write_outputs(
    output_dir: Path,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> None:
    """Write outputs only after every selected symbol passed validation."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as staging_name:
        staging_dir = Path(staging_name)
        for symbol, payload in payloads.items():
            (staging_dir / f"{symbol}.csv").write_bytes(payload)
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        # Each file is fully written before it reaches the final directory;
        # publish the manifest last so it never advertises an incomplete batch.
        output_dir.mkdir(parents=True, exist_ok=True)
        for symbol in payloads:
            os.replace(staging_dir / f"{symbol}.csv", output_dir / f"{symbol}.csv")
        os.replace(staging_dir / "manifest.json", output_dir / "manifest.json")


def run_pipeline(args: argparse.Namespace) -> int:
    start_day = parse_date_arg(args.start_date, "--start-date")
    end_day = parse_date_arg(args.end_date, "--end-date")
    if start_day > end_day:
        raise PipelineError("--start-date must not be later than --end-date.")
    if end_day > date.today():
        raise PipelineError("--end-date must not be in the future.")
    if args.timeout <= 0:
        raise PipelineError("--timeout must be positive.")
    if args.retries < 0:
        raise PipelineError("--retries must be non-negative.")
    if args.request_delay < 0:
        raise PipelineError("--request-delay must be non-negative.")
    start_date = start_day.strftime("%Y%m%d")
    end_date = end_day.strftime("%Y%m%d")

    provider_names = (
        ["adata", "akshare", "akshare_tencent"]
        if args.provider == "auto"
        else (["akshare", "akshare_tencent"] if args.provider == "akshare_tencent" else [args.provider])
    )
    providers: Dict[str, Any] = {}
    load_errors = []
    for provider_name in provider_names:
        try:
            providers[provider_name] = load_provider(provider_name)
        except ProviderError as exc:
            load_errors.append(str(exc))
    if not providers:
        raise ProviderError("No requested data provider is available: " + " | ".join(load_errors))

    universe = None
    eligibility_provider = None
    eligibility_errors = []
    for provider_name in (name for name in provider_names if name != "akshare_tencent"):
        provider = providers.get(provider_name)
        if provider is None:
            continue
        try:
            universe = fetch_current_eligibility(provider, provider_name)
            eligibility_provider = provider_name
            break
        except ProviderError as exc:
            eligibility_errors.append(f"{provider_name}: {exc}")
    if universe is None or eligibility_provider is None:
        raise ProviderError(
            "All eligibility providers failed: " + " | ".join(eligibility_errors)
        )

    history_provider_names = (
        ["akshare_tencent"]
        if args.provider == "akshare_tencent"
        else [eligibility_provider]
        + [name for name in provider_names if name != eligibility_provider and name in providers]
    )
    selected = select_symbols(universe, args.count, args.symbols)
    print(
        f"Current eligibility: CSI300={len(universe.csi300_codes)} unique listed members; "
        f"listed A-share codes={len(universe.listed_names)}; "
        f"eligibility_provider={eligibility_provider}."
    )
    for item in selected:
        print(
            f"Selected {item.candidate.symbol} {item.name} "
            f"[{', '.join(item.eligible_via)}] sector={item.candidate.sector}"
        )

    if args.dry_run:
        print("Dry run complete; no history files or manifest were written.")
        return 0

    frames: Dict[str, pd.DataFrame] = {}
    payloads: Dict[str, bytes] = {}
    providers_used: Dict[str, str] = {}
    for index, item in enumerate(selected, start=1):
        symbol = item.candidate.symbol
        print(f"Downloading {index}/{len(selected)}: {symbol} ...")
        history_errors = []
        frame = None
        for provider_name in history_provider_names:
            try:
                frame = download_history(
                    providers[provider_name],
                    item,
                    start_date,
                    end_date,
                    args.adjust,
                    args.timeout,
                    args.min_rows,
                    provider_name=provider_name,
                    retries=args.retries,
                    request_delay=args.request_delay,
                )
                providers_used[symbol] = provider_name
                break
            except PipelineError as exc:
                history_errors.append(f"{provider_name}: {exc}")
        if frame is None:
            raise ProviderError(
                f"All history providers failed for {symbol}: " + " | ".join(history_errors)
            )
        frames[symbol] = frame
        payloads[symbol] = csv_bytes(frame)
        print(
            f"  validated rows={len(frame)} range={frame['timestamps'].iloc[0]}"
            f"..{frame['timestamps'].iloc[-1]}"
        )

    manifest = build_manifest(
        providers[eligibility_provider],
        universe,
        selected,
        frames,
        payloads,
        start_date,
        end_date,
        args.adjust,
        args.min_rows,
        provider_name=eligibility_provider,
        providers_used=providers_used,
    )
    write_outputs(args.output_dir, manifest, payloads)
    print(f"Wrote {len(payloads)} symbol files and {args.output_dir / 'manifest.json'}")
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_date_arg(value: str, option_name: str) -> date:
    """Validate a CLI date in the provider's YYYYMMDD format."""

    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise PipelineError(f"{option_name} must use YYYYMMDD, got {value!r}.") from exc
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download validated daily A-share OHLCV for a small current CSI 300, "
            "ChiNext, or STAR Market universe."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-symbol CSV files and manifest.json.",
    )
    parser.add_argument("--count", type=positive_int, default=10)
    parser.add_argument(
        "--provider",
        choices=("auto", "adata", "akshare", "akshare_tencent"),
        default="auto",
        help=(
            "Eligibility/history provider order; auto tries Adata, AkShare Eastmoney, "
            "then AkShare Tencent history."
        ),
    )
    parser.add_argument(
        "--symbols",
        help="Optional comma-separated codes; each is still checked against live eligibility.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYYMMDD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="YYYYMMDD")
    parser.add_argument(
        "--adjust",
        choices=("", "qfq", "hfq"),
        default="qfq",
        help="Price adjustment; qfq is the default for split-aware training.",
    )
    parser.add_argument("--min-rows", type=positive_int, default=512)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per provider after an empty/error response (default: 2).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Seconds between retries, increasing per attempt (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch current eligibility and print selection without downloading or writing.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_pipeline(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "No static-data fallback was used because current index/board eligibility "
            "is required.",
            file=sys.stderr,
        )
        print(
            "Rerun when AkShare and network access are available: "
            f"{sys.executable} data/prepare_a_share_finetune.py --provider auto",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
