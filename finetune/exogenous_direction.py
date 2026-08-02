"""Leakage-safe exogenous features and market/residual return targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def build_live_origin_samples(
    frames: Mapping[str, pd.DataFrame],
    *,
    lookback: int,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build unlabeled latest origins using only rows observed by ``as_of``."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if not frames:
        raise ValueError("frames cannot be empty")

    latest_dates: list[pd.Timestamp] = []
    prepared_dates: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        if "date" not in frame.columns:
            raise ValueError(f"{symbol} is missing required column: date")
        dates = pd.to_datetime(frame["date"])
        if dates.empty or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError(f"{symbol} dates must be non-empty, sorted, and unique")
        prepared_dates[str(symbol)] = dates
        latest_dates.append(pd.Timestamp(dates.iloc[-1]))

    cutoff = pd.Timestamp(as_of) if as_of is not None else min(latest_dates)
    records: list[dict[str, object]] = []
    for symbol in sorted(prepared_dates):
        dates = prepared_dates[symbol]
        eligible = np.flatnonzero((dates <= cutoff).to_numpy())
        if len(eligible) < lookback:
            continue
        context_end = int(eligible[-1])
        records.append(
            {
                "symbol": symbol,
                "context_start": context_end - lookback + 1,
                "context_end": context_end,
                "context_end_date": pd.Timestamp(dates.iloc[context_end]),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["symbol", "context_start", "context_end", "context_end_date"],
    )


def sanitize_exogenous_frame(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize Tencent index/stock OHLC amount data to one stable schema."""
    frame = raw.rename(columns={column: str(column).lower() for column in raw.columns})
    required = ["date", "open", "high", "low", "close", "amount"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    valid = (
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["amount"] >= 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )
    frame = frame[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{name} has no structurally valid rows")
    return frame


def _prepare_exogenous_frame(
    frame: pd.DataFrame,
    prefix: str,
    horizons: Sequence[int],
) -> tuple[pd.DataFrame, list[str]]:
    required = {"date", "open", "high", "low", "close", "amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{prefix} is missing required columns: {sorted(missing)}")
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared = prepared.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if not prepared["date"].is_monotonic_increasing:
        raise ValueError(f"{prefix} dates must be sorted")
    numeric = prepared[["open", "high", "low", "close", "amount"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all() or np.any(prepared["close"].to_numpy() <= 0):
        raise ValueError(f"{prefix} values must be finite and close must be positive")

    feature_columns: list[str] = []
    log_close = np.log(prepared["close"].to_numpy(dtype=float))
    for horizon in horizons:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("feature horizons must be positive")
        column = f"{prefix}_return_{horizon}"
        prepared[column] = pd.Series(log_close).diff(horizon)
        feature_columns.append(column)

    daily_return = pd.Series(log_close).diff()
    volatility_column = f"{prefix}_volatility_20"
    range_column = f"{prefix}_range_20"
    amount_column = f"{prefix}_amount_z_20"
    prepared[volatility_column] = daily_return.rolling(20, min_periods=10).std(ddof=0)
    prepared[range_column] = (
        (prepared["high"] - prepared["low"])
        / np.maximum(prepared["close"], 1e-12)
    ).rolling(20, min_periods=10).mean()
    amount_mean = prepared["amount"].rolling(20, min_periods=10).mean()
    amount_std = prepared["amount"].rolling(20, min_periods=10).std(ddof=0)
    prepared[amount_column] = (prepared["amount"] - amount_mean) / (
        amount_std + 1e-12
    )
    feature_columns.extend([volatility_column, range_column, amount_column])
    observed_column = f"{prefix}_observed_date"
    prepared[observed_column] = prepared["date"]
    return prepared[["date", observed_column, *feature_columns]], feature_columns


def align_exogenous_features(
    samples: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    *,
    horizons: Sequence[int] = (1, 3, 5, 10, 20, 60),
    max_staleness_days: int = 7,
) -> pd.DataFrame:
    """As-of join exogenous features without ever reading after an origin date."""
    if "context_end_date" not in samples.columns:
        raise ValueError("samples must contain context_end_date")
    if max_staleness_days < 0:
        raise ValueError("max_staleness_days cannot be negative")
    result = samples.copy()
    result["context_end_date"] = pd.to_datetime(result["context_end_date"])
    result["_sample_order"] = np.arange(len(result))
    result = result.sort_values("context_end_date").reset_index(drop=True)

    for prefix, frame in frames.items():
        prepared, feature_columns = _prepare_exogenous_frame(
            frame, str(prefix), horizons
        )
        observed_column = f"{prefix}_observed_date"
        result = pd.merge_asof(
            result,
            prepared,
            left_on="context_end_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["date"])
        staleness_column = f"{prefix}_staleness_days"
        result[staleness_column] = (
            result["context_end_date"] - result[observed_column]
        ).dt.days
        stale = result[staleness_column] > int(max_staleness_days)
        result.loc[stale, feature_columns] = np.nan

    return (
        result.sort_values("_sample_order")
        .drop(columns=["_sample_order"])
        .reset_index(drop=True)
    )


def _asof_close(frame: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    market = frame[["date", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    market_dates = market["date"].to_numpy(dtype="datetime64[ns]")
    requested = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    indexes = np.searchsorted(market_dates, requested, side="right") - 1
    if np.any(indexes < 0):
        raise ValueError("market data starts after at least one requested date")
    return market["close"].to_numpy(dtype=float)[indexes]


def attach_return_decomposition_targets(
    samples: pd.DataFrame,
    stock_frames: Mapping[str, pd.DataFrame],
    market_frame: pd.DataFrame,
    *,
    market_beta: float = 1.0,
) -> pd.DataFrame:
    """Attach absolute, market, and residual log-return training targets."""
    required = {
        "symbol",
        "context_end",
        "label_index",
        "context_end_date",
        "label_end_date",
    }
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"samples are missing required columns: {sorted(missing)}")
    result = samples.copy()
    stock_returns = []
    for row in result.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in stock_frames:
            raise ValueError(f"missing stock frame for {symbol}")
        close = stock_frames[symbol]["close"].to_numpy(dtype=float)
        context_close = close[int(row.context_end)]
        future_close = close[int(row.label_index)]
        stock_returns.append(float(np.log(future_close / context_close)))
    stock_return = np.asarray(stock_returns, dtype=float)
    market_context = _asof_close(market_frame, result["context_end_date"])
    market_future = _asof_close(market_frame, result["label_end_date"])
    market_return = np.log(market_future / market_context)
    residual_return = stock_return - float(market_beta) * market_return
    result["stock_future_return"] = stock_return
    result["market_future_return"] = market_return
    result["residual_future_return"] = residual_return
    result["target_up"] = stock_return > 0.0
    result["market_up"] = market_return > 0.0
    result["residual_up"] = residual_return > 0.0
    return result


def combine_return_predictions(
    market_return,
    residual_return,
    market_beta=1.0,
) -> np.ndarray:
    market = np.asarray(market_return, dtype=float)
    residual = np.asarray(residual_return, dtype=float)
    beta = np.asarray(market_beta, dtype=float)
    try:
        return market * beta + residual
    except ValueError as exc:
        raise ValueError("market, residual, and beta values are not broadcastable") from exc


def estimate_market_betas(
    training_targets: pd.DataFrame,
    *,
    min_points: int = 60,
    lower: float = -1.0,
    upper: float = 3.0,
) -> dict[str, float]:
    """Estimate per-symbol OLS market beta from one training split only."""
    required = {"symbol", "market_future_return", "stock_future_return"}
    missing = required.difference(training_targets.columns)
    if missing:
        raise ValueError(f"training targets are missing columns: {sorted(missing)}")
    if min_points < 2 or lower >= upper:
        raise ValueError("invalid beta estimation settings")
    betas: dict[str, float] = {}
    for symbol, group in training_targets.groupby("symbol"):
        clean = group[["market_future_return", "stock_future_return"]].dropna()
        if len(clean) < min_points:
            betas[str(symbol)] = 1.0
            continue
        market = clean["market_future_return"].to_numpy(dtype=float)
        stock = clean["stock_future_return"].to_numpy(dtype=float)
        centred_market = market - market.mean()
        variance = float(np.dot(centred_market, centred_market))
        if variance <= 1e-12:
            beta = 1.0
        else:
            beta = float(
                np.dot(centred_market, stock - stock.mean()) / variance
            )
        betas[str(symbol)] = float(np.clip(beta, lower, upper))
    return betas


def retarget_sample_horizon(
    samples: pd.DataFrame,
    stock_frames: Mapping[str, pd.DataFrame],
    *,
    horizon: int,
) -> pd.DataFrame:
    """Reuse fixed contexts with a different complete future row horizon."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = {"symbol", "context_end", "label_index", "label_end_date"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"samples are missing required columns: {sorted(missing)}")
    records = []
    for row in samples.to_dict("records"):
        symbol = str(row["symbol"])
        if symbol not in stock_frames:
            raise ValueError(f"missing stock frame for {symbol}")
        frame = stock_frames[symbol]
        label_index = int(row["context_end"]) + int(horizon)
        if label_index >= len(frame):
            continue
        updated = dict(row)
        updated["label_index"] = label_index
        updated["label_end_date"] = pd.Timestamp(frame.iloc[label_index]["date"])
        records.append(updated)
    return pd.DataFrame.from_records(records, columns=samples.columns).reset_index(drop=True)
