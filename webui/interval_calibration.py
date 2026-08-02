"""Leak-free empirical price intervals and explicit forecast guardrails."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd


def historical_return_bands(
    context_close,
    *,
    max_horizon: int,
    confidence_level: float = 0.9,
    lookback: int = 252,
    min_samples: int = 60,
    adapt_volatility: bool = False,
) -> dict[str, object]:
    """Estimate horizon-wise close bands from returns observable at forecast time."""
    close = np.asarray(context_close, dtype=float).reshape(-1)
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if lookback < 1 or min_samples < 1:
        raise ValueError("lookback and min_samples must be positive")
    if len(close) <= max_horizon or not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("context_close must contain enough finite positive values")

    alpha = 1.0 - float(confidence_level)
    anchor = float(close[-1])
    lower = []
    median = []
    upper = []
    sample_count = []
    log_close = np.log(close)
    daily_returns = np.diff(log_close)
    volatility_scale = 1.0
    if adapt_volatility and len(daily_returns) >= 40:
        recent_volatility = float(np.std(daily_returns[-20:]))
        baseline_volatility = float(np.std(daily_returns[-int(lookback) :]))
        if baseline_volatility > 1e-12:
            volatility_scale = float(
                np.clip(recent_volatility / baseline_volatility, 0.75, 1.5)
            )
    for horizon in range(1, int(max_horizon) + 1):
        returns = log_close[horizon:] - log_close[:-horizon]
        returns = returns[-int(lookback) :]
        if len(returns) < min_samples:
            raise ValueError(
                f"horizon {horizon} has {len(returns)} historical returns; "
                f"need at least {min_samples}"
            )
        quantiles = np.quantile(returns, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])
        quantiles[[0, 2]] = quantiles[1] + (
            quantiles[[0, 2]] - quantiles[1]
        ) * volatility_scale
        prices = anchor * np.exp(quantiles)
        lower.append(float(prices[0]))
        median.append(float(prices[1]))
        upper.append(float(prices[2]))
        sample_count.append(int(len(returns)))

    return {
        "anchor_close": anchor,
        "confidence_level": float(confidence_level),
        "lower": np.asarray(lower, dtype=float),
        "median": np.asarray(median, dtype=float),
        "upper": np.asarray(upper, dtype=float),
        "sample_count": sample_count,
        "volatility_scale": volatility_scale,
    }


def _rescale_frame_close(frame: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy(deep=True)
    if "close" not in result.columns or len(result) != len(close):
        raise ValueError("forecast frames must contain close and match the horizon")
    original_close = result["close"].to_numpy(dtype=float)
    scale = close / np.where(np.abs(original_close) > 1e-12, original_close, 1.0)
    for column in ("open", "high", "low", "close", "amount"):
        if column in result.columns:
            result[column] = result[column].to_numpy(dtype=float) * scale
    result["close"] = close
    return result


def apply_return_band_guardrail(
    forecast: dict,
    context_close,
    *,
    confidence_level: float = 0.9,
    lookback: int = 252,
    min_samples: int = 60,
    adapt_volatility: bool = False,
    model_point_weight: float = 0.25,
) -> dict:
    """Clip point paths to empirical bands and replace uncalibrated close intervals.

    The returned metadata makes every change observable to API consumers. The
    input forecast is deep-copied and never mutated.
    """
    result = copy.deepcopy(forecast)
    if not 0.0 <= model_point_weight <= 1.0:
        raise ValueError("model_point_weight must be between zero and one")
    point_frame = result.get("median", result.get("prediction"))
    if not isinstance(point_frame, pd.DataFrame) or "close" not in point_frame:
        raise ValueError("forecast must contain a median or prediction close frame")
    horizon = len(point_frame)
    bands = historical_return_bands(
        context_close,
        max_horizon=horizon,
        confidence_level=confidence_level,
        lookback=lookback,
        min_samples=min_samples,
        adapt_volatility=adapt_volatility,
    )
    lower = bands["lower"]
    upper = bands["upper"]
    raw_median = point_frame["close"].to_numpy(dtype=float)
    clipped_median = np.clip(raw_median, lower, upper)
    guarded_median = np.clip(
        bands["anchor_close"]
        + float(model_point_weight) * (clipped_median - bands["anchor_close"]),
        lower,
        upper,
    )

    for key in ("prediction", "median", "mean"):
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            raw_close = frame["close"].to_numpy(dtype=float)
            clipped_close = np.clip(raw_close, lower, upper)
            guarded_close = np.clip(
                bands["anchor_close"]
                + float(model_point_weight)
                * (clipped_close - bands["anchor_close"]),
                lower,
                upper,
            )
            result[key] = _rescale_frame_close(
                frame, guarded_close
            )
    if isinstance(result.get("lower"), pd.DataFrame):
        result["lower"] = _rescale_frame_close(result["lower"], lower)
    if isinstance(result.get("upper"), pd.DataFrame):
        result["upper"] = _rescale_frame_close(result["upper"], upper)

    width = upper - lower
    result["interval_width"] = width
    result["relative_interval_width"] = width / np.maximum(
        np.abs(guarded_median), 1e-12
    )
    if isinstance(result.get("std"), pd.DataFrame):
        result["std"] = result["std"].reset_index(drop=True).copy(deep=True)
        result["std"]["close"] = width / 3.2897072539029444
    result["calibration"] = {
        "method": "historical_log_return_quantiles",
        "confidence_level": float(confidence_level),
        "lookback": int(lookback),
        "volatility_scale": bands["volatility_scale"],
        "sample_count": bands["sample_count"],
        "anchor_close": bands["anchor_close"],
        "lower_close": lower.tolist(),
        "median_close": bands["median"].tolist(),
        "upper_close": upper.tolist(),
        "raw_model_median_close": raw_median.tolist(),
        "band_clipped_model_median_close": clipped_median.tolist(),
        "guarded_model_median_close": guarded_median.tolist(),
        "model_point_weight": float(model_point_weight),
        "shrinkage_count": int(
            np.sum(~np.isclose(raw_median, guarded_median, rtol=1e-7, atol=1e-9))
        ),
        "guardrail_count": int(np.sum((raw_median < lower) | (raw_median > upper))),
    }
    return result
