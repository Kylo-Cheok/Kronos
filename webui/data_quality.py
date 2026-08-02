"""Data-quality helpers for financial time-series inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")


def _canonical_share_factors():
    # Common A-share bonus/share-conversion ratios are announced as
    # "10 shares become 10+n shares".  Their ex-right price factors are
    # therefore 10/(10+n).  Include reciprocals for reverse actions.
    downward = [10.0 / (10.0 + added) for added in range(1, 11)]
    return downward + [1.0 / factor for factor in downward]


def adjust_corporate_action_gaps(
    frame: pd.DataFrame,
    gap_ratio_threshold: float = 0.78,
    factor_tolerance: float = 0.03,
    max_intraday_move: float = 0.15,
):
    """Back-adjust likely split/bonus-share gaps without changing the input.

    Detection deliberately targets only very large overnight gaps that are
    close to a common share-conversion factor and whose same-day candle is not
    itself extreme.  Ordinary price-limit moves around 20% are left untouched.
    """
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"data is missing OHLC columns: {missing}")
    if not 0.0 < gap_ratio_threshold < 1.0:
        raise ValueError("gap_ratio_threshold must be between 0 and 1")
    if factor_tolerance <= 0.0:
        raise ValueError("factor_tolerance must be positive")

    adjusted = frame.copy(deep=True)
    if len(adjusted) < 2:
        return adjusted, {"event_count": 0, "events": []}

    prices = adjusted.loc[:, PRICE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("OHLC values must be finite and positive")

    candidates = np.asarray(_canonical_share_factors(), dtype=float)
    events = []
    for row_index in range(1, len(adjusted)):
        previous_close = float(adjusted.iloc[row_index - 1]["close"])
        current_open = float(adjusted.iloc[row_index]["open"])
        current_close = float(adjusted.iloc[row_index]["close"])
        observed_ratio = current_open / previous_close
        is_large_gap = (
            observed_ratio < gap_ratio_threshold
            or observed_ratio > 1.0 / gap_ratio_threshold
        )
        if not is_large_gap:
            continue
        intraday_move = current_close / current_open - 1.0
        if abs(intraday_move) > max_intraday_move:
            continue

        relative_errors = np.abs(candidates / observed_ratio - 1.0)
        candidate_index = int(np.argmin(relative_errors))
        if relative_errors[candidate_index] > factor_tolerance:
            continue
        factor = float(candidates[candidate_index])
        timestamp = None
        if "timestamps" in adjusted.columns:
            value = adjusted.iloc[row_index]["timestamps"]
            timestamp = pd.Timestamp(value).isoformat()
        events.append(
            {
                "row_index": int(row_index),
                "timestamp": timestamp,
                "observed_open_ratio": float(observed_ratio),
                "applied_factor": factor,
                "factor_relative_error": float(relative_errors[candidate_index]),
            }
        )

    if not events:
        return adjusted, {"event_count": 0, "events": []}

    event_factors = {event["row_index"]: event["applied_factor"] for event in events}
    multipliers = np.ones(len(adjusted), dtype=float)
    running_multiplier = 1.0
    for row_index in range(len(adjusted) - 1, -1, -1):
        multipliers[row_index] = running_multiplier
        if row_index in event_factors:
            running_multiplier *= event_factors[row_index]

    for column in PRICE_COLUMNS:
        adjusted[column] = adjusted[column].to_numpy(dtype=float) * multipliers
    if "volume" in adjusted.columns:
        volume = adjusted["volume"].to_numpy(dtype=float)
        adjusted["volume"] = volume / multipliers

    return adjusted, {
        "event_count": int(len(events)),
        "events": events,
        "historical_price_multiplier": float(running_multiplier),
    }
