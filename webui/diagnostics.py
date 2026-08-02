"""Pure helpers for comparing forecast stability and accuracy.

The Web UI uses these functions to separate model quality from chart rendering.
They intentionally report invalid OHLC output instead of silently repairing it.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


OHLC_COLUMNS = ("open", "high", "low", "close")


def _wilson_interval(correct: int, points: int, z: float = 1.959963984540054) -> list:
    """Return a two-sided Wilson score interval for a binary hit rate."""
    if points <= 0:
        return [None, None]
    rate = correct / points
    denominator = 1.0 + z * z / points
    centre = (rate + z * z / (2.0 * points)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / points + z * z / (4.0 * points * points))
        / denominator
    )
    return [float(centre - half_width), float(centre + half_width)]


def _binary_direction_summary(predicted_up: np.ndarray, actual_up: np.ndarray) -> dict:
    predicted_up = np.asarray(predicted_up, dtype=bool).reshape(-1)
    actual_up = np.asarray(actual_up, dtype=bool).reshape(-1)
    if len(predicted_up) != len(actual_up):
        raise ValueError("predicted and actual directions must have equal lengths")

    points = len(actual_up)
    correct = int(np.sum(predicted_up == actual_up))
    recalls = []
    for direction in (True, False):
        selected = actual_up == direction
        if np.any(selected):
            recalls.append(float(np.mean(predicted_up[selected] == actual_up[selected])))

    return {
        "points": int(points),
        "accuracy": float(correct / points) if points else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "predicted_up_rate": float(np.mean(predicted_up)) if points else None,
        "actual_up_rate": float(np.mean(actual_up)) if points else None,
        "wilson_95": _wilson_interval(correct, points),
    }


def build_direction_signal(
    context_close,
    predicted_close,
    cumulative_up_probability=None,
    horizon=None,
) -> dict:
    """Build a conservative direction signal from one independent forecast.

    The model vote is only actionable when it agrees with two lagged-only
    mean-reversion baselines.  Otherwise the function explicitly abstains.
    The model probability is exposed as raw and uncalibrated.
    """
    context = np.asarray(context_close, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_close, dtype=float).reshape(-1)
    if len(context) == 0 or len(predicted) == 0:
        raise ValueError("context_close and predicted_close cannot be empty")
    if not np.isfinite(np.concatenate([context, predicted])).all():
        raise ValueError("direction signal inputs must contain only finite values")

    selected_horizon = len(predicted) if horizon is None else int(horizon)
    if selected_horizon < 1 or selected_horizon > len(predicted):
        raise ValueError("horizon must be between 1 and the prediction length")
    reference_close = float(context[-1])
    if abs(reference_close) <= 1e-12:
        raise ValueError("the latest context close must be non-zero")

    predicted_end_close = float(predicted[selected_horizon - 1])
    predicted_return = predicted_end_close / reference_close - 1.0
    model_direction = "up" if predicted_return > 0.0 else "down"
    raw_probability = None
    if cumulative_up_probability is not None:
        probability = np.asarray(cumulative_up_probability, dtype=float).reshape(-1)
        if len(probability) < selected_horizon:
            raise ValueError("cumulative_up_probability is shorter than the horizon")
        if not np.isfinite(probability).all() or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError(
                "cumulative_up_probability must contain finite values between 0 and 1"
            )
        raw_probability = float(probability[selected_horizon - 1])

    result = {
        "horizon": int(selected_horizon),
        "reference_close": reference_close,
        "predicted_close": predicted_end_close,
        "predicted_return": float(predicted_return),
        "model_direction": model_direction,
        "prior_returns": {"5": None, "20": None},
        "votes": {
            "model": model_direction,
            "mean_reversion_5": None,
            "mean_reversion_20": None,
        },
        "status": "abstain",
        "candidate_direction": None,
        "direction": None,
        "reason": "insufficient_context_for_20_step_baseline",
        "raw_model_up_probability": raw_probability,
        "probability_is_calibrated": False,
    }
    if len(context) < 21:
        return result

    prior_5_return = float(context[-1] / context[-6] - 1.0)
    prior_20_return = float(context[-1] / context[-21] - 1.0)
    reversion_5_direction = "up" if prior_5_return <= 0.0 else "down"
    reversion_20_direction = "up" if prior_20_return <= 0.0 else "down"
    votes = {
        "model": model_direction,
        "mean_reversion_5": reversion_5_direction,
        "mean_reversion_20": reversion_20_direction,
    }
    agreed = len(set(votes.values())) == 1
    result.update(
        {
            "prior_returns": {"5": prior_5_return, "20": prior_20_return},
            "votes": votes,
            "status": "candidate" if agreed else "abstain",
            "candidate_direction": model_direction if agreed else None,
            "direction": None,
            "reason": "three_way_direction_agreement_uncalibrated"
            if agreed
            else "direction_votes_disagree",
        }
    )
    return result


def evaluate_direction_signals(signals, actual_close) -> dict:
    """Evaluate model and abstention-filtered chunk direction signals."""
    actual = np.asarray(actual_close, dtype=float).reshape(-1)
    if not np.isfinite(actual).all():
        raise ValueError("actual_close must contain only finite values")

    records = []
    for signal in signals:
        start = int(signal["target_start_index"])
        end = int(signal["target_end_index"])
        if start < 0 or end <= start or end > len(actual):
            raise ValueError("direction signal target indexes are outside actual_close")
        reference_close = float(signal["reference_close"])
        actual_return = float(actual[end - 1] / reference_close - 1.0)
        actual_up = actual_return > 0.0
        model_up = signal["model_direction"] == "up"
        candidate_selected = signal.get("status") in {"candidate", "actionable"}
        candidate_direction = signal.get("candidate_direction")
        if candidate_direction is None and signal.get("status") == "actionable":
            candidate_direction = signal.get("direction")
        candidate_up = candidate_direction == "up" if candidate_selected else None
        selected = signal.get("status") == "actionable"
        selected_up = signal.get("direction") == "up" if selected else None
        raw_probability = signal.get("raw_model_up_probability")
        if raw_probability is not None:
            raw_probability = float(raw_probability)
            if not np.isfinite(raw_probability) or not 0.0 <= raw_probability <= 1.0:
                raise ValueError(
                    "raw_model_up_probability must be finite and between 0 and 1"
                )
        records.append(
            {
                "target_start_index": start,
                "target_end_index": end,
                "actual_return": actual_return,
                "actual_up": bool(actual_up),
                "model_up": bool(model_up),
                "model_correct": bool(model_up == actual_up),
                "candidate_selected": bool(candidate_selected),
                "candidate_up": bool(candidate_up) if candidate_selected else None,
                "candidate_correct": bool(candidate_up == actual_up)
                if candidate_selected
                else None,
                "selected": bool(selected),
                "selected_up": bool(selected_up) if selected else None,
                "selected_correct": bool(selected_up == actual_up) if selected else None,
                "raw_model_up_probability": raw_probability,
            }
        )

    model_predicted = np.asarray([record["model_up"] for record in records], dtype=bool)
    model_actual = np.asarray([record["actual_up"] for record in records], dtype=bool)
    selected_records = [record for record in records if record["selected"]]
    candidate_records = [
        record for record in records if record["candidate_selected"]
    ]
    selected_predicted = np.asarray(
        [record["selected_up"] for record in selected_records], dtype=bool
    )
    selected_actual = np.asarray(
        [record["actual_up"] for record in selected_records], dtype=bool
    )
    candidate_predicted = np.asarray(
        [record["candidate_up"] for record in candidate_records], dtype=bool
    )
    candidate_actual = np.asarray(
        [record["actual_up"] for record in candidate_records], dtype=bool
    )
    candidate = _binary_direction_summary(candidate_predicted, candidate_actual)
    candidate["coverage"] = (
        float(len(candidate_records) / len(records)) if records else 0.0
    )
    filtered = _binary_direction_summary(selected_predicted, selected_actual)
    filtered["coverage"] = (
        float(len(selected_records) / len(records)) if records else 0.0
    )

    probability_records = [
        record
        for record in records
        if record["raw_model_up_probability"] is not None
    ]
    raw_probability = None
    if probability_records:
        probabilities = np.asarray(
            [record["raw_model_up_probability"] for record in probability_records],
            dtype=float,
        )
        probability_actual = np.asarray(
            [record["actual_up"] for record in probability_records], dtype=bool
        )
        bins = []
        bin_indexes = np.minimum((probabilities * 5.0).astype(int), 4)
        for bin_index in range(5):
            selected = bin_indexes == bin_index
            if not np.any(selected):
                continue
            bins.append(
                {
                    "lower": float(bin_index / 5.0),
                    "upper": float((bin_index + 1) / 5.0),
                    "points": int(np.sum(selected)),
                    "mean_probability": float(np.mean(probabilities[selected])),
                    "observed_up_rate": float(np.mean(probability_actual[selected])),
                }
            )
        raw_probability = {
            "points": int(len(probabilities)),
            "accuracy": float(
                np.mean((probabilities >= 0.5) == probability_actual)
            ),
            "brier_score": float(
                np.mean((probabilities - probability_actual.astype(float)) ** 2)
            ),
            "is_calibrated": False,
            "reliability_bins": bins,
        }

    return {
        "points": int(len(records)),
        "model": _binary_direction_summary(model_predicted, model_actual),
        "candidate": candidate,
        "filtered": filtered,
        "raw_probability": raw_probability,
        "records": records,
    }


def _as_frame(values) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.reset_index(drop=True).copy()
    return pd.DataFrame(values).reset_index(drop=True)


def _close_metrics(pred_close: np.ndarray, actual_close: np.ndarray) -> dict:
    if len(pred_close) == 0 or len(actual_close) == 0:
        return {"points": 0, "mae": None, "rmse": None, "mape": None}

    count = min(len(pred_close), len(actual_close))
    pred = pred_close[:count].astype(float)
    actual = actual_close[:count].astype(float)
    errors = pred - actual
    non_zero_actual = np.abs(actual) > 1e-12
    percentage_errors = np.abs(errors[non_zero_actual]) / np.abs(actual[non_zero_actual])

    return {
        "points": int(count),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(percentage_errors) * 100.0)
        if len(percentage_errors)
        else None,
    }


def _jump_metrics(close: np.ndarray, threshold: float) -> dict:
    if len(close) < 2:
        return {"count": 0, "max_abs_pct": 0.0}

    previous = close[:-1].astype(float)
    current = close[1:].astype(float)
    valid = np.abs(previous) > 1e-12
    returns = np.zeros_like(current, dtype=float)
    returns[valid] = current[valid] / previous[valid] - 1.0
    absolute_returns = np.abs(returns[valid])

    return {
        "count": int(np.sum(absolute_returns > threshold)),
        "max_abs_pct": float(np.max(absolute_returns) * 100.0)
        if len(absolute_returns)
        else 0.0,
    }


def summarize_direction_metrics(
    predicted_close,
    actual_close,
    reference_close,
    actual_reference_close=None,
    up_probability=None,
    confidence_thresholds=(0.0, 0.2, 0.4, 0.6, 0.8),
) -> dict:
    """Measure binary up/down accuracy using closes known at forecast time."""
    predicted = np.asarray(predicted_close, dtype=float).reshape(-1)
    actual = np.asarray(actual_close, dtype=float).reshape(-1)
    predicted_reference = np.asarray(reference_close, dtype=float).reshape(-1)
    actual_reference = (
        predicted_reference
        if actual_reference_close is None
        else np.asarray(actual_reference_close, dtype=float).reshape(-1)
    )

    lengths = {len(predicted), len(actual), len(predicted_reference), len(actual_reference)}
    if len(lengths) != 1:
        raise ValueError("predicted, actual, and reference closes must have equal lengths")
    if not np.isfinite(
        np.concatenate([predicted, actual, predicted_reference, actual_reference])
    ).all():
        raise ValueError("direction inputs must contain only finite values")

    predicted_up = predicted > predicted_reference
    actual_up = actual > actual_reference
    true_up_pred_up = int(np.sum(predicted_up & actual_up))
    true_up_pred_down = int(np.sum(~predicted_up & actual_up))
    true_down_pred_up = int(np.sum(predicted_up & ~actual_up))
    true_down_pred_down = int(np.sum(~predicted_up & ~actual_up))

    recalls = []
    if np.any(actual_up):
        recalls.append(float(np.mean(predicted_up[actual_up])))
    if np.any(~actual_up):
        recalls.append(float(np.mean(~predicted_up[~actual_up])))

    predicted_return = (predicted - predicted_reference) / (
        np.abs(predicted_reference) + 1e-12
    )
    actual_return = (actual - actual_reference) / (np.abs(actual_reference) + 1e-12)
    records = [
        {
            "predicted_return": float(predicted_return[index]),
            "actual_return": float(actual_return[index]),
            "predicted_up": bool(predicted_up[index]),
            "actual_up": bool(actual_up[index]),
            "correct": bool(predicted_up[index] == actual_up[index]),
            "up_probability": None,
            "confidence": None,
        }
        for index in range(len(predicted))
    ]

    result = {
        "points": int(len(predicted)),
        "accuracy": float(np.mean(predicted_up == actual_up)) if len(predicted) else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "predicted_up_rate": float(np.mean(predicted_up)) if len(predicted) else None,
        "actual_up_rate": float(np.mean(actual_up)) if len(actual) else None,
        "confusion_matrix": {
            "true_up_pred_up": true_up_pred_up,
            "true_up_pred_down": true_up_pred_down,
            "true_down_pred_up": true_down_pred_up,
            "true_down_pred_down": true_down_pred_down,
        },
        "records": records,
        "probability": None,
    }

    if up_probability is not None:
        probability = np.asarray(up_probability, dtype=float).reshape(-1)
        if len(probability) != len(actual):
            raise ValueError("up_probability must have the same length as actual_close")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("up_probability must contain finite values between 0 and 1")

        probability_up = probability >= 0.5
        confidence = np.abs(probability - 0.5) * 2.0
        for index, record in enumerate(records):
            record["up_probability"] = float(probability[index])
            record["confidence"] = float(confidence[index])
        selective_accuracy = {}
        for threshold in confidence_thresholds:
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence thresholds must be between 0 and 1")
            selected = confidence >= threshold
            selected_points = int(np.sum(selected))
            selective_accuracy[str(threshold)] = {
                "points": selected_points,
                "coverage": float(np.mean(selected)) if len(selected) else 0.0,
                "accuracy": float(np.mean(probability_up[selected] == actual_up[selected]))
                if selected_points
                else None,
            }

        result["probability"] = {
            "accuracy": float(np.mean(probability_up == actual_up)) if len(actual) else None,
            "brier_score": float(np.mean((probability - actual_up.astype(float)) ** 2))
            if len(actual)
            else None,
            "mean_up_probability": float(np.mean(probability)) if len(probability) else None,
            "selective_accuracy": selective_accuracy,
        }

    return result


def summarize_forecast_diagnostics(
    prediction,
    actual=None,
    horizons: Iterable[int] = (1, 5, 10, 20, 60, 120),
    prediction_reference_close=None,
    actual_reference_close=None,
    up_probability=None,
    direction_signals=None,
) -> dict:
    """Return accuracy and structural diagnostics for one forecast.

    ``actual`` is optional so the same report works for backtests and future
    forecasts.  The function never clips or reorders predicted candles.
    """
    pred = _as_frame(prediction)
    missing = [column for column in OHLC_COLUMNS if column not in pred.columns]
    if missing:
        raise ValueError(f"prediction is missing OHLC columns: {missing}")

    pred_close = pred["close"].to_numpy(dtype=float)
    pred_open = pred["open"].to_numpy(dtype=float)
    pred_high = pred["high"].to_numpy(dtype=float)
    pred_low = pred["low"].to_numpy(dtype=float)

    invalid_high = pred_high < np.maximum(pred_open, pred_close)
    invalid_low = pred_low > np.minimum(pred_open, pred_close)
    invalid_order = pred_high < pred_low
    actual_frame = _as_frame(actual) if actual is not None else None

    result = {
        "points": int(len(pred)),
        "close_range": [float(np.min(pred_close)), float(np.max(pred_close))]
        if len(pred_close)
        else [],
        "close_jumps_gt_10pct": _jump_metrics(pred_close, 0.10),
        "close_jumps_gt_20pct": _jump_metrics(pred_close, 0.20),
        "ohlc_violations": {
            "high_below_open_or_close": int(np.sum(invalid_high)),
            "low_above_open_or_close": int(np.sum(invalid_low)),
            "high_below_low": int(np.sum(invalid_order)),
            "total": int(np.sum(invalid_high | invalid_low | invalid_order)),
        },
        "metrics": None,
        "metrics_by_horizon": {},
        "direction": None,
        "chunk_direction": None,
    }

    if actual_frame is not None:
        if "close" not in actual_frame.columns:
            raise ValueError("actual is missing close column")
        actual_close = actual_frame["close"].to_numpy(dtype=float)
        result["metrics"] = _close_metrics(pred_close, actual_close)
        for horizon in horizons:
            horizon = int(horizon)
            if horizon < 1:
                continue
            result["metrics_by_horizon"][str(horizon)] = _close_metrics(
                pred_close[:horizon], actual_close[:horizon]
            )
        if prediction_reference_close is not None:
            result["direction"] = summarize_direction_metrics(
                predicted_close=pred_close,
                actual_close=actual_close,
                reference_close=prediction_reference_close,
                actual_reference_close=actual_reference_close,
                up_probability=up_probability,
            )
        if direction_signals is not None:
            result["chunk_direction"] = evaluate_direction_signals(
                direction_signals,
                actual_close,
            )

    return result


def build_ab_cases(
    temperature: float,
    top_p: float,
    sample_count: int,
) -> list[Mapping]:
    """Build the fixed A/B cases used by the diagnostic endpoint."""
    return [
        {
            "id": "current",
            "label": f"Current (T={temperature:g}, samples={sample_count})",
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sample_count": int(sample_count),
            "deterministic": False,
        },
        {
            "id": "conservative",
            "label": "Conservative (T=0.6, samples=5)",
            "temperature": 0.6,
            "top_p": 0.9,
            "sample_count": 5,
            "deterministic": False,
        },
        {
            "id": "deterministic",
            "label": "Deterministic (T=0.6, greedy)",
            "temperature": 0.6,
            "top_p": 0.9,
            "sample_count": 1,
            "deterministic": True,
        },
    ]
