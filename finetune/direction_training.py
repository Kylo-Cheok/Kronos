"""Leakage-safe utilities for training and evaluating direction heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit


SAMPLE_COLUMNS = [
    "symbol",
    "context_start",
    "context_end",
    "label_index",
    "context_end_date",
    "label_end_date",
    "target_up",
]


def build_direction_sample_index(
    frames: Mapping[str, pd.DataFrame],
    *,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> pd.DataFrame:
    """Index fixed-horizon direction samples without incomplete future labels."""
    if lookback < 1 or horizon < 1 or stride < 1:
        raise ValueError("lookback, horizon, and stride must all be positive")

    records: list[dict[str, object]] = []
    for symbol, source in frames.items():
        missing = {"date", "close"}.difference(source.columns)
        if missing:
            raise ValueError(f"{symbol} is missing required columns: {sorted(missing)}")

        frame = source.reset_index(drop=True).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
            raise ValueError(f"{symbol} dates must be sorted and unique")
        if not np.isfinite(frame["close"].to_numpy(dtype=float)).all():
            raise ValueError(f"{symbol} close values must be finite")

        last_start = len(frame) - lookback - horizon
        for context_start in range(0, last_start + 1, stride):
            context_end = context_start + lookback - 1
            label_index = context_end + horizon
            records.append(
                {
                    "symbol": str(symbol),
                    "context_start": context_start,
                    "context_end": context_end,
                    "label_index": label_index,
                    "context_end_date": frame.at[context_end, "date"],
                    "label_end_date": frame.at[label_index, "date"],
                    "target_up": bool(
                        frame.at[label_index, "close"] > frame.at[context_end, "close"]
                    ),
                }
            )

    return pd.DataFrame.from_records(records, columns=SAMPLE_COLUMNS)


def split_direction_sample_index(
    samples: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    test_symbol: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Create purged chronological splits based on label completion dates."""
    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)
    test_boundary = pd.Timestamp(test_end)
    if not train_boundary < validation_boundary < test_boundary:
        raise ValueError("split boundaries must be strictly increasing")

    indexed = samples.copy()
    for column in ("context_end_date", "label_end_date"):
        indexed[column] = pd.to_datetime(indexed[column])

    train = indexed[indexed["label_end_date"] <= train_boundary]
    validation = indexed[
        (indexed["context_end_date"] > train_boundary)
        & (indexed["label_end_date"] <= validation_boundary)
    ]
    test = indexed[
        (indexed["context_end_date"] > validation_boundary)
        & (indexed["label_end_date"] <= test_boundary)
    ]
    if test_symbol is not None:
        test = test[test["symbol"] == str(test_symbol)]

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _binary_log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
    )


def _as_binary_arrays(
    values: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value_array = np.asarray(values, dtype=float).reshape(-1)
    target_array = np.asarray(targets, dtype=float).reshape(-1)
    if value_array.size == 0 or value_array.shape != target_array.shape:
        raise ValueError("values and targets must be non-empty arrays of equal length")
    if not np.isfinite(value_array).all() or not np.isfinite(target_array).all():
        raise ValueError("values and targets must be finite")
    if not np.isin(target_array, (0.0, 1.0)).all():
        raise ValueError("targets must be binary")
    return value_array, target_array


def fit_temperature_scaling(
    logits: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    """Fit one positive temperature on validation logits only."""
    logit_array, target_array = _as_binary_arrays(logits, targets)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return _binary_log_loss(expit(logit_array / temperature), target_array)

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(100.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    temperature = float(np.exp(result.x))
    raw_probabilities = expit(logit_array)
    calibrated_probabilities = expit(logit_array / temperature)
    return {
        "temperature": temperature,
        "raw_log_loss": _binary_log_loss(raw_probabilities, target_array),
        "calibrated_log_loss": _binary_log_loss(
            calibrated_probabilities, target_array
        ),
        "calibrated_probabilities": calibrated_probabilities,
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def summarize_direction_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, object]:
    """Summarize discrimination and probability calibration metrics."""
    probability_array, target_array = _as_binary_arrays(probabilities, targets)
    if bins < 1:
        raise ValueError("bins must be positive")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must be between zero and one")

    predictions = probability_array >= 0.5
    target_bool = target_array.astype(bool)
    accuracy = float(np.mean(predictions == target_bool))
    recalls = [
        float(np.mean(predictions[target_bool == label] == label))
        for label in (False, True)
        if np.any(target_bool == label)
    ]
    balanced_accuracy = float(np.mean(recalls))
    brier_score = float(np.mean(np.square(probability_array - target_array)))

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.minimum(np.searchsorted(bin_edges, probability_array, side="right") - 1, bins - 1)
    reliability_bins: list[dict[str, object]] = []
    expected_calibration_error = 0.0
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            mean_probability = None
            observed_frequency = None
        else:
            mean_probability = float(np.mean(probability_array[mask]))
            observed_frequency = float(np.mean(target_array[mask]))
            expected_calibration_error += (
                count
                / len(probability_array)
                * abs(mean_probability - observed_frequency)
            )
        reliability_bins.append(
            {
                "lower": float(bin_edges[bin_index]),
                "upper": float(bin_edges[bin_index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
            }
        )

    successes = int(np.sum(predictions == target_bool))
    lower, upper = _wilson_interval(successes, len(target_array))
    return {
        "points": int(len(target_array)),
        "accuracy": round(accuracy, 12),
        "balanced_accuracy": round(balanced_accuracy, 12),
        "brier_score": round(brier_score, 12),
        "expected_calibration_error": round(expected_calibration_error, 12),
        "accuracy_wilson_95": (float(lower), float(upper)),
        "reliability_bins": reliability_bins,
    }


def _as_probability_matrix(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("probabilities must be a non-empty model-by-sample matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities must be finite")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError("probabilities must be between zero and one")
    return matrix


def apply_selective_direction_policy(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    policy: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a validation-fitted consensus/margin gate without using new labels."""
    matrix = _as_probability_matrix(probabilities)
    ensemble = np.mean(matrix, axis=0)
    if not bool(policy.get("enabled", False)):
        return ensemble, np.zeros(matrix.shape[1], dtype=bool)

    directions = matrix >= 0.5
    unanimous = np.all(directions == directions[0], axis=0)
    threshold = float(policy["margin_threshold"])
    selected = unanimous & (np.abs(ensemble - 0.5) >= threshold)
    return ensemble, selected


def fit_selective_direction_policy(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    min_coverage: float = 0.3,
    min_points: int = 30,
    require_wilson_above_chance: bool = True,
) -> dict[str, object]:
    """Fit a conservative signal gate using validation probabilities and labels only."""
    matrix = _as_probability_matrix(probabilities)
    _, target_array = _as_binary_arrays(np.mean(matrix, axis=0), targets)
    if matrix.shape[1] != len(target_array):
        raise ValueError("probabilities and targets must contain equal sample counts")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be between zero and one")
    if min_points < 1:
        raise ValueError("min_points must be positive")

    ensemble = np.mean(matrix, axis=0)
    directions = matrix >= 0.5
    unanimous = np.all(directions == directions[0], axis=0)
    margins = np.abs(ensemble - 0.5)
    required_points = max(min_points, int(np.ceil(min_coverage * len(target_array))))
    unanimous_margins = margins[unanimous]
    if len(unanimous_margins) < required_points:
        return {
            "enabled": False,
            "margin_threshold": None,
            "validation_coverage": 0.0,
            "validation_points": 0,
            "validation_accuracy": None,
            "validation_balanced_accuracy": None,
            "validation_wilson_95": (float("nan"), float("nan")),
        }

    candidate_thresholds = np.unique(
        np.quantile(unanimous_margins, [0.0, 0.25, 0.5, 0.75])
    )
    candidates: list[tuple[tuple[float, float, float, float], float, dict[str, object]]] = []
    for threshold in candidate_thresholds:
        selected = unanimous & (margins >= threshold)
        if int(selected.sum()) < required_points:
            continue
        summary = summarize_direction_probabilities(
            ensemble[selected], target_array[selected]
        )
        score = (
            float(summary["accuracy_wilson_95"][0]),
            float(summary["balanced_accuracy"]),
            float(summary["accuracy"]),
            float(selected.mean()),
        )
        candidates.append((score, float(threshold), summary))

    if not candidates:
        return {
            "enabled": False,
            "margin_threshold": None,
            "validation_coverage": 0.0,
            "validation_points": 0,
            "validation_accuracy": None,
            "validation_balanced_accuracy": None,
            "validation_wilson_95": (float("nan"), float("nan")),
        }

    _, threshold, summary = max(candidates, key=lambda candidate: candidate[0])
    wilson = summary["accuracy_wilson_95"]
    enabled = not require_wilson_above_chance or float(wilson[0]) > 0.5
    return {
        "enabled": enabled,
        "margin_threshold": threshold,
        "validation_coverage": float(summary["points"]) / len(target_array),
        "validation_points": int(summary["points"]),
        "validation_accuracy": float(summary["accuracy"]),
        "validation_balanced_accuracy": float(summary["balanced_accuracy"]),
        "validation_wilson_95": wilson,
    }


def summarize_non_overlapping_phases(
    probabilities: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    samples: pd.DataFrame,
    *,
    horizon: int,
) -> dict[str, object]:
    """Evaluate every non-overlapping sampling offset, not just a lucky first one."""
    probability_array, target_array = _as_binary_arrays(probabilities, targets)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if len(samples) != len(probability_array):
        raise ValueError("samples, probabilities, and targets must have equal length")
    required = {"symbol", "context_end"}
    if not required.issubset(samples.columns):
        raise ValueError(f"samples must contain {sorted(required)}")

    indexed = samples.reset_index(drop=True).copy()
    indexed["_position"] = np.arange(len(indexed))
    phase_summaries: list[dict[str, object]] = []
    for phase in range(horizon):
        positions: list[int] = []
        for _, group in indexed.sort_values(["symbol", "context_end"]).groupby(
            "symbol", sort=False
        ):
            group_positions = group["_position"].to_numpy(dtype=int)
            positions.extend(group_positions[phase::horizon].tolist())
        selected = np.asarray(sorted(positions), dtype=int)
        if len(selected) == 0:
            continue
        summary = summarize_direction_probabilities(
            probability_array[selected], target_array[selected]
        )
        phase_summaries.append({"phase": phase, **summary})

    accuracies = [float(summary["accuracy"]) for summary in phase_summaries]
    balanced = [
        float(summary["balanced_accuracy"]) for summary in phase_summaries
    ]
    return {
        "phases": phase_summaries,
        "mean_accuracy": float(np.mean(accuracies)),
        "worst_accuracy": float(np.min(accuracies)),
        "best_accuracy": float(np.max(accuracies)),
        "mean_balanced_accuracy": float(np.mean(balanced)),
        "worst_balanced_accuracy": float(np.min(balanced)),
    }
