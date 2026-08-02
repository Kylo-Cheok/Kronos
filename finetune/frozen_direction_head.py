"""Frozen-representation direction classifier utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
TIME_COLUMNS = ["minute", "hour", "weekday", "day", "month"]


class FrozenDirectionHead(nn.Module):
    """Small nonlinear head trained on detached Kronos representations."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _technical_snapshot(context: pd.DataFrame) -> np.ndarray:
    close = context["close"].to_numpy(dtype=np.float64)
    high = context["high"].to_numpy(dtype=np.float64)
    low = context["low"].to_numpy(dtype=np.float64)
    volume = context["volume"].to_numpy(dtype=np.float64)
    log_close = np.log(np.clip(close, 1e-12, None))
    log_returns = np.diff(log_close)

    features: list[float] = []
    for horizon in (1, 3, 5, 10, 20, 60):
        if len(close) > horizon:
            features.append(float(log_close[-1] - log_close[-1 - horizon]))
        else:
            features.append(0.0)
    for window in (5, 20, 60):
        values = log_returns[-window:]
        features.append(float(np.std(values)) if len(values) else 0.0)
    for window in (5, 20, 60):
        values = close[-window:]
        mean = float(np.mean(values)) if len(values) else float(close[-1])
        features.append(float(close[-1] / max(mean, 1e-12) - 1.0))
    for window in (5, 20):
        range_ratio = (high[-window:] - low[-window:]) / np.clip(
            close[-window:], 1e-12, None
        )
        features.append(float(np.mean(range_ratio)))
    volume_mean = float(np.mean(volume[-20:]))
    features.append(float(volume[-1] / max(volume_mean, 1e-12) - 1.0))
    return np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=5.0,
        neginf=-5.0,
    )


def prepare_context_arrays(
    frame: pd.DataFrame,
    context_start: int,
    context_end: int,
    *,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one context and derive time/technical features without future rows."""
    missing = set(FEATURE_COLUMNS + ["date"]).difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    if context_start < 0 or context_end < context_start or context_end >= len(frame):
        raise ValueError("invalid context bounds")

    context = frame.iloc[context_start : context_end + 1].copy()
    values = context[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("context feature values must be finite")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    values = np.clip((values - mean) / (std + 1e-5), -clip, clip).astype(
        np.float32
    )

    dates = pd.to_datetime(context["date"])
    stamps = np.column_stack(
        [
            dates.dt.minute,
            dates.dt.hour,
            dates.dt.weekday,
            dates.dt.day,
            dates.dt.month,
        ]
    ).astype(np.float32)
    return values, stamps, _technical_snapshot(context)


def select_non_overlapping_samples(
    samples: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Greedily retain sample origins separated by at least one forecast horizon."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = {"symbol", "context_end"}
    if not required.issubset(samples.columns):
        raise ValueError(f"samples must contain {sorted(required)}")

    selected_indices: list[int] = []
    for _, group in samples.sort_values(["symbol", "context_end"]).groupby(
        "symbol", sort=False
    ):
        last_context_end: int | None = None
        for index, row in group.iterrows():
            context_end = int(row["context_end"])
            if last_context_end is None or context_end - last_context_end >= horizon:
                selected_indices.append(index)
                last_context_end = context_end
    return samples.loc[selected_indices].reset_index(drop=True)


def assess_direction_candidate(
    summary: Mapping[str, float],
    independent_summary: Mapping[str, object],
    *,
    min_accuracy: float = 0.55,
    min_balanced_accuracy: float = 0.55,
    max_brier_score: float = 0.25,
    min_independent_points: int = 30,
    min_independent_accuracy: float = 0.55,
    min_wilson_lower: float = 0.40,
) -> dict[str, object]:
    """Apply explicit, conservative gates before a head can be called actionable."""
    failed_checks: list[str] = []
    if float(summary["accuracy"]) < min_accuracy:
        failed_checks.append("accuracy")
    if float(summary["balanced_accuracy"]) < min_balanced_accuracy:
        failed_checks.append("balanced_accuracy")
    if float(summary["brier_score"]) >= max_brier_score:
        failed_checks.append("brier_score")
    if int(independent_summary["points"]) < min_independent_points:
        failed_checks.append("independent_points")
    if float(independent_summary["accuracy"]) < min_independent_accuracy:
        failed_checks.append("independent_accuracy")
    wilson_lower = float(independent_summary["accuracy_wilson_95"][0])
    if wilson_lower < min_wilson_lower:
        failed_checks.append("wilson_lower")
    return {"accepted": not failed_checks, "failed_checks": failed_checks}
