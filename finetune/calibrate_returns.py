"""Fit affine return calibration on pre-test windows (leakage-safe vs test).

Uses context_end dates in [val_start, val_end] to fit per-horizon
``pred_cal = a * pred + b`` minimizing MAE, then applies to test predictions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_affine_per_horizon(
    pred: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    """Least-squares fit target ≈ a * pred + b; fallback identity if singular."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.size < 5:
        return 1.0, 0.0
    x = np.column_stack([pred, np.ones_like(pred)])
    try:
        coef, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if not np.isfinite(a) or not np.isfinite(b):
            return 1.0, 0.0
        # guard against pathological scales
        if abs(a) > 5.0 or abs(a) < 0.05:
            return 1.0, 0.0
        return a, b
    except np.linalg.LinAlgError:
        return 1.0, 0.0


def apply_affine(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return float(a) * np.asarray(pred, dtype=np.float64) + float(b)


def fit_and_apply_horizons(
    pred_by_h: dict[int, np.ndarray],
    target_by_h: dict[int, np.ndarray],
    horizons: Sequence[int],
    *,
    fit_mask: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, dict[str, float]]]:
    """Fit on fit_mask rows (or all), apply to full arrays."""
    calibrated: dict[int, np.ndarray] = {}
    params: dict[str, dict[str, float]] = {}
    for h in horizons:
        p = np.asarray(pred_by_h[h], dtype=np.float64)
        t = np.asarray(target_by_h[h], dtype=np.float64)
        if fit_mask is not None:
            a, b = fit_affine_per_horizon(p[fit_mask], t[fit_mask])
        else:
            a, b = fit_affine_per_horizon(p, t)
        calibrated[h] = apply_affine(p, a, b)
        params[str(h)] = {"a": a, "b": b}
    return calibrated, params
