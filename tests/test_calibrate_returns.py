"""Tests for affine return calibration."""

import numpy as np

from finetune.calibrate_returns import apply_affine, fit_affine_per_horizon, fit_and_apply_horizons


def test_fit_affine_recovers_scale_and_bias():
    rng = np.random.default_rng(0)
    true_a, true_b = 0.8, 0.01
    pred = rng.normal(0, 0.02, size=200)
    target = true_a * pred + true_b + rng.normal(0, 0.001, size=200)
    a, b = fit_affine_per_horizon(pred, target)
    assert abs(a - true_a) < 0.05
    assert abs(b - true_b) < 0.01


def test_apply_affine():
    out = apply_affine(np.array([1.0, 2.0]), 2.0, -0.5)
    np.testing.assert_allclose(out, [1.5, 3.5])


def test_fit_and_apply_horizons_with_mask():
    pred = {1: np.array([0.0, 0.1, 0.2, 0.3])}
    target = {1: np.array([0.0, 0.05, 0.10, 0.15])}
    mask = np.array([True, True, True, False])
    cal, params = fit_and_apply_horizons(pred, target, (1,), fit_mask=mask)
    assert "1" in params
    assert cal[1].shape == (4,)
