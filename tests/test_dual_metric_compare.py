"""Unit tests for dual-metric promotion helpers."""

import numpy as np

from finetune.dual_metric_compare import (
    blend_returns,
    dual_bar_decision,
    return_mae,
    summarize_horizons,
)
from finetune.selective_prediction import DOWN_CLASS, FLAT_CLASS, UP_CLASS


def test_return_mae_basic():
    pred = np.array([0.1, -0.1, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    assert abs(return_mae(pred, target) - (0.1 + 0.1 + 0.0) / 3) < 1e-9


def test_summarize_horizons_overall():
    pred_dir = {
        1: np.array([UP_CLASS, DOWN_CLASS]),
        3: np.array([UP_CLASS, UP_CLASS]),
    }
    t_dir = {
        1: np.array([UP_CLASS, UP_CLASS]),
        3: np.array([UP_CLASS, FLAT_CLASS]),
    }
    pred_ret = {1: np.array([0.0, 0.2]), 3: np.array([0.1, 0.1])}
    t_ret = {1: np.array([0.0, 0.0]), 3: np.array([0.0, 0.0])}
    s = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, horizons=(1, 3))
    # nonflat targets: h1 both, h3 only first → 3 nonflat points, correct: h1[0], h3[0] = 2/3
    assert abs(s["nonflat_accuracy_overall"] - 2 / 3) < 1e-9
    assert s["return_mae_overall"] > 0


def test_dual_bar_promotes_on_direction_and_mae():
    baseline = {
        "nonflat_accuracy_overall": 0.70,
        "return_mae_overall": 0.04,
        "by_horizon": {
            "1": {"return_mae": 0.02},
            "3": {"return_mae": 0.03},
            "5": {"return_mae": 0.05},
            "10": {"return_mae": 0.06},
        },
        "h1_gated_precision": 0.60,
        "h1_gated_coverage": 0.25,
    }
    candidate = {
        "nonflat_accuracy_overall": 0.71,  # +1pt
        "return_mae_overall": 0.038,
        "by_horizon": {
            "1": {"return_mae": 0.019},
            "3": {"return_mae": 0.029},
            "5": {"return_mae": 0.048},
            "10": {"return_mae": 0.058},
        },
        "h1_gated_precision": 0.62,
        "h1_gated_coverage": 0.25,  # in 20-40% band
    }
    d = dual_bar_decision(candidate, baseline)
    assert d["promote"] is True
    assert d["direction_win"] is True
    assert d["mae_win"] is True


def test_dual_bar_gate_path_requires_coverage_band():
    baseline = {
        "nonflat_accuracy_overall": 0.70,
        "return_mae_overall": 0.04,
        "by_horizon": {
            "1": {"return_mae": 0.02},
            "3": {"return_mae": 0.03},
            "5": {"return_mae": 0.05},
            "10": {"return_mae": 0.06},
        },
        "h1_gated_precision": 0.60,
        "h1_gated_nonflat": 0.80,
        "h1_gated_coverage": 0.25,
    }
    # Big gated gain but coverage outside 20-40% → no dir_via_gate
    oob = {
        "nonflat_accuracy_overall": 0.701,  # < +0.5pt
        "return_mae_overall": 0.038,
        "by_horizon": {
            "1": {"return_mae": 0.019},
            "3": {"return_mae": 0.029},
            "5": {"return_mae": 0.048},
            "10": {"return_mae": 0.058},
        },
        "h1_gated_precision": 0.70,
        "h1_gated_nonflat": 0.90,
        "h1_gated_coverage": 0.19,  # OOB
    }
    d = dual_bar_decision(oob, baseline)
    assert d["dir_via_gate"] is False
    assert d["gate_in_band"] is False
    # Still promotable via MAE win + direction non-regress
    assert d["mae_win"] is True
    assert d["promote"] is True
    assert d["reason"] == "mae_win_direction_nonregress"

def test_dual_bar_rejects_mae_blowup():
    baseline = {
        "nonflat_accuracy_overall": 0.70,
        "return_mae_overall": 0.04,
        "by_horizon": {
            "1": {"return_mae": 0.02},
            "3": {"return_mae": 0.03},
            "5": {"return_mae": 0.05},
            "10": {"return_mae": 0.06},
        },
    }
    candidate = {
        "nonflat_accuracy_overall": 0.72,
        "return_mae_overall": 0.035,
        "by_horizon": {
            "1": {"return_mae": 0.03},  # +50% relative — fail
            "3": {"return_mae": 0.028},
            "5": {"return_mae": 0.04},
            "10": {"return_mae": 0.05},
        },
    }
    d = dual_bar_decision(candidate, baseline)
    assert d["promote"] is False
    assert d["mae_nonregress"] is False


def test_blend_returns():
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 0.0])
    out = blend_returns(a, b, weight_a=0.25)
    np.testing.assert_allclose(out, [0.75, 0.25])
