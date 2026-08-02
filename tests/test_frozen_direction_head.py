import numpy as np
import pandas as pd
import torch

from finetune.frozen_direction_head import (
    FrozenDirectionHead,
    assess_direction_candidate,
    prepare_context_arrays,
    select_non_overlapping_samples,
)


def _frame(rows=100):
    close = np.linspace(10.0, 20.0, rows)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=rows),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(100.0, 300.0, rows),
            "amount": np.linspace(1000.0, 6000.0, rows),
        }
    )


def test_prepare_context_arrays_normalizes_context_without_future_rows():
    frame = _frame()
    values, stamps, technical = prepare_context_arrays(frame, 10, 39, clip=5.0)

    assert values.shape == (30, 6)
    assert stamps.shape == (30, 5)
    assert technical.ndim == 1
    np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=2e-5)
    np.testing.assert_allclose(values.std(axis=0), 1.0, atol=2e-5)
    assert stamps[-1].tolist() == [0.0, 0.0, 4.0, 23.0, 2.0]

    changed = frame.copy()
    changed.loc[40:, "close"] = 1_000_000.0
    changed_values, changed_stamps, changed_technical = prepare_context_arrays(
        changed, 10, 39, clip=5.0
    )
    np.testing.assert_array_equal(changed_values, values)
    np.testing.assert_array_equal(changed_stamps, stamps)
    np.testing.assert_array_equal(changed_technical, technical)


def test_direction_head_maps_frozen_features_to_one_logit_per_sample():
    head = FrozenDirectionHead(input_dim=20, hidden_dim=8, dropout=0.0)
    logits = head(torch.zeros(4, 20))
    assert logits.shape == (4,)


def test_non_overlapping_selection_keeps_independent_horizon_blocks():
    samples = pd.DataFrame(
        {
            "symbol": ["AAA"] * 8 + ["BBB"] * 8,
            "context_end": list(range(8)) * 2,
        }
    )
    selected = select_non_overlapping_samples(samples, horizon=3)

    assert selected[selected["symbol"] == "AAA"]["context_end"].tolist() == [0, 3, 6]
    assert selected[selected["symbol"] == "BBB"]["context_end"].tolist() == [0, 3, 6]


def test_candidate_acceptance_requires_discrimination_calibration_and_stability():
    good = {
        "accuracy": 0.58,
        "balanced_accuracy": 0.57,
        "brier_score": 0.235,
    }
    independent = {
        "points": 45,
        "accuracy": 0.58,
        "accuracy_wilson_95": (0.44, 0.70),
    }
    accepted = assess_direction_candidate(good, independent)
    assert accepted["accepted"] is True
    assert accepted["failed_checks"] == []

    overconfident = assess_direction_candidate(
        {**good, "brier_score": 0.31}, independent
    )
    assert overconfident["accepted"] is False
    assert "brier_score" in overconfident["failed_checks"]

    too_small = assess_direction_candidate(good, {**independent, "points": 12})
    assert too_small["accepted"] is False
    assert "independent_points" in too_small["failed_checks"]
