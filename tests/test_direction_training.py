import numpy as np
import pandas as pd

from finetune.direction_training import (
    apply_selective_direction_policy,
    build_direction_sample_index,
    fit_selective_direction_policy,
    fit_temperature_scaling,
    split_direction_sample_index,
    summarize_direction_probabilities,
    summarize_non_overlapping_phases,
)


def _frame(start="2024-01-01", closes=(10, 11, 12, 11, 13, 14)):
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(close), freq="D"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(len(close), dtype=float) + 100.0,
            "amount": close * 100.0,
        }
    )


def test_direction_sample_index_uses_only_complete_future_horizons():
    samples = build_direction_sample_index(
        {"AAA": _frame()},
        lookback=3,
        horizon=2,
        stride=1,
    )

    assert samples[["context_start", "context_end", "label_index"]].to_dict(
        "records"
    ) == [
        {"context_start": 0, "context_end": 2, "label_index": 4},
        {"context_start": 1, "context_end": 3, "label_index": 5},
    ]
    assert samples["target_up"].tolist() == [True, True]
    assert samples["context_end_date"].tolist() == list(
        pd.to_datetime(["2024-01-03", "2024-01-04"])
    )
    assert samples["label_end_date"].tolist() == list(
        pd.to_datetime(["2024-01-05", "2024-01-06"])
    )


def test_time_split_purges_labels_that_cross_split_boundaries():
    frames = {
        "AAA": _frame("2024-01-01", tuple(range(10, 22))),
        "BBB": _frame("2024-01-01", tuple(range(20, 32))),
    }
    samples = build_direction_sample_index(
        frames,
        lookback=3,
        horizon=2,
        stride=1,
    )

    split = split_direction_sample_index(
        samples,
        train_end="2024-01-06",
        validation_end="2024-01-09",
        test_end="2024-01-12",
        test_symbol="AAA",
    )

    assert (split["train"]["label_end_date"] <= pd.Timestamp("2024-01-06")).all()
    assert (split["validation"]["context_end_date"] > pd.Timestamp("2024-01-06")).all()
    assert (split["validation"]["label_end_date"] <= pd.Timestamp("2024-01-09")).all()
    assert (split["test"]["context_end_date"] > pd.Timestamp("2024-01-09")).all()
    assert split["test"]["symbol"].unique().tolist() == ["AAA"]


def test_temperature_scaling_reduces_overconfident_log_loss():
    logits = np.array([8.0, 7.0, -8.0, -7.0])
    targets = np.array([1.0, 0.0, 0.0, 1.0])

    result = fit_temperature_scaling(logits, targets)

    assert result["temperature"] > 1.0
    assert result["calibrated_log_loss"] < result["raw_log_loss"]
    np.testing.assert_array_equal(
        result["calibrated_probabilities"] >= 0.5,
        logits >= 0.0,
    )


def test_direction_probability_summary_reports_accuracy_brier_and_ece():
    summary = summarize_direction_probabilities(
        probabilities=[0.9, 0.8, 0.4, 0.1],
        targets=[1, 0, 1, 0],
        bins=2,
    )

    assert summary["points"] == 4
    assert summary["accuracy"] == 0.5
    assert summary["balanced_accuracy"] == 0.5
    assert summary["brier_score"] == 0.255
    assert summary["expected_calibration_error"] == 0.3
    assert len(summary["reliability_bins"]) == 2


def test_non_overlapping_phase_summary_exposes_lucky_sampling_offsets():
    samples = pd.DataFrame(
        {
            "symbol": ["AAA"] * 6,
            "context_end": list(range(6)),
        }
    )
    phases = summarize_non_overlapping_phases(
        probabilities=[0.9] * 6,
        targets=[1, 0, 1, 0, 1, 0],
        samples=samples,
        horizon=2,
    )

    assert [phase["accuracy"] for phase in phases["phases"]] == [1.0, 0.0]
    assert phases["mean_accuracy"] == 0.5
    assert phases["worst_accuracy"] == 0.0


def test_selective_policy_requires_agreement_and_keeps_validated_strong_margins():
    probabilities = np.array(
        [
            [0.80, 0.78, 0.22, 0.20, 0.75, 0.51, 0.49, 0.52, 0.48, 0.51],
            [0.76, 0.74, 0.25, 0.24, 0.71, 0.49, 0.51, 0.51, 0.49, 0.49],
            [0.72, 0.70, 0.28, 0.27, 0.68, 0.52, 0.48, 0.49, 0.51, 0.52],
        ]
    )
    targets = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])

    policy = fit_selective_direction_policy(
        probabilities,
        targets,
        min_coverage=0.4,
        min_points=4,
        require_wilson_above_chance=False,
    )
    ensemble, selected = apply_selective_direction_policy(probabilities, policy)

    assert policy["enabled"] is True
    assert policy["validation_points"] == 5
    assert policy["validation_accuracy"] == 1.0
    np.testing.assert_array_equal(selected, [True, True, True, True, True] + [False] * 5)
    np.testing.assert_allclose(ensemble[:5], probabilities[:, :5].mean(axis=0))


def test_selective_policy_disables_signal_when_validation_evidence_is_not_above_chance():
    probabilities = np.array(
        [
            [0.8, 0.8, 0.2, 0.2] * 20,
            [0.7, 0.7, 0.3, 0.3] * 20,
            [0.6, 0.6, 0.4, 0.4] * 20,
        ]
    )
    targets = np.array([1, 0, 0, 1] * 20)

    policy = fit_selective_direction_policy(
        probabilities,
        targets,
        min_coverage=0.3,
        min_points=20,
    )
    _, selected = apply_selective_direction_policy(probabilities, policy)

    assert policy["enabled"] is False
    assert not selected.any()
