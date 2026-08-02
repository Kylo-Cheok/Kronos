import numpy as np
import pytest

from model.kronos import summarize_prediction_samples
from webui.app import classify_uncertainty


def test_summarize_prediction_samples_returns_calibratable_statistics():
    samples = np.zeros((4, 3, 6), dtype=np.float32)
    samples[:, :, 3] = np.array(
        [
            [9.0, 10.5, 12.0],
            [11.0, 10.1, 9.0],
            [10.5, 9.5, 10.2],
            [8.5, 11.0, 10.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_prediction_samples(
        samples,
        last_close=10.0,
        confidence_level=0.8,
    )

    assert summary["mean"].shape == (3, 6)
    assert summary["lower"].shape == (3, 6)
    assert summary["median"].shape == (3, 6)
    assert summary["upper"].shape == (3, 6)
    np.testing.assert_allclose(summary["mean"][:, 3], [9.75, 10.275, 10.3])
    np.testing.assert_allclose(summary["median"][:, 3], [9.75, 10.3, 10.1])
    np.testing.assert_allclose(summary["up_probability"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(summary["cumulative_up_probability"], [0.5, 0.75, 0.5])
    assert np.all(summary["lower"] <= summary["median"])
    assert np.all(summary["median"] <= summary["upper"])
    assert np.all(summary["interval_width"] >= 0)


def test_summarize_prediction_samples_rejects_invalid_confidence_level():
    samples = np.zeros((2, 1, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="confidence_level"):
        summarize_prediction_samples(samples, last_close=10.0, confidence_level=1.0)


def test_classify_uncertainty_does_not_call_one_sample_low_uncertainty():
    assert classify_uncertainty(1.0, 0.0, sample_count=1) == "insufficient_samples"
    assert classify_uncertainty(1.0, 0.0, sample_count=16) == "low"
