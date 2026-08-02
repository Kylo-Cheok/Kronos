import numpy as np
import pandas as pd

from model.kronos import KronosPredictor


def test_predict_distribution_mode_exposes_mean_and_uncertainty_frames():
    predictor = object.__new__(KronosPredictor)
    predictor.price_cols = ["open", "high", "low", "close"]
    predictor.vol_col = "volume"
    predictor.amt_vol = "amount"
    predictor.clip = 5

    sampled_paths = np.zeros((1, 3, 2, 6), dtype=np.float32)
    sampled_paths[0, :, :, 3] = np.array(
        [
            [0.2, 0.3],
            [0.0, 0.1],
            [-0.2, -0.1],
        ],
        dtype=np.float32,
    )

    def fake_generate(*args, **kwargs):
        assert kwargs["return_samples"] is True
        return sampled_paths

    predictor.generate = fake_generate
    history = pd.DataFrame(
        {
            "open": [99.0, 100.0, 101.0],
            "high": [100.0, 101.0, 102.0],
            "low": [98.0, 99.0, 100.0],
            "close": [99.5, 100.5, 101.5],
        }
    )
    x_timestamp = pd.Series(pd.date_range("2026-01-01", periods=3, freq="h"))
    y_timestamp = pd.Series(pd.date_range("2026-01-01 03:00", periods=2, freq="h"))

    result = predictor.predict(
        history,
        x_timestamp,
        y_timestamp,
        pred_len=2,
        sample_count=3,
        verbose=False,
        return_distribution=True,
    )

    assert result["prediction"].shape == (2, 6)
    assert result["mean"].equals(result["prediction"])
    assert result["lower"].shape == (2, 6)
    assert result["upper"].shape == (2, 6)
    assert result["std"].shape == (2, 6)
    assert result["samples"].shape == (3, 2, 6)
    assert result["up_probability"].shape == (2,)
    assert result["cumulative_up_probability"].shape == (2,)


def test_predict_forwards_deterministic_inference_flag():
    predictor = object.__new__(KronosPredictor)
    predictor.price_cols = ["open", "high", "low", "close"]
    predictor.vol_col = "volume"
    predictor.amt_vol = "amount"
    predictor.clip = 5
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return np.zeros((1, 2, 6), dtype=np.float32)

    predictor.generate = fake_generate
    history = pd.DataFrame(
        {
            "open": [99.0, 100.0, 101.0],
            "high": [100.0, 101.0, 102.0],
            "low": [98.0, 99.0, 100.0],
            "close": [99.5, 100.5, 101.5],
        }
    )
    x_timestamp = pd.Series(pd.date_range("2026-01-01", periods=3, freq="h"))
    y_timestamp = pd.Series(pd.date_range("2026-01-01 03:00", periods=2, freq="h"))

    predictor.predict(
        history,
        x_timestamp,
        y_timestamp,
        pred_len=2,
        sample_count=1,
        verbose=False,
        deterministic=True,
    )

    assert calls[0]["deterministic"] is True
