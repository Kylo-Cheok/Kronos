import numpy as np
import pandas as pd

from webui.interval_calibration import (
    apply_return_band_guardrail,
    historical_return_bands,
)


def _frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(close), 100.0),
            "amount": close * 100.0,
        }
    )


def test_historical_return_bands_are_ordered_and_anchor_to_last_close():
    close = 100.0 * np.exp(np.linspace(0.0, 0.4, 120))
    bands = historical_return_bands(
        close, max_horizon=5, confidence_level=0.9, lookback=100, min_samples=30
    )

    assert bands["lower"].shape == (5,)
    assert np.all(bands["lower"] <= bands["median"])
    assert np.all(bands["median"] <= bands["upper"])
    assert np.all(np.asarray(bands["sample_count"]) >= 30)
    assert bands["anchor_close"] == close[-1]


def test_guardrail_shrinks_extreme_model_closes_and_preserves_candle_ratios():
    context_close = 100.0 * np.exp(np.sin(np.arange(180) / 7.0) * 0.02)
    extreme = _frame([300.0, 400.0, 500.0])
    forecast = {
        "prediction": extreme.copy(),
        "median": extreme.copy(),
        "mean": extreme.copy(),
        "lower": extreme.copy(),
        "upper": extreme.copy(),
        "interval_width": np.ones(3),
        "relative_interval_width": np.ones(3),
    }

    guarded = apply_return_band_guardrail(
        forecast,
        context_close,
        confidence_level=0.9,
        lookback=120,
        min_samples=30,
    )

    assert guarded["calibration"]["method"] == "historical_log_return_quantiles"
    assert guarded["calibration"]["guardrail_count"] == 3
    assert guarded["calibration"]["model_point_weight"] == 0.25
    assert guarded["calibration"]["raw_model_median_close"] == [300.0, 400.0, 500.0]
    assert np.all(
        guarded["median"]["close"].to_numpy()
        <= np.asarray(guarded["calibration"]["upper_close"])
    )
    assert np.all(guarded["median"]["close"].to_numpy() < 200.0)
    np.testing.assert_allclose(
        guarded["median"]["high"] / guarded["median"]["close"],
        extreme["high"] / extreme["close"],
    )
    np.testing.assert_allclose(
        guarded["interval_width"],
        np.asarray(guarded["calibration"]["upper_close"])
        - np.asarray(guarded["calibration"]["lower_close"]),
    )


def test_guardrail_does_not_mutate_input_forecast():
    context_close = np.linspace(90.0, 110.0, 150)
    median = _frame([100.0, 101.0])
    forecast = {
        "prediction": median.copy(),
        "median": median.copy(),
        "mean": median.copy(),
        "lower": median.copy(),
        "upper": median.copy(),
        "interval_width": np.ones(2),
        "relative_interval_width": np.ones(2),
    }
    original = forecast["median"].copy(deep=True)

    apply_return_band_guardrail(forecast, context_close, min_samples=20)

    pd.testing.assert_frame_equal(forecast["median"], original)


def test_volatility_adaptation_widens_bands_after_recent_volatility_jump():
    quiet = np.full(220, 0.001)
    volatile = np.tile([0.04, -0.035], 15)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([quiet, volatile])))

    static = historical_return_bands(
        close,
        max_horizon=10,
        lookback=220,
        min_samples=100,
        adapt_volatility=False,
    )
    adaptive = historical_return_bands(
        close,
        max_horizon=10,
        lookback=220,
        min_samples=100,
        adapt_volatility=True,
    )

    assert adaptive["volatility_scale"] > 1.0
    assert adaptive["upper"][-1] - adaptive["lower"][-1] > (
        static["upper"][-1] - static["lower"][-1]
    )
