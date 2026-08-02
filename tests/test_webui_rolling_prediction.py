import numpy as np
import pandas as pd

from webui.app import build_future_timestamps, run_rolling_prediction


def _forecast_frame(values):
    return pd.DataFrame({"close": values})


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.calls.append(
            {
                "context_close": df["close"].tolist(),
                "target_timestamps": list(pd.to_datetime(y_timestamp)),
                "pred_len": pred_len,
                "kwargs": kwargs,
            }
        )
        values = [100.0 + len(self.calls) + offset for offset in range(pred_len)]
        frame = _forecast_frame(values)
        return {
            "prediction": frame.copy(),
            "mean": frame.copy(),
            "lower": frame.copy(),
            "median": frame.copy(),
            "upper": frame.copy(),
            "std": _forecast_frame([1.0] * pred_len),
            "up_probability": np.full(pred_len, 0.75),
            "cumulative_up_probability": np.full(pred_len, 0.8),
            "interval_width": np.full(pred_len, 2.0),
            "relative_interval_width": np.full(pred_len, 0.02),
        }


def test_run_rolling_prediction_updates_context_with_observed_chunks():
    predictor = FakePredictor()
    context = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    context_timestamps = pd.date_range("2022-01-01", periods=3, freq="D")
    target = pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0]})
    target_timestamps = pd.date_range("2022-01-04", periods=5, freq="D")

    result = run_rolling_prediction(
        predictor=predictor,
        context_df=context,
        context_timestamps=context_timestamps,
        target_df=target,
        target_timestamps=target_timestamps,
        rolling_horizon=2,
        T=0.8,
        top_p=0.8,
        sample_count=4,
        confidence_level=0.9,
    )

    assert [call["context_close"] for call in predictor.calls] == [
        [1.0, 2.0, 3.0],
        [3.0, 10.0, 11.0],
        [11.0, 12.0, 13.0],
    ]
    assert [call["pred_len"] for call in predictor.calls] == [2, 2, 1]
    assert len(result["prediction"]) == 5
    assert result["prediction"]["close"].tolist() == [101.0, 102.0, 102.0, 103.0, 103.0]
    np.testing.assert_allclose(result["up_probability"], [0.75] * 5)
    np.testing.assert_allclose(result["cumulative_up_probability"], [0.8] * 5)
    np.testing.assert_allclose(
        result["prediction_reference_close"], [3.0, 101.0, 11.0, 102.0, 13.0]
    )
    np.testing.assert_allclose(
        result["actual_reference_close"], [3.0, 10.0, 11.0, 12.0, 13.0]
    )
    assert result["rolling_horizon"] == 2
    assert result["rolling_chunks"] == 3
    assert [
        (signal["target_start_index"], signal["target_end_index"])
        for signal in result["direction_signals"]
    ] == [(0, 2), (2, 4), (4, 5)]
    assert all(
        signal["reason"] == "insufficient_context_for_20_step_baseline"
        for signal in result["direction_signals"]
    )


def test_rolling_prediction_applies_empirical_guardrail_with_sufficient_context():
    class ExtremePredictor(FakePredictor):
        def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
            result = super().predict(
                df, x_timestamp, y_timestamp, pred_len, **kwargs
            )
            for key in ("prediction", "mean", "lower", "median", "upper"):
                result[key]["close"] = 1_000.0
            return result

    predictor = ExtremePredictor()
    context = pd.DataFrame(
        {"close": 100.0 * np.exp(np.sin(np.arange(120) / 8.0) * 0.03)}
    )
    context_timestamps = pd.bdate_range("2023-01-02", periods=120)
    target = pd.DataFrame({"close": [101.0, 102.0]})
    target_timestamps = pd.bdate_range(context_timestamps[-1], periods=3)[1:]

    result = run_rolling_prediction(
        predictor=predictor,
        context_df=context,
        context_timestamps=context_timestamps,
        target_df=target,
        target_timestamps=target_timestamps,
        rolling_horizon=2,
    )

    assert result["calibration"]["method"] == "historical_log_return_quantiles"
    assert result["calibration"]["guardrail_count"] == 2
    assert (result["median"]["close"] < 200.0).all()


def test_build_future_timestamps_starts_after_latest_observation():
    observed = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-05"])

    future = build_future_timestamps(observed, pred_len=2)

    assert list(future) == list(pd.to_datetime(["2022-01-07", "2022-01-09"]))
