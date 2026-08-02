import json

import pandas as pd

from webui.app import create_prediction_chart, resolve_prediction_timestamps


def test_resolve_prediction_timestamps_prefers_target_timestamps():
    target_timestamps = pd.Series(
        pd.to_datetime(["2022-06-07", "2022-06-08", "2022-06-09"])
    )

    resolved = resolve_prediction_timestamps(
        target_timestamps,
        pred_len=3,
        fallback_start=pd.Timestamp("2022-06-10"),
        fallback_freq=pd.Timedelta(days=1),
    )

    pd.testing.assert_index_equal(
        resolved,
        pd.DatetimeIndex(target_timestamps),
    )


def test_prediction_chart_uses_target_timestamps_for_both_series():
    historical_timestamps = pd.to_datetime(["2022-06-01", "2022-06-02"])
    target_timestamps = pd.to_datetime(["2022-06-07", "2022-06-08"])
    df = pd.DataFrame(
        {
            "timestamps": historical_timestamps,
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.0, 9.5],
            "close": [10.5, 11.0],
        }
    )
    pred_df = pd.DataFrame(
        {
            "open": [11.0, 11.2],
            "high": [11.5, 11.7],
            "low": [10.5, 10.7],
            "close": [11.2, 11.4],
        }
    )
    actual_df = pd.DataFrame(
        {
            "timestamps": target_timestamps,
            "open": [10.8, 11.0],
            "high": [11.3, 11.5],
            "low": [10.4, 10.6],
            "close": [11.0, 11.2],
        }
    )

    chart = json.loads(
        create_prediction_chart(
            df,
            pred_df,
            lookback=2,
            pred_len=2,
            actual_df=actual_df,
            historical_start_idx=0,
            prediction_timestamps=target_timestamps,
        )
    )

    traces = {trace["name"]: trace for trace in chart["data"]}
    assert traces["Prediction Data (120 data points)"]["x"] == [
        timestamp.isoformat()
        for timestamp in target_timestamps
    ]
    assert traces["Actual Data (120 data points)"]["x"] == traces[
        "Prediction Data (120 data points)"
    ]["x"]
