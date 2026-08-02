import numpy as np
import pandas as pd
import pytest

from webui.data_quality import adjust_corporate_action_gaps


def _split_frame():
    before = pd.DataFrame(
        {
            "timestamps": pd.date_range("2022-01-01", periods=25, freq="D"),
            "open": [140.0] * 25,
            "high": [142.0] * 25,
            "low": [138.0] * 25,
            "close": [140.0] * 25,
            "volume": [1000.0] * 25,
            "amount": [140000.0] * 25,
        }
    )
    after = pd.DataFrame(
        {
            "timestamps": pd.date_range("2022-01-26", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1400.0, 1400.0, 1400.0],
            "amount": [141400.0, 142800.0, 144200.0],
        }
    )
    return pd.concat([before, after], ignore_index=True)


def test_adjust_corporate_action_gaps_back_adjusts_price_and_volume():
    raw = _split_frame()

    adjusted, report = adjust_corporate_action_gaps(raw)

    assert report["event_count"] == 1
    event = report["events"][0]
    assert event["row_index"] == 25
    assert event["observed_open_ratio"] == pytest.approx(100.0 / 140.0)
    assert event["applied_factor"] == pytest.approx(10.0 / 14.0)
    assert adjusted.loc[24, "close"] == pytest.approx(100.0)
    assert adjusted.loc[25, "close"] == 101.0
    assert adjusted.loc[24, "volume"] == pytest.approx(1400.0)
    assert adjusted.loc[24, "amount"] == raw.loc[24, "amount"]
    assert raw.loc[24, "close"] == 140.0


def test_adjust_corporate_action_gaps_does_not_rewrite_normal_limit_move():
    raw = _split_frame().iloc[:26].copy()
    raw.loc[:24, ["open", "close"]] = 100.0
    raw.loc[:24, "high"] = 102.0
    raw.loc[:24, "low"] = 98.0
    raw.loc[25, ["open", "high", "low", "close"]] = [80.0, 82.0, 79.0, 81.0]

    adjusted, report = adjust_corporate_action_gaps(raw)

    assert report["event_count"] == 0
    pd.testing.assert_frame_equal(adjusted, raw)
