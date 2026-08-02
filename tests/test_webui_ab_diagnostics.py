import numpy as np
import pandas as pd
import pytest

import webui.app as app_module


class DirectionFakePredictor:
    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        close = np.array([101.0, 99.0], dtype=float)[:pred_len]
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(pred_len),
            }
        )
        return {
            "prediction": frame.copy(),
            "mean": frame.copy(),
            "lower": frame.copy(),
            "median": frame.copy(),
            "upper": frame.copy(),
            "std": frame.copy(),
            "up_probability": np.array([0.9, 0.1])[:pred_len],
            "cumulative_up_probability": np.array([0.9, 0.1])[:pred_len],
            "interval_width": np.ones(pred_len),
            "relative_interval_width": np.full(pred_len, 0.01),
        }


def test_ab_diagnostics_include_leak_free_direction_metrics(monkeypatch):
    data = pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open": [99.0, 100.0, 101.0, 97.0],
            "high": [100.0, 101.0, 103.0, 99.0],
            "low": [98.0, 99.0, 100.0, 96.0],
            "close": [99.0, 100.0, 102.0, 98.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(app_module, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(app_module, "predictor", DirectionFakePredictor())
    monkeypatch.setattr(app_module, "load_data_file", lambda _: (data.copy(), None))

    response = app_module.app.test_client().post(
        "/api/diagnostics/ab",
        json={
            "file_path": "unused.csv",
            "lookback": 2,
            "pred_len": 2,
            "start_date": "2024-01-01",
            "mode": "backtest",
            "rolling_horizon": 0,
            "case_ids": ["conservative"],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert [case["id"] for case in payload["cases"]] == ["conservative"]
    direction = payload["cases"][0]["diagnostics"]["direction"]
    assert direction["accuracy"] == 1.0
    assert direction["balanced_accuracy"] == 1.0
    assert direction["probability"]["brier_score"] == pytest.approx(0.01)
    chunk_direction = payload["cases"][0]["diagnostics"]["chunk_direction"]
    assert chunk_direction["points"] == 1
    assert chunk_direction["model"]["accuracy"] == 1.0
    assert chunk_direction["filtered"]["coverage"] == 0.0


def test_predict_exposes_short_cycle_direction_signal_and_backtest_evaluation(
    monkeypatch,
):
    data = pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open": [99.0, 100.0, 101.0, 97.0],
            "high": [100.0, 101.0, 103.0, 99.0],
            "low": [98.0, 99.0, 100.0, 96.0],
            "close": [99.0, 100.0, 102.0, 98.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(app_module, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(app_module, "predictor", DirectionFakePredictor())
    monkeypatch.setattr(app_module, "load_data_file", lambda _: (data.copy(), None))
    monkeypatch.setattr(app_module, "save_prediction_results", lambda **_: None)
    monkeypatch.setattr(
        app_module,
        "get_verified_direction",
        lambda origin_date, file_path: {
            "status": "abstain",
            "direction": None,
            "reason": "direction_model_outside_validity_period",
            "horizon": 1,
        },
    )

    response = app_module.app.test_client().post(
        "/api/predict",
        json={
            "file_path": "unused.csv",
            "lookback": 2,
            "pred_len": 2,
            "start_date": "2024-01-01",
            "mode": "backtest",
        },
    )

    assert response.status_code == 200
    summary = response.get_json()["direction_summary"]
    assert summary["policy"] == "verified_exogenous_one_day"
    assert summary["signals"][0]["horizon"] == 2
    assert summary["signals"][0]["probability_is_calibrated"] is False
    assert summary["evaluation"]["model"]["accuracy"] == 1.0
    assert summary["verified_signal"]["horizon"] == 1
    assert summary["verified_signal"]["status"] == "abstain"


def test_predict_prioritizes_independently_verified_one_day_signal(monkeypatch):
    data = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-07-28", periods=4, freq="D"),
            "open": [99.0, 100.0, 101.0, 100.0],
            "high": [100.0, 101.0, 102.0, 101.0],
            "low": [98.0, 99.0, 100.0, 99.0],
            "close": [99.0, 100.0, 101.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(app_module, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(app_module, "predictor", DirectionFakePredictor())
    monkeypatch.setattr(app_module, "load_data_file", lambda _: (data.copy(), None))
    monkeypatch.setattr(app_module, "save_prediction_results", lambda **_: None)
    monkeypatch.setattr(
        app_module,
        "get_verified_direction",
        lambda origin_date, file_path: {
            "status": "candidate",
            "direction": "down",
            "candidate_direction": "down",
            "reason": "verified_one_day_edge_low_confidence",
            "horizon": 1,
            "confidence_label": "low",
            "historical_accuracy": 0.5383,
            "historical_wilson_95": [0.5102, 0.5662],
            "raw_model_up_probability": 0.4999,
            "probability_is_calibrated": False,
        },
    )

    response = app_module.app.test_client().post(
        "/api/predict",
        json={
            "file_path": "688169_daily.csv",
            "lookback": 2,
            "pred_len": 2,
            "mode": "future",
        },
    )

    assert response.status_code == 200
    summary = response.get_json()["direction_summary"]
    assert summary["policy"] == "verified_exogenous_one_day"
    assert summary["verified_signal"]["direction"] == "down"
    assert summary["verified_signal"]["confidence_label"] == "low"
