import numpy as np
import pandas as pd
import pytest

from webui.diagnostics import (
    build_direction_signal,
    build_ab_cases,
    evaluate_direction_signals,
    summarize_direction_metrics,
    summarize_forecast_diagnostics,
)


def _frame():
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 104.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 103.0],
        }
    )


def test_summarize_forecast_diagnostics_reports_invalid_ohlc_and_jumps():
    prediction = _frame()
    prediction.loc[1, "close"] = 130.0
    prediction.loc[1, "low"] = 131.0
    actual = _frame()

    result = summarize_forecast_diagnostics(
        prediction,
        actual,
        horizons=(1, 2, 3),
        prediction_reference_close=[99.0, 100.0, 130.0],
        actual_reference_close=[99.0, 100.0, 102.0],
        up_probability=[0.8, 0.8, 0.2],
    )

    assert result["close_jumps_gt_20pct"]["count"] == 2
    assert result["ohlc_violations"]["low_above_open_or_close"] == 1
    assert result["ohlc_violations"]["total"] == 1
    assert result["metrics"]["points"] == 3
    assert result["metrics_by_horizon"]["1"]["mae"] == 0.0
    assert result["metrics_by_horizon"]["2"]["points"] == 2
    assert result["direction"]["points"] == 3
    assert result["direction"]["probability"]["brier_score"] is not None


def test_build_ab_cases_keeps_current_parameters_and_adds_controls():
    cases = build_ab_cases(1.0, 0.9, 16)

    assert [case["id"] for case in cases] == [
        "current",
        "conservative",
        "deterministic",
    ]
    assert cases[0]["sample_count"] == 16
    assert cases[1]["temperature"] == 0.6
    assert cases[2]["deterministic"] is True


def test_direction_metrics_use_per_forecast_reference_closes():
    result = summarize_direction_metrics(
        predicted_close=[101.0, 99.0, 102.0, 97.0],
        actual_close=[102.0, 98.0, 101.0, 99.0],
        reference_close=[100.0, 102.0, 98.0, 101.0],
    )

    assert result["points"] == 4
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["predicted_up_rate"] == 0.5
    assert result["actual_up_rate"] == 0.5
    assert result["confusion_matrix"] == {
        "true_up_pred_up": 2,
        "true_up_pred_down": 0,
        "true_down_pred_up": 0,
        "true_down_pred_down": 2,
    }
    assert len(result["records"]) == 4
    assert result["records"][0]["predicted_up"] is True
    assert result["records"][0]["actual_up"] is True
    assert result["records"][0]["correct"] is True


def test_direction_metrics_report_probability_calibration_and_selective_accuracy():
    result = summarize_direction_metrics(
        predicted_close=[101.0, 101.0, 99.0, 99.0],
        actual_close=[102.0, 98.0, 98.0, 102.0],
        reference_close=[100.0, 100.0, 100.0, 100.0],
        up_probability=[0.9, 0.6, 0.1, 0.4],
        confidence_thresholds=(0.0, 0.5),
    )

    probability = result["probability"]
    assert probability["accuracy"] == 0.5
    assert probability["brier_score"] == pytest.approx(0.185)
    assert probability["selective_accuracy"]["0.0"]["coverage"] == 1.0
    assert probability["selective_accuracy"]["0.5"] == {
        "points": 2,
        "coverage": 0.5,
        "accuracy": 1.0,
    }
    assert result["records"][0]["up_probability"] == 0.9
    assert result["records"][0]["confidence"] == pytest.approx(0.8)


def test_direction_signal_requires_model_and_mean_reversion_votes_to_agree():
    context_close = np.arange(100.0, 121.0)

    agreed = build_direction_signal(
        context_close=context_close,
        predicted_close=[118.0, 115.0],
        cumulative_up_probability=[0.2, 0.1],
    )
    disagreed = build_direction_signal(
        context_close=context_close,
        predicted_close=[123.0, 130.0],
        cumulative_up_probability=[0.8, 0.9],
    )

    assert agreed["model_direction"] == "down"
    assert agreed["votes"] == {
        "model": "down",
        "mean_reversion_5": "down",
        "mean_reversion_20": "down",
    }
    assert agreed["status"] == "candidate"
    assert agreed["candidate_direction"] == "down"
    assert agreed["direction"] is None
    assert agreed["reason"] == "three_way_direction_agreement_uncalibrated"
    assert agreed["raw_model_up_probability"] == pytest.approx(0.1)
    assert agreed["probability_is_calibrated"] is False

    assert disagreed["status"] == "abstain"
    assert disagreed["direction"] is None
    assert disagreed["reason"] == "direction_votes_disagree"


def test_direction_signal_abstains_when_context_is_too_short():
    signal = build_direction_signal(
        context_close=[100.0, 101.0, 102.0],
        predicted_close=[103.0, 104.0],
    )

    assert signal["status"] == "abstain"
    assert signal["reason"] == "insufficient_context_for_20_step_baseline"
    assert signal["direction"] is None


def test_direction_signal_evaluation_reports_filtered_coverage_and_wilson_interval():
    signals = [
        {
            "target_start_index": 0,
            "target_end_index": 2,
            "reference_close": 100.0,
            "model_direction": "up",
            "status": "actionable",
            "direction": "up",
            "raw_model_up_probability": 0.85,
        },
        {
            "target_start_index": 2,
            "target_end_index": 4,
            "reference_close": 103.0,
            "model_direction": "down",
            "status": "abstain",
            "direction": None,
            "raw_model_up_probability": 0.15,
        },
    ]

    result = evaluate_direction_signals(
        signals,
        actual_close=[101.0, 103.0, 102.0, 100.0],
    )

    assert result["points"] == 2
    assert result["model"]["accuracy"] == 1.0
    assert result["candidate"]["points"] == 1
    assert result["candidate"]["coverage"] == 0.5
    assert result["filtered"]["points"] == 1
    assert result["filtered"]["coverage"] == 0.5
    assert result["filtered"]["accuracy"] == 1.0
    assert result["filtered"]["wilson_95"][0] < 1.0
    assert result["records"][1]["selected"] is False
    assert result["raw_probability"]["accuracy"] == 1.0
    assert result["raw_probability"]["brier_score"] == pytest.approx(0.0225)
    assert result["raw_probability"]["is_calibrated"] is False
    assert result["raw_probability"]["reliability_bins"] == [
        {
            "lower": 0.0,
            "upper": 0.2,
            "points": 1,
            "mean_probability": 0.15,
            "observed_up_rate": 0.0,
        },
        {
            "lower": 0.8,
            "upper": 1.0,
            "points": 1,
            "mean_probability": 0.85,
            "observed_up_rate": 1.0,
        },
    ]
