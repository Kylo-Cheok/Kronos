import numpy as np
import pandas as pd
import pytest

from finetune.exogenous_direction import (
    align_exogenous_features,
    attach_return_decomposition_targets,
    combine_return_predictions,
    estimate_market_betas,
    build_live_origin_samples,
    sanitize_exogenous_frame,
    retarget_sample_horizon,
)


def _market_frame(start="2024-01-01", rows=120):
    dates = pd.bdate_range(start, periods=rows)
    close = 100.0 * np.exp(np.arange(rows) * 0.002)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "amount": np.linspace(1_000.0, 2_000.0, rows),
        }
    )


def test_exogenous_alignment_uses_latest_observation_at_or_before_origin():
    market = _market_frame()
    samples = pd.DataFrame(
        {
            "context_end_date": [market.loc[50, "date"], market.loc[80, "date"]],
        }
    )

    aligned = align_exogenous_features(
        samples,
        {"star50": market},
        horizons=(1, 5, 20),
    )

    assert aligned["star50_staleness_days"].tolist() == [0, 0]
    assert aligned.loc[0, "star50_return_5"] == pytest.approx(np.log(
        market.loc[50, "close"] / market.loc[45, "close"]
    ))
    assert np.isfinite(aligned.loc[0, "star50_volatility_20"])
    assert np.isfinite(aligned.loc[0, "star50_range_20"])
    assert np.isfinite(aligned.loc[0, "star50_amount_z_20"])

    changed = market.copy()
    changed.loc[51:, ["open", "high", "low", "close"]] *= 1000.0
    changed_alignment = align_exogenous_features(
        samples.iloc[[0]],
        {"star50": changed},
        horizons=(1, 5, 20),
    )
    pd.testing.assert_series_equal(
        aligned.iloc[0].drop(labels=["context_end_date"]),
        changed_alignment.iloc[0].drop(labels=["context_end_date"]),
        check_names=False,
    )


def test_exogenous_alignment_marks_stale_series_as_missing():
    market = _market_frame(rows=30)
    samples = pd.DataFrame(
        {"context_end_date": [market["date"].iloc[-1] + pd.Timedelta(days=10)]}
    )

    aligned = align_exogenous_features(
        samples,
        {"star50": market},
        horizons=(5,),
        max_staleness_days=5,
    )

    assert aligned.loc[0, "star50_staleness_days"] == 10
    assert np.isnan(aligned.loc[0, "star50_return_5"])


def test_return_targets_split_stock_move_into_market_and_residual():
    stock = _market_frame()
    market = _market_frame()
    stock["close"] *= np.exp(np.arange(len(stock)) * 0.001)
    samples = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "context_end": [40],
            "label_index": [50],
            "context_end_date": [stock.loc[40, "date"]],
            "label_end_date": [stock.loc[50, "date"]],
        }
    )

    targets = attach_return_decomposition_targets(
        samples,
        {"AAA": stock},
        market,
        market_beta=1.0,
    )

    expected_stock = np.log(stock.loc[50, "close"] / stock.loc[40, "close"])
    expected_market = np.log(market.loc[50, "close"] / market.loc[40, "close"])
    assert targets.loc[0, "stock_future_return"] == expected_stock
    assert targets.loc[0, "market_future_return"] == expected_market
    assert targets.loc[0, "residual_future_return"] == expected_stock - expected_market
    assert bool(targets.loc[0, "target_up"]) is True


def test_combined_return_is_market_beta_plus_residual():
    combined = combine_return_predictions(
        market_return=[0.02, -0.01],
        residual_return=[-0.005, 0.02],
        market_beta=[1.0, 0.5],
    )
    np.testing.assert_allclose(combined, [0.015, 0.015])


def test_sanitize_exogenous_frame_handles_index_data_without_volume():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "open": [101, 100, 100],
            "close": [102, 101, 101],
            "high": [103, 102, 102],
            "low": [100, 99, 99],
            "amount": [1200, 1000, 1000],
        }
    )

    clean = sanitize_exogenous_frame(raw, "star50")

    assert clean.columns.tolist() == ["date", "open", "high", "low", "close", "amount"]
    assert clean["date"].tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))


def test_market_beta_is_estimated_per_symbol_from_training_returns_only():
    market = np.linspace(-0.05, 0.05, 80)
    training = pd.DataFrame(
        {
            "symbol": ["AAA"] * 80 + ["BBB"] * 80,
            "market_future_return": np.concatenate([market, market]),
            "stock_future_return": np.concatenate([2.0 * market, 0.5 * market]),
        }
    )

    betas = estimate_market_betas(training, min_points=40)

    assert betas["AAA"] == pytest.approx(2.0)
    assert betas["BBB"] == pytest.approx(0.5)


def test_sample_index_can_be_retargeted_to_a_shorter_complete_horizon():
    frame = _market_frame(rows=8)
    samples = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "context_end": [3, 6],
            "label_index": [5, 7],
            "label_end_date": [frame.loc[5, "date"], frame.loc[7, "date"]],
        }
    )

    retargeted = retarget_sample_horizon(samples, {"AAA": frame}, horizon=1)

    assert retargeted["context_end"].tolist() == [3, 6]
    assert retargeted["label_index"].tolist() == [4, 7]
    assert retargeted["label_end_date"].tolist() == [
        frame.loc[4, "date"],
        frame.loc[7, "date"],
    ]


def test_live_origins_use_latest_rows_at_or_before_as_of_without_future_labels():
    first = _market_frame(rows=30)
    second = _market_frame(start="2024-01-03", rows=28)
    as_of = first.loc[25, "date"]

    origins = build_live_origin_samples(
        {"AAA": first, "BBB": second},
        lookback=20,
        as_of=as_of,
    )

    assert origins.columns.tolist() == [
        "symbol",
        "context_start",
        "context_end",
        "context_end_date",
    ]
    assert origins["symbol"].tolist() == ["AAA", "BBB"]
    assert origins["context_end_date"].max() <= as_of
    assert (origins["context_end"] - origins["context_start"] + 1).tolist() == [20, 20]
    assert "label_index" not in origins
