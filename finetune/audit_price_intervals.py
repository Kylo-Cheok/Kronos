"""Walk-forward audit for empirical short-horizon close intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from webui.interval_calibration import historical_return_bands


def summarize(records: pd.DataFrame, label: str) -> None:
    covered = records["covered"].to_numpy(dtype=bool)
    median_error = records["median_error"].to_numpy(dtype=float)
    persistence_error = records["persistence_error"].to_numpy(dtype=float)
    print(
        f"{label:20s} n={len(records):5d} coverage={covered.mean():.3f} "
        f"mean_width={records['relative_width'].mean():.3f} "
        f"median_MAE={np.mean(np.abs(median_error)):.3f} "
        f"persistence_MAE={np.mean(np.abs(persistence_error)):.3f}"
    )


def run_audit(frame: pd.DataFrame, *, adapt_volatility: bool) -> None:
    close = frame["close"].to_numpy(dtype=float)
    records = []
    context_length = 400
    horizon = 10
    for origin in range(context_length - 1, len(frame) - horizon):
        bands = historical_return_bands(
            close[origin - context_length + 1 : origin + 1],
            max_horizon=horizon,
            confidence_level=0.90,
            lookback=252,
            min_samples=120,
            adapt_volatility=adapt_volatility,
        )
        for step in range(horizon):
            actual = close[origin + step + 1]
            lower = bands["lower"][step]
            upper = bands["upper"][step]
            median = bands["median"][step]
            records.append(
                {
                    "origin": origin,
                    "origin_date": frame.at[origin, "date"],
                    "target_date": frame.at[origin + step + 1, "date"],
                    "horizon": step + 1,
                    "covered": lower <= actual <= upper,
                    "relative_width": (upper - lower) / bands["anchor_close"],
                    "median_error": median - actual,
                    "persistence_error": bands["anchor_close"] - actual,
                }
            )
    result = pd.DataFrame(records)
    print(f"\nvolatility_adaptation={adapt_volatility}")
    summarize(result, "all horizons")
    summarize(result[result["horizon"] == 10], "horizon 10")
    independent = result[
        (result["horizon"] == 10)
        & ((result["origin"] - (context_length - 1)) % horizon == 0)
    ]
    summarize(independent, "independent h10")
    for name, start, end in (
        ("problem 2022H2", "2022-07-01", "2022-12-31"),
        ("mid 2024", "2024-01-01", "2024-12-31"),
        ("recent", "2025-01-01", "2026-12-31"),
    ):
        selected = result[
            (result["target_date"] >= pd.Timestamp(start))
            & (result["target_date"] <= pd.Timestamp(end))
        ]
        summarize(selected, name)
    for step in range(1, horizon + 1):
        selected = result[result["horizon"] == step]
        print(
            f"h={step:2d} coverage={selected['covered'].mean():.3f} "
            f"width={selected['relative_width'].mean():.3f}"
        )


def main() -> None:
    frame = pd.read_csv(
        "data/direction_universe/688169_qfq.csv", parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)
    run_audit(frame, adapt_volatility=False)
    run_audit(frame, adapt_volatility=True)


if __name__ == "__main__":
    main()
