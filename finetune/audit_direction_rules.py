"""Audit leak-free target and market direction rules across chronological folds."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from finetune.direction_training import (
    split_direction_sample_index,
    summarize_direction_probabilities,
)
from finetune.train_frozen_direction_head import DEFAULT_FOLDS
from finetune.frozen_direction_head import select_non_overlapping_samples


HORIZONS = (1, 3, 5, 10, 20, 40, 60, 90)


def load_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(data_dir.glob("*_qfq.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        frames[path.name.split("_")[0]] = frame.sort_values("date").reset_index(drop=True)
    return frames


def build_market_returns(
    frames: dict[str, pd.DataFrame], horizons: tuple[int, ...]
) -> dict[int, dict[pd.Timestamp, float]]:
    by_horizon: dict[int, dict[pd.Timestamp, float]] = {}
    for horizon in horizons:
        records = []
        for frame in frames.values():
            close = frame["close"].to_numpy(dtype=float)
            values = np.full(len(frame), np.nan)
            values[horizon:] = np.log(close[horizon:] / close[:-horizon])
            records.append(pd.DataFrame({"date": frame["date"], "value": values}))
        combined = pd.concat(records, ignore_index=True).dropna()
        by_horizon[horizon] = combined.groupby("date")["value"].mean().to_dict()
    return by_horizon


def add_rule_features(
    samples: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    market_returns: dict[int, dict[pd.Timestamp, float]],
) -> pd.DataFrame:
    enriched = samples.copy()
    for horizon in HORIZONS:
        stock_values = []
        market_values = []
        for row in enriched.itertuples(index=False):
            close = frames[row.symbol]["close"].to_numpy(dtype=float)
            end = int(row.context_end)
            start = max(0, end - horizon)
            stock_values.append(float(np.log(close[end] / close[start])))
            market_values.append(
                float(market_returns[horizon].get(pd.Timestamp(row.context_end_date), 0.0))
            )
        enriched[f"stock_{horizon}"] = stock_values
        enriched[f"market_{horizon}"] = market_values
    return enriched


def probability_from_score(score: np.ndarray) -> np.ndarray:
    scale = float(np.nanstd(score))
    if scale <= 1e-12:
        return np.full(len(score), 0.5)
    return 1.0 / (1.0 + np.exp(-np.clip(score / scale, -10.0, 10.0)))


def candidate_rules() -> dict[str, tuple[str, int, int | None, float]]:
    rules = {}
    for source, horizon, orientation in product(
        ("stock", "market"), HORIZONS, (1.0, -1.0)
    ):
        rules[f"{source}_{horizon}_{'mom' if orientation > 0 else 'rev'}"] = (
            source,
            horizon,
            None,
            orientation,
        )
    for stock_horizon, market_horizon in product(HORIZONS, HORIZONS):
        rules[f"blend_s{stock_horizon}_m{market_horizon}"] = (
            "blend",
            stock_horizon,
            market_horizon,
            1.0,
        )
        rules[f"residual_s{stock_horizon}_m{market_horizon}"] = (
            "blend",
            stock_horizon,
            market_horizon,
            -1.0,
        )
    return rules


def rule_score(
    frame: pd.DataFrame, specification: tuple[str, int, int | None, float]
) -> np.ndarray:
    source, stock_horizon, market_horizon, orientation = specification
    if source in {"stock", "market"}:
        return orientation * frame[f"{source}_{stock_horizon}"].to_numpy(dtype=float)
    stock = frame[f"stock_{stock_horizon}"].to_numpy(dtype=float)
    market = frame[f"market_{market_horizon}"].to_numpy(dtype=float)
    if orientation > 0:
        return stock + market
    return stock - market


def robust_validation_score(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float, float]:
    overall = summarize_direction_probabilities(probabilities, targets)
    quarter_scores = []
    quarters = validation["context_end_date"].dt.to_period("Q")
    for quarter in quarters.unique():
        selected = (quarters == quarter).to_numpy()
        quarter_summary = summarize_direction_probabilities(
            probabilities[selected], targets[selected]
        )
        quarter_scores.append(float(quarter_summary["balanced_accuracy"]))
    worst_quarter = min(quarter_scores)
    robust_score = (
        0.50 * float(overall["balanced_accuracy"])
        + 0.30 * float(np.mean(quarter_scores))
        + 0.20 * worst_quarter
    )
    return robust_score, float(overall["balanced_accuracy"]), worst_quarter


def walk_forward_probabilities(
    samples: pd.DataFrame,
    test: pd.DataFrame,
    rules: dict[str, tuple[str, int, int | None, float]],
    *,
    validation_days: int = 365,
) -> tuple[np.ndarray, Counter]:
    probabilities = pd.Series(index=test.index, dtype=float)
    selections: Counter = Counter()
    test_months = test["context_end_date"].dt.to_period("M")
    for month in test_months.unique():
        month_mask = test_months == month
        month_test = test[month_mask]
        cutoff = pd.Timestamp(month_test["context_end_date"].min())
        history = samples[
            (samples["label_end_date"] < cutoff)
            & (samples["context_end_date"] >= cutoff - pd.Timedelta(days=validation_days))
        ]
        targets = history["target_up"].to_numpy(dtype=float)
        ranked = []
        for name, specification in rules.items():
            history_probability = probability_from_score(
                rule_score(history, specification)
            )
            robust_score, balanced_accuracy, worst_quarter = robust_validation_score(
                history, history_probability, targets
            )
            ranked.append((robust_score, balanced_accuracy, worst_quarter, name))
        selected_name = max(ranked)[-1]
        selections[selected_name] += len(month_test)
        selected_score = rule_score(month_test, rules[selected_name])
        probabilities.loc[month_test.index] = np.where(selected_score >= 0.0, 0.55, 0.45)
    return probabilities.loc[test.index].to_numpy(), selections


def semiannual_walk_forward(
    samples: pd.DataFrame,
    rules: dict[str, tuple[str, int, int | None, float]],
    *,
    target_symbol: str,
) -> None:
    all_probabilities = []
    all_targets = []
    gated_probabilities = []
    gated_targets = []
    gated_tests = []
    print("\nsemiannual walk-forward")
    for test_start in pd.date_range("2021-07-01", "2026-07-01", freq="6MS"):
        validation_start = test_start - pd.DateOffset(months=6)
        test_end = test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        validation = samples[
            (samples["context_end_date"] >= validation_start)
            & (samples["label_end_date"] < test_start)
        ]
        test = samples[
            (samples["symbol"] == target_symbol)
            & (samples["context_end_date"] >= test_start)
            & (samples["label_end_date"] <= test_end)
        ]
        if validation.empty or test.empty:
            continue
        validation_targets = validation["target_up"].to_numpy(dtype=float)
        ranked = []
        for name, specification in rules.items():
            probabilities = probability_from_score(rule_score(validation, specification))
            robust_score, balanced_accuracy, worst_quarter = robust_validation_score(
                validation, probabilities, validation_targets
            )
            ranked.append((robust_score, balanced_accuracy, worst_quarter, name))
        selected = max(ranked)
        _, validation_balanced, validation_worst_quarter, selected_name = selected
        actionable = validation_balanced >= 0.56 and validation_worst_quarter >= 0.52
        test_score = rule_score(test, rules[selected_name])
        probabilities = np.where(test_score >= 0.0, 0.55, 0.45)
        targets = test["target_up"].to_numpy(dtype=float)
        summary = summarize_direction_probabilities(probabilities, targets)
        all_probabilities.append(probabilities)
        all_targets.append(targets)
        if actionable:
            gated_probabilities.append(probabilities)
            gated_targets.append(targets)
            gated_tests.append(test)
        print(
            f"{test_start:%Y-%m}..{test_end:%Y-%m} {selected_name:24s} "
            f"n={len(test):3d} acc={summary['accuracy']:.3f} "
            f"bal={summary['balanced_accuracy']:.3f} "
            f"val_bal={validation_balanced:.3f} worstQ={validation_worst_quarter:.3f} "
            f"actionable={actionable}"
        )
    aggregate = summarize_direction_probabilities(
        np.concatenate(all_probabilities), np.concatenate(all_targets)
    )
    print(
        f"semiannual aggregate n={aggregate['points']} "
        f"acc={aggregate['accuracy']:.3f} bal={aggregate['balanced_accuracy']:.3f} "
        f"brier={aggregate['brier_score']:.3f} "
        f"wilson={aggregate['accuracy_wilson_95']}"
    )
    if gated_probabilities:
        gated = summarize_direction_probabilities(
            np.concatenate(gated_probabilities), np.concatenate(gated_targets)
        )
        selected_test = pd.concat(gated_tests, ignore_index=True)
        independent_test = select_non_overlapping_samples(selected_test, horizon=10)
        positions = {
            int(embedding_index): position
            for position, embedding_index in enumerate(selected_test["embedding_index"])
        }
        independent_positions = np.asarray(
            [positions[int(value)] for value in independent_test["embedding_index"]]
        )
        flat_probability = np.concatenate(gated_probabilities)
        flat_targets = np.concatenate(gated_targets)
        independent = summarize_direction_probabilities(
            flat_probability[independent_positions], flat_targets[independent_positions]
        )
        print(
            f"gated aggregate n={gated['points']} acc={gated['accuracy']:.3f} "
            f"bal={gated['balanced_accuracy']:.3f} brier={gated['brier_score']:.3f} "
            f"coverage={gated['points'] / aggregate['points']:.3f} "
            f"independent_n={independent['points']} "
            f"independent_acc={independent['accuracy']:.3f} "
            f"independent_wilson={independent['accuracy_wilson_95']}"
        )


def main() -> None:
    data_dir = Path("data/direction_universe")
    samples = pd.read_csv(
        "outputs/direction_head/sample_index.csv",
        parse_dates=["context_end_date", "label_end_date"],
        dtype={"symbol": str},
    )
    frames = load_frames(data_dir)
    print(f"frames={len(frames)} samples={len(samples)}")
    market_returns = build_market_returns(frames, HORIZONS)
    samples = add_rule_features(samples, frames, market_returns)
    rules = candidate_rules()

    for fold in DEFAULT_FOLDS:
        split = split_direction_sample_index(
            samples,
            train_end=fold["train_end"],
            validation_end=fold["validation_end"],
            test_end=fold["test_end"],
            test_symbol="688169",
        )
        validation = split["validation"]
        test = split["test"]
        validation_targets = validation["target_up"].to_numpy(dtype=float)
        test_targets = test["target_up"].to_numpy(dtype=float)
        ranked = []
        for name, specification in rules.items():
            probabilities = probability_from_score(rule_score(validation, specification))
            robust_score, balanced_accuracy, worst_quarter = robust_validation_score(
                validation, probabilities, validation_targets
            )
            ranked.append((robust_score, balanced_accuracy, worst_quarter, name))
        ranked.sort(reverse=True)
        print(
            f"\n{fold['name']} target_up_rate={test_targets.mean():.3f} "
            f"validation={len(validation)} test={len(test)}"
        )
        for robust_score, _, worst_quarter, name in ranked[:5]:
            specification = rules[name]
            validation_summary = summarize_direction_probabilities(
                probability_from_score(rule_score(validation, specification)),
                validation_targets,
            )
            test_summary = summarize_direction_probabilities(
                probability_from_score(rule_score(test, specification)), test_targets
            )
            print(
                f"{name:24s} val={validation_summary['accuracy']:.3f}/"
                f"{validation_summary['balanced_accuracy']:.3f} "
                f"robust={robust_score:.3f} worstQ={worst_quarter:.3f} "
                f"test={test_summary['accuracy']:.3f}/"
                f"{test_summary['balanced_accuracy']:.3f}"
            )
        for validation_days in (180, 365):
            walk_forward_probability, selections = walk_forward_probabilities(
                samples, test, rules, validation_days=validation_days
            )
            walk_forward_summary = summarize_direction_probabilities(
                walk_forward_probability, test_targets
            )
            print(
                f"walk_forward_{validation_days}d       "
                f"test={walk_forward_summary['accuracy']:.3f}/"
                f"{walk_forward_summary['balanced_accuracy']:.3f} "
                f"rules={selections.most_common(4)}"
            )
    semiannual_walk_forward(samples, rules, target_symbol="688169")


if __name__ == "__main__":
    main()
