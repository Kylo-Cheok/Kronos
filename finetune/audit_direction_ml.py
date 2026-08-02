"""Walk-forward audit of market-aware direction classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finetune.audit_direction_rules import (
    HORIZONS,
    add_rule_features,
    build_market_returns,
    load_frames,
)
from finetune.direction_training import summarize_direction_probabilities
from finetune.frozen_direction_head import select_non_overlapping_samples


def build_features(samples: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    columns = []
    for horizon in HORIZONS:
        stock_column = f"stock_{horizon}"
        market_column = f"market_{horizon}"
        samples[f"relative_{horizon}"] = samples[stock_column] - samples[market_column]
        samples[f"rank_{horizon}"] = (
            samples.groupby("context_end_date")[stock_column].rank(pct=True) - 0.5
        )
        columns.extend(
            [stock_column, market_column, f"relative_{horizon}", f"rank_{horizon}"]
        )
    engineered = samples[columns].to_numpy(dtype=np.float32)
    # The last 15 frozen-representation columns are deterministic technical
    # snapshots; excluding the 1664 hidden dimensions sharply reduces overfit.
    technical = embeddings[:, -15:].astype(np.float32)
    return np.nan_to_num(np.concatenate([technical, engineered], axis=1))


def candidate_models(seed: int):
    for c in (0.003, 0.01, 0.03, 0.1):
        yield (
            f"logistic_c{c}",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        )
    for depth in (2, 3):
        yield (
            f"hist_depth{depth}",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=120,
                max_depth=depth,
                min_samples_leaf=50,
                l2_regularization=2.0,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    for depth in (3, 5):
        yield (
            f"rf_depth{depth}",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=50,
                max_features=0.7,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        )


def robust_score(
    validation: pd.DataFrame, probabilities: np.ndarray, targets: np.ndarray
) -> tuple[float, dict]:
    overall = summarize_direction_probabilities(probabilities, targets)
    quarters = validation["context_end_date"].dt.to_period("Q")
    quarter_balanced = []
    for quarter in quarters.unique():
        selected = (quarters == quarter).to_numpy()
        summary = summarize_direction_probabilities(
            probabilities[selected], targets[selected]
        )
        quarter_balanced.append(summary["balanced_accuracy"])
    score = (
        0.6 * overall["balanced_accuracy"]
        + 0.2 * np.mean(quarter_balanced)
        + 0.2 * min(quarter_balanced)
        - 0.1 * overall["brier_score"]
    )
    return float(score), overall


def main() -> None:
    samples = pd.read_csv(
        "outputs/direction_head/sample_index.csv",
        parse_dates=["context_end_date", "label_end_date"],
        dtype={"symbol": str},
    )
    embeddings = np.load("outputs/direction_head/frozen_embeddings.npy", mmap_mode="r")
    frames = load_frames(Path("data/direction_universe"))
    samples = add_rule_features(samples, frames, build_market_returns(frames, HORIZONS))
    features = build_features(samples, embeddings)
    targets = samples["target_up"].to_numpy(dtype=int)
    all_probabilities = []
    ensemble_probabilities = []
    all_targets = []
    all_tests = []
    print(f"samples={len(samples)} features={features.shape[1]}")

    for test_start in pd.date_range("2021-07-01", "2026-07-01", freq="6MS"):
        validation_start = test_start - pd.DateOffset(months=6)
        train_start = validation_start - pd.DateOffset(years=2)
        test_end = test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        train_mask = (
            (samples["context_end_date"] >= train_start)
            & (samples["label_end_date"] < validation_start)
            & (samples["context_end"] % 3 == 0)
        )
        validation_mask = (
            (samples["context_end_date"] >= validation_start)
            & (samples["label_end_date"] < test_start)
            & (samples["context_end"] % 3 == 0)
        )
        test_mask = (
            (samples["symbol"] == "688169")
            & (samples["context_end_date"] >= test_start)
            & (samples["label_end_date"] <= test_end)
        )
        train_indexes = np.flatnonzero(train_mask)
        validation_indexes = np.flatnonzero(validation_mask)
        test_indexes = np.flatnonzero(test_mask)
        if min(len(train_indexes), len(validation_indexes), len(test_indexes)) == 0:
            continue
        ranked = []
        for name, model in candidate_models(seed=100):
            model.fit(features[train_indexes], targets[train_indexes])
            validation_probability = model.predict_proba(features[validation_indexes])[:, 1]
            global_score, validation_summary = robust_score(
                samples.iloc[validation_indexes],
                validation_probability,
                targets[validation_indexes],
            )
            target_validation = (
                samples.iloc[validation_indexes]["symbol"].to_numpy() == "688169"
            )
            target_summary = summarize_direction_probabilities(
                validation_probability[target_validation],
                targets[validation_indexes][target_validation],
            )
            score = 0.75 * global_score + 0.25 * target_summary["balanced_accuracy"]
            ranked.append(
                (
                    score,
                    name,
                    model,
                    validation_summary,
                    validation_probability,
                    target_summary,
                )
            )
        score, name, selected_model, validation_summary, _, target_validation_summary = max(
            ranked, key=lambda result: result[0]
        )
        test_probability = selected_model.predict_proba(features[test_indexes])[:, 1]
        top_candidates = sorted(ranked, key=lambda result: result[0], reverse=True)[:3]
        ensemble_probability = np.mean(
            [
                candidate[2].predict_proba(features[test_indexes])[:, 1]
                for candidate in top_candidates
            ],
            axis=0,
        )
        test_targets = targets[test_indexes]
        test_summary = summarize_direction_probabilities(test_probability, test_targets)
        ensemble_summary = summarize_direction_probabilities(
            ensemble_probability, test_targets
        )
        all_probabilities.append(test_probability)
        ensemble_probabilities.append(ensemble_probability)
        all_targets.append(test_targets)
        all_tests.append(samples.iloc[test_indexes])
        print(
            f"{test_start:%Y-%m}..{test_end:%Y-%m} {name:16s} "
            f"train={len(train_indexes):5d} val={len(validation_indexes):4d} "
            f"val={validation_summary['balanced_accuracy']:.3f}/"
            f"{target_validation_summary['balanced_accuracy']:.3f} "
            f"test={test_summary['accuracy']:.3f}/"
            f"{test_summary['balanced_accuracy']:.3f} "
            f"ens={ensemble_summary['accuracy']:.3f}/"
            f"{ensemble_summary['balanced_accuracy']:.3f} "
            f"top={[candidate[1] for candidate in top_candidates]}"
        )

    aggregate = summarize_direction_probabilities(
        np.concatenate(all_probabilities), np.concatenate(all_targets)
    )
    print(
        f"aggregate n={aggregate['points']} acc={aggregate['accuracy']:.3f} "
        f"bal={aggregate['balanced_accuracy']:.3f} "
        f"brier={aggregate['brier_score']:.3f} "
        f"wilson={aggregate['accuracy_wilson_95']}"
    )
    ensemble = summarize_direction_probabilities(
        np.concatenate(ensemble_probabilities), np.concatenate(all_targets)
    )
    test_frame = pd.concat(all_tests, ignore_index=True)
    independent = select_non_overlapping_samples(test_frame, horizon=10)
    positions = {
        int(embedding_index): position
        for position, embedding_index in enumerate(test_frame["embedding_index"])
    }
    independent_positions = np.asarray(
        [positions[int(value)] for value in independent["embedding_index"]]
    )
    flat_ensemble_probability = np.concatenate(ensemble_probabilities)
    flat_targets = np.concatenate(all_targets)
    independent_summary = summarize_direction_probabilities(
        flat_ensemble_probability[independent_positions],
        flat_targets[independent_positions],
    )
    print(
        f"ensemble n={ensemble['points']} acc={ensemble['accuracy']:.3f} "
        f"bal={ensemble['balanced_accuracy']:.3f} "
        f"brier={ensemble['brier_score']:.3f} "
        f"wilson={ensemble['accuracy_wilson_95']} "
        f"independent_n={independent_summary['points']} "
        f"independent_acc={independent_summary['accuracy']:.3f} "
        f"independent_wilson={independent_summary['accuracy_wilson_95']}"
    )


if __name__ == "__main__":
    main()
