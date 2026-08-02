"""Strict walk-forward audit of exogenous market/residual direction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finetune.audit_direction_ml import robust_score
from finetune.audit_direction_rules import (
    HORIZONS,
    add_rule_features,
    build_market_returns,
    load_frames,
)
from finetune.direction_training import (
    apply_selective_direction_policy,
    fit_selective_direction_policy,
    summarize_direction_probabilities,
    summarize_non_overlapping_phases,
)
from finetune.exogenous_direction import (
    align_exogenous_features,
    attach_return_decomposition_targets,
    combine_return_predictions,
    estimate_market_betas,
    retarget_sample_horizon,
    sanitize_exogenous_frame,
)
from finetune.frozen_direction_head import select_non_overlapping_samples


TARGET_SYMBOL = "688169"
EXOGENOUS_HORIZONS = (1, 3, 5, 10, 20, 60)
SECTOR_NAMES = ("ecovacs", "midea", "gree", "supor", "flyco", "bear", "haier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=10, choices=(1, 3, 5, 10))
    parser.add_argument(
        "--exclude-embeddings",
        action="store_true",
        help="Use only features that are available for a new, unlabeled origin.",
    )
    parser.add_argument(
        "--save-deployment-artifact",
        type=Path,
        default=None,
        help="Save the latest strictly walk-forward one-day ensemble.",
    )
    return parser.parse_args()


def load_exogenous_frames(path: Path) -> dict[str, pd.DataFrame]:
    return {
        file.stem: sanitize_exogenous_frame(pd.read_csv(file), file.stem)
        for file in sorted(path.glob("*.csv"))
    }


def build_feature_matrix(
    samples: pd.DataFrame,
    embeddings: np.ndarray | None,
    exogenous_frames: dict[str, pd.DataFrame],
    stock_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    enriched = add_rule_features(
        samples,
        stock_frames,
        build_market_returns(stock_frames, HORIZONS),
    )
    enriched = align_exogenous_features(
        enriched,
        exogenous_frames,
        horizons=EXOGENOUS_HORIZONS,
    )

    rule_columns = []
    for horizon in HORIZONS:
        enriched[f"relative_{horizon}"] = (
            enriched[f"stock_{horizon}"] - enriched[f"market_{horizon}"]
        )
        enriched[f"rank_{horizon}"] = (
            enriched.groupby("context_end_date")[f"stock_{horizon}"].rank(pct=True)
            - 0.5
        )
        rule_columns.extend(
            [
                f"stock_{horizon}",
                f"market_{horizon}",
                f"relative_{horizon}",
                f"rank_{horizon}",
            ]
        )

    for horizon in EXOGENOUS_HORIZONS:
        sector_columns = [f"{name}_return_{horizon}" for name in SECTOR_NAMES]
        enriched[f"sector_return_{horizon}"] = enriched[sector_columns].mean(axis=1)
        enriched[f"sector_dispersion_{horizon}"] = enriched[sector_columns].std(
            axis=1, ddof=0
        )
        enriched[f"star_style_spread_{horizon}"] = (
            enriched[f"star50_return_{horizon}"]
            - enriched[f"csi300_return_{horizon}"]
        )

    exogenous_columns = [
        column
        for column in enriched.columns
        if any(
            token in column
            for token in ("_return_", "_volatility_", "_range_", "_amount_z_")
        )
        and column not in {
            "stock_future_return",
            "market_future_return",
            "residual_future_return",
        }
    ]
    derived_columns = [
        column
        for column in enriched.columns
        if column.startswith(("sector_return_", "sector_dispersion_", "star_style_spread_"))
    ]
    full_columns = list(dict.fromkeys(rule_columns + exogenous_columns + derived_columns))
    engineered = enriched[full_columns].to_numpy(dtype=np.float32)
    if embeddings is None:
        technical = np.empty((len(enriched), 0), dtype=np.float32)
    else:
        technical = np.asarray(embeddings[:, -15:], dtype=np.float32)
    features = np.nan_to_num(
        np.concatenate([technical, engineered], axis=1),
        nan=0.0,
        posinf=5.0,
        neginf=-5.0,
    )
    market_columns = [
        column
        for column in full_columns
        if not column.startswith(("stock_", "relative_", "rank_"))
    ]
    market_positions = [
        technical.shape[1] + full_columns.index(column) for column in market_columns
    ]
    return enriched, features, full_columns, market_positions


def make_regressor(name: str, seed: int):
    if name.startswith("ridge"):
        alpha = float(name.removeprefix("ridge"))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "hist2":
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=150,
            max_depth=2,
            min_samples_leaf=50,
            l2_regularization=4.0,
            random_state=seed,
        )
    if name == "hist3":
        return HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=60,
            l2_regularization=6.0,
            random_state=seed,
        )
    if name == "rf3":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=50,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(name)


def fit_probability_calibrator(
    predicted_return: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
) -> LogisticRegression:
    calibrator = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    sample_weight = np.ones(len(targets), dtype=float)
    sample_weight[target_mask] = 5.0
    calibrator.fit(predicted_return.reshape(-1, 1), targets, sample_weight=sample_weight)
    return calibrator


def fit_direct_candidate(
    name: str,
    features: np.ndarray,
    targets: pd.DataFrame,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    seed: int,
) -> tuple[object, np.ndarray]:
    model = make_regressor(name, seed)
    model.fit(
        features[train_indexes],
        targets.iloc[train_indexes]["stock_future_return"].to_numpy(),
    )
    return model, model.predict(features[validation_indexes])


def fit_decomposed_candidate(
    name: str,
    features: np.ndarray,
    market_positions: list[int],
    targets: pd.DataFrame,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    training = targets.iloc[train_indexes]
    betas = estimate_market_betas(training, min_points=60)
    training_beta = training["symbol"].map(betas).fillna(1.0).to_numpy(dtype=float)
    residual_target = (
        training["stock_future_return"].to_numpy(dtype=float)
        - training_beta * training["market_future_return"].to_numpy(dtype=float)
    )
    residual_model = make_regressor(name, seed)
    residual_model.fit(features[train_indexes], residual_target)

    unique_market = (
        training.assign(_position=train_indexes)
        .sort_values("context_end_date")
        .drop_duplicates("context_end_date")
    )
    unique_positions = unique_market["_position"].to_numpy(dtype=int)
    market_model = make_regressor(name, seed + 17)
    market_model.fit(
        features[unique_positions][:, market_positions],
        unique_market["market_future_return"].to_numpy(dtype=float),
    )
    validation = targets.iloc[validation_indexes]
    validation_beta = validation["symbol"].map(betas).fillna(1.0).to_numpy(dtype=float)
    market_prediction = market_model.predict(
        features[validation_indexes][:, market_positions]
    )
    residual_prediction = residual_model.predict(features[validation_indexes])
    prediction = combine_return_predictions(
        market_prediction, residual_prediction, validation_beta
    )
    return {
        "market": market_model,
        "residual": residual_model,
        "betas": betas,
    }, prediction


def candidate_predict(
    candidate: dict[str, object],
    features: np.ndarray,
    market_positions: list[int],
    rows: pd.DataFrame,
) -> np.ndarray:
    if candidate["kind"] == "direct":
        return candidate["model"].predict(features)
    models = candidate["model"]
    beta = rows["symbol"].map(models["betas"]).fillna(1.0).to_numpy(dtype=float)
    market = models["market"].predict(features[:, market_positions])
    residual = models["residual"].predict(features)
    return combine_return_predictions(market, residual, beta)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    suffix = "_engineered" if args.exclude_embeddings else ""
    output_dir = Path(f"outputs/exogenous_direction_h{args.horizon}{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(
        "outputs/direction_head/sample_index.csv",
        parse_dates=["context_end_date", "label_end_date"],
        dtype={"symbol": str},
    )
    embeddings = None
    if not args.exclude_embeddings:
        embeddings = np.load(
            "outputs/direction_head/frozen_embeddings.npy", mmap_mode="r"
        )
    stock_frames = load_frames(Path("data/direction_universe"))
    exogenous_frames = load_exogenous_frames(Path("data/exogenous"))
    if args.horizon != 10:
        samples = retarget_sample_horizon(
            samples, stock_frames, horizon=args.horizon
        )
    market_start = exogenous_frames["star50"]["date"].min()
    market_end = exogenous_frames["star50"]["date"].max()
    samples = samples[
        (samples["context_end_date"] >= market_start)
        & (samples["label_end_date"] <= market_end)
    ].reset_index(drop=True)
    if embeddings is not None:
        embeddings = np.asarray(embeddings[samples["embedding_index"].to_numpy()])
    samples["embedding_index"] = np.arange(len(samples), dtype=np.int64)
    samples = attach_return_decomposition_targets(
        samples, stock_frames, exogenous_frames["star50"]
    )
    samples, features, feature_columns, market_positions = build_feature_matrix(
        samples, embeddings, exogenous_frames, stock_frames
    )
    target_up = samples["target_up"].to_numpy(dtype=int)
    print(
        f"samples={len(samples)} features={features.shape[1]} "
        f"market_features={len(market_positions)} exogenous={len(exogenous_frames)}"
    )

    fold_results = []
    selected_probability_parts = []
    ensemble_probability_parts = []
    selective_probability_parts = []
    selective_target_parts = []
    selective_total_points = 0
    deployment_artifact = None
    target_parts = []
    test_parts = []
    model_names = ("ridge1", "ridge10", "ridge100", "hist2", "hist3", "rf3")
    for test_start in pd.date_range("2021-07-01", "2026-07-01", freq="6MS"):
        validation_start = test_start - pd.DateOffset(months=6)
        train_start = validation_start - pd.DateOffset(years=3)
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
            (samples["symbol"] == TARGET_SYMBOL)
            & (samples["context_end_date"] >= test_start)
            & (samples["label_end_date"] <= test_end)
        )
        train_indexes = np.flatnonzero(train_mask)
        validation_indexes = np.flatnonzero(validation_mask)
        test_indexes = np.flatnonzero(test_mask)
        if min(len(train_indexes), len(validation_indexes), len(test_indexes)) == 0:
            continue
        validation_rows = samples.iloc[validation_indexes]
        validation_targets = target_up[validation_indexes]
        target_validation_mask = (
            validation_rows["symbol"].to_numpy() == TARGET_SYMBOL
        )
        ranked = []
        for model_name in model_names:
            for kind in ("direct", "decomposed"):
                if kind == "direct":
                    model, validation_return = fit_direct_candidate(
                        model_name,
                        features,
                        samples,
                        train_indexes,
                        validation_indexes,
                        seed=100,
                    )
                else:
                    model, validation_return = fit_decomposed_candidate(
                        model_name,
                        features,
                        market_positions,
                        samples,
                        train_indexes,
                        validation_indexes,
                        seed=100,
                    )
                calibrator = fit_probability_calibrator(
                    validation_return, validation_targets, target_validation_mask
                )
                validation_probability = calibrator.predict_proba(
                    validation_return.reshape(-1, 1)
                )[:, 1]
                global_score, global_summary = robust_score(
                    validation_rows,
                    validation_probability,
                    validation_targets,
                )
                target_summary = summarize_direction_probabilities(
                    validation_probability[target_validation_mask],
                    validation_targets[target_validation_mask],
                )
                score = (
                    0.65 * global_score
                    + 0.35 * target_summary["balanced_accuracy"]
                    - 0.10 * target_summary["brier_score"]
                )
                ranked.append(
                    {
                        "score": float(score),
                        "name": f"{kind}_{model_name}",
                        "kind": kind,
                        "model": model,
                        "calibrator": calibrator,
                        "validation_probability": validation_probability,
                        "global_validation": global_summary,
                        "target_validation": target_summary,
                    }
                )
        ranked.sort(key=lambda candidate: candidate["score"], reverse=True)
        test_rows = samples.iloc[test_indexes]
        test_features = features[test_indexes]
        test_targets = target_up[test_indexes]
        top = ranked[:3]
        target_validation_probabilities = np.stack(
            [
                candidate["validation_probability"][target_validation_mask]
                for candidate in top
            ]
        )
        selective_policy = fit_selective_direction_policy(
            target_validation_probabilities,
            validation_targets[target_validation_mask],
            min_coverage=0.3,
            min_points=30,
        )
        deployment_artifact = {
            "schema_version": 1,
            "target_symbol": TARGET_SYMBOL,
            "horizon": int(args.horizon),
            "lookback": int(
                samples.iloc[test_indexes[0]]["context_end"]
                - samples.iloc[test_indexes[0]]["context_start"]
                + 1
            ),
            "valid_from": str(test_start.date()),
            "valid_to": str(test_end.date()),
            "training_start": str(train_start.date()),
            "training_label_cutoff": str(validation_start.date()),
            "calibration_start": str(validation_start.date()),
            "calibration_end": str((test_start - pd.Timedelta(days=1)).date()),
            "feature_columns": feature_columns,
            "feature_count": int(features.shape[1]),
            "market_positions": market_positions,
            "candidates": [
                {
                    "name": candidate["name"],
                    "kind": candidate["kind"],
                    "model": candidate["model"],
                    "calibrator": candidate["calibrator"],
                }
                for candidate in top
            ],
            "selective_policy": selective_policy,
        }
        test_probabilities = []
        for candidate in top:
            predicted_return = candidate_predict(
                candidate, test_features, market_positions, test_rows
            )
            test_probabilities.append(
                candidate["calibrator"].predict_proba(
                    predicted_return.reshape(-1, 1)
                )[:, 1]
            )
        selected_probability = test_probabilities[0]
        ensemble_probability, selective_mask = apply_selective_direction_policy(
            np.stack(test_probabilities), selective_policy
        )
        selected_summary = summarize_direction_probabilities(
            selected_probability, test_targets
        )
        ensemble_summary = summarize_direction_probabilities(
            ensemble_probability, test_targets
        )
        selective_summary = None
        if selective_mask.any():
            selective_summary = summarize_direction_probabilities(
                ensemble_probability[selective_mask], test_targets[selective_mask]
            )
        fold_result = {
            "start": str(test_start.date()),
            "end": str(test_end.date()),
            "train_points": len(train_indexes),
            "validation_points": len(validation_indexes),
            "test_points": len(test_indexes),
            "top_models": [candidate["name"] for candidate in top],
            "top_validation_balanced": [
                candidate["target_validation"]["balanced_accuracy"]
                for candidate in top
            ],
            "selected": selected_summary,
            "ensemble": ensemble_summary,
            "selective_policy": selective_policy,
            "selective": selective_summary,
            "selective_test_coverage": float(np.mean(selective_mask)),
        }
        fold_results.append(fold_result)
        selected_probability_parts.append(selected_probability)
        ensemble_probability_parts.append(ensemble_probability)
        selective_total_points += len(test_targets)
        if selective_mask.any():
            selective_probability_parts.append(ensemble_probability[selective_mask])
            selective_target_parts.append(test_targets[selective_mask])
        target_parts.append(test_targets)
        test_parts.append(test_rows)
        selective_text = (
            "off"
            if selective_summary is None
            else f"{selective_summary['accuracy']:.3f}@{np.mean(selective_mask):.2f}"
        )
        print(
            f"{test_start:%Y-%m}..{test_end:%Y-%m} "
            f"top={top[0]['name']:20s} "
            f"val={top[0]['target_validation']['balanced_accuracy']:.3f} "
            f"test={selected_summary['accuracy']:.3f}/"
            f"{selected_summary['balanced_accuracy']:.3f} "
            f"ens={ensemble_summary['accuracy']:.3f}/"
            f"{ensemble_summary['balanced_accuracy']:.3f} "
            f"selective={selective_text}"
        )

    flat_selected = np.concatenate(selected_probability_parts)
    flat_ensemble = np.concatenate(ensemble_probability_parts)
    flat_targets = np.concatenate(target_parts)
    selected_summary = summarize_direction_probabilities(flat_selected, flat_targets)
    ensemble_summary = summarize_direction_probabilities(flat_ensemble, flat_targets)
    test_frame = pd.concat(test_parts, ignore_index=True)
    independent = select_non_overlapping_samples(
        test_frame, horizon=args.horizon
    )
    position_by_embedding = {
        int(value): position
        for position, value in enumerate(test_frame["embedding_index"])
    }
    independent_positions = np.asarray(
        [position_by_embedding[int(value)] for value in independent["embedding_index"]]
    )
    independent_summary = summarize_direction_probabilities(
        flat_ensemble[independent_positions], flat_targets[independent_positions]
    )
    phase_summary = summarize_non_overlapping_phases(
        flat_ensemble,
        flat_targets,
        test_frame,
        horizon=args.horizon,
    )
    selective_summary = None
    if selective_probability_parts:
        selective_summary = summarize_direction_probabilities(
            np.concatenate(selective_probability_parts),
            np.concatenate(selective_target_parts),
        )
        selective_summary["coverage"] = (
            int(selective_summary["points"]) / selective_total_points
        )
    report = {
        "horizon": args.horizon,
        "feature_points": features.shape[1],
        "feature_columns": feature_columns,
        "market_feature_points": len(market_positions),
        "folds": fold_results,
        "selected_aggregate": selected_summary,
        "ensemble_aggregate": ensemble_summary,
        "independent_ensemble": independent_summary,
        "non_overlapping_phases": phase_summary,
        "selective_ensemble": selective_summary,
    }
    if args.save_deployment_artifact is not None:
        if args.horizon != 1 or args.exclude_embeddings:
            raise ValueError(
                "deployment artifact requires horizon=1 with technical snapshots"
            )
        if deployment_artifact is None:
            raise RuntimeError("no deployment fold was produced")
        wilson_lower = float(independent_summary["accuracy_wilson_95"][0])
        if (
            wilson_lower <= 0.5
            or float(independent_summary["balanced_accuracy"]) < 0.52
        ):
            raise RuntimeError(
                "deployment gate failed: one-day walk-forward direction edge "
                "is not independently above chance"
            )
        deployment_artifact["walk_forward_evidence"] = independent_summary
        deployment_artifact["probability_is_calibrated"] = False
        deployment_artifact["confidence_label"] = "low"
        args.save_deployment_artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(deployment_artifact, args.save_deployment_artifact)
        print(f"saved deployment artifact: {args.save_deployment_artifact}")
    (output_dir / "audit_report.json").write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"aggregate selected={selected_summary['accuracy']:.3f}/"
        f"{selected_summary['balanced_accuracy']:.3f} "
        f"ensemble={ensemble_summary['accuracy']:.3f}/"
        f"{ensemble_summary['balanced_accuracy']:.3f} "
        f"brier={ensemble_summary['brier_score']:.3f}"
    )
    print(
        f"independent n={independent_summary['points']} "
        f"accuracy={independent_summary['accuracy']:.3f} "
        f"balanced={independent_summary['balanced_accuracy']:.3f} "
        f"brier={independent_summary['brier_score']:.3f} "
        f"wilson={independent_summary['accuracy_wilson_95']}"
    )
    print(
        f"phase accuracy mean={phase_summary['mean_accuracy']:.3f} "
        f"worst={phase_summary['worst_accuracy']:.3f} "
        f"best={phase_summary['best_accuracy']:.3f}; "
        f"balanced mean={phase_summary['mean_balanced_accuracy']:.3f} "
        f"worst={phase_summary['worst_balanced_accuracy']:.3f}"
    )
    if selective_summary is None:
        print("selective signal disabled in every fold")
    else:
        print(
            f"selective n={selective_summary['points']} "
            f"coverage={selective_summary['coverage']:.3f} "
            f"accuracy={selective_summary['accuracy']:.3f} "
            f"balanced={selective_summary['balanced_accuracy']:.3f} "
            f"wilson={selective_summary['accuracy_wilson_95']}"
        )


if __name__ == "__main__":
    main()
