"""Safe live inference for the independently audited one-day direction ensemble."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from finetune.audit_exogenous_direction import (
    build_feature_matrix,
    candidate_predict,
    load_exogenous_frames,
)
from finetune.audit_direction_rules import load_frames
from finetune.exogenous_direction import build_live_origin_samples
from finetune.frozen_direction_head import prepare_context_arrays


def _validated_evidence(artifact: Mapping[str, object]) -> Mapping[str, object]:
    if int(artifact.get("schema_version", 0)) != 1:
        raise ValueError("unsupported direction artifact schema")
    if int(artifact.get("horizon", 0)) != 1:
        raise ValueError("only an independently verified one-day artifact is allowed")
    evidence = artifact.get("walk_forward_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("direction artifact is missing walk-forward evidence")
    wilson = evidence.get("accuracy_wilson_95")
    if not isinstance(wilson, Sequence) or len(wilson) != 2:
        raise ValueError("direction artifact has invalid Wilson evidence")
    if float(wilson[0]) <= 0.5 or float(evidence["balanced_accuracy"]) < 0.52:
        raise ValueError("direction evidence is not independently above chance")
    return evidence


def make_direction_decision(
    model_probabilities,
    artifact: Mapping[str, object],
    *,
    origin_date: str | pd.Timestamp,
) -> dict[str, object]:
    """Convert ensemble outputs into a conservative, evidence-labelled decision."""
    evidence = _validated_evidence(artifact)
    probability = np.asarray(model_probabilities, dtype=float).reshape(-1)
    if probability.size == 0 or not np.isfinite(probability).all():
        raise ValueError("model probabilities must be non-empty and finite")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("model probabilities must be between zero and one")

    origin = pd.Timestamp(origin_date)
    valid_from = pd.Timestamp(str(artifact["valid_from"]))
    valid_to = pd.Timestamp(str(artifact["valid_to"]))
    base = {
        "horizon": 1,
        "origin_date": str(origin.date()),
        "confidence_label": "low",
        "probability_is_calibrated": False,
        "historical_points": int(evidence["points"]),
        "historical_accuracy": float(evidence["accuracy"]),
        "historical_balanced_accuracy": float(evidence["balanced_accuracy"]),
        "historical_wilson_95": [
            float(evidence["accuracy_wilson_95"][0]),
            float(evidence["accuracy_wilson_95"][1]),
        ],
    }
    if origin < valid_from or origin > valid_to:
        return {
            **base,
            "status": "abstain",
            "direction": None,
            "candidate_direction": None,
            "raw_model_up_probability": None,
            "reason": "direction_model_outside_validity_period",
        }

    mean_probability = float(np.mean(probability))
    direction = "up" if mean_probability >= 0.5 else "down"
    return {
        **base,
        "status": "candidate",
        "direction": direction,
        "candidate_direction": direction,
        "raw_model_up_probability": mean_probability,
        "reason": "verified_one_day_edge_low_confidence",
    }


def predict_live_direction(
    artifact_path: str | Path,
    *,
    stock_data_dir: str | Path,
    exogenous_data_dir: str | Path,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Generate one live target signal with the exact audited feature schema."""
    artifact = joblib.load(Path(artifact_path))
    _validated_evidence(artifact)
    if as_of is not None:
        requested_origin = pd.Timestamp(as_of)
        valid_from = pd.Timestamp(str(artifact["valid_from"]))
        valid_to = pd.Timestamp(str(artifact["valid_to"]))
        if requested_origin < valid_from or requested_origin > valid_to:
            return make_direction_decision(
                [0.5], artifact, origin_date=requested_origin
            )
    stock_frames = load_frames(Path(stock_data_dir))
    exogenous_frames = load_exogenous_frames(Path(exogenous_data_dir))
    target_symbol = str(artifact["target_symbol"])
    if target_symbol not in stock_frames:
        raise ValueError(f"target symbol {target_symbol} is missing from stock data")

    target_dates = pd.to_datetime(stock_frames[target_symbol]["date"])
    origin = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(target_dates.iloc[-1])
    valid_from = pd.Timestamp(str(artifact["valid_from"]))
    valid_to = pd.Timestamp(str(artifact["valid_to"]))
    if origin < valid_from or origin > valid_to:
        return make_direction_decision([0.5], artifact, origin_date=origin)

    origins = build_live_origin_samples(
        stock_frames,
        lookback=int(artifact["lookback"]),
        as_of=origin,
    )
    if target_symbol not in set(origins["symbol"]):
        raise ValueError("target stock has insufficient live context")
    technical = []
    for row in origins.itertuples(index=False):
        _, _, snapshot = prepare_context_arrays(
            stock_frames[str(row.symbol)],
            int(row.context_start),
            int(row.context_end),
        )
        technical.append(snapshot)
    enriched, features, feature_columns, market_positions = build_feature_matrix(
        origins,
        np.asarray(technical, dtype=np.float32),
        exogenous_frames,
        stock_frames,
    )
    if feature_columns != list(artifact["feature_columns"]):
        raise ValueError("live engineered feature schema differs from the artifact")
    if features.shape[1] != int(artifact["feature_count"]):
        raise ValueError("live feature count differs from the artifact")
    if market_positions != list(artifact["market_positions"]):
        raise ValueError("live market feature positions differ from the artifact")

    target_positions = np.flatnonzero(
        enriched["symbol"].astype(str).to_numpy() == target_symbol
    )
    if len(target_positions) != 1:
        raise ValueError("live feature matrix must contain exactly one target origin")
    position = target_positions[0]
    target_features = features[[position]]
    target_rows = enriched.iloc[[position]]
    probabilities = []
    for candidate in artifact["candidates"]:
        predicted_return = candidate_predict(
            candidate,
            target_features,
            market_positions,
            target_rows,
        )
        probabilities.append(
            float(
                candidate["calibrator"].predict_proba(
                    predicted_return.reshape(-1, 1)
                )[0, 1]
            )
        )
    decision = make_direction_decision(
        probabilities,
        artifact,
        origin_date=enriched.iloc[position]["context_end_date"],
    )
    decision["models"] = [candidate["name"] for candidate in artifact["candidates"]]
    decision["model_probabilities"] = probabilities
    return decision
