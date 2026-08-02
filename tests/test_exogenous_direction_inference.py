import pytest

from finetune.exogenous_direction_inference import make_direction_decision


def _artifact():
    return {
        "schema_version": 1,
        "horizon": 1,
        "valid_from": "2026-07-01",
        "valid_to": "2026-12-31",
        "confidence_label": "low",
        "probability_is_calibrated": False,
        "walk_forward_evidence": {
            "points": 1213,
            "accuracy": 0.5383,
            "balanced_accuracy": 0.5322,
            "accuracy_wilson_95": [0.5102, 0.5662],
        },
    }


def test_verified_one_day_decision_is_low_confidence_candidate_not_probability_claim():
    decision = make_direction_decision(
        [0.501, 0.502, 0.4995],
        _artifact(),
        origin_date="2026-08-03",
    )

    assert decision["status"] == "candidate"
    assert decision["direction"] == "up"
    assert decision["confidence_label"] == "low"
    assert decision["probability_is_calibrated"] is False
    assert decision["historical_accuracy"] == pytest.approx(0.5383)
    assert decision["historical_wilson_95"] == [0.5102, 0.5662]


def test_direction_decision_abstains_outside_artifact_validity_period():
    decision = make_direction_decision(
        [0.6, 0.7, 0.8],
        _artifact(),
        origin_date="2027-01-02",
    )

    assert decision["status"] == "abstain"
    assert decision["direction"] is None
    assert decision["reason"] == "direction_model_outside_validity_period"


def test_direction_decision_rejects_unverified_horizon_or_evidence():
    artifact = _artifact()
    artifact["horizon"] = 5
    with pytest.raises(ValueError, match="one-day"):
        make_direction_decision([0.6], artifact, origin_date="2026-08-03")

    artifact = _artifact()
    artifact["walk_forward_evidence"]["accuracy_wilson_95"] = [0.49, 0.56]
    with pytest.raises(ValueError, match="above chance"):
        make_direction_decision([0.6], artifact, origin_date="2026-08-03")
