"""Structural + behavioral tests for the promoted h=1 selective gate."""

from pathlib import Path

import numpy as np

from finetune.promoted_config import (
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    model_dir,
    resolve_promoted_model_dirs,
)
from finetune.selective_prediction import (
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)


def test_promoted_model_dirs_exist():
    dirs = resolve_promoted_model_dirs()
    assert set(dirs) == {1, 3, 5, 10}
    for h, path in dirs.items():
        assert path.is_dir(), f"missing model for h={h}: {path}"
        assert (path / "multihorizon_head.pt").is_file()
        assert PROMOTED_MODEL_BY_HORIZON[h] in str(path)


def test_promoted_gate_constants_match_log_winner():
    assert PROMOTED_H1_GATE["confidence_key"] == "actionable_score"
    assert PROMOTED_H1_GATE["confidence_threshold"] == 0.45
    assert PROMOTED_H1_GATE["min_abs_return"] >= 0.0
    from finetune.promoted_config import PROMOTED_MODEL_BY_HORIZON, PROMOTED_RETURN_BLEND

    assert PROMOTED_MODEL_BY_HORIZON[1] == "r10_joint_splitlr"
    assert PROMOTED_MODEL_BY_HORIZON[5] == "r5_frozen_pool48"
    assert PROMOTED_RETURN_BLEND["primary_weight"] == 0.85
    assert PROMOTED_RETURN_BLEND["primary"] == "r10_joint_splitlr"
    assert PROMOTED_RETURN_BLEND["secondary"] == "r5_frozen_pool48"


def test_promoted_gate_improves_precision_on_synthetic_mixture():
    """Drive the real gate function: low-conf calls abstain; wrong tiny moves drop with mag>0."""
    hard = np.array(
        [UP_CLASS, UP_CLASS, DOWN_CLASS, DOWN_CLASS, UP_CLASS, DOWN_CLASS, UP_CLASS, DOWN_CLASS, UP_CLASS, DOWN_CLASS]
    )
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.3])
    pred_ret = np.array([0.02, 0.02, -0.02, -0.02, 0.02, -0.02, 0.001, -0.001, 0.02, -0.02])
    target = np.array(
        [UP_CLASS, UP_CLASS, DOWN_CLASS, DOWN_CLASS, UP_CLASS, DOWN_CLASS, DOWN_CLASS, UP_CLASS, UP_CLASS, DOWN_CLASS]
    )
    ungated = gated_actionable_metrics(hard, target)
    # Confidence-only gate (matches Phase-3 default min_abs_return=0 when set to 0)
    gated_conf = apply_consistency_and_magnitude_gate(
        hard,
        conf,
        pred_ret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=0.0,
        require_sign_agree=False,
    )
    metrics_conf = gated_actionable_metrics(gated_conf, target)
    assert metrics_conf["coverage"] < ungated["coverage"]
    assert gated_conf[8] == FLAT_CLASS and gated_conf[9] == FLAT_CLASS
    # Magnitude gate still works as a shipped helper when min_abs_return>0
    gated_mag = apply_consistency_and_magnitude_gate(
        hard,
        conf,
        pred_ret,
        confidence_threshold=0.45,
        min_abs_return=0.005,
        require_sign_agree=False,
    )
    assert gated_mag[6] == FLAT_CLASS and gated_mag[7] == FLAT_CLASS
    metrics_mag = gated_actionable_metrics(gated_mag, target)
    assert metrics_mag["precision_on_calls"] > ungated["precision_on_calls"]


def test_logits_confidence_used_by_promoted_key():
    logits = np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
    conf = direction_confidence_from_logits(logits)
    key = PROMOTED_H1_GATE["confidence_key"]
    assert key in conf
    assert conf[key].shape == (2,)
    assert conf[key][0] > 0.5


def test_run_promoted_eval_wires_return_blend():
    """Structural: shipped entry must apply PROMOTED_RETURN_BLEND, not dir-only ret."""
    import inspect

    from finetune import run_promoted_eval as mod
    from finetune.dual_metric_compare import blend_returns

    src = inspect.getsource(mod)
    assert "PROMOTED_RETURN_BLEND" in src
    assert "blend_returns" in src
    assert "use_return_blend" in src
    # blend_returns is the real helper used by the eval path
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    np.testing.assert_allclose(blend_returns(a, b, 0.85), [0.85, 0.15])
