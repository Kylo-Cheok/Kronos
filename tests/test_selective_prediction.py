"""Unit tests for confidence gating and absolute-direction backtest helpers."""

import numpy as np

from finetune.selective_prediction import (
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    absolute_direction_backtest,
    accuracy_vs_coverage_curve,
    apply_confidence_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
    select_threshold_for_coverage_band,
)


def test_direction_confidence_from_logits_softmax_and_scores():
    # Strong UP, strong DOWN, flat-ish
    logits = np.array(
        [
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    conf = direction_confidence_from_logits(logits)
    assert conf["hard_pred"].tolist() == [UP_CLASS, DOWN_CLASS, FLAT_CLASS]
    assert conf["probs"].shape == (3, 3)
    assert conf["max_prob"][0] > 0.9
    assert conf["margin"][0] > conf["margin"][2]
    np.testing.assert_allclose(conf["probs"].sum(axis=-1), 1.0, atol=1e-9)


def test_apply_confidence_gate_abstains_below_threshold():
    hard = np.array([UP_CLASS, DOWN_CLASS, UP_CLASS])
    confidence = np.array([0.9, 0.4, 0.7])
    gated = apply_confidence_gate(hard, confidence, 0.6)
    assert gated.tolist() == [UP_CLASS, FLAT_CLASS, UP_CLASS]


def test_nonflat_accuracy_ignores_true_flat():
    pred = np.array([UP_CLASS, DOWN_CLASS, UP_CLASS, FLAT_CLASS])
    target = np.array([UP_CLASS, UP_CLASS, FLAT_CLASS, DOWN_CLASS])
    # nonflat targets: indices 0,1,3 → correct only index 0
    assert abs(nonflat_accuracy(pred, target) - (1 / 3)) < 1e-9


def test_gated_metrics_precision_and_coverage():
    pred = np.array([UP_CLASS, FLAT_CLASS, DOWN_CLASS, UP_CLASS])
    target = np.array([UP_CLASS, UP_CLASS, DOWN_CLASS, FLAT_CLASS])
    m = gated_actionable_metrics(pred, target)
    assert abs(m["coverage"] - 0.75) < 1e-9  # 3 calls
    # calls at 0,2,3: correct, correct, wrong → 2/3
    assert abs(m["precision_on_calls"] - (2 / 3)) < 1e-9


def test_accuracy_vs_coverage_curve_and_band_selection():
    # 12 samples so band selection's n_calls>=5 floor can still fire.
    hard = np.array(
        [UP_CLASS, UP_CLASS, DOWN_CLASS, DOWN_CLASS, UP_CLASS, UP_CLASS] * 2
    )
    target = np.array(
        [UP_CLASS, DOWN_CLASS, DOWN_CLASS, UP_CLASS, UP_CLASS, FLAT_CLASS] * 2
    )
    conf = np.array([0.95, 0.55, 0.90, 0.50, 0.80, 0.40] * 2)
    curve = accuracy_vs_coverage_curve(
        conf, hard, target, thresholds=[0.0, 0.6, 0.85, 0.99]
    )
    assert len(curve) == 4
    assert curve[0]["coverage"] == 1.0  # thr=0 keeps all hard preds
    best = select_threshold_for_coverage_band(
        curve, min_coverage=0.2, max_coverage=0.9, score_key="precision_on_calls"
    )
    assert best is not None
    assert 0.2 <= best["coverage"] <= 0.9


def test_absolute_direction_backtest_long_short_flat():
    # UP captures +10%, DOWN captures -10% move as profit, FLAT skips
    pred = np.array([UP_CLASS, DOWN_CLASS, FLAT_CLASS])
    log_rets = np.log(np.array([1.10, 0.90, 1.05]))
    bt = absolute_direction_backtest(pred, log_rets, transaction_cost=0.0)
    assert bt["n_trades"] == 2.0
    assert bt["total_return"] > 0.0
    assert bt["hit_rate"] == 1.0


def test_consistency_and_magnitude_gate_filters_disagreement():
    from finetune.selective_prediction import apply_consistency_and_magnitude_gate

    hard = np.array([UP_CLASS, UP_CLASS, DOWN_CLASS, DOWN_CLASS])
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    # second UP has negative return (disagree); fourth DOWN abs return too small
    pred_ret = np.array([0.02, -0.01, -0.03, -0.001])
    gated = apply_consistency_and_magnitude_gate(
        hard,
        conf,
        pred_ret,
        confidence_threshold=0.5,
        min_abs_return=0.005,
        require_sign_agree=True,
    )
    assert gated.tolist() == [UP_CLASS, FLAT_CLASS, DOWN_CLASS, FLAT_CLASS]
