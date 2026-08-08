"""Confidence-gated selective prediction and absolute-direction backtest helpers.

Pure functions only — no model I/O — so unit tests and offline sweeps can reuse
the same metric definitions as production evaluation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DOWN_CLASS = 0
FLAT_CLASS = 1
UP_CLASS = 2


def direction_confidence_from_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Convert [..., 3] logits into softmax probs and confidence scores.

    Confidence definitions:
    - max_prob: max_c softmax(logits)_c
    - margin: P(up) - P(down) absolute value |P(up)-P(down)|
    - nonflat_prob: P(up) + P(down)
    - actionable_score: max(P(up), P(down))  (strength of the better side)
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[-1] != 3:
        raise ValueError("logits last dim must be 3 (down/flat/up)")
    # stable softmax
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    p_down = probs[..., DOWN_CLASS]
    p_flat = probs[..., FLAT_CLASS]
    p_up = probs[..., UP_CLASS]
    max_prob = probs.max(axis=-1)
    margin = np.abs(p_up - p_down)
    nonflat_prob = p_up + p_down
    actionable_score = np.maximum(p_up, p_down)
    hard_pred = probs.argmax(axis=-1).astype(np.int64)
    return {
        "probs": probs.astype(np.float64),
        "max_prob": max_prob.astype(np.float64),
        "margin": margin.astype(np.float64),
        "nonflat_prob": nonflat_prob.astype(np.float64),
        "actionable_score": actionable_score.astype(np.float64),
        "hard_pred": hard_pred,
        "p_down": p_down.astype(np.float64),
        "p_flat": p_flat.astype(np.float64),
        "p_up": p_up.astype(np.float64),
    }


def apply_confidence_gate(
    hard_pred: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    *,
    abstain_class: int = FLAT_CLASS,
) -> np.ndarray:
    """Keep hard_pred only when confidence >= threshold; else abstain (FLAT)."""
    hard_pred = np.asarray(hard_pred)
    confidence = np.asarray(confidence, dtype=np.float64)
    if hard_pred.shape != confidence.shape:
        raise ValueError("hard_pred and confidence must share shape")
    gated = hard_pred.copy()
    gated[confidence < float(threshold)] = int(abstain_class)
    return gated


def apply_consistency_and_magnitude_gate(
    hard_pred: np.ndarray,
    confidence: np.ndarray,
    pred_return: np.ndarray,
    *,
    confidence_threshold: float,
    min_abs_return: float = 0.0,
    require_sign_agree: bool = True,
    abstain_class: int = FLAT_CLASS,
    down_class: int = DOWN_CLASS,
    up_class: int = UP_CLASS,
) -> np.ndarray:
    """Abstain unless confidence, optional |return|, and dir/return sign agree.

    Sign agreement: UP requires pred_return > 0, DOWN requires pred_return < 0.
    """
    hard_pred = np.asarray(hard_pred).astype(np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    pred_return = np.asarray(pred_return, dtype=np.float64)
    if hard_pred.shape != confidence.shape or hard_pred.shape != pred_return.shape:
        raise ValueError("hard_pred, confidence, pred_return must share shape")
    gated = hard_pred.copy()
    keep = confidence >= float(confidence_threshold)
    if min_abs_return > 0.0:
        keep = keep & (np.abs(pred_return) >= float(min_abs_return))
    if require_sign_agree:
        agree_up = (hard_pred == up_class) & (pred_return > 0.0)
        agree_down = (hard_pred == down_class) & (pred_return < 0.0)
        keep = keep & (agree_up | agree_down | (hard_pred == abstain_class))
    # also drop flat hard preds from "calls" naturally
    gated[~keep] = int(abstain_class)
    return gated


def nonflat_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    flat_class: int = FLAT_CLASS,
) -> float:
    """Accuracy on samples whose *true* label is non-flat."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    mask = target != flat_class
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float((pred[mask] == target[mask]).sum() / total)


def gated_actionable_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    flat_class: int = FLAT_CLASS,
) -> dict[str, float]:
    """Metrics when the model is allowed to abstain (predict FLAT).

    - coverage: fraction of samples where pred is UP or DOWN
    - gated_nonflat_acc: among predictions that are non-flat AND true is non-flat,
      share that match (strict: only score when both actionable)
    - precision_on_calls: among model non-flat calls, fraction correct vs true label
      (true FLAT counted as wrong — realistic trading precision)
    - recall_on_true_moves: among true non-flat, fraction correctly called
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    n = len(pred)
    if n == 0:
        return {
            "coverage": 0.0,
            "precision_on_calls": 0.0,
            "recall_on_true_moves": 0.0,
            "gated_nonflat_acc": 0.0,
            "n_calls": 0.0,
            "n_true_moves": 0.0,
        }
    call_mask = pred != flat_class
    true_move = target != flat_class
    n_calls = int(call_mask.sum())
    n_true = int(true_move.sum())
    coverage = n_calls / n
    if n_calls > 0:
        precision = float((pred[call_mask] == target[call_mask]).sum() / n_calls)
    else:
        precision = 0.0
    if n_true > 0:
        recall = float(((pred == target) & true_move).sum() / n_true)
    else:
        recall = 0.0
    # gated nonflat: true nonflat and model made a call (or evaluate only true moves
    # with model's possibly-abstained pred — same as recall if we require match)
    both = true_move & call_mask
    if both.sum() > 0:
        gated_nf = float((pred[both] == target[both]).sum() / both.sum())
    else:
        gated_nf = 0.0
    return {
        "coverage": float(coverage),
        "precision_on_calls": precision,
        "recall_on_true_moves": recall,
        "gated_nonflat_acc": gated_nf,
        "n_calls": float(n_calls),
        "n_true_moves": float(n_true),
    }


def accuracy_vs_coverage_curve(
    confidence: np.ndarray,
    hard_pred: np.ndarray,
    target: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    """Sweep confidence thresholds and report gated metrics at each point."""
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        gated = apply_confidence_gate(hard_pred, confidence, thr)
        m = gated_actionable_metrics(gated, target)
        m["threshold"] = float(thr)
        m["ungated_nonflat_acc"] = nonflat_accuracy(hard_pred, target)
        rows.append(m)
    return rows


def select_threshold_for_coverage_band(
    curve: list[dict[str, float]],
    *,
    min_coverage: float = 0.20,
    max_coverage: float = 0.40,
    score_key: str = "precision_on_calls",
) -> dict[str, float] | None:
    """Pick the threshold in [min_coverage, max_coverage] maximizing score_key."""
    candidates = [
        row
        for row in curve
        if min_coverage <= row["coverage"] <= max_coverage and row["n_calls"] >= 5
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r[score_key], r["coverage"]))


def absolute_direction_backtest(
    pred_direction: np.ndarray,
    future_log_returns: np.ndarray,
    *,
    flat_class: int = FLAT_CLASS,
    down_class: int = DOWN_CLASS,
    up_class: int = UP_CLASS,
    transaction_cost: float = 0.0,
    stride: int = 1,
) -> dict[str, float]:
    """Simple absolute long/flat/short PnL from direction calls.

    Position: +1 for UP, -1 for DOWN, 0 for FLAT/abstain.
    Realized PnL per step: position * (exp(log_return)-1) - cost if position != 0.

    When ``stride`` > 1, only every ``stride``-th sample is kept (non-overlapping
    origins for horizon≈stride comparisons).
    """
    pred = np.asarray(pred_direction).astype(np.int64)
    rets = np.asarray(future_log_returns, dtype=np.float64)
    if pred.shape != rets.shape:
        raise ValueError("pred_direction and future_log_returns must match")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    if stride > 1:
        pred = pred[::stride]
        rets = rets[::stride]
    position = np.zeros_like(rets)
    position[pred == up_class] = 1.0
    position[pred == down_class] = -1.0
    simple_ret = np.exp(rets) - 1.0
    active = position != 0.0
    step_pnl = position * simple_ret
    step_pnl = step_pnl - float(transaction_cost) * active.astype(np.float64)
    equity = np.cumprod(1.0 + step_pnl)
    total_return = float(equity[-1] - 1.0) if len(equity) else 0.0
    n_trades = int(active.sum())
    if n_trades > 0:
        hit = ((position > 0) & (simple_ret > 0)) | ((position < 0) & (simple_ret < 0))
        hit_rate = float(hit[active].mean())
        avg_trade = float(step_pnl[active].mean())
    else:
        hit_rate = 0.0
        avg_trade = 0.0
    return {
        "total_return": total_return,
        "n_trades": float(n_trades),
        "coverage": float(n_trades / len(pred)) if len(pred) else 0.0,
        "hit_rate": hit_rate,
        "avg_trade_pnl": avg_trade,
        "final_equity": float(equity[-1]) if len(equity) else 1.0,
        "stride": float(stride),
        "n_origins": float(len(pred)),
    }
