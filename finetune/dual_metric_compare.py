"""Dual-metric helpers: direction nonflat + return MAE promotion rules.

Pure functions for unit tests and offline ensemble strategy comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from finetune.selective_prediction import (
    FLAT_CLASS,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)


def return_mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.size == 0:
        raise ValueError("pred and target must be non-empty and same shape")
    return float(np.mean(np.abs(pred - target)))


def summarize_horizons(
    pred_dir: dict[int, np.ndarray],
    pred_ret: dict[int, np.ndarray],
    t_dir: dict[int, np.ndarray],
    t_ret: dict[int, np.ndarray],
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    by_h: dict[str, dict[str, float]] = {}
    nf_c = nf_t = 0
    maes: list[float] = []
    for h in horizons:
        pd_ = pred_dir[h]
        pr = pred_ret[h]
        td = t_dir[h]
        tr = t_ret[h]
        nf = nonflat_accuracy(pd_, td)
        mae = return_mae(pr, tr)
        da = float((pd_ == td).mean())
        mask = td != FLAT_CLASS
        nf_c += int(((pd_ == td) & mask).sum())
        nf_t += int(mask.sum())
        maes.append(mae)
        by_h[str(h)] = {
            "nonflat_accuracy": nf,
            "direction_accuracy": da,
            "return_mae": mae,
        }
    return {
        "nonflat_accuracy_overall": (nf_c / nf_t) if nf_t else 0.0,
        "return_mae_overall": float(np.mean(maes)),
        "by_horizon": by_h,
    }


def dual_bar_decision(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    min_nonflat_gain: float = 0.005,
    min_gated_prec_gain: float = 0.01,
    max_horizon_mae_rel_worsen: float = 0.05,
    max_ungated_nonflat_collapse: float = 0.03,
    gate_coverage_min: float = 0.20,
    gate_coverage_max: float = 0.40,
) -> dict[str, Any]:
    """Apply Phase-3 dual promotion bar.

    Direction win: overall nonflat +min_nonflat_gain OR h1 gated precision /
    gated_nonflat +min_gated_prec_gain **only when candidate coverage is inside
    [gate_coverage_min, gate_coverage_max]** (default 20–40%).
    MAE win: overall MAE strictly lower AND no horizon MAE worsens by more
    than max_horizon_mae_rel_worsen relative.
    Promote if direction win and MAE win/non-regression, OR MAE win with
    direction nonflat collapse <= max_ungated_nonflat_collapse.
    """
    c_nf = float(candidate["nonflat_accuracy_overall"])
    b_nf = float(baseline["nonflat_accuracy_overall"])
    c_mae = float(candidate["return_mae_overall"])
    b_mae = float(baseline["return_mae_overall"])

    dir_via_nonflat = (c_nf - b_nf) >= min_nonflat_gain
    dir_via_gate = False
    gate_in_band = False
    c_cov = candidate.get("h1_gated_coverage")
    if c_cov is not None:
        c_cov_f = float(c_cov)
        gate_in_band = gate_coverage_min - 1e-12 <= c_cov_f <= gate_coverage_max + 1e-12
    if (
        gate_in_band
        and candidate.get("h1_gated_precision") is not None
        and baseline.get("h1_gated_precision") is not None
    ):
        dir_via_gate = (
            float(candidate["h1_gated_precision"])
            - float(baseline["h1_gated_precision"])
        ) >= min_gated_prec_gain
        # also allow gated_nonflat path (same band requirement)
        if candidate.get("h1_gated_nonflat") is not None and baseline.get(
            "h1_gated_nonflat"
        ) is not None:
            dir_via_gate = dir_via_gate or (
                float(candidate["h1_gated_nonflat"])
                - float(baseline["h1_gated_nonflat"])
            ) >= min_gated_prec_gain
    direction_win = dir_via_nonflat or dir_via_gate

    mae_overall_better = c_mae < b_mae - 1e-12
    horizon_ok = True
    rel_worsen: dict[str, float] = {}
    for h, b_row in baseline["by_horizon"].items():
        b_h = float(b_row["return_mae"])
        c_h = float(candidate["by_horizon"][h]["return_mae"])
        if b_h <= 0:
            rel = 0.0 if c_h <= b_h else 1.0
        else:
            rel = (c_h - b_h) / b_h
        rel_worsen[h] = rel
        if rel > max_horizon_mae_rel_worsen + 1e-12:
            horizon_ok = False
    mae_win = mae_overall_better and horizon_ok
    mae_nonregress = (c_mae <= b_mae + 1e-12) and horizon_ok

    nonflat_collapse = b_nf - c_nf
    direction_nonregress = nonflat_collapse <= max_ungated_nonflat_collapse + 1e-12

    # Dual bar: need both families OK
    promote = False
    reason = ""
    if direction_win and mae_win:
        promote = True
        reason = "direction_win_and_mae_win"
    elif direction_win and mae_nonregress:
        promote = True
        reason = "direction_win_mae_nonregress"
    elif mae_win and direction_nonregress:
        promote = True
        reason = "mae_win_direction_nonregress"
    else:
        reason = "dual_bar_not_met"

    return {
        "promote": promote,
        "reason": reason,
        "direction_win": direction_win,
        "dir_via_nonflat": dir_via_nonflat,
        "dir_via_gate": dir_via_gate,
        "gate_in_band": gate_in_band,
        "gate_coverage_band": [gate_coverage_min, gate_coverage_max],
        "mae_win": mae_win,
        "mae_nonregress": mae_nonregress,
        "direction_nonregress": direction_nonregress,
        "nonflat_delta": c_nf - b_nf,
        "mae_delta": c_mae - b_mae,
        "horizon_mae_rel_worsen": rel_worsen,
    }


def blend_returns(
    rets_a: np.ndarray,
    rets_b: np.ndarray,
    weight_a: float = 0.5,
) -> np.ndarray:
    """Convex blend of two return predictions."""
    w = float(weight_a)
    if not 0.0 <= w <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    a = np.asarray(rets_a, dtype=np.float64)
    b = np.asarray(rets_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("return arrays must match")
    return w * a + (1.0 - w) * b


def blend_direction_logits(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    weight_a: float = 0.5,
) -> np.ndarray:
    """Convex blend of direction logits then hard-argmax labels.

    Returns hard class ids from blended logits (same shape as argmax over classes).
    """
    w = float(weight_a)
    if not 0.0 <= w <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    a = np.asarray(logits_a, dtype=np.float64)
    b = np.asarray(logits_b, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("logits must match and end with dim 3")
    blended = w * a + (1.0 - w) * b
    return blended.argmax(axis=-1).astype(np.int64), blended

def pick_per_horizon(
    per_model: dict[str, dict[int, dict[str, np.ndarray]]],
    *,
    direction_selector: str = "nonflat",
    return_selector: str = "mae",
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Pick direction model and return model independently per horizon.

    per_model[name][h] has keys pred_dir, pred_ret, t_dir, t_ret, logits (optional).
    """
    mapping_dir: dict[int, str] = {}
    mapping_ret: dict[int, str] = {}
    pred_dir: dict[int, np.ndarray] = {}
    pred_ret: dict[int, np.ndarray] = {}
    t_dir: dict[int, np.ndarray] = {}
    t_ret: dict[int, np.ndarray] = {}
    logits: dict[int, np.ndarray] = {}

    for h in horizons:
        best_nf = -1.0
        best_nf_name = None
        best_mae = float("inf")
        best_mae_name = None
        for name, by_h in per_model.items():
            td = by_h[h]["t_dir"]
            pd_ = by_h[h]["pred_dir"]
            pr = by_h[h]["pred_ret"]
            tr = by_h[h]["t_ret"]
            nf = nonflat_accuracy(pd_, td)
            mae = return_mae(pr, tr)
            # direction leader: max nonflat, MAE as tie-break
            if best_nf_name is None or nf > best_nf + 1e-12 or (
                abs(nf - best_nf) <= 1e-12 and mae < return_mae(
                    per_model[best_nf_name][h]["pred_ret"],
                    per_model[best_nf_name][h]["t_ret"],
                )
            ):
                best_nf = nf
                best_nf_name = name
            # return leader: min MAE, nonflat as tie-break
            if best_mae_name is None or mae < best_mae - 1e-12 or (
                abs(mae - best_mae) <= 1e-12
                and nf
                > nonflat_accuracy(
                    per_model[best_mae_name][h]["pred_dir"],
                    per_model[best_mae_name][h]["t_dir"],
                )
            ):
                best_mae = mae
                best_mae_name = name

        dir_name = best_nf_name if direction_selector == "nonflat" else best_mae_name
        ret_name = best_mae_name if return_selector == "mae" else best_nf_name
        assert dir_name is not None and ret_name is not None
        mapping_dir[h] = dir_name
        mapping_ret[h] = ret_name
        pred_dir[h] = per_model[dir_name][h]["pred_dir"]
        pred_ret[h] = per_model[ret_name][h]["pred_ret"]
        t_dir[h] = per_model[dir_name][h]["t_dir"]
        t_ret[h] = per_model[dir_name][h]["t_ret"]
        if "logits" in per_model[dir_name][h]:
            logits[h] = per_model[dir_name][h]["logits"]

    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, horizons)
    return {
        "mapping_direction": {str(k): v for k, v in mapping_dir.items()},
        "mapping_return": {str(k): v for k, v in mapping_ret.items()},
        "summary": summary,
        "pred_dir": pred_dir,
        "pred_ret": pred_ret,
        "t_dir": t_dir,
        "t_ret": t_ret,
        "logits": logits,
    }


def h1_gate_metrics(
    logits: np.ndarray,
    pred_ret: np.ndarray,
    t_dir: np.ndarray,
    *,
    conf_thr: float = 0.45,
    min_abs_return: float = 0.003,
    conf_key: str = "actionable_score",
) -> dict[str, float]:
    conf = direction_confidence_from_logits(logits)
    hard = conf["hard_pred"]
    score = conf[conf_key]
    gated = apply_consistency_and_magnitude_gate(
        hard,
        score,
        pred_ret,
        confidence_threshold=conf_thr,
        min_abs_return=min_abs_return,
        require_sign_agree=False,
    )
    m = gated_actionable_metrics(gated, t_dir)
    return {
        "coverage": m["coverage"],
        "precision_on_calls": m["precision_on_calls"],
        "gated_nonflat_acc": m["gated_nonflat_acc"],
        "n_calls": m["n_calls"],
    }
