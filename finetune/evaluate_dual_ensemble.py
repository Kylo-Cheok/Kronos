"""Phase-3 dual-metric ensemble evaluation on 688169.

Compares ensemble strategies that can improve direction and/or return MAE:
  - historical R10@h1 + R5@h3/5/10
  - auto best-nonflat per horizon
  - auto best-MAE per horizon
  - decoupled: best-nonflat direction + best-MAE return
  - return blend of R5 and R10 with best-nonflat direction
  - Phase-2 promoted gate metrics on h=1

Usage::
    python finetune/evaluate_dual_ensemble.py --output outputs/eval_dual_phase3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import (  # noqa: E402
    blend_returns,
    dual_bar_decision,
    h1_gate_metrics,
    pick_per_horizon,
    return_mae,
    summarize_horizons,
)
from evaluate_gated_ensemble import (  # noqa: E402
    HORIZONS,
    load_csv,
    load_model,
    make_windows,
    predict_window,
)
from multihorizon_objective import FLAT_CLASS  # noqa: E402
from selective_prediction import direction_confidence_from_logits, nonflat_accuracy  # noqa: E402

DEFAULT_CANDIDATES = {
    "r5_frozen_pool48": ROOT
    / "outputs"
    / "models"
    / "a_share_multihorizon_predictor_r5_frozen_pool48"
    / "checkpoints"
    / "best_model",
    "r10_joint_splitlr": ROOT
    / "outputs"
    / "models"
    / "a_share_multihorizon_predictor_r10_joint_splitlr"
    / "checkpoints"
    / "best_model",
    "r9_joint_pool48": ROOT
    / "outputs"
    / "models"
    / "a_share_multihorizon_predictor_r9_joint_pool48"
    / "checkpoints"
    / "best_model",
    "r4_joint_lr2e6": ROOT
    / "outputs"
    / "models"
    / "a_share_multihorizon_predictor_r4_joint_lr2e6"
    / "checkpoints"
    / "best_model",
}

HISTORICAL = {1: "r10_joint_splitlr", 3: "r5_frozen_pool48", 5: "r5_frozen_pool48", 10: "r5_frozen_pool48"}
PHASE2_GATE = {"conf_thr": 0.45, "min_abs_return": 0.003}


def collect(candidates: dict[str, Path], symbol: str = "688169") -> dict:
    df = load_csv(symbol)
    windows = make_windows(df)
    print(f"Loaded {len(windows)} windows for {symbol}")
    loaded = {}
    for name, path in candidates.items():
        if not path.exists():
            print(f"SKIP {name}")
            continue
        print(f"Loading {name}...")
        loaded[name] = load_model(path)
    if not loaded:
        raise FileNotFoundError("no models")

    per_model: dict = {
        name: {
            h: {
                "pred_dir": [],
                "pred_ret": [],
                "t_dir": [],
                "t_ret": [],
                "logits": [],
            }
            for h in HORIZONS
        }
        for name in loaded
    }
    for w in windows:
        for name, mod in loaded.items():
            pred = predict_window(mod, w)
            logits = pred["logits"]
            conf = direction_confidence_from_logits(logits)
            for hi, h in enumerate(HORIZONS):
                per_model[name][h]["pred_dir"].append(int(conf["hard_pred"][hi]))
                per_model[name][h]["pred_ret"].append(float(pred["pred_return"][hi]))
                per_model[name][h]["t_dir"].append(int(pred["target_direction"][hi]))
                per_model[name][h]["t_ret"].append(float(pred["target_return"][hi]))
                per_model[name][h]["logits"].append(logits[hi])

    for name in loaded:
        for h in HORIZONS:
            for k in ("pred_dir", "pred_ret", "t_dir", "t_ret"):
                per_model[name][h][k] = np.asarray(per_model[name][h][k])
            per_model[name][h]["logits"] = np.stack(per_model[name][h]["logits"], axis=0)
    return per_model


def fixed_mapping_summary(per_model: dict, mapping: dict[int, str]) -> dict:
    pred_dir, pred_ret, t_dir, t_ret, logits = {}, {}, {}, {}, {}
    for h, name in mapping.items():
        pred_dir[h] = per_model[name][h]["pred_dir"]
        pred_ret[h] = per_model[name][h]["pred_ret"]
        t_dir[h] = per_model[name][h]["t_dir"]
        t_ret[h] = per_model[name][h]["t_ret"]
        logits[h] = per_model[name][h]["logits"]
    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))
    gate = h1_gate_metrics(
        logits[1],
        pred_ret[1],
        t_dir[1],
        conf_thr=PHASE2_GATE["conf_thr"],
        min_abs_return=PHASE2_GATE["min_abs_return"],
    )
    summary["h1_gated_precision"] = gate["precision_on_calls"]
    summary["h1_gated_nonflat"] = gate["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = gate["coverage"]
    summary["h1_gate"] = gate
    summary["mapping"] = {str(k): v for k, v in mapping.items()}
    return summary


def blend_strategy(per_model: dict, weight_r10: float) -> dict:
    """Direction from best-nonflat pick; returns = blend R5/R10 when both exist."""
    picked = pick_per_horizon(per_model, direction_selector="nonflat", return_selector="mae")
    pred_dir = picked["pred_dir"]
    t_dir = picked["t_dir"]
    t_ret = picked["t_ret"]
    pred_ret = {}
    for h in HORIZONS:
        if "r10_joint_splitlr" in per_model and "r5_frozen_pool48" in per_model:
            pred_ret[h] = blend_returns(
                per_model["r10_joint_splitlr"][h]["pred_ret"],
                per_model["r5_frozen_pool48"][h]["pred_ret"],
                weight_a=weight_r10,
            )
        else:
            pred_ret[h] = picked["pred_ret"][h]
    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))
    # gate uses direction model logits for h=1
    dir_name = picked["mapping_direction"]["1"]
    gate = h1_gate_metrics(
        per_model[dir_name][1]["logits"],
        pred_ret[1],
        t_dir[1],
        conf_thr=PHASE2_GATE["conf_thr"],
        min_abs_return=PHASE2_GATE["min_abs_return"],
    )
    summary["h1_gated_precision"] = gate["precision_on_calls"]
    summary["h1_gated_nonflat"] = gate["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = gate["coverage"]
    summary["h1_gate"] = gate
    summary["mapping_direction"] = picked["mapping_direction"]
    summary["return_blend_r10_weight"] = weight_r10
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_dual_phase3.json")
    parser.add_argument("--symbol", default="688169")
    parser.add_argument(
        "--candidates",
        default=None,
        help="comma exp names; default R5/R10/R9/R4",
    )
    args = parser.parse_args()

    if args.candidates:
        cands = {}
        for part in args.candidates.split(","):
            part = part.strip()
            if not part:
                continue
            cands[part] = (
                ROOT
                / "outputs"
                / "models"
                / f"a_share_multihorizon_predictor_{part}"
                / "checkpoints"
                / "best_model"
            )
    else:
        cands = DEFAULT_CANDIDATES

    per_model = collect(cands, symbol=args.symbol)

    strategies = {}

    # 1 historical phase-2 mapping
    hist_map = {h: HISTORICAL[h] for h in HORIZONS if HISTORICAL[h] in per_model}
    if len(hist_map) == len(HORIZONS):
        strategies["historical_r10_r5"] = fixed_mapping_summary(per_model, hist_map)

    # 2 auto nonflat (same model for dir+ret)
    auto_nf = {}
    for h in HORIZONS:
        best, best_name = -1.0, None
        for name in per_model:
            nf = nonflat_accuracy(per_model[name][h]["pred_dir"], per_model[name][h]["t_dir"])
            if nf > best:
                best, best_name = nf, name
        auto_nf[h] = best_name  # type: ignore[assignment]
    strategies["auto_nonflat"] = fixed_mapping_summary(per_model, auto_nf)

    # 3 auto MAE
    auto_mae = {}
    for h in HORIZONS:
        best, best_name = float("inf"), None
        for name in per_model:
            mae = return_mae(per_model[name][h]["pred_ret"], per_model[name][h]["t_ret"])
            if mae < best:
                best, best_name = mae, name
        auto_mae[h] = best_name  # type: ignore[assignment]
    strategies["auto_mae"] = fixed_mapping_summary(per_model, auto_mae)

    # 4 decoupled dir nonflat / ret mae
    decoupled = pick_per_horizon(per_model)
    summary = decoupled["summary"]
    dir_h1 = decoupled["mapping_direction"]["1"]
    gate = h1_gate_metrics(
        per_model[dir_h1][1]["logits"],
        decoupled["pred_ret"][1],
        decoupled["t_dir"][1],
        conf_thr=PHASE2_GATE["conf_thr"],
        min_abs_return=PHASE2_GATE["min_abs_return"],
    )
    summary["h1_gated_precision"] = gate["precision_on_calls"]
    summary["h1_gated_nonflat"] = gate["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = gate["coverage"]
    summary["h1_gate"] = gate
    summary["mapping_direction"] = decoupled["mapping_direction"]
    summary["mapping_return"] = decoupled["mapping_return"]
    strategies["decoupled_dir_nf_ret_mae"] = summary

    # 5 return blends
    for w in (0.3, 0.5, 0.7, 0.85, 1.0):
        strategies[f"blend_r10w{w:.2f}"] = blend_strategy(per_model, w)

    # 6 R10 everywhere (joint)
    if "r10_joint_splitlr" in per_model:
        strategies["r10_all"] = fixed_mapping_summary(
            per_model, {h: "r10_joint_splitlr" for h in HORIZONS}
        )
    if "r5_frozen_pool48" in per_model:
        strategies["r5_all"] = fixed_mapping_summary(
            per_model, {h: "r5_frozen_pool48" for h in HORIZONS}
        )

    baseline_name = "historical_r10_r5"
    if baseline_name not in strategies:
        baseline_name = "auto_nonflat"
    baseline = strategies[baseline_name]

    decisions = {}
    print("\n=== Dual-metric strategies vs baseline", baseline_name, "===")
    print(
        f"{'strategy':<32} {'nonflat':>8} {'mae':>8} {'g_prec':>8} {'g_nf':>8} {'promote':>8} reason"
    )
    best_promote = None
    for name, s in strategies.items():
        d = dual_bar_decision(s, baseline)
        decisions[name] = d
        print(
            f"{name:<32} {s['nonflat_accuracy_overall']:>8.2%} {s['return_mae_overall']:>8.4f} "
            f"{s.get('h1_gated_precision', 0):>8.2%} {s.get('h1_gated_nonflat', 0):>8.2%} "
            f"{str(d['promote']):>8} {d['reason']}"
        )
        if d["promote"]:
            score = (
                (s["nonflat_accuracy_overall"] - baseline["nonflat_accuracy_overall"]) * 2
                + (baseline["return_mae_overall"] - s["return_mae_overall"]) * 20
                + (s.get("h1_gated_precision", 0) - baseline.get("h1_gated_precision", 0))
            )
            if best_promote is None or score > best_promote[0]:
                best_promote = (score, name, s, d)

    out = {
        "symbol": args.symbol,
        "baseline_name": baseline_name,
        "baseline": baseline,
        "strategies": strategies,
        "decisions": decisions,
        "promoted": None
        if best_promote is None
        else {
            "name": best_promote[1],
            "summary": best_promote[2],
            "decision": best_promote[3],
            "score": best_promote[0],
        },
        "phase2_gate": PHASE2_GATE,
        "n_windows": int(len(next(iter(per_model.values()))[1]["t_dir"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPromoted: {out['promoted']['name'] if out['promoted'] else None}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
