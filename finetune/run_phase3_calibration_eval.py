"""Phase-3: calibrate blended returns on pre-test windows; score on test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from calibrate_returns import fit_and_apply_horizons
from dual_metric_compare import (
    blend_returns,
    dual_bar_decision,
    h1_gate_metrics,
    summarize_horizons,
)
from evaluate_gated_ensemble import (
    DEVICE,
    FEATURES,
    HORIZONS,
    LOOKBACK,
    VAL_END,
    WINDOW,
    derive_time_features,
    load_csv,
    load_model,
    normalize_window,
)
from multihorizon_objective import make_multihorizon_targets
from selective_prediction import direction_confidence_from_logits

VAL_START = "2025-06-01"


def make_windows(df: pd.DataFrame, min_end: str) -> list[dict]:
    min_cut = pd.Timestamp(min_end)
    val_cut = pd.Timestamp(VAL_END)
    out = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        ced = df["timestamps"].iloc[start + LOOKBACK - 1]
        if ced <= min_cut:
            continue
        window = df.iloc[start : start + WINDOW].copy()
        out.append(
            {
                "context_end_date": str(ced.date()),
                "features": window[FEATURES].to_numpy(dtype=np.float32),
                "raw_close": window["close"].to_numpy(dtype=np.float32),
                "timestamps": window["timestamps"],
                "is_test": bool(ced > val_cut),
            }
        )
    return out


@torch.no_grad()
def predict_all(mod: dict, windows: list[dict]) -> dict:
    rows = {
        h: {
            "pred_dir": [],
            "pred_ret": [],
            "t_dir": [],
            "t_ret": [],
            "logits": [],
            "is_test": [],
        }
        for h in HORIZONS
    }
    for w in windows:
        x = torch.from_numpy(normalize_window(w["features"])).unsqueeze(0).to(DEVICE)
        stamp = torch.from_numpy(derive_time_features(w["timestamps"])).unsqueeze(0).to(
            DEVICE
        )
        raw = torch.from_numpy(w["raw_close"]).unsqueeze(0).to(DEVICE)
        t0, t1 = mod["tokenizer"].encode(x, half=True)
        _, _, hidden = mod["model"](
            t0[:, :-1], t1[:, :-1], stamp[:, :-1, :], return_context=True
        )
        out = mod["head"](hidden, context_length=LOOKBACK)
        logits = out["direction_logits"][0].cpu().numpy()
        pret = out["return_prediction"][0].cpu().numpy()
        conf = direction_confidence_from_logits(logits)
        tgt = make_multihorizon_targets(
            raw,
            context_length=LOOKBACK,
            horizons=mod["horizons"],
            min_deadzone=mod["min_deadzone"],
            volatility_multiplier=mod["vol_mult"],
        )
        for hi, hh in enumerate(HORIZONS):
            rows[hh]["pred_dir"].append(int(conf["hard_pred"][hi]))
            rows[hh]["pred_ret"].append(float(pret[hi]))
            rows[hh]["t_dir"].append(int(tgt["direction"][0, hi].item()))
            rows[hh]["t_ret"].append(float(tgt["returns"][0, hi].item()))
            rows[hh]["logits"].append(logits[hi])
            rows[hh]["is_test"].append(w["is_test"])
    for hh in HORIZONS:
        for k in rows[hh]:
            if k == "logits":
                rows[hh][k] = np.stack(rows[hh][k])
            else:
                rows[hh][k] = np.asarray(rows[hh][k])
    return rows


def slice_mask(d: dict, mask: np.ndarray) -> dict:
    return {h: d[h][mask] for h in HORIZONS}


def attach_gate(summary: dict, logits: np.ndarray, pret: np.ndarray, t_dir: np.ndarray) -> dict:
    g = h1_gate_metrics(
        logits, pret, t_dir, conf_thr=0.45, min_abs_return=0.003
    )
    summary = dict(summary)
    summary["h1_gated_precision"] = g["precision_on_calls"]
    summary["h1_gated_nonflat"] = g["gated_nonflat_acc"]
    summary["h1_gated_coverage"] = g["coverage"]
    summary["h1_gate"] = g
    return summary


def main() -> int:
    df = load_csv("688169")
    windows = make_windows(df, VAL_START)
    n_test = sum(1 for w in windows if w["is_test"])
    n_fit = len(windows) - n_test
    print(f"windows={len(windows)} fit={n_fit} test={n_test}")

    r5 = load_model(
        ROOT
        / "outputs/models/a_share_multihorizon_predictor_r5_frozen_pool48/checkpoints/best_model"
    )
    r10 = load_model(
        ROOT
        / "outputs/models/a_share_multihorizon_predictor_r10_joint_splitlr/checkpoints/best_model"
    )
    print("predict r5...")
    p5 = predict_all(r5, windows)
    print("predict r10...")
    p10 = predict_all(r10, windows)

    is_test = p10[1]["is_test"].astype(bool)
    is_fit = ~is_test

    # historical baseline mapping
    base_dir = {
        1: p10[1]["pred_dir"],
        3: p5[3]["pred_dir"],
        5: p5[5]["pred_dir"],
        10: p5[10]["pred_dir"],
    }
    base_ret = {
        1: p10[1]["pred_ret"],
        3: p5[3]["pred_ret"],
        5: p5[5]["pred_ret"],
        10: p5[10]["pred_ret"],
    }
    # phase3 direction: nf-best (R10 ties take MAE) + R5@h5
    prom_dir = {
        1: p10[1]["pred_dir"],
        3: p10[3]["pred_dir"],
        5: p5[5]["pred_dir"],
        10: p10[10]["pred_dir"],
    }
    t_dir = {h: p10[h]["t_dir"] for h in HORIZONS}
    t_ret = {h: p10[h]["t_ret"] for h in HORIZONS}

    blend = {
        h: blend_returns(p10[h]["pred_ret"], p5[h]["pred_ret"], 0.85) for h in HORIZONS
    }
    calibrated, params = fit_and_apply_horizons(
        blend, t_ret, HORIZONS, fit_mask=is_fit
    )
    print("calibration params:", params)

    base = attach_gate(
        summarize_horizons(
            slice_mask(base_dir, is_test),
            slice_mask(base_ret, is_test),
            slice_mask(t_dir, is_test),
            slice_mask(t_ret, is_test),
            tuple(HORIZONS),
        ),
        p10[1]["logits"][is_test],
        base_ret[1][is_test],
        t_dir[1][is_test],
    )
    prom = attach_gate(
        summarize_horizons(
            slice_mask(prom_dir, is_test),
            slice_mask(blend, is_test),
            slice_mask(t_dir, is_test),
            slice_mask(t_ret, is_test),
            tuple(HORIZONS),
        ),
        p10[1]["logits"][is_test],
        blend[1][is_test],
        t_dir[1][is_test],
    )
    prom_cal = attach_gate(
        summarize_horizons(
            slice_mask(prom_dir, is_test),
            slice_mask(calibrated, is_test),
            slice_mask(t_dir, is_test),
            slice_mask(t_ret, is_test),
            tuple(HORIZONS),
        ),
        p10[1]["logits"][is_test],
        calibrated[1][is_test],
        t_dir[1][is_test],
    )

    d_blend = dual_bar_decision(prom, base)
    d_cal = dual_bar_decision(prom_cal, base)

    def show(name: str, s: dict, d: dict) -> None:
        print(
            f"{name}: nf={s['nonflat_accuracy_overall']:.2%} mae={s['return_mae_overall']:.5f} "
            f"g_prec={s['h1_gated_precision']:.2%} promote={d['promote']} ({d['reason']})"
        )
        for h in HORIZONS:
            row = s["by_horizon"][str(h)]
            print(
                f"  h={h}: nf={row['nonflat_accuracy']:.2%} mae={row['return_mae']:.5f}"
            )

    show("baseline", base, {"promote": False, "reason": "baseline"})
    show("blend085", prom, d_blend)
    show("blend085_cal", prom_cal, d_cal)

    out = {
        "n_fit": int(is_fit.sum()),
        "n_test": int(is_test.sum()),
        "calibration_params": params,
        "baseline": base,
        "blend_r10w0.85": prom,
        "blend_r10w0.85_calibrated": prom_cal,
        "decision_blend": d_blend,
        "decision_calibrated": d_cal,
        "promoted_name": (
            "blend_r10w0.85_calibrated"
            if d_cal["promote"]
            else ("blend_r10w0.85" if d_blend["promote"] else None)
        ),
    }
    path = ROOT / "outputs" / "eval_dual_phase3_calibrated.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
