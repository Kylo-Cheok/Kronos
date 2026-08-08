"""Phase 6 — Round P6-1: evaluate the FULL model zoo per-horizon.

For every candidate checkpoint we:
  * search the 9 TTA lookback configs (P5-1 style) and keep the best
    direction (max ungated nonflat) per (model, horizon);
  * pick the best DIRECTION model per horizon;
  * build a candidate config: direction = best-model-per-horizon (TTA),
    returns = promoted R10/R5 blend (kept identical to baseline so the
    dual-bar isolates the *direction* gain and MAE cannot regress);
  * apply the full P5-8 h=1 gate (conf + mag=0.002 + strict_h5 consistency)
    and compare against the P5-8 baseline (r10/r5 + P5-2 per-h lookbacks +
    P5-8 gate) computed from the SAME cache.

Output: outputs/eval_p6_r1_zoo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from selective_prediction import (  # noqa: E402
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)

# 9 TTA lookback configs (P5-1).
LOOKBACK_CONFIGS = [
    ("3lb_tight", (126, 128, 130)),
    ("4lb_left", (124, 126, 128, 130)),
    ("4lb_right", (126, 128, 130, 132)),
    ("5lb_current", (124, 126, 128, 130, 132)),
    ("5lb_left", (122, 124, 126, 128, 130)),
    ("5lb_right", (126, 128, 130, 132, 134)),
    ("6lb", (124, 126, 128, 130, 132, 134)),
    ("7lb_full", (122, 124, 126, 128, 130, 132, 134)),
    ("single_128", (128,)),
]
LB_NAME_TO_TUPLE = dict(LOOKBACK_CONFIGS)

# Promoted return blend (P5-5).
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
PRIMARY_W_BY_H = {1: 0.925, 3: 0.775, 5: 1.0, 10: 0.65}

# P5-2 per-horizon TTA lookbacks (P5-8 baseline direction).
P5_TTA_BY_H = {
    1: (124, 126, 128, 130),
    3: (124, 126, 128, 130),
    5: (124, 126, 128, 130, 132, 134),
    10: (126, 128, 130, 132),
}
P5_BASE_MAP = {1: "r10_joint_splitlr", 3: "r5_frozen_pool48",
               5: "r5_frozen_pool48", 10: "r10_joint_splitlr"}

GATE_THR_GRID = [0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]
GATE_MAG = 0.002


def tta_logits(store, model, h, lookbacks):
    lb_idx = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [store[model][h]["per_lb_logits"][i] for i in lb_idx]
    return average_logits(per_lb, None)


def gate_sweep(dir_by_h, score_by_h, gate_ret, t_dir_1, variant_consistency=True):
    """Sweep thr + strict_h5 consistency; return (best_row, all_rows)."""
    h1_hard = dir_by_h[1]
    h1_score = score_by_h[1]
    h5_hard = dir_by_h[5]
    rows = []
    for thr in GATE_THR_GRID:
        base = apply_consistency_and_magnitude_gate(
            h1_hard, h1_score, gate_ret,
            confidence_threshold=thr, min_abs_return=GATE_MAG,
            require_sign_agree=False,
        )
        gated = apply_consistency_filter(base, h1_hard, h5_hard, "strict") \
            if variant_consistency else base
        m = gated_actionable_metrics(gated, t_dir_1)
        if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
            continue
        if m["n_calls"] < 8:
            continue
        rows.append({"thr": thr, "gated": gated, **m})
    if not rows:
        return None, []
    rows.sort(key=lambda r: (r["precision_on_calls"], r["gated_nonflat_acc"]), reverse=True)
    return rows[0], rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-lb", default=str(ROOT / "outputs" / "p6_per_lb_store.npz"))
    ap.add_argument("--sl", default=str(ROOT / "outputs" / "p6_sl_store.npz"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "eval_p6_r1_zoo.json"))
    args = ap.parse_args()

    tta = load_per_lb_store(Path(args.per_lb), ALL_LOOKBACKS)
    sl = load_sl_store(Path(args.sl))
    models = sorted(tta.keys())
    n = len(tta[models[0]][1]["td"])
    print(f"Loaded {len(models)} models, {n} windows")

    # ---- 1. Per-model per-horizon best TTA direction ----
    print("\n=== Per-model per-horizon best-TTA nonflat ===")
    best_tta: dict[str, dict] = {}
    zoo_table: dict[str, dict] = {str(h): {} for h in HORIZONS}
    for m in models:
        best_tta[m] = {}
        for h in HORIZONS:
            td = tta[m][h]["td"]
            best_nf = -1.0
            best_cfg = None
            best_logits = None
            for cfg_name, lbs in LOOKBACK_CONFIGS:
                avg = tta_logits(tta, m, h, lbs)
                conf = direction_confidence_from_logits(avg)
                nf = nonflat_accuracy(conf["hard_pred"], td)
                if nf > best_nf:
                    best_nf = nf
                    best_cfg = cfg_name
                    best_logits = avg
            conf = direction_confidence_from_logits(best_logits)
            best_tta[m][h] = {
                "config": best_cfg, "nonflat": best_nf, "logits": best_logits,
                "hard": conf["hard_pred"], "score": conf["actionable_score"], "td": td,
                "tr": tta[m][h]["tr"],
            }
            zoo_table[str(h)][m] = best_nf

    # ---- 2. Pick best DIRECTION model per horizon ----
    mapping_dir: dict[int, str] = {}
    lookbacks_by_h: dict[int, tuple] = {}
    for h in HORIZONS:
        ranked = sorted(zoo_table[str(h)].items(), key=lambda kv: kv[1], reverse=True)
        best_m = ranked[0][0]
        mapping_dir[h] = best_m
        lookbacks_by_h[h] = LB_NAME_TO_TUPLE[best_tta[best_m][h]["config"]]
        print(f"\n h={h}: BEST direction = {best_m} "
              f"(nf={ranked[0][1]:.2%}, cfg={best_tta[best_m][h]['config']})")
        print("   top5: " + ", ".join(f"{nm}={nf:.1%}" for nm, nf in ranked[:5]))

    # ---- 3. Candidate: direction=best-model TTA; returns=promoted blend ----
    pred_dir, t_dir, t_ret, pred_ret = {}, {}, {}, {}
    cand_hard_by_h, cand_score_by_h = {}, {}
    for h in HORIZONS:
        m = mapping_dir[h]
        pred_dir[h] = best_tta[m][h]["hard"]
        t_dir[h] = best_tta[m][h]["td"]
        t_ret[h] = best_tta[m][h]["tr"]
        pred_ret[h] = blend_returns(
            sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"], PRIMARY_W_BY_H[h])
        cand_hard_by_h[h] = best_tta[m][h]["hard"]
        cand_score_by_h[h] = best_tta[m][h]["score"]
    cand_summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)
    print(f"\nCANDIDATE (best-model dir + promoted ret): "
          f"nf={cand_summary['nonflat_accuracy_overall']:.2%} "
          f"mae={cand_summary['return_mae_overall']:.6f}")

    # ---- 4. Baseline P5-8: r10/r5 + P5-2 lookbacks + P5-8 gate ----
    bp_dir, bt_dir, bt_ret, bpred_ret = {}, {}, {}, {}
    base_score_by_h = {}
    for h in HORIZONS:
        m = P5_BASE_MAP[h]
        avg = tta_logits(tta, m, h, P5_TTA_BY_H[h])
        conf = direction_confidence_from_logits(avg)
        bp_dir[h] = conf["hard_pred"]
        base_score_by_h[h] = conf["actionable_score"]
        bt_dir[h] = tta[m][h]["td"]
        bt_ret[h] = tta[m][h]["tr"]
        bpred_ret[h] = blend_returns(
            sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"], PRIMARY_W_BY_H[h])
    base_summary = summarize_horizons(bp_dir, bpred_ret, bt_dir, bt_ret, HORIZONS)
    print(f"BASELINE P5-8: nf={base_summary['nonflat_accuracy_overall']:.2%} "
          f"mae={base_summary['return_mae_overall']:.6f}")

    # ---- 5. h=1 gate ----
    cand_best, _ = gate_sweep(cand_hard_by_h, cand_score_by_h,
                              sl[PRIMARY][1]["pret"], t_dir[1], True)
    if cand_best:
        cm = gated_actionable_metrics(cand_best["gated"], t_dir[1])
        cand_summary["h1_gated_precision"] = cm["precision_on_calls"]
        cand_summary["h1_gated_nonflat"] = cm["gated_nonflat_acc"]
        cand_summary["h1_gated_coverage"] = cm["coverage"]
        cand_summary["h1_gate"] = {"thr": cand_best["thr"], "mag": GATE_MAG,
                                   "consistency": "strict_h5"}
        cand_bt = absolute_direction_backtest(cand_best["gated"], t_ret[1], transaction_cost=0.0005)
        cand_summary["h1_gated_backtest"] = cand_bt
        print(f"CANDIDATE gated: thr={cand_best['thr']:.2f} cov={cm['coverage']:.2%} "
              f"prec={cm['precision_on_calls']:.2%} gnf={cm['gated_nonflat_acc']:.2%} "
              f"bt_ret={cand_bt['total_return']:.2%}")

    base_gated = apply_consistency_and_magnitude_gate(
        bp_dir[1], base_score_by_h[1], sl[PRIMARY][1]["pret"],
        confidence_threshold=0.45, min_abs_return=GATE_MAG, require_sign_agree=False)
    base_gated = apply_consistency_filter(base_gated, bp_dir[1], bp_dir[5], "strict")
    bm = gated_actionable_metrics(base_gated, bt_dir[1])
    base_summary["h1_gated_precision"] = bm["precision_on_calls"]
    base_summary["h1_gated_nonflat"] = bm["gated_nonflat_acc"]
    base_summary["h1_gated_coverage"] = bm["coverage"]
    base_summary["h1_gate"] = {"thr": 0.45, "mag": GATE_MAG, "consistency": "strict_h5"}
    base_bt = absolute_direction_backtest(base_gated, bt_ret[1], transaction_cost=0.0005)
    base_summary["h1_gated_backtest"] = base_bt
    print(f"BASELINE gated: thr=0.45 cov={bm['coverage']:.2%} "
          f"prec={bm['precision_on_calls']:.2%} gnf={bm['gated_nonflat_acc']:.2%} "
          f"bt_ret={base_bt['total_return']:.2%}")

    # ---- 6. Dual-bar decision ----
    decision = dual_bar_decision(cand_summary, base_summary)
    print(f"\nDECISION: promote={decision['promote']} reason={decision['reason']} "
          f"d_nf={decision['nonflat_delta']:+.4f} d_mae={decision['mae_delta']:+.6f}")

    result = {
        "symbol": "688169",
        "n_windows": n,
        "models_evaluated": models,
        "zoo_table_best_tta_nonflat_by_horizon": zoo_table,
        "best_direction_mapping": {str(h): mapping_dir[h] for h in HORIZONS},
        "best_direction_lookbacks": {str(h): list(lookbacks_by_h[h]) for h in HORIZONS},
        "candidate_summary": cand_summary,
        "baseline_summary": base_summary,
        "decision_vs_p5_8": decision,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
