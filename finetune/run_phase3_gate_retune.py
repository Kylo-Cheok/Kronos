"""Retune h=1 gate thresholds on Phase-3 blend ensemble; dual-metric aware."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons
from evaluate_gated_ensemble import HORIZONS, load_csv, load_model, make_windows, predict_window
from selective_prediction import (
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)


def main() -> int:
    cands = {
        "r5": ROOT
        / "outputs/models/a_share_multihorizon_predictor_r5_frozen_pool48/checkpoints/best_model",
        "r10": ROOT
        / "outputs/models/a_share_multihorizon_predictor_r10_joint_splitlr/checkpoints/best_model",
    }
    loaded = {k: load_model(v) for k, v in cands.items()}
    windows = make_windows(load_csv("688169"))
    store = {
        name: {h: {"logits": [], "pret": [], "td": [], "tr": []} for h in HORIZONS}
        for name in loaded
    }
    for w in windows:
        for name, mod in loaded.items():
            pred = predict_window(mod, w)
            conf = direction_confidence_from_logits(pred["logits"])
            for hi, h in enumerate(HORIZONS):
                store[name][h]["logits"].append(pred["logits"][hi])
                store[name][h]["pret"].append(float(pred["pred_return"][hi]))
                store[name][h]["td"].append(int(pred["target_direction"][hi]))
                store[name][h]["tr"].append(float(pred["target_return"][hi]))
    for name in loaded:
        for h in HORIZONS:
            store[name][h]["logits"] = np.stack(store[name][h]["logits"])
            for k in ("pret", "td", "tr"):
                store[name][h][k] = np.asarray(store[name][h][k])

    # Phase-3 direction + blend returns
    dir_src = {1: "r10", 3: "r10", 5: "r5", 10: "r10"}
    pred_dir = {
        h: direction_confidence_from_logits(store[dir_src[h]][h]["logits"])["hard_pred"]
        for h in HORIZONS
    }
    pred_ret = {
        h: blend_returns(store["r10"][h]["pret"], store["r5"][h]["pret"], 0.85)
        for h in HORIZONS
    }
    t_dir = {h: store["r10"][h]["td"] for h in HORIZONS}
    t_ret = {h: store["r10"][h]["tr"] for h in HORIZONS}
    summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, tuple(HORIZONS))

    logits1 = store["r10"][1]["logits"]
    conf = direction_confidence_from_logits(logits1)
    hard = conf["hard_pred"]
    score = conf["actionable_score"]
    pret1 = pred_ret[1]
    td1 = t_dir[1]

    baseline_gate = gated_actionable_metrics(
        apply_consistency_and_magnitude_gate(
            hard, score, pret1, confidence_threshold=0.45, min_abs_return=0.003
        ),
        td1,
    )
    print(
        f"Phase2 gate: cov={baseline_gate['coverage']:.1%} "
        f"prec={baseline_gate['precision_on_calls']:.1%} "
        f"gnf={baseline_gate['gated_nonflat_acc']:.1%}"
    )

    rows = []
    for thr in [0.40, 0.42, 0.45, 0.48, 0.50, 0.52]:
        for mag in [0.0, 0.002, 0.003, 0.004, 0.005, 0.006]:
            gated = apply_consistency_and_magnitude_gate(
                hard,
                score,
                pret1,
                confidence_threshold=thr,
                min_abs_return=mag,
            )
            m = gated_actionable_metrics(gated, td1)
            if not (0.15 <= m["coverage"] <= 0.45) or m["n_calls"] < 8:
                continue
            rows.append(
                {
                    "thr": thr,
                    "mag": mag,
                    "coverage": m["coverage"],
                    "precision": m["precision_on_calls"],
                    "gated_nf": m["gated_nonflat_acc"],
                    "n_calls": m["n_calls"],
                }
            )
    rows.sort(key=lambda r: (r["precision"], r["gated_nf"], r["coverage"]), reverse=True)
    print("Top gate configs (15-45% coverage):")
    for r in rows[:12]:
        print(
            f"  thr={r['thr']} mag={r['mag']}: cov={r['coverage']:.1%} "
            f"prec={r['precision']:.1%} gnf={r['gated_nf']:.1%} n={r['n_calls']:.0f}"
        )

    best = rows[0] if rows else None
    base_summary = dict(summary)
    base_summary["h1_gated_precision"] = baseline_gate["precision_on_calls"]
    base_summary["h1_gated_nonflat"] = baseline_gate["gated_nonflat_acc"]
    cand_summary = dict(summary)
    if best:
        cand_summary["h1_gated_precision"] = best["precision"]
        cand_summary["h1_gated_nonflat"] = best["gated_nf"]
        cand_summary["h1_gated_coverage"] = best["coverage"]
    decision = dual_bar_decision(cand_summary, base_summary) if best else None

    out = {
        "ungated": summary,
        "phase2_gate": baseline_gate,
        "top_gates": rows[:15],
        "best_gate": best,
        "dual_decision_vs_phase2_gate": decision,
    }
    path = ROOT / "outputs" / "eval_phase3_gate_retune.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}")
    print("best", best, "decision", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
