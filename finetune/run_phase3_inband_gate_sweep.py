"""Sweep h=1 gates that stay inside the frozen 20-40% coverage band."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns
from evaluate_gated_ensemble import load_csv, load_model, make_windows, predict_window
from promoted_config import model_dir
from selective_prediction import (
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)


def main() -> int:
    r5 = load_model(model_dir("r5_frozen_pool48"))
    r10 = load_model(model_dir("r10_joint_splitlr"))
    windows = make_windows(load_csv("688169"))
    logits = []
    pret_r10 = []
    pret_r5 = []
    td = []
    for w in windows:
        p10 = predict_window(r10, w)
        p5 = predict_window(r5, w)
        logits.append(p10["logits"][0])
        pret_r10.append(float(p10["pred_return"][0]))
        pret_r5.append(float(p5["pred_return"][0]))
        td.append(int(p10["target_direction"][0]))
    logits = np.stack(logits)
    pret_r10 = np.asarray(pret_r10)
    pret_r5 = np.asarray(pret_r5)
    td = np.asarray(td)
    blend = blend_returns(pret_r10, pret_r5, 0.85)
    conf = direction_confidence_from_logits(logits)
    hard = conf["hard_pred"]
    score = conf["actionable_score"]

    base = gated_actionable_metrics(
        apply_consistency_and_magnitude_gate(
            hard, score, pret_r10, confidence_threshold=0.45, min_abs_return=0.003
        ),
        td,
    )
    print(
        "BASE R10 gate thr=0.45 mag=0.003:",
        f"cov={base['coverage']:.1%} prec={base['precision_on_calls']:.1%} "
        f"gnf={base['gated_nonflat_acc']:.1%}",
    )

    def sweep(pret: np.ndarray, label: str) -> list[dict]:
        rows = []
        for thr in np.round(np.linspace(0.35, 0.55, 21), 3):
            for mag in [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]:
                m = gated_actionable_metrics(
                    apply_consistency_and_magnitude_gate(
                        hard,
                        score,
                        pret,
                        confidence_threshold=float(thr),
                        min_abs_return=float(mag),
                    ),
                    td,
                )
                if not (0.20 - 1e-9 <= m["coverage"] <= 0.40 + 1e-9):
                    continue
                if m["n_calls"] < 8:
                    continue
                rows.append(
                    {
                        "source": label,
                        "thr": float(thr),
                        "mag": float(mag),
                        "coverage": m["coverage"],
                        "precision": m["precision_on_calls"],
                        "gated_nf": m["gated_nonflat_acc"],
                        "n_calls": m["n_calls"],
                        "d_prec_pt": (m["precision_on_calls"] - base["precision_on_calls"])
                        * 100,
                        "d_gnf_pt": (m["gated_nonflat_acc"] - base["gated_nonflat_acc"])
                        * 100,
                    }
                )
        rows.sort(key=lambda r: (r["precision"], r["gated_nf"]), reverse=True)
        return rows

    blend_rows = sweep(blend, "blend")
    r10_rows = sweep(pret_r10, "r10")
    print("Top in-band on BLEND |ret|:")
    for r in blend_rows[:12]:
        print(
            f"  thr={r['thr']} mag={r['mag']}: cov={r['coverage']:.1%} "
            f"prec={r['precision']:.1%} gnf={r['gated_nf']:.1%} "
            f"d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt"
        )
    print("Top in-band on R10 |ret|:")
    for r in r10_rows[:12]:
        print(
            f"  thr={r['thr']} mag={r['mag']}: cov={r['coverage']:.1%} "
            f"prec={r['precision']:.1%} gnf={r['gated_nf']:.1%} "
            f"d_prec={r['d_prec_pt']:+.2f}pt d_gnf={r['d_gnf_pt']:+.2f}pt"
        )

    # Best that also achieves +1pt gated precision or gated_nf
    winners = [
        r
        for r in blend_rows + r10_rows
        if r["d_prec_pt"] >= 1.0 - 1e-9 or r["d_gnf_pt"] >= 1.0 - 1e-9
    ]
    winners.sort(key=lambda r: (r["d_prec_pt"], r["d_gnf_pt"]), reverse=True)
    print(f"In-band +≥1pt winners: {len(winners)}")
    for r in winners[:8]:
        print(r)

    out = {
        "baseline_gate": base,
        "blend_in_band": blend_rows[:20],
        "r10_in_band": r10_rows[:20],
        "plus1pt_winners": winners[:20],
        "best_plus1pt": winners[0] if winners else None,
    }
    path = ROOT / "outputs" / "eval_phase3_inband_gate_sweep.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
