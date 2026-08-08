"""Agreement ensemble: only trade when multiple models agree on UP/DOWN.

Compares:
  - single best h=1 model (R10)
  - agreement of R10+R5 (and optional R4/R9) on h=1
  - agreement + confidence gate

Goal: raise precision / hit / gated quality at 15-45% coverage without retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import (  # noqa: E402
    load_csv,
    load_model,
    make_windows,
    predict_window,
)
from selective_prediction import (  # noqa: E402
    FLAT_CLASS,
    absolute_direction_backtest,
    apply_confidence_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

CANDIDATES = ["r10_joint_splitlr", "r5_frozen_pool48", "r9_joint_pool48", "r4_joint_lr2e6"]


def model_path(name: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "models"
        / f"a_share_multihorizon_predictor_{name}"
        / "checkpoints"
        / "best_model"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_agreement_h1.json")
    parser.add_argument("--horizon-index", type=int, default=0, help="0=h1,1=h3,2=h5,3=h10")
    args = parser.parse_args()
    hi = args.horizon_index

    loaded = {}
    for name in CANDIDATES:
        p = model_path(name)
        if p.exists():
            print(f"Loading {name}")
            loaded[name] = load_model(p)
    df = load_csv("688169")
    windows = make_windows(df)

    per = {
        name: {"hard": [], "conf": [], "t_dir": [], "t_ret": [], "pred_ret": []}
        for name in loaded
    }
    for w in windows:
        for name, mod in loaded.items():
            pred = predict_window(mod, w)
            conf = direction_confidence_from_logits(pred["logits"][hi : hi + 1])
            per[name]["hard"].append(int(conf["hard_pred"][0]))
            per[name]["conf"].append(float(conf["actionable_score"][0]))
            per[name]["t_dir"].append(int(pred["target_direction"][hi]))
            per[name]["t_ret"].append(float(pred["target_return"][hi]))
            per[name]["pred_ret"].append(float(pred["pred_return"][hi]))

    for name in loaded:
        for k in per[name]:
            per[name][k] = np.asarray(per[name][k])

    t_dir = per["r10_joint_splitlr"]["t_dir"]
    t_ret = per["r10_joint_splitlr"]["t_ret"]

    def report(label: str, pred: np.ndarray) -> dict:
        m = gated_actionable_metrics(pred, t_dir)
        bt = absolute_direction_backtest(pred, t_ret, transaction_cost=0.0005)
        nf = nonflat_accuracy(pred, t_dir)
        row = {
            "label": label,
            "coverage": m["coverage"],
            "precision": m["precision_on_calls"],
            "gated_nf": m["gated_nonflat_acc"],
            "ungated_style_nf": nf,
            "hit": bt["hit_rate"],
            "total_return": bt["total_return"],
            "n_trades": bt["n_trades"],
            "avg_trade": bt["avg_trade_pnl"],
        }
        print(
            f"{label:40s} cov={row['coverage']:.1%} prec={row['precision']:.1%} "
            f"hit={row['hit']:.1%} ret={row['total_return']:.1%} trades={row['n_trades']:.0f}"
        )
        return row

    results = []
    # single models
    for name in loaded:
        results.append(report(f"single/{name}", per[name]["hard"]))
        gated = apply_confidence_gate(per[name]["hard"], per[name]["conf"], 0.45)
        results.append(report(f"single_gate0.45/{name}", gated))

    # pairwise agreement
    names = list(loaded.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ha, hb = per[a]["hard"], per[b]["hard"]
            agree = ha.copy()
            disagree = ha != hb
            agree[disagree] = FLAT_CLASS
            # also abstain if either is flat
            agree[(ha == FLAT_CLASS) | (hb == FLAT_CLASS)] = FLAT_CLASS
            results.append(report(f"agree/{a}+{b}", agree))
            # agreement + min confidence
            min_conf = np.minimum(per[a]["conf"], per[b]["conf"])
            gated = apply_confidence_gate(agree, min_conf, 0.40)
            results.append(report(f"agree_gate0.40/{a}+{b}", gated))
            gated45 = apply_confidence_gate(agree, min_conf, 0.45)
            results.append(report(f"agree_gate0.45/{a}+{b}", gated45))

    # triple: r10+r5+r4
    if all(n in loaded for n in ["r10_joint_splitlr", "r5_frozen_pool48", "r4_joint_lr2e6"]):
        h10 = per["r10_joint_splitlr"]["hard"]
        h5 = per["r5_frozen_pool48"]["hard"]
        h4 = per["r4_joint_lr2e6"]["hard"]
        agree = h10.copy()
        mask = (h10 == h5) & (h10 == h4) & (h10 != FLAT_CLASS)
        agree[~mask] = FLAT_CLASS
        results.append(report("agree/r10+r5+r4", agree))
        min_conf = np.minimum(
            np.minimum(per["r10_joint_splitlr"]["conf"], per["r5_frozen_pool48"]["conf"]),
            per["r4_joint_lr2e6"]["conf"],
        )
        results.append(report("agree_gate0.40/r10+r5+r4", apply_confidence_gate(agree, min_conf, 0.40)))

    # pick best in 15-45% coverage by precision then hit then return
    band = [r for r in results if 0.15 <= r["coverage"] <= 0.45 and r["n_trades"] >= 8]
    band_sorted = sorted(band, key=lambda r: (r["precision"], r["hit"], r["total_return"]), reverse=True)
    by_ret = sorted(band, key=lambda r: r["total_return"], reverse=True)

    baseline = next(r for r in results if r["label"] == "single_gate0.45/r10_joint_splitlr")
    promoted = None
    for r in band_sorted:
        if (
            r["precision"] > baseline["precision"] + 1e-9
            or r["hit"] > baseline["hit"] + 1e-9
            or r["total_return"] > baseline["total_return"] + 1e-9
        ):
            # require not worse precision by more than 2pt if using return
            if r["precision"] >= baseline["precision"] - 0.02:
                promoted = r
                break
    if promoted is None and band_sorted:
        # fallback: best precision
        promoted = band_sorted[0]

    out = {
        "horizon_index": hi,
        "results": results,
        "baseline_gate": baseline,
        "top_precision_band": band_sorted[:10],
        "top_return_band": by_ret[:10],
        "promoted": promoted,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBaseline gate:", baseline)
    print("Promoted:", promoted)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
