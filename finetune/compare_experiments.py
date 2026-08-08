"""Compare all multihorizon experiment results on the 688169 test set.

Reads evaluation JSONs produced by evaluate_multihorizon.py and prints a
side-by-side comparison table.  Also saves a combined JSON report.

Usage::

    python finetune/compare_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "A_baseline",
        "label": "Joint dir=0.25 (baseline)",
        "eval_json": ROOT / "outputs" / "eval_688169_baseline.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor" / "summary.json",
    },
    {
        "name": "B_joint_d005",
        "label": "Joint dir=0.05",
        "eval_json": ROOT / "outputs" / "eval_688169_joint_d005.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_joint_d005" / "summary.json",
    },
    {
        "name": "smoke_frozen",
        "label": "Frozen dir=0.25 (1ep smoke)",
        "eval_json": ROOT / "outputs" / "eval_688169_smoke_frozen.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_smoke_frozen" / "summary.json",
    },
    {
        "name": "C_frozen_d025",
        "label": "Frozen dir=0.25",
        "eval_json": ROOT / "outputs" / "eval_688169_frozen_d025.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_frozen_d025" / "summary.json",
    },
    {
        "name": "D_frozen_d005",
        "label": "Frozen dir=0.05",
        "eval_json": ROOT / "outputs" / "eval_688169_frozen_d005.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_frozen_d005" / "summary.json",
    },
    {
        "name": "E_joint_d025_cw",
        "label": "Joint dir=0.25 + cw[2,.5,2]",
        "eval_json": ROOT / "outputs" / "eval_688169_joint_d025_cw.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_joint_d025_cw" / "summary.json",
    },
    {
        "name": "F_frozen_d025_cw",
        "label": "Frozen dir=0.25 + cw[2,.5,2]",
        "eval_json": ROOT / "outputs" / "eval_688169_frozen_d025_cw.json",
        "summary_json": ROOT / "outputs" / "models" / "a_share_multihorizon_predictor_frozen_d025_cw" / "summary.json",
    },
]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    results = []
    for exp in EXPERIMENTS:
        eval_data = load_json(exp["eval_json"])
        summary = load_json(exp["summary_json"])
        if eval_data is None:
            print(f"[SKIP] {exp['name']}: {exp['eval_json']} not found")
            results.append({"name": exp["name"], "label": exp["label"], "status": "not_evaluated"})
            continue
        entry = {
            "name": exp["name"],
            "label": exp["label"],
            "n_test_windows": eval_data["n_test_windows"],
            "direction_accuracy_overall": eval_data["direction_accuracy_overall"],
            "nonflat_accuracy_overall": eval_data["nonflat_accuracy_overall"],
            "direction_accuracy_by_horizon": eval_data["direction_accuracy_by_horizon"],
            "nonflat_accuracy_by_horizon": eval_data["nonflat_accuracy_by_horizon"],
            "return_mae_by_horizon": eval_data["return_mae_by_horizon"],
            "return_rmse_by_horizon": eval_data["return_rmse_by_horizon"],
            "class_distribution_by_horizon": eval_data["class_distribution_by_horizon"],
        }
        if summary and "final_result" in summary:
            fr = summary["final_result"]
            entry["best_val_loss"] = fr.get("best_val_loss")
            entry["best_val_direction_accuracy"] = fr.get("best_val_direction_accuracy")
            entry["best_val_nonflat_accuracy"] = fr.get("best_val_nonflat_accuracy")
            entry["train_direction_loss_weight"] = fr.get("direction_loss_weight")
            entry["direction_class_weights"] = fr.get("direction_class_weights")
            entry["freeze_backbone"] = fr.get("freeze_backbone", False)
        results.append(entry)

    # Print comparison table
    print("=" * 110)
    print(f"{'Experiment':<35} {'Dir Acc':>8} {'NonFlat':>8} {'h1':>7} {'h3':>7} {'h5':>7} {'h10':>7} {'RetMAE(h10)':>11}")
    print("=" * 110)
    for r in results:
        if "direction_accuracy_overall" not in r:
            print(f"{r['label']:<35} {'N/A':>8}")
            continue
        da = r["direction_accuracy_overall"]
        nf = r["nonflat_accuracy_overall"]
        h1 = r["direction_accuracy_by_horizon"].get("1", 0)
        h3 = r["direction_accuracy_by_horizon"].get("3", 0)
        h5 = r["direction_accuracy_by_horizon"].get("5", 0)
        h10 = r["direction_accuracy_by_horizon"].get("10", 0)
        mae10 = r["return_mae_by_horizon"].get("10", 0)
        print(f"{r['label']:<35} {da:>7.1%} {nf:>7.1%} {h1:>6.1%} {h3:>6.1%} {h5:>6.1%} {h10:>6.1%} {mae10:>10.4f}")
    print("=" * 110)
    print("Dir Acc = overall 3-class direction accuracy (down/flat/up)")
    print("NonFlat = direction accuracy on non-FLAT true labels (actionable up/down calls)")
    print("h1/h3/h5/h10 = per-horizon direction accuracy")
    print("RetMAE(h10) = mean absolute error of log-return prediction at h=10")

    # Per-horizon non-flat accuracy detail
    print("\nNon-flat accuracy (actionable up/down calls, excluding FLAT true labels):")
    print("-" * 80)
    for r in results:
        if "nonflat_accuracy_by_horizon" not in r:
            continue
        nf = r["nonflat_accuracy_by_horizon"]
        print(f"  {r['label']:<35} h1={nf.get('1',0):.1%}  h3={nf.get('3',0):.1%}  h5={nf.get('5',0):.1%}  h10={nf.get('10',0):.1%}")

    # Class distribution
    print("\nClass distribution (true labels) at h=10:")
    print("-" * 80)
    for r in results:
        if "class_distribution_by_horizon" not in r:
            continue
        dist = r["class_distribution_by_horizon"].get("10", {})
        print(f"  {r['label']:<35} down={dist.get('down',0)}  flat={dist.get('flat',0)}  up={dist.get('up',0)}")

    # Save combined report
    report_path = ROOT / "outputs" / "experiment_comparison_688169.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved comparison report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
