"""Sweep inference gates on a fixed ensemble and report backtest vs baseline.

Uses the historical R10@h1 + R5@h3/5/10 mapping by default and evaluates:
  - confidence-only gate
  - confidence + sign-consistency with return head
  - confidence + |return| magnitude
  - combined

Primary promotion metric: h=1 gated precision in 20-40% coverage and h=1
backtest hit_rate / total_return (stride=1 and stride=horizon).
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

from evaluate_gated_ensemble import (  # noqa: E402
    HORIZONS,
    load_csv,
    load_model,
    make_windows,
    predict_window,
)
from selective_prediction import (  # noqa: E402
    absolute_direction_backtest,
    apply_confidence_gate,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

DEFAULT_MAP = {
    1: "r10_joint_splitlr",
    3: "r5_frozen_pool48",
    5: "r5_frozen_pool48",
    10: "r5_frozen_pool48",
}


def model_path(name: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "models"
        / f"a_share_multihorizon_predictor_{name}"
        / "checkpoints"
        / "best_model"
    )


def collect_predictions(mapping: dict[int, str], symbol: str = "688169") -> dict:
    names = sorted(set(mapping.values()))
    loaded = {n: load_model(model_path(n)) for n in names}
    df = load_csv(symbol)
    windows = make_windows(df)
    # store per-horizon arrays
    store = {
        h: {
            "logits": [],
            "pred_ret": [],
            "t_dir": [],
            "t_ret": [],
            "dates": [],
        }
        for h in HORIZONS
    }
    for w in windows:
        # cache preds per model once per window
        cache = {}
        for name, mod in loaded.items():
            cache[name] = predict_window(mod, w)
        for hi, h in enumerate(HORIZONS):
            name = mapping[h]
            pred = cache[name]
            store[h]["logits"].append(pred["logits"][hi])
            store[h]["pred_ret"].append(float(pred["pred_return"][hi]))
            store[h]["t_dir"].append(int(pred["target_direction"][hi]))
            store[h]["t_ret"].append(float(pred["target_return"][hi]))
            store[h]["dates"].append(pred["context_end_date"])
    for h in HORIZONS:
        store[h]["logits"] = np.stack(store[h]["logits"], axis=0)
        store[h]["pred_ret"] = np.asarray(store[h]["pred_ret"], dtype=np.float64)
        store[h]["t_dir"] = np.asarray(store[h]["t_dir"], dtype=np.int64)
        store[h]["t_ret"] = np.asarray(store[h]["t_ret"], dtype=np.float64)
        conf = direction_confidence_from_logits(store[h]["logits"])
        store[h]["hard_pred"] = conf["hard_pred"]
        store[h]["actionable_score"] = conf["actionable_score"]
        store[h]["margin"] = conf["margin"]
        store[h]["max_prob"] = conf["max_prob"]
    return store


def evaluate_gate_config(
    store: dict,
    *,
    conf_key: str,
    conf_thr: float,
    min_abs_return: float,
    require_sign: bool,
    cost: float = 0.0005,
) -> dict:
    by_h = {}
    all_pred = []
    all_tgt = []
    for h in HORIZONS:
        hard = store[h]["hard_pred"]
        conf = store[h][conf_key]
        pret = store[h]["pred_ret"]
        if require_sign or min_abs_return > 0:
            gated = apply_consistency_and_magnitude_gate(
                hard,
                conf,
                pret,
                confidence_threshold=conf_thr,
                min_abs_return=min_abs_return,
                require_sign_agree=require_sign,
            )
        else:
            gated = apply_confidence_gate(hard, conf, conf_thr)
        m = gated_actionable_metrics(gated, store[h]["t_dir"])
        bt = absolute_direction_backtest(
            gated, store[h]["t_ret"], transaction_cost=cost, stride=1
        )
        bt_nonoverlap = absolute_direction_backtest(
            gated, store[h]["t_ret"], transaction_cost=cost, stride=max(1, h)
        )
        by_h[str(h)] = {
            "metrics": m,
            "backtest": bt,
            "backtest_nonoverlap": bt_nonoverlap,
            "ungated_nonflat": nonflat_accuracy(hard, store[h]["t_dir"]),
        }
        all_pred.append(gated)
        all_tgt.append(store[h]["t_dir"])
    overall = gated_actionable_metrics(np.concatenate(all_pred), np.concatenate(all_tgt))
    return {
        "config": {
            "conf_key": conf_key,
            "conf_thr": conf_thr,
            "min_abs_return": min_abs_return,
            "require_sign": require_sign,
        },
        "overall_metrics": overall,
        "by_horizon": by_h,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_promoted_gates.json")
    parser.add_argument("--symbol", default="688169")
    args = parser.parse_args()

    print("Collecting ensemble predictions...")
    store = collect_predictions(DEFAULT_MAP, symbol=args.symbol)

    # Ungated baseline h=1
    base_h1 = absolute_direction_backtest(
        store[1]["hard_pred"], store[1]["t_ret"], transaction_cost=0.0005
    )
    base_h1_metrics = gated_actionable_metrics(store[1]["hard_pred"], store[1]["t_dir"])

    conf_thrs = [0.35, 0.40, 0.45, 0.50, 0.55]
    mag_thrs = [0.0, 0.005, 0.01, 0.015]
    results = []
    for conf_key in ["actionable_score", "margin"]:
        for thr in conf_thrs:
            for mag in mag_thrs:
                for sign in [False, True]:
                    if conf_key == "margin" and thr >= 0.45:
                        # margin rarely exceeds 0.45 for h=1 in baseline; skip
                        continue
                    row = evaluate_gate_config(
                        store,
                        conf_key=conf_key,
                        conf_thr=thr,
                        min_abs_return=mag,
                        require_sign=sign,
                    )
                    h1 = row["by_horizon"]["1"]
                    cov = h1["metrics"]["coverage"]
                    # keep only candidates with some trades
                    if h1["metrics"]["n_calls"] < 8:
                        continue
                    results.append(
                        {
                            **row["config"],
                            "h1_coverage": cov,
                            "h1_precision": h1["metrics"]["precision_on_calls"],
                            "h1_gated_nf": h1["metrics"]["gated_nonflat_acc"],
                            "h1_hit": h1["backtest"]["hit_rate"],
                            "h1_total_return": h1["backtest"]["total_return"],
                            "h1_avg_trade": h1["backtest"]["avg_trade_pnl"],
                            "h1_n_trades": h1["backtest"]["n_trades"],
                            "h1_ret_stride1": h1["backtest"]["total_return"],
                            "full": row,
                        }
                    )

    # Rank by h1 precision within 15-45% coverage, then hit_rate, then total_return
    band = [r for r in results if 0.15 <= r["h1_coverage"] <= 0.45]
    band_sorted = sorted(
        band,
        key=lambda r: (r["h1_precision"], r["h1_hit"], r["h1_total_return"]),
        reverse=True,
    )
    # Also rank by total_return in band
    by_return = sorted(band, key=lambda r: r["h1_total_return"], reverse=True)

    print("\n=== Baseline ungated h=1 ===")
    print(
        f"cov={base_h1_metrics['coverage']:.1%} prec={base_h1_metrics['precision_on_calls']:.1%} "
        f"hit={base_h1['hit_rate']:.1%} ret={base_h1['total_return']:.1%}"
    )
    print("\n=== Top by precision (15-45% cov) ===")
    for r in band_sorted[:8]:
        print(
            f"  {r['conf_key']} thr={r['conf_thr']} mag={r['min_abs_return']} sign={r['require_sign']}: "
            f"cov={r['h1_coverage']:.1%} prec={r['h1_precision']:.1%} hit={r['h1_hit']:.1%} "
            f"ret={r['h1_total_return']:.1%} trades={r['h1_n_trades']:.0f}"
        )
    print("\n=== Top by total_return (15-45% cov) ===")
    for r in by_return[:8]:
        print(
            f"  {r['conf_key']} thr={r['conf_thr']} mag={r['min_abs_return']} sign={r['require_sign']}: "
            f"cov={r['h1_coverage']:.1%} prec={r['h1_precision']:.1%} hit={r['h1_hit']:.1%} "
            f"ret={r['h1_total_return']:.1%} trades={r['h1_n_trades']:.0f}"
        )

    # Promoted: best precision with hit >= baseline hit and cov in band; else best return with prec>=0.55
    promoted = None
    for r in band_sorted:
        if r["h1_hit"] >= base_h1["hit_rate"] and r["h1_precision"] >= 0.55:
            promoted = r
            break
    if promoted is None and by_return:
        for r in by_return:
            if r["h1_precision"] >= 0.55:
                promoted = r
                break
    if promoted is None and band_sorted:
        promoted = band_sorted[0]

    # Confidence-only baseline gate for comparison (actionable 0.45)
    conf_only = evaluate_gate_config(
        store,
        conf_key="actionable_score",
        conf_thr=0.45,
        min_abs_return=0.0,
        require_sign=False,
    )

    out = {
        "symbol": args.symbol,
        "mapping": {str(k): v for k, v in DEFAULT_MAP.items()},
        "ungated_h1": {
            "metrics": base_h1_metrics,
            "backtest": base_h1,
            "nonflat": nonflat_accuracy(store[1]["hard_pred"], store[1]["t_dir"]),
        },
        "confidence_only_h1": {
            "metrics": conf_only["by_horizon"]["1"]["metrics"],
            "backtest": conf_only["by_horizon"]["1"]["backtest"],
            "config": conf_only["config"],
        },
        "top_by_precision": [
            {k: v for k, v in r.items() if k != "full"} for r in band_sorted[:10]
        ],
        "top_by_return": [
            {k: v for k, v in r.items() if k != "full"} for r in by_return[:10]
        ],
        "promoted": (
            {k: v for k, v in promoted.items() if k != "full"} if promoted else None
        ),
        "promoted_full": promoted["full"] if promoted else None,
        "n_candidates_scored": len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPromoted: {out['promoted']}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
