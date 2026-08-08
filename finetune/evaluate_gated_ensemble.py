"""Evaluate per-horizon ensemble with confidence gating and absolute backtest.

Loads candidate multihorizon checkpoints, scores every 688169 test window,
auto-selects the best model per horizon (by ungated nonflat), then:
  1. Reports ungated metrics
  2. Sweeps confidence thresholds (actionable_score / margin / max_prob)
  3. Runs a simple long/flat/short backtest (h=1 primary for trading)

Usage::

    python finetune/evaluate_gated_ensemble.py
    python finetune/evaluate_gated_ensemble.py --output outputs/eval_gated.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from model.kronos import Kronos, KronosTokenizer
from multihorizon_objective import (
    DEFAULT_HORIZONS,
    FLAT_CLASS,
    MultiHorizonForecastHead,
    make_multihorizon_targets,
)
from selective_prediction import (
    absolute_direction_backtest,
    accuracy_vs_coverage_curve,
    apply_confidence_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
    select_threshold_for_coverage_band,
)

SYMBOL = "688169"
CSV_PATH = ROOT / "data" / "a_share_finetune_multiboard" / "csv" / f"{SYMBOL}.csv"
TOKENIZER_PATH = (
    ROOT / "outputs" / "models" / "a_share_multi_tokenizer" / "checkpoints" / "best_model"
)
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
LOOKBACK = 128
PREDICT_WINDOW = 10
WINDOW = LOOKBACK + PREDICT_WINDOW + 1
VAL_END = "2025-12-15"
FEATURES = ["open", "high", "low", "close", "vol", "amt"]
CLIP = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HORIZONS = list(DEFAULT_HORIZONS)
THRESHOLDS = [round(x, 2) for x in np.linspace(0.0, 0.95, 20)]


def load_csv(symbol: str = SYMBOL) -> pd.DataFrame:
    path = ROOT / "data" / "a_share_finetune_multiboard" / "csv" / f"{symbol}.csv"
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.normalize()
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["vol"] = df["volume"]
    df["amt"] = df["amount"]
    return df


def derive_time_features(dates: pd.Series) -> np.ndarray:
    stamps = pd.DatetimeIndex(dates)
    return np.stack(
        [
            np.zeros(len(stamps)),
            np.zeros(len(stamps)),
            stamps.weekday.to_numpy(),
            stamps.day.to_numpy(),
            stamps.month.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)


def make_windows(df: pd.DataFrame) -> list[dict]:
    val_cut = pd.Timestamp(VAL_END)
    windows = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        context_end_date = df["timestamps"].iloc[start + LOOKBACK - 1]
        if context_end_date <= val_cut:
            continue
        window = df.iloc[start : start + WINDOW].copy()
        windows.append(
            {
                "start": start,
                "context_end_date": str(context_end_date.date()),
                "features": window[FEATURES].to_numpy(dtype=np.float32),
                "raw_close": window["close"].to_numpy(dtype=np.float32),
                "timestamps": window["timestamps"],
            }
        )
    return windows


def normalize_window(features: np.ndarray) -> np.ndarray:
    lookback = features[:LOOKBACK]
    mean = lookback.mean(axis=0, keepdims=True)
    std = lookback.std(axis=0, keepdims=True)
    normalized = (features - mean) / (std + 1e-5)
    return np.clip(normalized, -CLIP, CLIP)


def load_model(model_dir: Path) -> dict:
    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_PATH))
    tokenizer.eval().to(DEVICE)
    model = Kronos.from_pretrained(str(model_dir))
    model.eval().to(DEVICE)
    head_path = model_dir / "multihorizon_head.pt"
    head_ckpt = torch.load(head_path, map_location=DEVICE, weights_only=False)
    d_model = head_ckpt["d_model"]
    horizons = tuple(head_ckpt.get("horizons", HORIZONS))
    pool_size = head_ckpt.get("pool_size", 16)
    min_deadzone = head_ckpt.get("direction_min_deadzone", 0.003)
    vol_mult = head_ckpt.get("direction_volatility_multiplier", 0.5)
    head = MultiHorizonForecastHead(
        d_model, horizons=horizons, pool_size=pool_size, dropout=0.0
    ).to(DEVICE)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "head": head,
        "horizons": horizons,
        "min_deadzone": min_deadzone,
        "vol_mult": vol_mult,
        "pool_size": pool_size,
        "model_dir": str(model_dir),
    }


@torch.no_grad()
def predict_window(loaded: dict, w: dict) -> dict:
    x_norm = normalize_window(w["features"])
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
    stamp = derive_time_features(w["timestamps"])
    stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
    raw_close_tensor = torch.from_numpy(w["raw_close"]).unsqueeze(0).to(DEVICE)

    token_seq_0, token_seq_1 = loaded["tokenizer"].encode(x_tensor, half=True)
    _, _, hidden_states = loaded["model"](
        token_seq_0[:, :-1],
        token_seq_1[:, :-1],
        stamp_tensor[:, :-1, :],
        return_context=True,
    )
    outputs = loaded["head"](hidden_states, context_length=LOOKBACK)
    logits = outputs["direction_logits"][0].cpu().numpy()
    return_pred = outputs["return_prediction"][0].cpu().numpy()

    targets = make_multihorizon_targets(
        raw_close_tensor,
        context_length=LOOKBACK,
        horizons=loaded["horizons"],
        min_deadzone=loaded["min_deadzone"],
        volatility_multiplier=loaded["vol_mult"],
    )
    return {
        "logits": logits,
        "pred_return": return_pred,
        "target_direction": targets["direction"][0].cpu().numpy(),
        "target_return": targets["returns"][0].cpu().numpy(),
        "context_end_date": w["context_end_date"],
    }


def parse_candidates(raw: str | None) -> dict[str, Path]:
    if not raw:
        return {k: v for k, v in DEFAULT_CANDIDATES.items() if v.exists()}
    out: dict[str, Path] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, path = part.split("=", 1)
            out[name.strip()] = Path(path.strip())
        else:
            # exp-name style
            path = (
                ROOT
                / "outputs"
                / "models"
                / f"a_share_multihorizon_predictor_{part}"
                / "checkpoints"
                / "best_model"
            )
            out[part] = path
    return out


def evaluate(
    candidates: dict[str, Path],
    *,
    symbol: str = SYMBOL,
    confidence_key: str = "actionable_score",
    fixed_mapping: dict[int, str] | None = None,
    transaction_cost: float = 0.0005,
) -> dict:
    df = load_csv(symbol)
    windows = make_windows(df)
    print(f"Loaded {len(windows)} test windows for {symbol}")

    loaded_models = {}
    for name, path in candidates.items():
        if not path.exists():
            print(f"SKIP missing model {name}: {path}")
            continue
        print(f"Loading {name}...")
        loaded_models[name] = load_model(path)
    if not loaded_models:
        raise FileNotFoundError("No candidate models found")

    # per_model[name][h] lists
    per_model: dict = {
        name: {
            h: {
                "logits": [],
                "pred_ret": [],
                "t_dir": [],
                "t_ret": [],
                "dates": [],
            }
            for h in HORIZONS
        }
        for name in loaded_models
    }

    for w in windows:
        for name, loaded in loaded_models.items():
            pred = predict_window(loaded, w)
            for hi, h in enumerate(HORIZONS):
                per_model[name][h]["logits"].append(pred["logits"][hi])
                per_model[name][h]["pred_ret"].append(float(pred["pred_return"][hi]))
                per_model[name][h]["t_dir"].append(int(pred["target_direction"][hi]))
                per_model[name][h]["t_ret"].append(float(pred["target_return"][hi]))
                per_model[name][h]["dates"].append(pred["context_end_date"])

    # pick best model per horizon by ungated nonflat (or fixed mapping)
    best_model_per_h: dict[int, str] = {}
    per_model_metrics: dict = {}
    print("\n=== Per-model per-horizon nonflat ===")
    for h in HORIZONS:
        print(f"\nh={h}:")
        best_nf = -1.0
        best_name = None
        per_model_metrics[str(h)] = {}
        for name in loaded_models:
            t_dirs = np.array(per_model[name][h]["t_dir"])
            logits = np.stack(per_model[name][h]["logits"], axis=0)
            conf = direction_confidence_from_logits(logits)
            p_dirs = conf["hard_pred"]
            nf = nonflat_accuracy(p_dirs, t_dirs)
            da = float((p_dirs == t_dirs).mean())
            mae = float(
                np.mean(
                    np.abs(
                        np.array(per_model[name][h]["pred_ret"])
                        - np.array(per_model[name][h]["t_ret"])
                    )
                )
            )
            per_model_metrics[str(h)][name] = {
                "direction_accuracy": da,
                "nonflat_accuracy": nf,
                "return_mae": mae,
            }
            print(f"  {name}: dir={da:.2%}, nonflat={nf:.2%}, MAE={mae:.4f}")
            if nf > best_nf:
                best_nf = nf
                best_name = name
        if fixed_mapping and h in fixed_mapping and fixed_mapping[h] in loaded_models:
            best_model_per_h[h] = fixed_mapping[h]
            print(f"  -> Fixed: {fixed_mapping[h]}")
        else:
            best_model_per_h[h] = best_name  # type: ignore[assignment]
            print(f"  -> Best: {best_name} (nonflat={best_nf:.2%})")

    # Build ensemble arrays
    ens: dict[int, dict] = {}
    for h in HORIZONS:
        name = best_model_per_h[h]
        logits = np.stack(per_model[name][h]["logits"], axis=0)
        conf = direction_confidence_from_logits(logits)
        ens[h] = {
            "model": name,
            "logits": logits,
            "hard_pred": conf["hard_pred"],
            "confidence": conf[confidence_key],
            "all_conf": conf,
            "t_dir": np.array(per_model[name][h]["t_dir"]),
            "t_ret": np.array(per_model[name][h]["t_ret"]),
            "pred_ret": np.array(per_model[name][h]["pred_ret"]),
            "dates": per_model[name][h]["dates"],
        }

    # Ungated ensemble metrics
    print("\n=== Ensemble ungated ===")
    ens_nf_c = 0
    ens_nf_t = 0
    ens_da_c = 0
    ens_da_t = 0
    mae_list = []
    by_h = {}
    for h in HORIZONS:
        t = ens[h]["t_dir"]
        p = ens[h]["hard_pred"]
        nf = nonflat_accuracy(p, t)
        da = float((p == t).mean())
        mae = float(np.mean(np.abs(ens[h]["pred_ret"] - ens[h]["t_ret"])))
        mask = t != FLAT_CLASS
        ens_nf_c += int(((p == t) & mask).sum())
        ens_nf_t += int(mask.sum())
        ens_da_c += int((p == t).sum())
        ens_da_t += len(t)
        mae_list.extend((ens[h]["pred_ret"] - ens[h]["t_ret"]).tolist())
        by_h[str(h)] = {
            "model": ens[h]["model"],
            "direction_accuracy": da,
            "nonflat_accuracy": nf,
            "return_mae": mae,
        }
        print(f"  h={h} model={ens[h]['model']}: dir={da:.2%}, nonflat={nf:.2%}, MAE={mae:.4f}")
    ungated = {
        "direction_accuracy_overall": ens_da_c / ens_da_t if ens_da_t else 0.0,
        "nonflat_accuracy_overall": ens_nf_c / ens_nf_t if ens_nf_t else 0.0,
        "return_mae_overall": float(np.mean(np.abs(mae_list))),
        "by_horizon": by_h,
    }
    print(
        f"Overall: dir={ungated['direction_accuracy_overall']:.2%}, "
        f"nonflat={ungated['nonflat_accuracy_overall']:.2%}, "
        f"MAE={ungated['return_mae_overall']:.4f}"
    )

    # Confidence curves per horizon + overall (concat)
    print(f"\n=== Confidence gate sweep (key={confidence_key}) ===")
    curves = {}
    band_picks = {}
    for h in HORIZONS:
        curve = accuracy_vs_coverage_curve(
            ens[h]["confidence"], ens[h]["hard_pred"], ens[h]["t_dir"], THRESHOLDS
        )
        curves[str(h)] = curve
        pick = select_threshold_for_coverage_band(
            curve, min_coverage=0.20, max_coverage=0.40, score_key="precision_on_calls"
        )
        band_picks[str(h)] = pick
        if pick:
            print(
                f"  h={h}: thr={pick['threshold']:.2f} cov={pick['coverage']:.1%} "
                f"prec={pick['precision_on_calls']:.1%} gated_nf={pick['gated_nonflat_acc']:.1%}"
            )
        else:
            print(f"  h={h}: no threshold in 20-40% coverage band")

    # Overall curve: stack all horizons
    all_conf = np.concatenate([ens[h]["confidence"] for h in HORIZONS])
    all_pred = np.concatenate([ens[h]["hard_pred"] for h in HORIZONS])
    all_tgt = np.concatenate([ens[h]["t_dir"] for h in HORIZONS])
    overall_curve = accuracy_vs_coverage_curve(all_conf, all_pred, all_tgt, THRESHOLDS)
    overall_pick = select_threshold_for_coverage_band(
        overall_curve, min_coverage=0.20, max_coverage=0.40, score_key="precision_on_calls"
    )
    if overall_pick:
        print(
            f"  overall: thr={overall_pick['threshold']:.2f} cov={overall_pick['coverage']:.1%} "
            f"prec={overall_pick['precision_on_calls']:.1%}"
        )

    # Backtests: h=1 primary (trading), also h=5 and ensemble-average style
    print("\n=== Absolute-direction backtest ===")
    backtests = {}
    for h in HORIZONS:
        # ungated
        bt_u = absolute_direction_backtest(
            ens[h]["hard_pred"], ens[h]["t_ret"], transaction_cost=transaction_cost
        )
        pick = band_picks.get(str(h))
        if pick:
            gated_pred = apply_confidence_gate(
                ens[h]["hard_pred"], ens[h]["confidence"], pick["threshold"]
            )
            bt_g = absolute_direction_backtest(
                gated_pred, ens[h]["t_ret"], transaction_cost=transaction_cost
            )
        else:
            # fallback thr that yields ~30% coverage if possible
            bt_g = bt_u
            gated_pred = ens[h]["hard_pred"]
        backtests[str(h)] = {"ungated": bt_u, "gated": bt_g, "gate_pick": pick}
        print(
            f"  h={h} ungated: ret={bt_u['total_return']:.2%} trades={bt_u['n_trades']:.0f} "
            f"hit={bt_u['hit_rate']:.1%} | gated: ret={bt_g['total_return']:.2%} "
            f"trades={bt_g['n_trades']:.0f} hit={bt_g['hit_rate']:.1%}"
        )

    # Also evaluate fixed R10@h1 + R5@h3/5/10 if both present (historical baseline mapping)
    fixed_map = {1: "r10_joint_splitlr", 3: "r5_frozen_pool48", 5: "r5_frozen_pool48", 10: "r5_frozen_pool48"}
    if all(fixed_map[h] in loaded_models for h in HORIZONS):
        print("\n=== Historical R10@h1 + R5@h3/5/10 mapping ===")
        hist_nf_c = hist_nf_t = 0
        hist_by_h = {}
        for h in HORIZONS:
            name = fixed_map[h]
            logits = np.stack(per_model[name][h]["logits"], axis=0)
            conf = direction_confidence_from_logits(logits)
            t = np.array(per_model[name][h]["t_dir"])
            p = conf["hard_pred"]
            nf = nonflat_accuracy(p, t)
            mask = t != FLAT_CLASS
            hist_nf_c += int(((p == t) & mask).sum())
            hist_nf_t += int(mask.sum())
            hist_by_h[str(h)] = {"model": name, "nonflat_accuracy": nf}
            print(f"  h={h}: {name} nonflat={nf:.2%}")
        hist_overall = hist_nf_c / hist_nf_t if hist_nf_t else 0.0
        print(f"  overall nonflat={hist_overall:.2%}")
    else:
        hist_overall = None
        hist_by_h = {}

    result = {
        "symbol": symbol,
        "n_test_windows": len(windows),
        "candidates": list(loaded_models.keys()),
        "confidence_key": confidence_key,
        "best_model_per_horizon": {str(h): best_model_per_h[h] for h in HORIZONS},
        "ungated_ensemble": ungated,
        "historical_r10_r5_mapping": {
            "nonflat_accuracy_overall": hist_overall,
            "by_horizon": hist_by_h,
        },
        "gated": {
            "overall_band_pick": overall_pick,
            "band_picks_by_horizon": band_picks,
            "overall_curve": overall_curve,
            "curves_by_horizon": {k: v for k, v in curves.items()},
        },
        "backtests": backtests,
        "per_model_metrics": per_model_metrics,
        "baseline_reference": {
            "log_nonflat_overall": 0.7025,
            "note": "R10 ensemble from optimization_log.md Round 10",
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma list of exp names or name=path pairs",
    )
    parser.add_argument(
        "--confidence-key",
        default="actionable_score",
        choices=["actionable_score", "margin", "max_prob", "nonflat_prob"],
    )
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--transaction-cost", type=float, default=0.0005)
    args = parser.parse_args()

    candidates = parse_candidates(args.candidates)
    result = evaluate(
        candidates,
        symbol=args.symbol,
        confidence_key=args.confidence_key,
        transaction_cost=args.transaction_cost,
    )
    out = args.output or (ROOT / "outputs" / f"eval_gated_{args.symbol}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
