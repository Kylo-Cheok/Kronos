"""Phase 9 R5 — regime-conditional confidence threshold.

R4 finding: binary index fade (idx5) lifts prec +6.58pt but drops beat-B&H
8/9→7/9 (000333's up-regime bounces get fully filtered).

R5 hypothesis: instead of binary fade, use regime-conditional confidence
threshold — relax gate when index is down (catch bounces), tighten gate when
index is up (keep only very strong signals). This should recover some up-regime
calls while keeping the prec gain.

Design (predict once, sweep gate logic — zero model cost):
  * idx_n=5 (R4 optimal)
  * For each prediction i:
      thr_i = low_thr  if idx_ret_i < 0  else  high_thr
  * Sweep low_thr ∈ {0.30, 0.35, 0.40} × high_thr ∈ {0.50, 0.55, 0.60}
  * Baselines: uniform thr=0.45 (R1 base), binary fade (R4 optimal)

Leakage-safe: index return uses only closes up to context_end_date.

Output: outputs/eval_p9_panel_r5_conditional_gate.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import (  # noqa: E402
    FEATURES, LOOKBACK, PREDICT_WINDOW, WINDOW, load_csv,
)
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p4_r7_expanded_tta import (  # noqa: E402
    average_logits, predict_per_lookback,
)
from run_p6_build_store import load_model, model_dir  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS, DOWN_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

HORIZONS = (1, 3, 5, 10)
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
GATE_MAG = 0.002
MODEL = "r5_frozen_pool48"
IDX_N = 5  # R4 optimal
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]

# Sweep grid
LOW_THRS = (0.30, 0.35, 0.40)
HIGH_THRS = (0.50, 0.55, 0.60)


def load_index_returns() -> dict[int, pd.Series]:
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return {n: np.log(close / close.shift(n)) for n in (5, 10, 20)}, close


def get_index_ret(idx_returns: pd.Series, date: pd.Timestamp) -> float:
    try:
        loc = idx_returns.index.get_indexer([date], method="pad")[0]
    except (KeyError, IndexError):
        return float("nan")
    if loc < 0:
        return float("nan")
    val = idx_returns.iloc[loc]
    return float(val) if pd.notna(val) else float("nan")


def make_test_windows(df: pd.DataFrame, lookbacks) -> list[dict]:
    lo = pd.Timestamp(TEST_LO)
    max_lb = max(lookbacks)
    needed = max_lb + PREDICT_WINDOW + 1
    out = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        ced = df["timestamps"].iloc[start + LOOKBACK - 1]
        if not (ced > lo):
            continue
        extra = max_lb - LOOKBACK
        if start - extra < 0:
            continue
        big_start = start - extra
        window = df.iloc[big_start: big_start + needed].copy()
        out.append({
            "start": start,
            "context_end_date": ced,
            "features": window[FEATURES].to_numpy(dtype=np.float32),
            "raw_close": window["close"].to_numpy(dtype=np.float32),
            "timestamps": window["timestamps"],
        })
    return out


def gate_uniform(hard, score, trh1, thr: float, mag: float) -> np.ndarray:
    """Uniform threshold gate (R1/R4 base style, require_sign_agree=False)."""
    gated = hard.copy()
    keep = (score >= thr) & (np.abs(trh1) >= mag)
    gated[~keep] = FLAT_CLASS
    return gated


def gate_fade(hard, score, trh1, idx_arr, thr: float, mag: float) -> np.ndarray:
    """Binary fade: keep UP only when index down (R4 optimal)."""
    base = gate_uniform(hard, score, trh1, thr, mag)
    valid = ~np.isnan(idx_arr)
    base[(base == UP_CLASS) & (idx_arr >= 0)] = FLAT_CLASS
    base[(base == UP_CLASS) & ~valid] = FLAT_CLASS
    return base


def gate_conditional(hard, score, trh1, idx_arr,
                     low_thr: float, high_thr: float, mag: float) -> np.ndarray:
    """Regime-conditional: low_thr when index down, high_thr when index up."""
    gated = hard.copy()
    thr_arr = np.where(idx_arr < 0, low_thr, high_thr)
    thr_arr = np.where(np.isnan(idx_arr), high_thr, thr_arr)  # NaN → strict
    keep = (score >= thr_arr) & (np.abs(trh1) >= mag)
    gated[~keep] = FLAT_CLASS
    return gated


def backtest(g, trh1, tdh1) -> dict:
    pos = (g == UP_CLASS).astype(np.float32)
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    ret = float(np.sum(pos * trh1) - turnover * COST)
    gm = gated_actionable_metrics(g, tdh1)
    return {"ret": ret, "n_calls": int(gm["n_calls"]),
            "prec": gm["precision_on_calls"],
            "cov": gm["coverage"], "hit": gm["gated_nonflat_acc"]}


def main() -> int:
    t0 = time.time()
    idx_returns_dict, _ = load_index_returns()
    idx_ret_series = idx_returns_dict[IDX_N]
    loaded = load_model(model_dir(MODEL))
    lookbacks = (124, 126, 128, 130, 132, 134)

    # Predict once per symbol, store everything needed for gate sweep
    store = []  # list of per-symbol dicts
    print(f"=== Predicting panel (idx_n={IDX_N}) ===", flush=True)
    for sym in PANEL:
        df = load_csv(sym)
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, per_lb_logits, idx_rets = [], [], [[] for _ in lookbacks], []
        for w in windows:
            drop = max(lookbacks) - REF_CTX
            close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
            tgt = make_multihorizon_targets(
                torch.from_numpy(close_ref).unsqueeze(0),
                context_length=REF_CTX, horizons=HORIZONS,
                min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
            td_all.append(tgt["direction"][0].cpu().numpy())
            tr_all.append(tgt["returns"][0].cpu().numpy())
            idx_rets.append(get_index_ret(idx_ret_series, w["context_end_date"]))
            pred = predict_per_lookback(loaded, w, lookbacks)
            for lb_i, lg in enumerate(pred["per_lb_logits"]):
                per_lb_logits[lb_i].append(lg)
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        idx_arr = np.asarray(idx_rets)
        avg_logits = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        conf = direction_confidence_from_logits(avg_logits)
        hard = conf["hard_pred"]
        score = conf["actionable_score"]
        tdh1 = td[:, 0]
        trh1 = tr[:, 0]
        bh = float(np.sum(trh1))
        store.append({
            "symbol": sym, "n": len(windows), "bh": bh,
            "hard": hard, "score": score, "trh1": trh1, "tdh1": tdh1,
            "idx_arr": idx_arr,
        })
        print(f"  {sym}: bh={bh:.3f} n={len(windows)} "
              f"(idx_down {int((idx_arr < 0).sum())}/{len(idx_arr)})", flush=True)

    # Baselines
    print("\n=== Baselines ===", flush=True)
    base_g = [gate_uniform(s["hard"], s["score"], s["trh1"], 0.45, GATE_MAG)
              for s in store]
    fade_g = [gate_fade(s["hard"], s["score"], s["trh1"], s["idx_arr"], 0.45, GATE_MAG)
              for s in store]
    base_bt = [backtest(g, s["trh1"], s["tdh1"]) for g, s in zip(base_g, store)]
    fade_bt = [backtest(g, s["trh1"], s["tdh1"]) for g, s in zip(fade_g, store)]
    bh = np.array([s["bh"] for s in store])
    base_ret = np.array([b["ret"] for b in base_bt])
    fade_ret = np.array([b["ret"] for b in fade_bt])
    base_prec = np.array([b["prec"] for b in base_bt])
    fade_prec = np.array([b["prec"] for b in fade_bt])
    print(f"  base (uniform 0.45): ret {base_ret.mean():.3f} "
          f"beat {int((base_ret > bh).sum())}/9 prec {base_prec.mean():.2%}", flush=True)
    print(f"  fade (R4 optimal):   ret {fade_ret.mean():.3f} "
          f"beat {int((fade_ret > bh).sum())}/9 prec {fade_prec.mean():.2%}", flush=True)

    # Sweep conditional thresholds
    results = []
    print("\n=== Conditional threshold sweep ===", flush=True)
    for low_thr in LOW_THRS:
        for high_thr in HIGH_THRS:
            cond_g = [gate_conditional(s["hard"], s["score"], s["trh1"], s["idx_arr"],
                                       low_thr, high_thr, GATE_MAG) for s in store]
            cond_bt = [backtest(g, s["trh1"], s["tdh1"]) for g, s in zip(cond_g, store)]
            cond_ret = np.array([b["ret"] for b in cond_bt])
            cond_prec = np.array([b["prec"] for b in cond_bt])
            beat = int((cond_ret > bh).sum())
            summary = {
                "low_thr": low_thr, "high_thr": high_thr,
                "ret_mean": float(cond_ret.mean()),
                "beat_bh": beat,
                "prec_mean": float(cond_prec.mean()),
            }
            results.append({"summary": summary, "per_symbol": [
                {"symbol": s["symbol"], **b} for s, b in zip(store, cond_bt)]})
            print(f"  low={low_thr:.2f} high={high_thr:.2f}: "
                  f"ret {summary['ret_mean']:.3f} beat {beat}/9 "
                  f"prec {summary['prec_mean']:.2%}", flush=True)

            # per-symbol detail for promising configs
            if summary["ret_mean"] > fade_ret.mean() and summary["prec_mean"] > base_prec.mean():
                sym_idx = {s["symbol"]: i for i, s in enumerate(store)}
                for s, b in zip(store, cond_bt):
                    i = sym_idx[s["symbol"]]
                    print(f"    {s['symbol']}: bh={s['bh']:.3f} "
                          f"base={base_bt[i]['ret']:.3f} "
                          f"fade={fade_bt[i]['ret']:.3f} "
                          f"cond={b['ret']:.3f} prec={b['prec']:.2%}", flush=True)

    out = {
        "baselines": {
            "base": {"ret_mean": float(base_ret.mean()),
                     "beat_bh": int((base_ret > bh).sum()),
                     "prec_mean": float(base_prec.mean())},
            "fade": {"ret_mean": float(fade_ret.mean()),
                     "beat_bh": int((fade_ret > bh).sum()),
                     "prec_mean": float(fade_prec.mean())},
        },
        "conditional_sweep": results,
    }
    out_path = ROOT / "outputs" / "eval_p9_panel_r5_conditional_gate.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
