"""Phase 9 R4 — exogenous index regime gate.

Hypothesis: the model sees only single-stock OHLCV. A market-wide trend
filter from CSI300 (the stock's index) is a NEW information source.

Design:
  * For each test window's context_end_date, compute CSI300's past-N-day
    log return (N in {5, 10, 20}).
  * Gate variants:
    (a) trend-confirm: only keep UP calls when index return > 0 (up market).
    (b) trend-fade: only keep UP calls when index return < 0 (counter-trend,
        testing the R3 finding that r5's alpha comes from bounce-timing).
  * Compare against r5 base (no index gate).

Leakage-safe: index return uses only closes up to context_end_date.

Output: outputs/eval_p9_panel_r4_indexgate.json
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
    FLAT_CLASS, UP_CLASS,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

HORIZONS = (1, 3, 5, 10)
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
MODEL = "r5_frozen_pool48"
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]


def load_index_returns() -> dict[int, pd.Series]:
    """Load CSI300 close, return dict of N-day log return series indexed by date."""
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    out = {}
    for n in (5, 10, 20):
        out[n] = np.log(close / close.shift(n))
    return out, close


def get_index_ret(idx_returns: pd.Series, date: pd.Timestamp) -> float:
    """Get index N-day return at or before `date`. NaN if unavailable."""
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


def run_index_gate(idx_n: int, mode: str, gate_thr: float, gate_mag: float) -> dict:
    """Run r5 with index regime filter."""
    t0 = time.time()
    idx_returns_dict, _ = load_index_returns()
    idx_ret_series = idx_returns_dict[idx_n]
    loaded = load_model(model_dir(MODEL))
    lookbacks = (124, 126, 128, 130, 132, 134)

    per_symbol = []
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
        n = len(windows)

        avg_logits = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        conf = direction_confidence_from_logits(avg_logits)
        hard = conf["hard_pred"]
        score = conf["actionable_score"]
        tdh1 = td[:, 0]
        trh1 = tr[:, 0]

        base_g = apply_consistency_and_magnitude_gate(
            hard, score, trh1,
            confidence_threshold=gate_thr, min_abs_return=gate_mag,
            require_sign_agree=False)

        # index gate
        idx_g = base_g.copy()
        if mode == "confirm":
            # keep UP only when index up; abstain when index down/NaN
            valid_idx = ~np.isnan(idx_arr)
            idx_g[(base_g == UP_CLASS) & (idx_arr <= 0)] = FLAT_CLASS
            idx_g[(base_g == UP_CLASS) & ~valid_idx] = FLAT_CLASS
        elif mode == "fade":
            # keep UP only when index down (counter-trend bounce)
            valid_idx = ~np.isnan(idx_arr)
            idx_g[(base_g == UP_CLASS) & (idx_arr >= 0)] = FLAT_CLASS
            idx_g[(base_g == UP_CLASS) & ~valid_idx] = FLAT_CLASS

        def backtest(g):
            pos = (g == UP_CLASS).astype(np.float32)
            turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
            ret = float(np.sum(pos * trh1) - turnover * COST)
            gm = gated_actionable_metrics(g, tdh1)
            return {"ret": ret, "n_calls": int(gm["n_calls"]),
                    "prec": gm["precision_on_calls"],
                    "cov": gm["coverage"],
                    "hit": gm["gated_nonflat_acc"]}

        bh = float(np.sum(trh1))
        n_idx_avail = int((~np.isnan(idx_arr)).sum())
        per_symbol.append({
            "symbol": sym, "n": n, "buy_hold": bh,
            "base": backtest(base_g),
            "index_gate": backtest(idx_g),
            "idx_ret_mean": float(np.nanmean(idx_arr)) if n_idx_avail > 0 else None,
            "n_idx_avail": n_idx_avail,
        })
        print(f"  {sym}: bh={bh:.3f} base={per_symbol[-1]['base']['ret']:.3f} "
              f"idx={per_symbol[-1]['index_gate']['ret']:.3f} "
              f"(idx_avail {n_idx_avail}/{n})", flush=True)

    bh = np.array([r["buy_hold"] for r in per_symbol])
    base_ret = np.array([r["base"]["ret"] for r in per_symbol])
    idx_ret = np.array([r["index_gate"]["ret"] for r in per_symbol])
    base_prec = np.array([r["base"]["prec"] for r in per_symbol])
    idx_prec = np.array([r["index_gate"]["prec"] for r in per_symbol])
    summary = {
        "idx_n": idx_n, "mode": mode,
        "base_ret_mean": float(base_ret.mean()),
        "idx_ret_mean": float(idx_ret.mean()),
        "bh_ret_mean": float(bh.mean()),
        "beat_bh_base": int((base_ret > bh).sum()),
        "beat_bh_idx": int((idx_ret > bh).sum()),
        "base_prec_mean": float(base_prec.mean()),
        "idx_prec_mean": float(idx_prec.mean()),
        "n_symbols": len(per_symbol),
    }
    print(f"\n  idx_n={idx_n} mode={mode}: base {summary['base_ret_mean']:.3f} "
          f"→ idx {summary['idx_ret_mean']:.3f} (beat B&H {summary['beat_bh_idx']}/{summary['n_symbols']}) "
          f"prec {summary['base_prec_mean']:.2%}→{summary['idx_prec_mean']:.2%} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"summary": summary, "per_symbol": per_symbol}


def main() -> int:
    out_all = []
    for idx_n in (5, 10, 20):
        for mode in ("confirm", "fade"):
            print(f"\n=== idx_n={idx_n} mode={mode} ===", flush=True)
            r = run_index_gate(idx_n, mode, gate_thr=0.45, gate_mag=0.002)
            out_all.append(r)

    out = ROOT / "outputs" / "eval_p9_panel_r4_indexgate.json"
    out.write_text(json.dumps(out_all, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
