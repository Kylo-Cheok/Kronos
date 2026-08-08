"""Phase 9 R3 — cross-sectional ranking gate.

Hypothesis: the model predicts each stock in isolation. A cross-sectional
trend filter (the stock's recent-return rank within the panel) is a NEW
information source the model cannot see from its single-stock context.

Design:
  * For each test window's context_end_date, compute each symbol's past-N-day
    log return (N in {5, 10, 20}) and rank it within the panel.
  * Gate: only go long when (a) the model says UP and (b) the symbol's rank
    is in the top half (strong stocks). Abstain otherwise.
  * Compare against the R2 r5 baseline (no rank gate).

This is a pure inference-time gate; no retraining. Leakage-safe: rank uses
only past returns up to context_end_date.

Output: outputs/eval_p9_panel_r3_rankgate.json
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
    ALL_LOOKBACKS, average_logits, predict_per_lookback,
)
from run_p6_build_store import load_model, model_dir  # noqa: E402
from selective_prediction import (  # noqa: E402
    DOWN_CLASS, FLAT_CLASS, UP_CLASS,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
MODEL = "r5_frozen_pool48"
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]


def load_all_close() -> dict[str, pd.DataFrame]:
    """Load close series for all panel symbols, indexed by date."""
    out = {}
    for sym in PANEL:
        df = load_csv(sym)[["timestamps", "close"]].set_index("timestamps")
        out[sym] = df["close"]
    return out


def compute_rank_at_date(all_close: dict[str, pd.Series],
                         sym: str, date: pd.Timestamp, lookback_n: int) -> float:
    """Rank of sym's past-lookback_n log return within panel, in [0,1].

    1.0 = strongest, 0.0 = weakest. Leakage-safe: uses only closes up to and
    including `date`.
    """
    ranks = {}
    for s, series in all_close.items():
        # find the close at `date` (last available <= date)
        try:
            loc = series.index.get_indexer([date], method="pad")[0]
        except (KeyError, IndexError):
            return 0.5
        if loc < 0 or loc < lookback_n:
            return 0.5
        recent = series.iloc[loc - lookback_n + 1: loc + 1]
        if len(recent) < lookback_n:
            return 0.5
        ret = float(np.log(recent.iloc[-1] / recent.iloc[0]))
        ranks[s] = ret
    if sym not in ranks:
        return 0.5
    vals = sorted(ranks.values())
    pos = vals.index(ranks[sym])
    return pos / max(len(vals) - 1, 1)


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


def run_rank_gate(rank_lookback: int, rank_thr: float,
                  gate_thr: float, gate_mag: float) -> dict:
    """Run r5 with a cross-sectional rank filter on top of the confidence gate."""
    t0 = time.time()
    all_close = load_all_close()
    loaded = load_model(model_dir(MODEL))
    lookbacks = (124, 126, 128, 130, 132, 134)

    per_symbol = []
    for sym in PANEL:
        df = load_csv(sym)
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, per_lb_logits, ranks = [], [], [[] for _ in lookbacks], []
        for w in windows:
            drop = max(lookbacks) - REF_CTX
            close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
            tgt = make_multihorizon_targets(
                torch.from_numpy(close_ref).unsqueeze(0),
                context_length=REF_CTX, horizons=HORIZONS,
                min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
            td_all.append(tgt["direction"][0].cpu().numpy())
            tr_all.append(tgt["returns"][0].cpu().numpy())
            ranks.append(compute_rank_at_date(all_close, sym, w["context_end_date"], rank_lookback))
            pred = predict_per_lookback(loaded, w, lookbacks)
            for lb_i, lg in enumerate(pred["per_lb_logits"]):
                per_lb_logits[lb_i].append(lg)
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        ranks_arr = np.asarray(ranks)
        n = len(windows)

        # h=1 direction from TTA
        avg_logits = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        conf = direction_confidence_from_logits(avg_logits)
        hard = conf["hard_pred"]
        score = conf["actionable_score"]
        tdh1 = td[:, 0]
        trh1 = tr[:, 0]

        # base gate (R2 steady: thr0.45/mag0.002)
        base_g = apply_consistency_and_magnitude_gate(
            hard, score, trh1,
            confidence_threshold=gate_thr, min_abs_return=gate_mag,
            require_sign_agree=False)
        # rank filter: only keep UP calls where rank >= rank_thr
        rank_g = base_g.copy()
        rank_g[(base_g == UP_CLASS) & (ranks_arr < rank_thr)] = FLAT_CLASS

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
        per_symbol.append({
            "symbol": sym, "n": n, "buy_hold": bh,
            "base": backtest(base_g),
            "rank_gate": backtest(rank_g),
            "rank_mean": float(ranks_arr.mean()),
            "rank_above_thr_frac": float((ranks_arr >= rank_thr).mean()),
        })
        print(f"  {sym}: bh={bh:.3f} base={per_symbol[-1]['base']['ret']:.3f} "
              f"rank={per_symbol[-1]['rank_gate']['ret']:.3f} "
              f"(rank≥{rank_thr}: {(ranks_arr>=rank_thr).mean():.0%})",
              flush=True)

    # panel summary
    bh = np.array([r["buy_hold"] for r in per_symbol])
    base_ret = np.array([r["base"]["ret"] for r in per_symbol])
    rank_ret = np.array([r["rank_gate"]["ret"] for r in per_symbol])
    base_prec = np.array([r["base"]["prec"] for r in per_symbol])
    rank_prec = np.array([r["rank_gate"]["prec"] for r in per_symbol])
    summary = {
        "rank_lookback": rank_lookback, "rank_thr": rank_thr,
        "gate_thr": gate_thr, "gate_mag": gate_mag,
        "base_ret_mean": float(base_ret.mean()),
        "rank_ret_mean": float(rank_ret.mean()),
        "bh_ret_mean": float(bh.mean()),
        "beat_bh_base": int((base_ret > bh).sum()),
        "beat_bh_rank": int((rank_ret > bh).sum()),
        "base_prec_mean": float(base_prec.mean()),
        "rank_prec_mean": float(rank_prec.mean()),
        "n_symbols": len(per_symbol),
    }
    print("\n=== RANK GATE SUMMARY ===")
    print(f"rank_lookback={rank_lookback} rank_thr={rank_thr}")
    print(f"base ret mean: {summary['base_ret_mean']:.3f} (beat B&H {summary['beat_bh_base']}/{summary['n_symbols']})")
    print(f"rank ret mean: {summary['rank_ret_mean']:.3f} (beat B&H {summary['beat_bh_rank']}/{summary['n_symbols']})")
    print(f"base prec mean: {summary['base_prec_mean']:.2%} → rank prec mean: {summary['rank_prec_mean']:.2%}")
    print(f"elapsed: {time.time()-t0:.0f}s")
    return {"summary": summary, "per_symbol": per_symbol}


def main() -> int:
    out_all = []
    # sweep rank lookback and threshold
    for rl in (5, 10, 20):
        for rt in (0.3, 0.5, 0.7):
            print(f"\n=== rank_lookback={rl} rank_thr={rt} ===", flush=True)
            r = run_rank_gate(rl, rt, gate_thr=0.45, gate_mag=0.002)
            r["config"] = {"rank_lookback": rl, "rank_thr": rt}
            out_all.append(r)

    out = ROOT / "outputs" / "eval_p9_panel_r3_rankgate.json"
    out.write_text(json.dumps(out_all, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\nSaved {out}")
    # print best
    best = max(out_all, key=lambda x: x["summary"]["rank_ret_mean"])
    print(f"\nBEST: rank_lookback={best['summary']['rank_lookback']} "
          f"rank_thr={best['summary']['rank_thr']} "
          f"ret_mean={best['summary']['rank_ret_mean']:.3f} "
          f"beat_bh={best['summary']['beat_bh_rank']}/{best['summary']['n_symbols']} "
          f"prec={best['summary']['rank_prec_mean']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
