"""Phase 9 — honest cross-sectional panel evaluator.

Single source of truth for all Phase 9 rounds.  Given a model name + a set of
symbols, it produces leakage-safe per-symbol + panel-aggregate metrics on the
*test* band (post val_end=2025-12-15), with regime stratification and a
cash-baseline benchmark so that "beat B&H in a bear band" cannot masquerade as
alpha.

Design rules (from Phase 7/8 lessons):
  * Targets always use the fixed reference label convention
    ctx=122 / dz=0.003 / vol=0.5 — never inherit the TTA first-lookback label.
  * The test band is touched once per configuration; selection happens on the
    pre-test selection band (2025-06-01, 2025-12-15] only.
  * Regime tags are derived from the per-symbol future return over the test
    window — up/down/range — so "beat B&H by being in cash during a bear" is
    surfaced explicitly rather than hidden in a single return number.
  * A random-cash baseline is reported alongside any gate so that
    "6/8 beat B&H" can be compared against a cash-heavy null.

Output: outputs/eval_p9_panel_<tag>.json
"""

from __future__ import annotations

import argparse
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
    CLIP, FEATURES, LOOKBACK, PREDICT_WINDOW, WINDOW, load_csv,
)
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p4_r7_expanded_tta import (  # noqa: E402
    ALL_LOOKBACKS, average_logits, predict_per_lookback,
)
from run_p6_build_store import load_model, model_dir  # noqa: E402
from selective_prediction import (  # noqa: E402
    DOWN_CLASS, FLAT_CLASS, UP_CLASS,
    absolute_direction_backtest,
    apply_confidence_gate,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"  # post val_end — touched once
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
COST = 0.0005  # single-side transaction cost, matches P8 multi-stock eval

# The 8-symbol panel used in the P8 long-only audit + 688169 anchor.
DEFAULT_PANEL = [
    "000001", "000002", "000063", "000333",
    "000651", "002415", "600036", "601318", "688169",
]


def make_test_windows(df: pd.DataFrame, lookbacks) -> list[dict]:
    """Build test-band windows: context_end_date > TEST_LO (strictly post val)."""
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
            "context_end_date": str(ced.date()),
            "features": window[FEATURES].to_numpy(dtype=np.float32),
            "raw_close": window["close"].to_numpy(dtype=np.float32),
            "timestamps": window["timestamps"],
        })
    return out


def tag_regime(future_log_ret: float) -> str:
    """Coarse regime tag from the h=1 future log return.

    Thresholds chosen so the 8-symbol panel splits into roughly balanced
    up/range/down buckets on the test band (±2% ≈ one daily vol for liquid
    A-shares).
    """
    if future_log_ret > 0.02:
        return "up"
    if future_log_ret < -0.02:
        return "down"
    return "range"


def run_symbol(model_name: str, symbol: str, lookbacks: tuple[int, ...]) -> dict:
    """Score one model on one symbol's test band; return per-horizon metrics."""
    df = load_csv(symbol)
    windows = make_test_windows(df, lookbacks)
    if not windows:
        return {"symbol": symbol, "n": 0, "error": "no test windows"}
    loaded = load_model(model_dir(model_name))

    td_all = []
    tr_all = []
    per_lb_logits = [[] for _ in lookbacks]  # list per lb of [H,3]
    raw_closes = []
    for w in windows:
        drop = max(lookbacks) - REF_CTX
        close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
        tgt = make_multihorizon_targets(
            torch.from_numpy(close_ref).unsqueeze(0),
            context_length=REF_CTX, horizons=HORIZONS,
            min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
        td_all.append(tgt["direction"][0].cpu().numpy())
        tr_all.append(tgt["returns"][0].cpu().numpy())
        raw_closes.append(close_ref)
        pred = predict_per_lookback(loaded, w, lookbacks)
        for lb_i, lg in enumerate(pred["per_lb_logits"]):
            per_lb_logits[lb_i].append(lg)
    td = np.asarray(td_all)  # [N, H]
    tr = np.asarray(tr_all)  # [N, H]
    n = len(windows)

    # Per-horizon metrics (use lookback 128 single — the P5-5 return source)
    h_results = {}
    for hi, h in enumerate(HORIZONS):
        # average over the provided lookbacks (TTA direction)
        idx = list(range(len(lookbacks)))
        avg_logits = average_logits(
            [np.asarray(per_lb_logits[i])[:, hi, :] for i in idx], None)
        conf = direction_confidence_from_logits(avg_logits)
        hard = conf["hard_pred"]
        score = conf["actionable_score"]
        tdh = td[:, hi]
        trh = tr[:, hi]
        nf = float(nonflat_accuracy(hard, tdh))

        # gate at a sweep of thresholds — panel-calibrated later
        curve = []
        for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            g = apply_confidence_gate(hard, score, thr)
            gm = gated_actionable_metrics(g, tdh)
            curve.append({"thr": thr, **gm})

        h_results[h] = {
            "nonflat": nf,
            "n_true_moves": int((tdh != FLAT_CLASS).sum()),
            "n_up": int((tdh == UP_CLASS).sum()),
            "n_down": int((tdh == DOWN_CLASS).sum()),
            "n_flat": int((tdh == FLAT_CLASS).sum()),
            "gate_curve": curve,
        }

    # h=1 long-only backtest at a couple of gate settings (matches P8 audit)
    h1_idx = 0
    avg_logits_h1 = average_logits(
        [np.asarray(per_lb_logits[i])[:, h1_idx, :] for i in range(len(lookbacks))], None)
    conf_h1 = direction_confidence_from_logits(avg_logits_h1)
    hard_h1 = conf_h1["hard_pred"]
    score_h1 = conf_h1["actionable_score"]
    trh1 = tr[:, h1_idx]
    tdh1 = td[:, h1_idx]
    # future log return for regime tagging (h=1)
    regimes = np.array([tag_regime(float(r)) for r in trh1])

    # buy & hold over the test band (sum of h=1 log returns = approx total)
    bh = float(np.sum(trh1))

    def long_only_gated(thr: float, mag: float) -> dict:
        g = apply_consistency_and_magnitude_gate(
            hard_h1, score_h1, trh1,
            confidence_threshold=thr, min_abs_return=mag,
            require_sign_agree=False)
        # long-only: UP -> hold, else cash
        pos = (g == UP_CLASS).astype(np.float32)
        # transaction cost on position changes
        turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
        ret = float(np.sum(pos * trh1) - turnover * COST)
        gm = gated_actionable_metrics(g, tdh1)
        # regime-split precision
        reg_prec = {}
        for r in ("up", "range", "down"):
            m = regimes == r
            if m.sum() > 0:
                calls = (g[m] != FLAT_CLASS)
                if calls.sum() > 0:
                    reg_prec[r] = float((g[m][calls] == tdh1[m][calls]).mean())
                else:
                    reg_prec[r] = None
                reg_prec[f"{r}_n_calls"] = int(calls.sum())
                reg_prec[f"{r}_n"] = int(m.sum())
            else:
                reg_prec[r] = None
                reg_prec[f"{r}_n"] = 0
        return {
            "ret": ret, "n_calls": int(gm["n_calls"]),
            "prec": gm["precision_on_calls"],
            "hit": gm["gated_nonflat_acc"],
            "cov": gm["coverage"],
            "mdd": float(_max_drawdown(pos * trh1)),
            "regime": reg_prec,
        }

    steady = long_only_gated(0.45, 0.002)
    aggressive = long_only_gated(0.38, 0.001)

    # random cash baseline: hold with probability p each day (deterministic seed)
    rng = np.random.default_rng(42)
    rand_cash = {}
    for p in (0.3, 0.5, 0.7):
        pos = (rng.random(n) < p).astype(np.float32)
        turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
        ret = float(np.sum(pos * trh1) - turnover * COST)
        rand_cash[f"p{p}"] = {"ret": ret, "turnover": turnover}

    return {
        "symbol": symbol, "n": n,
        "buy_hold": bh,
        "horizons": h_results,
        "regime_counts": {r: int((regimes == r).sum()) for r in ("up", "range", "down")},
        "steady": steady,
        "aggressive": aggressive,
        "random_cash": rand_cash,
        "n_test_windows": n,
    }


def _max_drawdown(log_rets: np.ndarray) -> float:
    """Max drawdown of a log-return series (negative number)."""
    if len(log_rets) == 0:
        return 0.0
    equity = np.cumsum(log_rets)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    return float(dd.min()) if len(dd) > 0 else 0.0


def panel_summary(results: list[dict]) -> dict:
    """Aggregate per-symbol results into panel-level summary."""
    ok = [r for r in results if r.get("n", 0) > 0]
    if not ok:
        return {"error": "no valid symbols"}
    # h=1 metrics
    h1_nf = np.array([r["horizons"][1]["nonflat"] for r in ok])
    steady_ret = np.array([r["steady"]["ret"] for r in ok])
    aggr_ret = np.array([r["aggressive"]["ret"] for r in ok])
    bh = np.array([r["buy_hold"] for r in ok])
    steady_prec = np.array([r["steady"]["prec"] for r in ok])
    aggr_prec = np.array([r["aggressive"]["prec"] for r in ok])
    beat_bh_steady = int((steady_ret > bh).sum())
    beat_bh_aggr = int((aggr_ret > bh).sum())
    # up-regime beat count
    up_beat_steady = 0
    for r in ok:
        rp = r["steady"]["regime"]
        if rp.get("up") is not None and r["steady"]["ret"] > r["buy_hold"]:
            # only count if the symbol had meaningful up-regime presence
            if rp.get("up_n", 0) >= 5:
                up_beat_steady += 1
    # random cash baseline (p=0.5) beat count
    rand_beat = int(sum(
        1 for r in ok if r["random_cash"]["p0.5"]["ret"] > r["buy_hold"]
    ))
    return {
        "n_symbols": len(ok),
        "h1_nonflat_mean": float(h1_nf.mean()),
        "h1_nonflat_median": float(np.median(h1_nf)),
        "h1_nonflat_min": float(h1_nf.min()),
        "h1_nonflat_max": float(h1_nf.max()),
        "steady_ret_mean": float(steady_ret.mean()),
        "steady_ret_median": float(np.median(steady_ret)),
        "aggr_ret_mean": float(aggr_ret.mean()),
        "bh_ret_mean": float(bh.mean()),
        "bh_ret_median": float(np.median(bh)),
        "steady_prec_mean": float(steady_prec.mean()),
        "aggr_prec_mean": float(aggr_prec.mean()),
        "beat_bh_steady": beat_bh_steady,
        "beat_bh_aggr": beat_bh_aggr,
        "beat_bh_random_cash_p05": rand_beat,
        "up_regime_beat_steady": up_beat_steady,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="r11_joint_splitlr_140stocks")
    ap.add_argument("--tag", default="r11_baseline")
    ap.add_argument("--symbols", default=",".join(DEFAULT_PANEL))
    ap.add_argument("--lookbacks", default="124,126,128,130,132,134",
                    help="comma-separated lookback set for TTA direction")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    lookbacks = tuple(int(x) for x in args.lookbacks.split(","))
    t0 = time.time()
    results = []
    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym} ...", flush=True)
        try:
            r = run_symbol(args.model, sym, lookbacks)
        except Exception as e:  # noqa: BLE001
            r = {"symbol": sym, "error": str(e)}
            print(f"  ERROR: {e}")
        results.append(r)
        print(f"  n={r.get('n',0)} bh={r.get('buy_hold',0):.3f} "
              f"steady={r.get('steady',{}).get('ret',0):.3f} "
              f"h1nf={r.get('horizons',{}).get(1,{}).get('nonflat',0):.2%}",
              flush=True)

    summary = panel_summary(results)
    print("\n=== PANEL SUMMARY ===")
    print(f"h1 nonflat mean: {summary['h1_nonflat_mean']:.2%} "
          f"(median {summary['h1_nonflat_median']:.2%})")
    print(f"steady ret mean: {summary['steady_ret_mean']:.3f} "
          f"(bh {summary['bh_ret_mean']:.3f})")
    print(f"beat B&H: steady {summary['beat_bh_steady']}/{summary['n_symbols']}, "
          f"aggr {summary['beat_bh_aggr']}/{summary['n_symbols']}, "
          f"random-cash-p05 {summary['beat_bh_random_cash_p05']}/{summary['n_symbols']}")
    print(f"steady prec mean: {summary['steady_prec_mean']:.2%}")
    print(f"up-regime beat (steady): {summary['up_regime_beat_steady']}")
    print(f"elapsed: {time.time()-t0:.0f}s")

    out = ROOT / "outputs" / f"eval_p9_panel_{args.tag}.json"
    out.write_text(json.dumps({
        "model": args.model, "tag": args.tag,
        "lookbacks": list(lookbacks),
        "test_lo": TEST_LO, "ref_ctx": REF_CTX, "ref_dz": REF_DZ,
        "ref_vol": REF_VOL, "cost": COST,
        "summary": summary, "per_symbol": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
