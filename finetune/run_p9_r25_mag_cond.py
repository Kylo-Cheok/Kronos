"""Phase 9 R25 — Magnitude-conditional threshold (solve 688169 crash).

R24 found lb20_cond is panel-optimal (+3.027) but 688169 regresses
(+0.020 vs fixed +0.218). Hypothesis: cond thr lowers threshold in ALL
down-regimes, but crash-regime (idx<-0.05) bounces are dead-cat; only
mild-down (-0.05<=idx<0) bounces are real alpha.

R25 tests magnitude-conditional: crash → high thr (0.45/0.50),
mild-down → low thr (0.40), up → high thr (0.50). Uses 20d lookback.

Output: outputs/eval_p9_panel_r25_mag_cond.json
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

from evaluate_gated_ensemble import load_csv  # noqa: E402
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p4_r7_expanded_tta import average_logits  # noqa: E402
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p9_r19_bigpanel import (  # noqa: E402
    PANEL, REF_CTX, REF_DZ, REF_VOL, COST,
    make_test_windows, predict_all_horizons,
)
from run_tta_eval import DEVICE, HORIZONS  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)


def load_index_returns(n: int):
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return np.log(close / close.shift(n))


def get_index_ret(series, date):
    try:
        loc = series.index.get_indexer([date], method="pad")[0]
    except (KeyError, IndexError):
        return float("nan")
    if loc < 0:
        return float("nan")
    val = series.iloc[loc]
    return float(val) if pd.notna(val) else float("nan")


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def gate_mag_cond(hard, score, idx_arr, thr_crash, thr_milddn, thr_up, crash_bound):
    """Magnitude-conditional: crash (idx<crash_bound) → thr_crash,
    mild-down (crash_bound<=idx<0) → thr_milddn, up (idx>=0) → thr_up."""
    g = hard.copy()
    crash = (idx_arr < crash_bound) & ~np.isnan(idx_arr)
    milddn = (idx_arr >= crash_bound) & (idx_arr < 0) & ~np.isnan(idx_arr)
    up = (idx_arr >= 0) & ~np.isnan(idx_arr)
    nan = np.isnan(idx_arr)
    thr_map = np.full_like(score, thr_up, dtype=np.float64)
    thr_map[crash] = thr_crash
    thr_map[milddn] = thr_milddn
    thr_map[nan] = thr_up  # conservative
    g[score < thr_map] = FLAT_CLASS
    return g


def backtest(g, trh1, tdh1):
    pos = (g == UP_CLASS).astype(np.float32)
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    ret = float(np.sum(pos * trh1) - turnover * COST)
    gm = gated_actionable_metrics(g, tdh1)
    return {"ret": ret, "n_calls": int(gm["n_calls"]),
            "prec": gm["precision_on_calls"], "cov": gm["coverage"]}


def eval_config(store, bh, gate_fn):
    bt = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in store]
    ret = np.array([b["ret"] for b in bt])
    prec = np.array([b["prec"] for b in bt])
    cov = np.array([b["cov"] for b in bt])
    return {"ret_mean": float(ret.mean()), "beat_bh": int((ret > bh).sum()),
            "n_symbols": len(store),
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
            "total_ret": float(ret.sum()),
            "per_symbol": [{"symbol": s["symbol"], **b} for s, b in zip(store, bt)]}


def main():
    t0 = time.time()
    lookbacks = (124, 126, 128, 130, 132, 134)
    idx_series = load_index_returns(20)  # R24 found 20d optimal

    md = model_dir("r5_frozen_pool48")
    print(f"Loading r5_frozen_pool48...", flush=True)
    loaded = load_model(md)

    store = []
    print(f"=== Predicting panel ({len(PANEL)} symbols) ===", flush=True)
    for sym in PANEL:
        try:
            df = load_csv(sym)
        except Exception as e:
            print(f"  SKIP {sym}: {e}", flush=True)
            continue
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, idx_rets = [], [], []
        per_lb_logits = [[] for _ in lookbacks]
        for w in windows:
            drop = max(lookbacks) - REF_CTX
            close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
            tgt = make_multihorizon_targets(
                torch.from_numpy(close_ref).unsqueeze(0),
                context_length=REF_CTX, horizons=HORIZONS,
                min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
            td_all.append(tgt["direction"][0].cpu().numpy())
            tr_all.append(tgt["returns"][0].cpu().numpy())
            idx_rets.append(get_index_ret(idx_series, w["context_end_date"]))
            pred = predict_all_horizons(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        avg_lg = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        conf = direction_confidence_from_logits(avg_lg)
        store.append({
            "symbol": sym, "bh": float(np.sum(tr[:, 0])),
            "hard": conf["hard_pred"], "score": conf["actionable_score"],
            "trh1": tr[:, 0], "tdh1": td[:, 0], "idx_arr": np.asarray(idx_rets),
        })
        print(f"  {sym}: bh={store[-1]['bh']:.3f}", flush=True)

    n_sym = len(store)
    bh = np.array([s["bh"] for s in store])

    # Baselines + magnitude-conditional grid
    configs = [
        ("fixed_0.45", lambda s: gate_conf(s["hard"], s["score"], 0.45)),
        ("lb20_cond_0.40_0.50", lambda s: gate_mag_cond(
            s["hard"], s["score"], s["idx_arr"], 0.40, 0.40, 0.50, 0.0)),
    ]

    # Magnitude-conditional: crash_bound sweep + thr combos
    mag_configs = [
        # (thr_crash, thr_milddn, thr_up, crash_bound)
        (0.45, 0.40, 0.50, -0.05),
        (0.50, 0.40, 0.50, -0.05),
        (0.45, 0.35, 0.50, -0.05),
        (0.50, 0.35, 0.50, -0.05),
        (0.45, 0.40, 0.50, -0.03),
        (0.50, 0.40, 0.50, -0.03),
        (0.45, 0.40, 0.50, -0.08),
        (0.50, 0.40, 0.50, -0.08),
        (0.45, 0.40, 0.55, -0.05),
        (0.50, 0.40, 0.55, -0.05),
        (0.45, 0.35, 0.55, -0.05),
    ]
    for tc, tm, tu, cb in mag_configs:
        name = f"mag_{tc:.2f}_{tm:.2f}_{tu:.2f}_{cb:.2f}"
        configs.append((name, lambda s, tc=tc, tm=tm, tu=tu, cb=cb: gate_mag_cond(
            s["hard"], s["score"], s["idx_arr"], tc, tm, tu, cb)))

    results = {}
    print(f"\n=== R25 configs ({n_sym} symbols, 20d lookback) ===", flush=True)
    print(f"{'config':<30} {'ret_mean':>9} {'total':>8} {'beat':>7} {'prec':>7} {'cov':>7}", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        print(f"{name:<30} {r['ret_mean']:>+9.3f} {r['total_ret']:>+8.3f} "
              f"{r['beat_bh']:>3}/{n_sym}  {r['prec_mean']:>6.2%} {r['cov_mean']:>6.2%}", flush=True)

    # 688169 detail
    print(f"\n=== 688169 detail (bh=-0.412) ===", flush=True)
    sym688 = next((s for s in store if s["symbol"] == "688169"), None)
    if sym688:
        for name, gate_fn in configs:
            r = backtest(gate_fn(sym688), sym688["trh1"], sym688["tdh1"])
            print(f"  {name:<30} ret={r['ret']:+.3f} prec={r['prec']:.2%} cov={r['cov']:.2%}", flush=True)

    # Regime split
    print(f"\n=== Regime split ===", flush=True)
    up_syms = [s for s in store if s["bh"] > 0]
    dn_syms = [s for s in store if s["bh"] <= 0]
    bh_up = np.array([s["bh"] for s in up_syms])
    bh_dn = np.array([s["bh"] for s in dn_syms])
    print(f"{'config':<30} {'up_ret':>8} {'up_beat':>8} | {'dn_ret':>8} {'dn_beat':>8}", flush=True)
    for name, gate_fn in configs:
        bt_up = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in up_syms]
        bt_dn = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in dn_syms]
        ret_up = np.mean([b["ret"] for b in bt_up])
        ret_dn = np.mean([b["ret"] for b in bt_dn])
        beat_up = int(sum(b["ret"] > b_b for b, b_b in zip(bt_up, bh_up)))
        beat_dn = int(sum(b["ret"] > b_b for b, b_b in zip(bt_dn, bh_dn)))
        print(f"{name:<30} {ret_up:>+8.3f} {beat_up:>3}/{len(up_syms)}  | "
              f"{ret_dn:>+8.3f} {beat_dn:>3}/{len(dn_syms)}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r25_mag_cond.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
