"""Phase 9 R22 — Multi-regime conditional threshold refinement.

R21 broke through: cond_0.40_0.50 (CSI300<0 → 0.40, >=0 → 0.50) beat
csi300_fade on total_ret (+2.372 vs +2.305) AND cov (43.49% vs 25.67%).
R22 refines: (a) 3-regime split (strong_down/neutral/strong_up with
different thr); (b) continuous thr mapping; (c) different CSI300 lookback.

Hypothesis: binary split is coarse. A 3-regime or continuous mapping may
capture the "strong down = best bounce" effect better.

Output: outputs/eval_p9_panel_r22_multi_regime.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
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
    get_index_ret, load_index_returns, make_test_windows, predict_all_horizons,
)
from run_tta_eval import DEVICE, HORIZONS  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def gate_3regime(hard, score, idx_arr, thr_sd, thr_n, thr_su, lo_bound, hi_bound):
    """3-regime: strong_down (idx<lo) / neutral / strong_up (idx>hi)."""
    g = hard.copy()
    sd = (idx_arr < lo_bound) & ~np.isnan(idx_arr)
    su = (idx_arr > hi_bound) & ~np.isnan(idx_arr)
    neutral = ~(sd | su) & ~np.isnan(idx_arr)
    nan = np.isnan(idx_arr)
    # Apply strongest threshold first (neutral=su default), then lower for regimes
    g[score < thr_n] = FLAT_CLASS
    g[sd & (score < thr_sd)] = UP_CLASS  # re-enable if was wrongly killed? No.
    # Actually need per-point thr. Let's do it directly:
    g = hard.copy()
    thr_map = np.full_like(score, thr_n, dtype=np.float64)
    thr_map[sd] = thr_sd
    thr_map[su] = thr_su
    thr_map[nan] = thr_su  # conservative
    g[score < thr_map] = FLAT_CLASS
    return g


def gate_continuous(hard, score, idx_arr, thr_lo, thr_hi, idx_lo, idx_hi):
    """Continuous linear mapping: idx in [idx_lo, idx_hi] → thr in [thr_lo, thr_hi]."""
    g = hard.copy()
    valid = ~np.isnan(idx_arr)
    # Clip idx to [idx_lo, idx_hi], then linear interp
    clipped = np.clip(idx_arr, idx_lo, idx_hi)
    frac = (clipped - idx_lo) / (idx_hi - idx_lo)
    thr_map = thr_lo + frac * (thr_hi - thr_lo)
    nan = np.isnan(idx_arr)
    thr_map[nan] = thr_hi  # conservative
    g[score < thr_map] = FLAT_CLASS
    return g


def apply_fade(g, idx_arr):
    g = g.copy()
    valid = ~np.isnan(idx_arr)
    g[(g == UP_CLASS) & (idx_arr >= 0)] = FLAT_CLASS
    g[(g == UP_CLASS) & ~valid] = FLAT_CLASS
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
    idx_series = load_index_returns()
    lookbacks = (124, 126, 128, 130, 132, 134)

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

    # Baselines
    configs = [
        ("fixed_0.45", lambda s: gate_conf(s["hard"], s["score"], 0.45)),
        ("csi300_fade", lambda s: apply_fade(
            gate_conf(s["hard"], s["score"], 0.45), s["idx_arr"])),
        ("cond_0.40_0.50", lambda s: gate_3regime(
            s["hard"], s["score"], s["idx_arr"], 0.40, 0.40, 0.50, 0.0, 0.0)),
    ]

    # 3-regime: strong_down (idx<-0.02) / neutral (-0.02..0.02) / strong_up (>0.02)
    # Sweep thr for each regime
    three_regime_configs = [
        # (sd, n, su, lo, hi)
        (0.35, 0.45, 0.50, -0.02, 0.02),
        (0.35, 0.40, 0.50, -0.02, 0.02),
        (0.38, 0.40, 0.50, -0.02, 0.02),
        (0.35, 0.45, 0.55, -0.02, 0.02),
        (0.40, 0.40, 0.50, -0.03, 0.03),
        (0.35, 0.45, 0.50, -0.03, 0.03),
        (0.35, 0.45, 0.55, -0.03, 0.03),
        (0.30, 0.40, 0.50, -0.03, 0.03),
    ]
    for sd, n, su, lo, hi in three_regime_configs:
        name = f"3r_{sd:.2f}_{n:.2f}_{su:.2f}_{lo:.2f}_{hi:.2f}"
        configs.append((name, lambda s, sd=sd, n=n, su=su, lo=lo, hi=hi: gate_3regime(
            s["hard"], s["score"], s["idx_arr"], sd, n, su, lo, hi)))

    # Continuous: linear mapping idx in [idx_lo, idx_hi] → thr in [thr_lo, thr_hi]
    cont_configs = [
        # (thr_lo, thr_hi, idx_lo, idx_hi)
        (0.35, 0.50, -0.03, 0.03),
        (0.35, 0.55, -0.03, 0.03),
        (0.38, 0.50, -0.02, 0.02),
        (0.40, 0.50, -0.02, 0.02),
        (0.35, 0.50, -0.05, 0.05),
    ]
    for tl, th, il, ih in cont_configs:
        name = f"cont_{tl:.2f}_{th:.2f}_{il:.2f}_{ih:.2f}"
        configs.append((name, lambda s, tl=tl, th=th, il=il, ih=ih: gate_continuous(
            s["hard"], s["score"], s["idx_arr"], tl, th, il, ih)))

    results = {}
    print(f"\n=== R22 configs ({n_sym} symbols) ===", flush=True)
    print(f"{'config':<32} {'ret_mean':>9} {'total':>8} {'beat':>7} {'prec':>7} {'cov':>7}", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        print(f"{name:<32} {r['ret_mean']:>+9.3f} {r['total_ret']:>+8.3f} "
              f"{r['beat_bh']:>3}/{n_sym}  {r['prec_mean']:>6.2%} {r['cov_mean']:>6.2%}", flush=True)

    # Regime split for all
    print(f"\n=== Regime split ===", flush=True)
    up_syms = [s for s in store if s["bh"] > 0]
    dn_syms = [s for s in store if s["bh"] <= 0]
    bh_up = np.array([s["bh"] for s in up_syms])
    bh_dn = np.array([s["bh"] for s in dn_syms])
    print(f"{'config':<32} {'up_ret':>8} {'up_beat':>8} | {'dn_ret':>8} {'dn_beat':>8}", flush=True)
    for name, gate_fn in configs:
        bt_up = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in up_syms]
        bt_dn = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in dn_syms]
        ret_up = np.mean([b["ret"] for b in bt_up])
        ret_dn = np.mean([b["ret"] for b in bt_dn])
        beat_up = int(sum(b["ret"] > b_b for b, b_b in zip(bt_up, bh_up)))
        beat_dn = int(sum(b["ret"] > b_b for b, b_b in zip(bt_dn, bh_dn)))
        print(f"{name:<32} {ret_up:>+8.3f} {beat_up:>3}/{len(up_syms)}  | "
              f"{ret_dn:>+8.3f} {beat_dn:>3}/{len(dn_syms)}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r22_multi_regime.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
