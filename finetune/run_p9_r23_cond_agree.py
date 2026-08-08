"""Phase 9 R23 — cond thr + cross-horizon agree combination.

R21 found cond_0.40_0.50 is optimal (regime-based). R19 showed h3_agree
(cross-horizon consistency) beat 22/28 with different mechanism. R23 tests
if combining them is complementary: cond thr sets regime-aware threshold,
h3_agree filters UP calls where h3 disagrees.

Also tests: cond thr + selective h3 agree (only in up-regime), and
cond thr + magnitude gate (using h1 return pred, re-tested in panel context).

Output: outputs/eval_p9_panel_r23_cond_agree.json
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


def gate_cond_thr(hard, score, idx_arr, thr_lo, thr_hi):
    g = hard.copy()
    dn = (idx_arr < 0) & ~np.isnan(idx_arr)
    up = (idx_arr >= 0) & ~np.isnan(idx_arr)
    nan = np.isnan(idx_arr)
    g[score < thr_lo] = FLAT_CLASS
    g[up & (score < thr_hi)] = FLAT_CLASS
    g[nan & (score < thr_hi)] = FLAT_CLASS
    return g


def apply_h3_agree(g, hard_h3):
    """Filter UP calls where h3 hard pred disagrees."""
    g = g.copy()
    up_mask = (g == UP_CLASS)
    disagree = up_mask & (hard_h3 != UP_CLASS)
    g[disagree] = FLAT_CLASS
    return g


def apply_h3_agree_dn_only(g, hard_h3, idx_arr):
    """h3 agree only in down-regime (CSI300<0)."""
    g = g.copy()
    up_mask = (g == UP_CLASS)
    dn = (idx_arr < 0) & ~np.isnan(idx_arr)
    disagree = up_mask & dn & (hard_h3 != UP_CLASS)
    g[disagree] = FLAT_CLASS
    return g


def apply_h3_agree_up_only(g, hard_h3, idx_arr):
    """h3 agree only in up-regime (CSI300>=0)."""
    g = g.copy()
    up_mask = (g == UP_CLASS)
    up = (idx_arr >= 0) & ~np.isnan(idx_arr)
    disagree = up_mask & up & (hard_h3 != UP_CLASS)
    g[disagree] = FLAT_CLASS
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
        # h=0 and h=1 (h3) confidences
        confs = {}
        for h_idx in [0, 1]:
            avg_lg = average_logits(
                [np.asarray(per_lb_logits[i])[:, h_idx, :] for i in range(len(lookbacks))], None)
            confs[h_idx] = direction_confidence_from_logits(avg_lg)
        store.append({
            "symbol": sym, "bh": float(np.sum(tr[:, 0])),
            "hard": confs[0]["hard_pred"], "hard_h3": confs[1]["hard_pred"],
            "score": confs[0]["actionable_score"],
            "trh1": tr[:, 0], "tdh1": td[:, 0], "idx_arr": np.asarray(idx_rets),
        })
        print(f"  {sym}: bh={store[-1]['bh']:.3f}", flush=True)

    n_sym = len(store)
    bh = np.array([s["bh"] for s in store])

    configs = [
        ("fixed_0.45", lambda s: gate_conf(s["hard"], s["score"], 0.45)),
        ("cond_0.40_0.50", lambda s: gate_cond_thr(
            s["hard"], s["score"], s["idx_arr"], 0.40, 0.50)),
        ("cond+h3agree", lambda s: apply_h3_agree(
            gate_cond_thr(s["hard"], s["score"], s["idx_arr"], 0.40, 0.50),
            s["hard_h3"])),
        ("cond+h3agree_dn", lambda s: apply_h3_agree_dn_only(
            gate_cond_thr(s["hard"], s["score"], s["idx_arr"], 0.40, 0.50),
            s["hard_h3"], s["idx_arr"])),
        ("cond+h3agree_up", lambda s: apply_h3_agree_up_only(
            gate_cond_thr(s["hard"], s["score"], s["idx_arr"], 0.40, 0.50),
            s["hard_h3"], s["idx_arr"])),
        # Also test h3_agree alone on fixed 0.45 for reference
        ("fixed+h3agree", lambda s: apply_h3_agree(
            gate_conf(s["hard"], s["score"], 0.45), s["hard_h3"])),
        # cond with different thr + h3agree
        ("cond_0.40_0.48+h3", lambda s: apply_h3_agree(
            gate_cond_thr(s["hard"], s["score"], s["idx_arr"], 0.40, 0.48),
            s["hard_h3"])),
        ("cond_0.35_0.50+h3", lambda s: apply_h3_agree(
            gate_cond_thr(s["hard"], s["score"], s["idx_arr"], 0.35, 0.50),
            s["hard_h3"])),
    ]

    results = {}
    print(f"\n=== R23 configs ({n_sym} symbols) ===", flush=True)
    print(f"{'config':<24} {'ret_mean':>9} {'total':>8} {'beat':>7} {'prec':>7} {'cov':>7}", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        print(f"{name:<24} {r['ret_mean']:>+9.3f} {r['total_ret']:>+8.3f} "
              f"{r['beat_bh']:>3}/{n_sym}  {r['prec_mean']:>6.2%} {r['cov_mean']:>6.2%}", flush=True)

    # Regime split
    print(f"\n=== Regime split ===", flush=True)
    up_syms = [s for s in store if s["bh"] > 0]
    dn_syms = [s for s in store if s["bh"] <= 0]
    bh_up = np.array([s["bh"] for s in up_syms])
    bh_dn = np.array([s["bh"] for s in dn_syms])
    print(f"{'config':<24} {'up_ret':>8} {'up_beat':>8} | {'dn_ret':>8} {'dn_beat':>8}", flush=True)
    for name, gate_fn in configs:
        bt_up = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in up_syms]
        bt_dn = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in dn_syms]
        ret_up = np.mean([b["ret"] for b in bt_up])
        ret_dn = np.mean([b["ret"] for b in bt_dn])
        beat_up = int(sum(b["ret"] > b_b for b, b_b in zip(bt_up, bh_up)))
        beat_dn = int(sum(b["ret"] > b_b for b, b_b in zip(bt_dn, bh_dn)))
        print(f"{name:<24} {ret_up:>+8.3f} {beat_up:>3}/{len(up_syms)}  | "
              f"{ret_dn:>+8.3f} {beat_dn:>3}/{len(dn_syms)}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r23_cond_agree.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
