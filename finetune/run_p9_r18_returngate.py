"""Phase 9 R18 — Return-head prediction as gate signal.

Key finding from r5 summary.json: r5 was trained WITHOUT consistency loss
(consistency_loss_weight=0). The return head is therefore independently
trained (sharing only the trunk representation with the direction head) and
may carry magnitude information that direction confidence does not.

R6 tested sign-agree (ineffective) and magnitude-threshold (ineffective
because |pred|~0.04 all exceed the 0.002 threshold). R6 did NOT test
return-pred RANK as the gate signal. R18 fills that gap:
  (a) return_pred quantile gate: keep UP calls only when pred_ret_h1 is in
      the top-k% within each symbol (per-symbol ranking to handle scale
      differences across stocks)
  (b) return_pred threshold gate: keep UP calls when pred_ret_h1 > thr
      (absolute threshold sweep)
  (c) direction_conf AND return_pred quantile (intersection)
  (d) csi300_fade AND return_pred quantile

All gates start from the direction UP calls (hard_pred==UP at thr 0.45),
then further filter by return pred. Zero training cost.

Output: outputs/eval_p9_panel_r18_returngate.json
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
    average_logits, derive_time_features, normalize_with_lookback,
)
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_tta_eval import DEVICE, HORIZONS, PREDICT_WINDOW_LEN  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
MODEL = "r5_frozen_pool48"
IDX_N = 5
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]


def load_index_returns():
    df = pd.read_csv(ROOT / "data" / "exogenous" / "csi300.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return np.log(close / close.shift(IDX_N))


def get_index_ret(series, date):
    try:
        loc = series.index.get_indexer([date], method="pad")[0]
    except (KeyError, IndexError):
        return float("nan")
    if loc < 0:
        return float("nan")
    val = series.iloc[loc]
    return float(val) if pd.notna(val) else float("nan")


def make_test_windows(df, lookbacks):
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


@torch.no_grad()
def predict_all_horizons(loaded, w, lookbacks):
    max_lb = max(lookbacks)
    per_lb_logits = []
    per_lb_ret = []
    target_dir = target_ret = None
    for lb in lookbacks:
        drop = max_lb - lb
        x_full = w["features"][drop: drop + lb + PREDICT_WINDOW_LEN + 1]
        close_full = w["raw_close"][drop: drop + lb + PREDICT_WINDOW_LEN + 1]
        ts_full = w["timestamps"].iloc[drop: drop + lb + PREDICT_WINDOW_LEN + 1]
        x_norm = normalize_with_lookback(x_full, lb)
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
        stamp = derive_time_features(ts_full)
        stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
        raw_close_tensor = torch.from_numpy(close_full).unsqueeze(0).to(DEVICE)
        token_seq_0, token_seq_1 = loaded["tokenizer"].encode(x_tensor, half=True)
        _, _, hidden_states = loaded["model"](
            token_seq_0[:, :-1], token_seq_1[:, :-1],
            stamp_tensor[:, :-1, :], return_context=True,
        )
        outputs = loaded["head"](hidden_states, context_length=lb)
        logits = outputs["direction_logits"][0].cpu().numpy()
        ret_pred = outputs["return_prediction"][0].cpu().numpy()  # [H]
        per_lb_logits.append(logits.astype(np.float64))
        per_lb_ret.append(ret_pred.astype(np.float64))
        if target_dir is None:
            targets = make_multihorizon_targets(
                raw_close_tensor, context_length=lb,
                horizons=loaded["horizons"],
                min_deadzone=loaded["min_deadzone"],
                volatility_multiplier=loaded["vol_mult"],
            )
            target_dir = targets["direction"][0].cpu().numpy()
            target_ret = targets["returns"][0].cpu().numpy()
    return {"per_lb_logits": per_lb_logits,
            "per_lb_ret": per_lb_ret,
            "target_direction": target_dir,
            "target_return": target_ret}


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def apply_fade(g, idx_arr):
    g = g.copy()
    valid = ~np.isnan(idx_arr)
    g[(g == UP_CLASS) & (idx_arr >= 0)] = FLAT_CLASS
    g[(g == UP_CLASS) & ~valid] = FLAT_CLASS
    return g


def gate_ret_quantile(base_g, ret_pred, q):
    """Keep UP calls only when pred_ret is in top-(q*100)% within the symbol."""
    g = base_g.copy()
    up_mask = (g == UP_CLASS)
    if not up_mask.any():
        return g
    # Rank all pred_ret; keep UP calls whose pred_ret is in top q fraction.
    thr_val = np.nanquantile(ret_pred, 1.0 - q)
    g[up_mask & (ret_pred < thr_val)] = FLAT_CLASS
    return g


def gate_ret_thr(base_g, ret_pred, thr):
    g = base_g.copy()
    up_mask = (g == UP_CLASS)
    g[up_mask & (ret_pred < thr)] = FLAT_CLASS
    return g


def backtest(g, trh1, tdh1):
    pos = (g == UP_CLASS).astype(np.float32)
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum())
    ret = float(np.sum(pos * trh1) - turnover * COST)
    gm = gated_actionable_metrics(g, tdh1)
    return {"ret": ret, "n_calls": int(gm["n_calls"]),
            "prec": gm["precision_on_calls"],
            "cov": gm["coverage"]}


def eval_config(store, bh, gate_fn):
    bt = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in store]
    ret = np.array([b["ret"] for b in bt])
    prec = np.array([b["prec"] for b in bt])
    cov = np.array([b["cov"] for b in bt])
    return {"ret_mean": float(ret.mean()), "beat_bh": int((ret > bh).sum()),
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
            "per_symbol": [{"symbol": s["symbol"], **b} for s, b in zip(store, bt)]}


def main():
    t0 = time.time()
    idx_series = load_index_returns()
    lookbacks = (124, 126, 128, 130, 132, 134)

    md = model_dir(MODEL)
    print(f"Loading {MODEL}...", flush=True)
    loaded = load_model(md)

    store = []
    print(f"=== Predicting panel ({MODEL}) ===", flush=True)
    for sym in PANEL:
        df = load_csv(sym)
        windows = make_test_windows(df, lookbacks)
        if not windows:
            continue
        td_all, tr_all, idx_rets = [], [], []
        per_lb_logits = [[] for _ in lookbacks]
        per_lb_ret = [[] for _ in lookbacks]
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
                per_lb_ret[lb_i].append(pred["per_lb_ret"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)

        # TTA-average direction logits.
        avg_lg_h1 = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        confs = direction_confidence_from_logits(avg_lg_h1)
        # TTA-average return predictions (h=1).
        ret_pred_h1 = np.mean([np.asarray(per_lb_ret[i])[:, 0] for i in range(len(lookbacks))], axis=0)

        tdh1 = td[:, 0]
        trh1 = tr[:, 0]
        bh = float(np.sum(trh1))
        store.append({
            "symbol": sym, "n": len(windows), "bh": bh,
            "hard": confs["hard_pred"],
            "score": confs["actionable_score"],
            "ret_pred": ret_pred_h1,
            "trh1": trh1, "tdh1": tdh1,
            "idx_arr": np.asarray(idx_rets),
        })
        print(f"  {sym}: bh={bh:.3f} ret_pred mean={ret_pred_h1.mean():.4f} "
              f"std={ret_pred_h1.std():.4f}", flush=True)

    bh = np.array([s["bh"] for s in store])

    configs = [
        ("base", lambda s: gate_conf(s["hard"], s["score"], 0.45)),
        ("csi300_fade", lambda s: apply_fade(
            gate_conf(s["hard"], s["score"], 0.45), s["idx_arr"])),
    ]
    # Return-pred quantile gates (per-symbol ranking).
    for q in (0.30, 0.50, 0.70):
        configs.append((f"ret_q{int(q*100)}",
                        lambda s, qq=q: gate_ret_quantile(
                            gate_conf(s["hard"], s["score"], 0.45), s["ret_pred"], qq)))
    # Return-pred quantile + csi300 fade.
    for q in (0.30, 0.50, 0.70):
        configs.append((f"fade_ret_q{int(q*100)}",
                        lambda s, qq=q: gate_ret_quantile(
                            apply_fade(gate_conf(s["hard"], s["score"], 0.45), s["idx_arr"]),
                            s["ret_pred"], qq)))
    # Return-pred absolute threshold sweep.
    for thr in (0.0, 0.005, 0.01):
        configs.append((f"ret_thr{thr}",
                        lambda s, tt=thr: gate_ret_thr(
                            gate_conf(s["hard"], s["score"], 0.45), s["ret_pred"], tt)))

    results = {}
    print(f"\n=== {MODEL} return-gate configs ===", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        print(f"  {name:<16} ret={r['ret_mean']:.3f} beat={r['beat_bh']}/9 "
              f"prec={r['prec_mean']:.2%} cov={r['cov_mean']:.2%}", flush=True)

    # Correlation between direction confidence and return pred (diagnostic).
    print(f"\n=== Diagnostic: dir_conf vs ret_pred correlation ===", flush=True)
    for s in store:
        up_mask = (s["hard"] == UP_CLASS)
        if up_mask.sum() > 5:
            corr = float(np.corrcoef(s["score"][up_mask], s["ret_pred"][up_mask])[0, 1])
        else:
            corr = float("nan")
        print(f"  {s['symbol']}: corr={corr:.3f} (n_up={int(up_mask.sum())})", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r18_returngate.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
