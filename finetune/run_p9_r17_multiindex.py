"""Phase 9 R17 — Multi-index fade + sector-matched index + relative weakness.

R4 proved CSI300 5d fade is a free-lunch gate (prec +6.58pt, ret maintained).
But 688169 is a STAR-board stock; STAR50 may be a more relevant regime signal
than CSI300 for it. R17 tests:
  (a) each of 6 indices as the fade signal (CSI300/CSI500/ChiNext/STAR50/SSE/SZSE)
  (b) sector-matched fade: 688169 -> STAR50, others -> CSI300
  (c) relative-weakness fade for 688169: keep UP only when STAR50 5d ret <
      CSI300 5d ret (STAR50 relatively weak -> bounce likely), others use
      CSI300 fade

Leakage-safe: all index returns use closes up to context_end_date only.
Prediction runs once (r5); all gate variants are zero-cost post-hoc.

Output: outputs/eval_p9_panel_r17_multiindex.json
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
INDICES = ["csi300", "csi500", "chinext", "star50", "sse", "szse"]
# Sector mapping: STAR-board stock -> star50, everything else -> csi300.
SECTOR_MAP = {sym: ("star50" if sym == "688169" else "csi300") for sym in PANEL}


def load_index_returns(name: str, n: int = IDX_N) -> pd.Series:
    df = pd.read_csv(ROOT / "data" / "exogenous" / f"{name}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    close = df["close"]
    return np.log(close / close.shift(n))


def get_index_ret(series: pd.Series, date: pd.Timestamp) -> float:
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
        per_lb_logits.append(logits.astype(np.float64))
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
            "target_direction": target_dir,
            "target_return": target_ret}


def gate_conf(hard, score, thr):
    g = hard.copy()
    g[score < thr] = FLAT_CLASS
    return g


def apply_fade_arr(g, idx_arr):
    """Binary fade: drop UP calls when idx_arr >= 0 (up/flat market)."""
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
    # Load all index return series once.
    idx_series = {name: load_index_returns(name) for name in INDICES}
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
        td_all, tr_all = [], []
        idx_rets_by_name = {name: [] for name in INDICES}
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
            for name in INDICES:
                idx_rets_by_name[name].append(
                    get_index_ret(idx_series[name], w["context_end_date"]))
            pred = predict_all_horizons(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)

        confs = {}
        for h_idx in range(4):
            avg_lg = average_logits(
                [np.asarray(per_lb_logits[i])[:, h_idx, :] for i in range(len(lookbacks))], None)
            confs[h_idx] = direction_confidence_from_logits(avg_lg)

        tdh1 = td[:, 0]
        trh1 = tr[:, 0]
        bh = float(np.sum(trh1))
        entry = {
            "symbol": sym, "n": len(windows), "bh": bh,
            "hard": confs[0]["hard_pred"],
            "score": confs[0]["actionable_score"],
            "trh1": trh1, "tdh1": tdh1,
            "idx": {name: np.asarray(idx_rets_by_name[name]) for name in INDICES},
        }
        store.append(entry)
        print(f"  {sym}: bh={bh:.3f}", flush=True)

    bh = np.array([s["bh"] for s in store])

    # Build gate configurations.
    configs = [("base", lambda s: gate_conf(s["hard"], s["score"], 0.45))]
    # Single-index fade for each index.
    for name in INDICES:
        configs.append((f"{name}_fade",
                        lambda s, nm=name: apply_fade_arr(
                            gate_conf(s["hard"], s["score"], 0.45), s["idx"][nm])))
    # Sector-matched fade: 688169 -> star50, others -> csi300.
    def sector_fade(s):
        g = gate_conf(s["hard"], s["score"], 0.45)
        nm = SECTOR_MAP.get(s["symbol"], "csi300")
        return apply_fade_arr(g, s["idx"][nm])
    configs.append(("sector_fade", sector_fade))
    # Relative-weakness fade for 688169: keep UP only when STAR50 5d < CSI300 5d
    # (STAR50 relatively weak -> bounce likely); others use csi300 fade.
    def relweak_fade(s):
        g = gate_conf(s["hard"], s["score"], 0.45)
        if s["symbol"] == "688169":
            star = s["idx"]["star50"]
            csi = s["idx"]["csi300"]
            rel = star - csi  # relative weakness: <0 means STAR50 underperforms
            return apply_fade_arr(g, rel)
        return apply_fade_arr(g, s["idx"]["csi300"])
    configs.append(("relweak_fade", relweak_fade))

    results = {}
    print(f"\n=== {MODEL} multi-index configs ===", flush=True)
    for name, gate_fn in configs:
        r = eval_config(store, bh, gate_fn)
        results[name] = r
        print(f"  {name:<18} ret={r['ret_mean']:.3f} beat={r['beat_bh']}/9 "
              f"prec={r['prec_mean']:.2%} cov={r['cov_mean']:.2%}", flush=True)

    # Per-symbol breakdown for top configs.
    print(f"\n=== Per-symbol (csi300_fade vs sector_fade vs relweak_fade) ===", flush=True)
    for cfg in ["csi300_fade", "sector_fade", "relweak_fade"]:
        print(f"  -- {cfg} --", flush=True)
        for ps in results[cfg]["per_symbol"]:
            print(f"    {ps['symbol']}: ret={ps['ret']:.3f} prec={ps['prec']:.2%} "
                  f"cov={ps['cov']:.2%} n={ps['n_calls']}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r17_multiindex.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
