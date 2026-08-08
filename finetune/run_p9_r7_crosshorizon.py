"""Phase 9 R7 — cross-horizon consistency gate.

R6 proved legal sign-agree (within h=1) ineffective. R7 tests cross-horizon
consistency: h=1 UP call requires h=3/5/10 to also predict UP.

Phase 5 P5-8 used strict_h5 on 688169 (worked). Untested on panel.

Configs (all use |trh1|>=mag for magnitude, matching R4 baseline):
  1. base:     h=1 only, conf>=0.45
  2. fade:     base + idx5 fade (R4 optimal)
  3. h3_agree: h=1 UP requires h=3 UP
  4. h5_agree: h=1 UP requires h=5 UP
  5. h10_agree: h=1 UP requires h=10 UP
  6. h3_agree + fade
  7. h5_agree + fade
  8. h10_agree + fade

Output: outputs/eval_p9_panel_r7_crosshorizon.json
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
from run_tta_eval import CLIP, DEVICE, HORIZONS, PREDICT_WINDOW_LEN  # noqa: E402
from selective_prediction import (  # noqa: E402
    FLAT_CLASS, UP_CLASS, DOWN_CLASS,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
TEST_LO = "2025-12-15"
COST = 0.0005
GATE_MAG = 0.002
GATE_THR = 0.45
MODEL = "r5_frozen_pool48"
IDX_N = 5
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]
HORIZON_NAMES = {0: "h1", 1: "h3", 2: "h5", 3: "h10"}


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


@torch.no_grad()
def predict_all_horizons(loaded: dict, w: dict, lookbacks: tuple[int, ...]) -> dict:
    """Predict and return per-lookback logits for ALL horizons."""
    max_lb = max(lookbacks)
    per_lb_logits = []
    target_dir = None
    target_ret = None
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

    return {
        "per_lb_logits": per_lb_logits,
        "target_direction": target_dir,
        "target_return": target_ret,
    }


def gate_base(hard_h1, score_h1, trh1, thr, mag):
    """Base gate on h=1 only."""
    gated = hard_h1.copy()
    keep = (score_h1 >= thr) & (np.abs(trh1) >= mag)
    gated[~keep] = FLAT_CLASS
    return gated


def gate_cross_horizon(hard_h1, score_h1, trh1, hard_hX, thr, mag):
    """h=1 UP requires h=X also UP. DOWN calls unaffected (long-only)."""
    gated = gate_base(hard_h1, score_h1, trh1, thr, mag)
    # Only filter UP calls where h=X disagrees
    up_mask = (gated == UP_CLASS)
    disagree = up_mask & (hard_hX != UP_CLASS)
    gated[disagree] = FLAT_CLASS
    return gated


def apply_fade(g, idx_arr):
    g = g.copy()
    valid = ~np.isnan(idx_arr)
    g[(g == UP_CLASS) & (idx_arr >= 0)] = FLAT_CLASS
    g[(g == UP_CLASS) & ~valid] = FLAT_CLASS
    return g


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

    store = []
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
            pred = predict_all_horizons(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        idx_arr = np.asarray(idx_rets)

        # Compute confidence for each horizon
        confs = {}
        for h_idx in range(4):
            avg_lg = average_logits(
                [np.asarray(per_lb_logits[i])[:, h_idx, :] for i in range(len(lookbacks))], None)
            c = direction_confidence_from_logits(avg_lg)
            confs[h_idx] = c

        tdh1 = td[:, 0]
        trh1 = tr[:, 0]
        bh = float(np.sum(trh1))
        store.append({
            "symbol": sym, "n": len(windows), "bh": bh,
            "hard": {h: confs[h]["hard_pred"] for h in range(4)},
            "score": confs[0]["actionable_score"],  # h=1 score
            "trh1": trh1, "tdh1": tdh1, "idx_arr": idx_arr,
        })
        # Print cross-horizon agreement stats
        h1_up = (confs[0]["hard_pred"] == UP_CLASS)
        agree = {h: int(((confs[h]["hard_pred"] == UP_CLASS) & h1_up).sum()) for h in range(4)}
        print(f"  {sym}: bh={bh:.3f} h1_up={int(h1_up.sum())} "
              f"agree h3={agree[1]} h5={agree[2]} h10={agree[3]}", flush=True)

    bh = np.array([s["bh"] for s in store])

    configs = [
        ("base",        lambda s: gate_base(s["hard"][0], s["score"], s["trh1"], GATE_THR, GATE_MAG)),
        ("fade",        lambda s: apply_fade(gate_base(s["hard"][0], s["score"], s["trh1"], GATE_THR, GATE_MAG), s["idx_arr"])),
        ("h3_agree",    lambda s: gate_cross_horizon(s["hard"][0], s["score"], s["trh1"], s["hard"][1], GATE_THR, GATE_MAG)),
        ("h5_agree",    lambda s: gate_cross_horizon(s["hard"][0], s["score"], s["trh1"], s["hard"][2], GATE_THR, GATE_MAG)),
        ("h10_agree",   lambda s: gate_cross_horizon(s["hard"][0], s["score"], s["trh1"], s["hard"][3], GATE_THR, GATE_MAG)),
        ("h3+fade",     lambda s: apply_fade(gate_cross_horizon(s["hard"][0], s["score"], s["trh1"], s["hard"][1], GATE_THR, GATE_MAG), s["idx_arr"])),
        ("h5+fade",     lambda s: apply_fade(gate_cross_horizon(s["hard"][0], s["score"], s["trh1"], s["hard"][2], GATE_THR, GATE_MAG), s["idx_arr"])),
        ("h10+fade",    lambda s: apply_fade(gate_cross_horizon(s["hard"][0], s["score"], s["trh1"], s["hard"][3], GATE_THR, GATE_MAG), s["idx_arr"])),
    ]

    results = {}
    print("\n=== Config comparison ===", flush=True)
    for name, gate_fn in configs:
        bt = [backtest(gate_fn(s), s["trh1"], s["tdh1"]) for s in store]
        ret = np.array([b["ret"] for b in bt])
        prec = np.array([b["prec"] for b in bt])
        cov = np.array([b["cov"] for b in bt])
        beat = int((ret > bh).sum())
        results[name] = {
            "ret_mean": float(ret.mean()), "beat_bh": beat,
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
            "per_symbol": [{"symbol": s["symbol"], **b} for s, b in zip(store, bt)],
        }
        print(f"  {name:12s}: ret {ret.mean():.3f} beat {beat}/9 "
              f"prec {prec.mean():.2%} cov {cov.mean():.2%}", flush=True)

    # Per-symbol detail for best config
    best_name = max(results, key=lambda k: results[k]["prec_mean"] if results[k]["cov_mean"] > 0.05 else 0)
    print(f"\n=== {best_name} per-symbol ===", flush=True)
    best_fn = dict(configs)[best_name]
    for s in store:
        b = backtest(best_fn(s), s["trh1"], s["tdh1"])
        print(f"  {s['symbol']}: bh={s['bh']:.3f} ret={b['ret']:.3f} "
              f"prec={b['prec']:.2%} cov={b['cov']:.2%} n_calls={b['n_calls']}", flush=True)

    out_path = ROOT / "outputs" / "eval_p9_panel_r7_crosshorizon.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
