"""Phase 9 R6 — legal sign-agree gate using model return prediction.

R5 bug version (sign agree with TRUE returns) hit prec 71.90% — leakage.
R6 tests LEGAL sign agree: use model's return_head prediction (not true return)
to check direction/return consistency. No future info leaked.

Also separates magnitude source:
  - R1/R4 baseline: |true_return| >= mag  (LEAKAGE — uses future)
  - R6 legal:       |pred_return| >= mag  (no leakage)

Sweep configs:
  1. base_trh1:   conf>=0.45 & |trh1|>=0.002              (R1/R4 baseline, leaked mag)
  2. fade_trh1:   base_trh1 + idx5 fade                    (R4 optimal, leaked mag)
  3. base_pred:   conf>=0.45 & |pred_ret|>=0.002           (legal mag, no sign agree)
  4. sign_agree:  base_pred & sign_agree(hard, pred_ret)   (legal sign agree)
  5. sa_fade:     sign_agree + idx5 fade                   (legal sign agree + fade)
  6. sa_fade_sweep: sign_agree + fade, sweep conf threshold

Output: outputs/eval_p9_panel_r6_signagree.json
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
    average_logits, derive_time_features, make_tta_windows,
    normalize_with_lookback,
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
MODEL = "r5_frozen_pool48"
IDX_N = 5
PANEL = ["000001", "000002", "000063", "000333", "000651",
         "002415", "600036", "601318", "688169"]


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
def predict_with_returns(loaded: dict, w: dict, lookbacks: tuple[int, ...]) -> dict:
    """Like predict_per_lookback but also returns return_prediction per lookback."""
    max_lb = max(lookbacks)
    per_lb_logits = []
    per_lb_returns = []
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
        ret_pred = outputs["return_prediction"][0].cpu().numpy()
        per_lb_logits.append(logits.astype(np.float64))
        per_lb_returns.append(ret_pred.astype(np.float64))

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
        "per_lb_returns": per_lb_returns,
        "target_direction": target_dir,
        "target_return": target_ret,
        "context_end_date": w["context_end_date"],
    }


def gate_base(hard, score, mag_source, thr, mag):
    """Base gate: conf >= thr & |mag_source| >= mag. No sign agree."""
    gated = hard.copy()
    keep = (score >= thr) & (np.abs(mag_source) >= mag)
    gated[~keep] = FLAT_CLASS
    return gated


def gate_sign_agree(hard, score, pred_ret, thr, mag):
    """Sign agree gate: conf >= thr & |pred_ret| >= mag & sign(hard)==sign(pred_ret)."""
    gated = hard.copy()
    keep = (score >= thr) & (np.abs(pred_ret) >= mag)
    agree_up = (hard == UP_CLASS) & (pred_ret > 0.0)
    agree_dn = (hard == DOWN_CLASS) & (pred_ret < 0.0)
    keep = keep & (agree_up | agree_dn | (hard == FLAT_CLASS))
    gated[~keep] = FLAT_CLASS
    return gated


def apply_fade(g, idx_arr):
    """Binary fade: keep UP only when index down."""
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
        td_all, tr_all, per_lb_logits, per_lb_returns, idx_rets = [], [], [[] for _ in lookbacks], [[] for _ in lookbacks], []
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
            pred = predict_with_returns(loaded, w, lookbacks)
            for lb_i in range(len(lookbacks)):
                per_lb_logits[lb_i].append(pred["per_lb_logits"][lb_i])
                per_lb_returns[lb_i].append(pred["per_lb_returns"][lb_i])
        td = np.asarray(td_all)
        tr = np.asarray(tr_all)
        idx_arr = np.asarray(idx_rets)
        avg_logits = average_logits(
            [np.asarray(per_lb_logits[i])[:, 0, :] for i in range(len(lookbacks))], None)
        # TTA-average return predictions (h=1 only)
        avg_returns = np.mean([np.asarray(per_lb_returns[i])[:, 0] for i in range(len(lookbacks))], axis=0)
        conf = direction_confidence_from_logits(avg_logits)
        hard = conf["hard_pred"]
        score = conf["actionable_score"]
        tdh1 = td[:, 0]
        trh1 = tr[:, 0]
        bh = float(np.sum(trh1))
        store.append({
            "symbol": sym, "n": len(windows), "bh": bh,
            "hard": hard, "score": score, "trh1": trh1, "tdh1": tdh1,
            "pred_ret": avg_returns, "idx_arr": idx_arr,
        })
        print(f"  {sym}: bh={bh:.3f} pred_ret range [{avg_returns.min():.4f},{avg_returns.max():.4f}] "
              f"mean|pred|={np.abs(avg_returns).mean():.4f}", flush=True)

    bh = np.array([s["bh"] for s in store])

    # Configs to evaluate
    configs = [
        ("base_trh1", lambda s: gate_base(s["hard"], s["score"], s["trh1"], 0.45, GATE_MAG)),
        ("fade_trh1", lambda s: apply_fade(gate_base(s["hard"], s["score"], s["trh1"], 0.45, GATE_MAG), s["idx_arr"])),
        ("base_pred", lambda s: gate_base(s["hard"], s["score"], s["pred_ret"], 0.45, GATE_MAG)),
        ("sign_agree", lambda s: gate_sign_agree(s["hard"], s["score"], s["pred_ret"], 0.45, GATE_MAG)),
        ("sa_fade", lambda s: apply_fade(gate_sign_agree(s["hard"], s["score"], s["pred_ret"], 0.45, GATE_MAG), s["idx_arr"])),
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

    # Sweep sign_agree+fade confidence threshold
    print("\n=== sa_fade conf sweep ===", flush=True)
    sa_fade_sweep = []
    for thr in (0.35, 0.40, 0.45, 0.50, 0.55):
        bt = [backtest(
            apply_fade(gate_sign_agree(s["hard"], s["score"], s["pred_ret"], thr, GATE_MAG), s["idx_arr"]),
            s["trh1"], s["tdh1"]) for s in store]
        ret = np.array([b["ret"] for b in bt])
        prec = np.array([b["prec"] for b in bt])
        cov = np.array([b["cov"] for b in bt])
        beat = int((ret > bh).sum())
        sa_fade_sweep.append({
            "thr": thr, "ret_mean": float(ret.mean()), "beat_bh": beat,
            "prec_mean": float(prec.mean()), "cov_mean": float(cov.mean()),
        })
        print(f"  thr={thr:.2f}: ret {ret.mean():.3f} beat {beat}/9 "
              f"prec {prec.mean():.2%} cov {cov.mean():.2%}", flush=True)

    # Per-symbol detail for best sa_fade config
    best_sweep = max(sa_fade_sweep, key=lambda x: x["prec_mean"] if x["cov_mean"] > 0.05 else 0)
    best_thr = best_sweep["thr"]
    print(f"\n=== sa_fade per-symbol (thr={best_thr}) ===", flush=True)
    for s in store:
        g = apply_fade(gate_sign_agree(s["hard"], s["score"], s["pred_ret"], best_thr, GATE_MAG), s["idx_arr"])
        b = backtest(g, s["trh1"], s["tdh1"])
        print(f"  {s['symbol']}: bh={s['bh']:.3f} ret={b['ret']:.3f} "
              f"prec={b['prec']:.2%} cov={b['cov']:.2%} n_calls={b['n_calls']}", flush=True)

    out = {"configs": results, "sa_fade_sweep": sa_fade_sweep}
    out_path = ROOT / "outputs" / "eval_p9_panel_r6_signagree.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
