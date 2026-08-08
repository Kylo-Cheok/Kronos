"""Port the long-only strategies (P5-8 params / aggressive) to other symbols.

Parameters were tuned on 688169; the 8 panel symbols are untouched by that
tuning, so this is a frozen-parameter transfer test.

Per symbol we cache ONLY what the long-only gate needs (compact npz):
  hard1/score1 (r10, 4lb_left TTA), hard3 (r5, 4lb_left), hard5 (r5, 6lb),
  hard10 (r10, 4lb_right), sl r10 h=1 pret (lb=128), tr1, td1, dates.
Targets: fixed reference (ctx=122, dz=0.003, vol=0.5).

Usage:
  python finetune/run_p8_ms_long_only.py --symbols 000001   # build cache
  python finetune/run_p8_ms_long_only.py --report           # aggregate table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import (  # noqa: E402
    FEATURES, LOOKBACK, PREDICT_WINDOW, WINDOW, load_csv,
)
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits, predict_per_lookback  # noqa: E402
from run_tta_eval import make_tta_windows, normalize_with_lookback, derive_time_features  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from selective_prediction import (  # noqa: E402
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
)

HORIZONS = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
COST = 0.0005
OUT_DIR = ROOT / "outputs" / "p8_ms_gate_inputs"

TTA = {1: (124, 126, 128, 130), 3: (124, 126, 128, 130),
       5: (124, 126, 128, 130, 132, 134), 10: (126, 128, 130, 132)}
MODEL_OF = {1: PRIMARY, 3: SECONDARY, 5: SECONDARY, 10: PRIMARY}

STRATEGIES = {
    "steady": {"thr": 0.45, "mag": 0.002},      # 稳健型
    "aggressive": {"thr": 0.38, "mag": 0.001},  # 进取型
}


@torch.no_grad()
def _unused():  # placeholder removed
    pass


def build_symbol(sym: str, loaded: dict) -> Path | None:
    df = load_csv(sym)
    windows = make_tta_windows(df, ALL_LOOKBACKS)
    if len(windows) < 60:
        print(f"{sym}: only {len(windows)} windows, skip")
        return None
    dates, hards, scores = [], {h: [] for h in HORIZONS}, {1: [], 5: []}
    tr1, td1 = [], []
    for w in windows:
        dates.append(w["context_end_date"])
        drop = max(ALL_LOOKBACKS) - REF_CTX
        close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
        tgt = make_multihorizon_targets(
            torch.from_numpy(close_ref).unsqueeze(0),
            context_length=REF_CTX, horizons=HORIZONS,
            min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
        td1.append(int(tgt["direction"][0, 0].cpu()))
        tr1.append(float(tgt["returns"][0, 0].cpu()))
        for h in HORIZONS:
            mod = loaded[MODEL_OF[h]]
            pred = predict_per_lookback(mod, w, TTA[h])
            avg = average_logits([p[HORIZONS.index(h)] for p in pred["per_lb_logits"]], None)
            conf = direction_confidence_from_logits(avg[None, :])
            hards[h].append(int(conf["hard_pred"][0]))
            if h in (1, 5):
                scores[h].append(float(conf["actionable_score"][0]))
    # SL returns need the return head: do a dedicated pass at lb=128
    sl_ret_vals = []
    mod = loaded[PRIMARY]
    for w in windows:
        drop = max(ALL_LOOKBACKS) - 128
        x_full = w["features"][drop: drop + 128 + PREDICT_WINDOW + 1]
        ts_full = w["timestamps"].iloc[drop: drop + 128 + PREDICT_WINDOW + 1]
        x_norm = normalize_with_lookback(x_full, 128)
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(
            next(mod["model"].parameters()).device)
        stamp = derive_time_features(ts_full)
        stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(x_tensor.device)
        tok0, tok1 = mod["tokenizer"].encode(x_tensor, half=True)
        _, _, hidden = mod["model"](tok0[:, :-1], tok1[:, :-1],
                                    stamp_tensor[:, :-1, :], return_context=True)
        out = mod["head"](hidden, context_length=128)
        sl_ret_vals.append(float(out["return_prediction"][0, 0].cpu()))

    out = OUT_DIR / f"{sym}.npz"
    np.savez_compressed(
        out, dates=np.asarray(dates),
        hard1=np.asarray(hards[1]), hard3=np.asarray(hards[3]),
        hard5=np.asarray(hards[5]), hard10=np.asarray(hards[10]),
        score1=np.asarray(scores[1]), score5=np.asarray(scores[5]),
        sl_r10_h1_pret=np.asarray(sl_ret_vals),
        tr1=np.asarray(tr1), td1=np.asarray(td1))
    print(f"{sym}: {len(dates)} windows -> {out}", flush=True)
    return out


def report():
    rows = []
    for path in sorted(OUT_DIR.glob("*.npz")):
        sym = path.stem
        d = np.load(path, allow_pickle=False)
        hard1, hard5 = d["hard1"], d["hard5"]
        score1, mag_ret = d["score1"], d["sl_r10_h1_pret"]
        tr1, td1 = d["tr1"], d["td1"]
        bh = float(np.exp(np.sum(tr1)) - 1)
        row = {"symbol": sym, "n": len(tr1), "buy_hold": bh}
        for tag, cfg in STRATEGIES.items():
            g = apply_consistency_and_magnitude_gate(
                hard1, score1, mag_ret, confidence_threshold=cfg["thr"],
                min_abs_return=cfg["mag"], require_sign_agree=False)
            g = apply_consistency_filter(g, hard1, hard5, "strict")
            pos = (np.asarray(g) == 2).astype(float)
            simple = np.exp(tr1) - 1.0
            pnl = pos * simple - COST * (pos > 0)
            eq = np.cumprod(1 + pnl)
            n = int((pos > 0).sum())
            peak = np.maximum.accumulate(eq)
            row[tag] = {
                "ret": float(eq[-1] - 1), "n": n,
                "prec": float(np.mean(td1[pos > 0] == 2)) if n else None,
                "hit": float(np.mean(pnl[pos > 0] > 0)) if n else None,
                "mdd": float(np.min(eq / peak - 1)),
            }
        rows.append(row)
        parts = []
        for t in STRATEGIES:
            if row[t]["n"]:
                parts.append(f"{t}: {row[t]['ret']:+.1%} n={row[t]['n']} "
                             f"prec={row[t]['prec']:.0%}")
            else:
                parts.append(f"{t}: no trades")
        print(f"{sym}: bh={bh:+.1%} | " + " | ".join(parts), flush=True)
    valid = [r for r in rows]
    for tag in STRATEGIES:
        rets = [r[tag]["ret"] for r in valid]
        print(f"\n[{tag}] mean ret {np.mean(rets):+.1%}, median "
              f"{np.median(rets):+.1%}, beat bh on "
              f"{sum(1 for r in valid if r[tag]['ret'] > r['buy_hold'])}/{len(valid)}")
    bh_all = [r["buy_hold"] for r in valid]
    print(f"[buy&hold] mean {np.mean(bh_all):+.1%}, median {np.median(bh_all):+.1%}")
    out = ROOT / "outputs" / "eval_p8_ms_long_only.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    if args.report:
        report()
        return 0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    loaded = {m: load_model(model_dir(m)) for m in (PRIMARY, SECONDARY)}
    for sym in symbols:
        if (OUT_DIR / f"{sym}.npz").exists():
            print(f"{sym}: cached")
            continue
        build_symbol(sym, loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
