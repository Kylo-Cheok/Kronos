"""Forward prediction for 688169 with the promoted P5-8 production config.

One-command usage (from repo root, project venv):

    .venv/Scripts/python.exe finetune/predict_688169_promoted.py
    .venv/Scripts/python.exe finetune/predict_688169_promoted.py --symbol 688169

Promoted recipe (source of truth: finetune/promoted_config.py):
  * direction models:  h=1,10 -> r10_joint_splitlr ; h=3,5 -> r5_frozen_pool48
  * direction TTA:     h=1 (124,126,128,130), h=3 (124,126,128,130),
                       h=5 (124..134), h=10 (126,128,130,132)  [logit average]
  * returns:           single lookback lb=128, per-h blend R10/R5:
                       h=1 0.925/0.075, h=3 0.775/0.225, h=5 1.0/0.0, h=10 0.65/0.35
  * h=1 trade gate:    actionable_score >= 0.45 AND |R10 h=1 ret| >= 0.002
                       AND h=5 predicts the SAME direction (strict_h5)

Output: stdout table + outputs/pred_688169_promoted.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import DEVICE, FEATURES, load_csv, load_model  # noqa: E402
from promoted_config import (  # noqa: E402
    PROMOTED_H1_GATE,
    PROMOTED_MODEL_BY_HORIZON,
    PROMOTED_RETURN_BLEND,
    PROMOTED_TTA_LOOKBACKS_BY_HORIZON,
    model_dir,
)
from run_tta_eval import derive_time_features, normalize_with_lookback  # noqa: E402

HORIZONS = (1, 3, 5, 10)
DIR_NAMES = {0: "DOWN 跌", 1: "FLAT 平", 2: "UP 涨"}
PRIMARY = PROMOTED_RETURN_BLEND["primary"]
SECONDARY = PROMOTED_RETURN_BLEND["secondary"]
BLEND_W = PROMOTED_RETURN_BLEND["primary_weight_by_horizon"]


@torch.no_grad()
def forward_logits(loaded: dict, feats: np.ndarray, ts, lb: int) -> dict:
    """One forward pass on the LAST lb bars; returns logits[H,3] + ret[H].

    The eval path feeds lb+11 rows and drops the last token (`[:, :-1]`).
    For forward prediction we append ONE dummy row (copy of the last bar);
    it lands in the dropped position and does not affect the context tokens.
    """
    x = np.concatenate([feats[-lb:], feats[-1:]], axis=0)
    x_norm = normalize_with_lookback(x, lb)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE)
    # dummy next-day timestamp in the dropped position (content irrelevant)
    import pandas as pd
    ts_ext = pd.concat([ts.iloc[-lb:],
                        pd.Series([ts.iloc[-1] + pd.Timedelta(days=1)],
                                  index=[ts.index[-1] + 1])])
    stamp = derive_time_features(ts_ext)
    stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(DEVICE)
    tok0, tok1 = loaded["tokenizer"].encode(x_tensor, half=True)
    _, _, hidden = loaded["model"](
        tok0[:, :-1], tok1[:, :-1], stamp_tensor[:, :-1, :], return_context=True)
    out = loaded["head"](hidden, context_length=lb)
    return {
        "logits": out["direction_logits"][0].cpu().numpy().astype(np.float64),
        "ret": out["return_prediction"][0].cpu().numpy().astype(np.float64),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="688169")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "pred_688169_promoted.json"))
    args = ap.parse_args()

    df = load_csv(args.symbol)
    feats = df[FEATURES].to_numpy(dtype=np.float32)
    ts = df["timestamps"]
    last_close = float(df["close"].iloc[-1])
    last_date = str(df["timestamps"].iloc[-1].date())
    print(f"{args.symbol} 截至 {last_date}，收盘价 {last_close:.2f}")

    loaded = {name: load_model(model_dir(name))
              for name in set(PROMOTED_MODEL_BY_HORIZON.values())}

    # direction: per-horizon TTA-averaged logits
    dir_out = {}
    for h in HORIZONS:
        m = PROMOTED_MODEL_BY_HORIZON[h]
        lbs = PROMOTED_TTA_LOOKBACKS_BY_HORIZON[h]
        stack = np.stack([forward_logits(loaded[m], feats, ts, lb)["logits"]
                          for lb in lbs], axis=0).mean(axis=0)
        logits_h = stack[HORIZONS.index(h)]
        probs = np.exp(logits_h - logits_h.max())
        probs = probs / probs.sum()
        hard = int(np.argmax(logits_h))
        actionable = float(max(probs[0], probs[2]))  # P(down)/P(up) 较大者
        dir_out[h] = {"model": m, "tta": list(lbs), "hard": hard,
                      "probs": probs.tolist(), "actionable_score": actionable,
                      "logits": logits_h.tolist()}

    # returns: single lb=128, per-horizon blend
    ret_sl = {m: forward_logits(loaded[m], feats, ts, 128)["ret"] for m in loaded}
    ret_out = {}
    for h in HORIZONS:
        i = HORIZONS.index(h)
        w = BLEND_W[str(h)] if isinstance(list(BLEND_W.keys())[0], str) else BLEND_W[h]
        r = w * ret_sl[PRIMARY][i] + (1 - w) * ret_sl[SECONDARY][i]
        ret_out[h] = float(r)

    # h=1 gate: conf + magnitude + strict_h5 consistency
    gate = PROMOTED_H1_GATE
    mag = abs(ret_sl[PRIMARY][0])
    gate_pass = (
        dir_out[1]["actionable_score"] >= gate["confidence_threshold"]
        and mag >= gate["min_abs_return"]
        and dir_out[5]["hard"] == dir_out[1]["hard"]
    )
    # A-share spot is LONG-ONLY: only UP calls are buy signals;
    # a DOWN call means stay flat / avoid (never short).
    is_up = dir_out[1]["hard"] == 2
    long_only_buy = gate_pass and is_up
    if long_only_buy:
        action = "✅ 买入信号（纯多头门控通过）"
    elif gate_pass and not is_up:
        action = "⏸️ 门控通过但方向为跌 → 空仓/回避（A股不做空）"
    else:
        action = "❌ 观望（门控未过）"

    print("\n horizon  direction        conf    pred_ret   target_px   model")
    for h in HORIZONS:
        d = dir_out[h]
        px = last_close * float(np.exp(ret_out[h]))
        print(f"  T+{h:<3d}   {DIR_NAMES[d['hard']]:12s}  "
              f"{d['actionable_score']:.3f}   {ret_out[h]:+.4f}   {px:9.2f}   {d['model']}")
    print(f"\n[h=1 门控（纯多头）] conf={dir_out[1]['actionable_score']:.3f} "
          f"(thr {gate['confidence_threshold']}) | mag|ret|={mag:.4f} "
          f"(thr {gate['min_abs_return']}) | "
          f"h5 一致={'是' if dir_out[5]['hard'] == dir_out[1]['hard'] else '否'}"
          f"  => {action}")

    result = {
        "symbol": args.symbol, "as_of": last_date, "last_close": last_close,
        "config": "P5-8 promoted (phase5_log.md / promoted_config.py)",
        "per_horizon": {
            str(h): {
                "direction": DIR_NAMES[dir_out[h]["hard"]].split()[0],
                "prob_down": dir_out[h]["probs"][0],
                "prob_flat": dir_out[h]["probs"][1],
                "prob_up": dir_out[h]["probs"][2],
                "actionable_score": dir_out[h]["actionable_score"],
                "pred_log_return": ret_out[h],
                "pred_price": last_close * float(np.exp(ret_out[h])),
                "model": dir_out[h]["model"], "tta_lookbacks": dir_out[h]["tta"],
            } for h in HORIZONS
        },
        "h1_gate": {
            "long_only": True,
            "buy_signal": bool(long_only_buy),
            "gate_pass_any_direction": bool(gate_pass),
            "direction": DIR_NAMES[dir_out[1]["hard"]].split()[0],
            "pass": bool(gate_pass),
            "confidence": dir_out[1]["actionable_score"],
            "confidence_threshold": gate["confidence_threshold"],
            "abs_ret": mag, "mag_threshold": gate["min_abs_return"],
            "h5_agrees": bool(dir_out[5]["hard"] == dir_out[1]["hard"]),
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
