"""688169 latest-window three-way forecast comparison.

Compares:
  1) Public Kronos-base (generative path)
  2) Prior best: R10@h1 + R5@h3/5/10 multihorizon ensemble (raw head calls)
  3) Phase-2 promoted: same ensemble + h=1 selective gate
     (actionable_score>=0.45 and |return_pred|>=0.003)

Outputs:
  outputs/pred_688169_threeway.json
  outputs/pred_688169_threeway.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from model import KronosPredictor, load_model, load_tokenizer
from multihorizon_objective import (
    DEFAULT_HORIZONS,
    DOWN_CLASS,
    FLAT_CLASS,
    UP_CLASS,
    MultiHorizonForecastHead,
    make_direction_deadzones,
)
from promoted_config import PROMOTED_H1_GATE, PROMOTED_MODEL_BY_HORIZON, model_dir
from selective_prediction import (
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
)

SYMBOL = "688169"
CSV_PATH = ROOT / "data" / "a_share_finetune_multiboard" / "csv" / f"{SYMBOL}.csv"
TOKENIZER_FT_PATH = (
    ROOT / "outputs" / "models" / "a_share_multi_tokenizer" / "checkpoints" / "best_model"
)
PUBLIC_TOKENIZER = "Kronos-Tokenizer-base"
PUBLIC_MODEL = "Kronos-base"
LOOKBACK = 128
HORIZONS = list(DEFAULT_HORIZONS)
CLIP = 5.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LIMIT_RATE = 0.20
CLASS_NAMES = {DOWN_CLASS: "DOWN", FLAT_CLASS: "FLAT", UP_CLASS: "UP"}


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.normalize()
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["vol"] = df["volume"]
    df["amt"] = df["amount"]
    return df


def derive_time_features(dates: pd.Series) -> np.ndarray:
    stamps = pd.DatetimeIndex(dates)
    return np.stack(
        [
            np.zeros(len(stamps)),
            np.zeros(len(stamps)),
            stamps.weekday.to_numpy(),
            stamps.day.to_numpy(),
            stamps.month.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)


def normalize_window(features: np.ndarray) -> np.ndarray:
    lookback = features[:LOOKBACK]
    mean = lookback.mean(axis=0, keepdims=True)
    std = lookback.std(axis=0, keepdims=True)
    return np.clip((features - mean) / (std + 1e-5), -CLIP, CLIP)


def load_head_model(path: Path) -> dict:
    from model.kronos import Kronos, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_FT_PATH))
    tokenizer.eval().to(DEVICE)
    model = Kronos.from_pretrained(str(path))
    model.eval().to(DEVICE)
    head_ckpt = torch.load(path / "multihorizon_head.pt", map_location=DEVICE, weights_only=False)
    head = MultiHorizonForecastHead(
        head_ckpt["d_model"],
        horizons=tuple(head_ckpt.get("horizons", HORIZONS)),
        pool_size=head_ckpt.get("pool_size", 16),
        dropout=0.0,
    ).to(DEVICE)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "head": head,
        "horizons": tuple(head_ckpt.get("horizons", HORIZONS)),
        "min_deadzone": head_ckpt.get("direction_min_deadzone", 0.003),
        "vol_mult": head_ckpt.get("direction_volatility_multiplier", 0.5),
        "pool_size": head_ckpt.get("pool_size", 16),
        "model_dir": str(path),
    }


@torch.no_grad()
def predict_multihorizon(loaded: dict, df: pd.DataFrame) -> dict:
    window = df.iloc[-LOOKBACK:].copy()
    features = window[["open", "high", "low", "close", "vol", "amt"]].to_numpy(dtype=np.float32)
    raw_close = window["close"].to_numpy(dtype=np.float32)
    x_tensor = torch.from_numpy(normalize_window(features)).unsqueeze(0).to(DEVICE)
    stamp = torch.from_numpy(derive_time_features(window["timestamps"])).unsqueeze(0).to(DEVICE)
    raw_close_t = torch.from_numpy(raw_close).unsqueeze(0).to(DEVICE)

    tok0, tok1 = loaded["tokenizer"].encode(x_tensor, half=True)
    _, _, hidden = loaded["model"](tok0, tok1, stamp, return_context=True)
    outputs = loaded["head"](hidden, context_length=LOOKBACK)
    logits = outputs["direction_logits"][0].cpu().numpy()
    rets = outputs["return_prediction"][0].cpu().numpy()
    conf = direction_confidence_from_logits(logits)
    deadzones = make_direction_deadzones(
        raw_close_t,
        context_length=LOOKBACK,
        horizons=loaded["horizons"],
        min_deadzone=loaded["min_deadzone"],
        volatility_multiplier=loaded["vol_mult"],
    )[0].cpu().numpy()

    current_close = float(raw_close[-1])
    last_date = window["timestamps"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max(HORIZONS))
    out = {
        "context_end_date": str(last_date.date()),
        "current_close": current_close,
        "horizons": {},
    }
    for hi, h in enumerate(loaded["horizons"]):
        log_ret = float(rets[hi])
        # cumulative limit envelope
        max_up = current_close * ((1 + LIMIT_RATE) ** h)
        max_dn = current_close * ((1 - LIMIT_RATE) ** h)
        pred_close = float(np.clip(current_close * np.exp(log_ret), max_dn, max_up))
        log_ret = float(np.log(pred_close / current_close))
        hard = int(conf["hard_pred"][hi])
        probs = conf["probs"][hi]
        out["horizons"][str(h)] = {
            "direction": CLASS_NAMES[hard],
            "direction_class": hard,
            "direction_probs": {
                "down": float(probs[0]),
                "flat": float(probs[1]),
                "up": float(probs[2]),
            },
            "actionable_score": float(conf["actionable_score"][hi]),
            "margin": float(conf["margin"][hi]),
            "max_prob": float(conf["max_prob"][hi]),
            "predicted_log_return": log_ret,
            "predicted_return_pct": log_ret * 100.0,
            "predicted_close": pred_close,
            "deadzone_log_return": float(deadzones[hi]),
            "target_date": str(future_dates[h - 1].date()),
        }
    return out


def build_prior_ensemble(per_model: dict[str, dict], df: pd.DataFrame) -> dict:
    """Raw ensemble using prior-best mapping (no gate)."""
    base_date = df["timestamps"].iloc[-1]
    current_close = float(df["close"].iloc[-1])
    ens = {
        "label": "prior_best_R10_R5_ensemble",
        "context_end_date": str(base_date.date()),
        "current_close": current_close,
        "mapping": {str(h): PROMOTED_MODEL_BY_HORIZON[h] for h in HORIZONS},
        "horizons": {},
    }
    for h in HORIZONS:
        name = PROMOTED_MODEL_BY_HORIZON[h]
        src = per_model[name]["horizons"][str(h)]
        ens["horizons"][str(h)] = {
            **src,
            "model": name,
            "trade_action": src["direction"],  # raw call is the trade signal
            "gated": False,
            "abstain": src["direction"] == "FLAT",
        }
    return ens


def build_promoted(per_model: dict[str, dict], df: pd.DataFrame) -> dict:
    """Same ensemble + Phase-2 h=1 selective gate."""
    ens = build_prior_ensemble(per_model, df)
    ens["label"] = "phase2_promoted_gated"
    ens["gate"] = PROMOTED_H1_GATE

    h1 = ens["horizons"]["1"]
    hard = np.array([h1["direction_class"]], dtype=np.int64)
    conf = np.array([h1["actionable_score"]], dtype=np.float64)
    pret = np.array([h1["predicted_log_return"]], dtype=np.float64)
    gated = apply_consistency_and_magnitude_gate(
        hard,
        conf,
        pret,
        confidence_threshold=PROMOTED_H1_GATE["confidence_threshold"],
        min_abs_return=PROMOTED_H1_GATE["min_abs_return"],
        require_sign_agree=PROMOTED_H1_GATE["require_sign_agree"],
    )[0]
    action = CLASS_NAMES[int(gated)]
    pass_conf = h1["actionable_score"] >= PROMOTED_H1_GATE["confidence_threshold"]
    pass_mag = abs(h1["predicted_log_return"]) >= PROMOTED_H1_GATE["min_abs_return"]
    h1["trade_action"] = action
    h1["gated"] = True
    h1["abstain"] = action == "FLAT"
    h1["gate_detail"] = {
        "pass_confidence": bool(pass_conf),
        "pass_magnitude": bool(pass_mag),
        "raw_direction": h1["direction"],
        "promoted_action": action,
    }
    # longer horizons: keep research predictions, but mark trading uses h=1 by default
    for h in (3, 5, 10):
        ens["horizons"][str(h)]["trade_action"] = ens["horizons"][str(h)]["direction"]
        ens["horizons"][str(h)]["gated"] = False
        ens["horizons"][str(h)]["note"] = "research horizon; primary trading signal is h=1"
    return ens


@torch.no_grad()
def predict_public(df: pd.DataFrame) -> dict:
    history = df.iloc[-LOOKBACK:].copy()
    x_df = history[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = history["timestamps"].reset_index(drop=True)
    last_date = history["timestamps"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=max(HORIZONS))
    y_timestamp = pd.Series(future_dates)

    tokenizer = load_tokenizer(PUBLIC_TOKENIZER)
    model = load_model(PUBLIC_MODEL)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=max(HORIZONS),
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
        deterministic=True,
    )
    current_close = float(history["close"].iloc[-1])
    prev = current_close
    limited = []
    for c in pred_df["close"].to_numpy(dtype=float):
        c = float(np.clip(c, prev * (1 - LIMIT_RATE), prev * (1 + LIMIT_RATE)))
        limited.append(c)
        prev = c
    limited = np.asarray(limited)
    hist = history["close"].to_numpy(dtype=float)
    daily_vol = float(np.std(np.log(hist[1:] / hist[:-1])))

    out = {
        "label": "public_kronos_base",
        "context_end_date": str(last_date.date()),
        "current_close": current_close,
        "horizons": {},
    }
    for h in HORIZONS:
        idx = h - 1
        future_c = float(limited[idx])
        log_ret = float(np.log(future_c / current_close))
        deadzone = max(daily_vol * np.sqrt(h) * 0.5, 0.003)
        if log_ret > deadzone:
            direction = "UP"
        elif log_ret < -deadzone:
            direction = "DOWN"
        else:
            direction = "FLAT"
        out["horizons"][str(h)] = {
            "direction": direction,
            "trade_action": direction,
            "predicted_log_return": log_ret,
            "predicted_return_pct": log_ret * 100.0,
            "predicted_close": future_c,
            "deadzone_log_return": deadzone,
            "target_date": str(future_dates[idx].date()),
            "gated": False,
            "abstain": direction == "FLAT",
        }
    return out


def recent_context(df: pd.DataFrame, n: int = 10) -> list[dict]:
    tail = df.iloc[-n:]
    rows = []
    for _, r in tail.iterrows():
        rows.append(
            {
                "date": str(pd.Timestamp(r["timestamps"]).date()),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
        )
    return rows


def render_report(public: dict, prior: dict, promoted: dict, recent: list[dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 108)
    lines.append("688169 石头科技 · 三方最新预测对比")
    lines.append(
        f"Context end: {prior['context_end_date']}    Last close: {prior['current_close']:.2f}"
    )
    lines.append("=" * 108)
    lines.append("")
    lines.append("【最近 10 日收盘】")
    for r in recent:
        lines.append(f"  {r['date']}  close={r['close']:.2f}  vol={r['volume']:.0f}")
    lines.append("")
    lines.append("【三方定义】")
    lines.append("  A. 公开权重 Public Kronos-base（生成式路径 + deadzone 方向）")
    lines.append("  B. 此前最优 Prior best：R10@h1 + R5@h3/5/10 多 horizon 集成（原始 head 方向）")
    lines.append(
        "  C. 本次优化 Promoted：同 B 的权重 + h=1 选择性门控 "
        f"(score≥{PROMOTED_H1_GATE['confidence_threshold']}, "
        f"|ret|≥{PROMOTED_H1_GATE['min_abs_return']})"
    )
    lines.append("")
    lines.append("-" * 108)
    lines.append(
        f"{'H':<4}{'Date':<12}{'Source':<16}{'Dir/Action':<12}{'Ret%':>8}{'Close':>10}"
        f"{'Conf':>8}{'|ret|':>8}{'Note':<28}"
    )
    lines.append("-" * 108)

    sources = [
        ("Public", public),
        ("PriorBest", prior),
        ("Promoted", promoted),
    ]
    for h in HORIZONS:
        for label, blob in sources:
            x = blob["horizons"][str(h)]
            action = x.get("trade_action", x["direction"])
            conf = x.get("actionable_score")
            conf_s = f"{conf:.3f}" if conf is not None else "  -  "
            abs_ret = abs(x["predicted_log_return"])
            note = ""
            if label == "Promoted" and h == 1:
                gd = x.get("gate_detail", {})
                if gd.get("promoted_action") == "FLAT" and gd.get("raw_direction") != "FLAT":
                    note = f"ABSTAIN raw={gd.get('raw_direction')}"
                elif gd:
                    note = "TAKE trade (gate pass)" if not x.get("abstain") else "ABSTAIN"
            elif label == "Promoted" and h != 1:
                note = "research only"
            lines.append(
                f"{'h='+str(h):<4}{x['target_date']:<12}{label:<16}{action:<12}"
                f"{x['predicted_return_pct']:>7.2f}%{x['predicted_close']:>10.2f}"
                f"{conf_s:>8}{abs_ret:>8.4f}  {note}"
            )
        lines.append("-" * 108)

    # h=1 trading summary
    lines.append("")
    lines.append("【h=1 可交易信号（主决策）】")
    p1 = public["horizons"]["1"]
    b1 = prior["horizons"]["1"]
    c1 = promoted["horizons"]["1"]
    lines.append(
        f"  Public   : {p1['trade_action']:<6}  ret={p1['predicted_return_pct']:+.2f}%  "
        f"close={p1['predicted_close']:.2f}  target={p1['target_date']}"
    )
    lines.append(
        f"  PriorBest: {b1['trade_action']:<6}  ret={b1['predicted_return_pct']:+.2f}%  "
        f"close={b1['predicted_close']:.2f}  conf={b1['actionable_score']:.3f}  "
        f"probs D/F/U="
        f"{b1['direction_probs']['down']:.2f}/"
        f"{b1['direction_probs']['flat']:.2f}/"
        f"{b1['direction_probs']['up']:.2f}"
    )
    gd = c1.get("gate_detail", {})
    lines.append(
        f"  Promoted : {c1['trade_action']:<6}  ret={c1['predicted_return_pct']:+.2f}%  "
        f"close={c1['predicted_close']:.2f}  conf={c1['actionable_score']:.3f}  "
        f"gate conf_ok={gd.get('pass_confidence')} mag_ok={gd.get('pass_magnitude')}  "
        f"raw={gd.get('raw_direction')}"
    )
    lines.append("")
    if c1["trade_action"] == "FLAT" and b1["trade_action"] != "FLAT":
        lines.append(
            "  → 解读：优化后配置在 h=1 **弃权**（置信度或幅度未过门），"
            "避免在弱信号上强行交易；价格预测仍与 PriorBest 同源。"
        )
    elif c1["trade_action"] == b1["trade_action"]:
        lines.append(
            "  → 解读：门控通过，Promoted 与 PriorBest 给出相同交易方向；"
            "优化价值体现在「只在高置信出手」。"
        )
    else:
        lines.append("  → 解读：门控改变了交易动作（通常是 raw→abstain）。")

    lines.append("")
    lines.append("【一致性 / 分歧速览】")
    for h in HORIZONS:
        dirs = {
            "Public": public["horizons"][str(h)]["trade_action"],
            "PriorBest": prior["horizons"][str(h)]["trade_action"],
            "Promoted": promoted["horizons"][str(h)]["trade_action"],
        }
        uniq = set(dirs.values())
        status = "一致" if len(uniq) == 1 else ("部分分歧" if len(uniq) == 2 else "三方分歧")
        lines.append(
            f"  h={h}: Public={dirs['Public']:<5} Prior={dirs['PriorBest']:<5} "
            f"Promoted={dirs['Promoted']:<5}  [{status}]"
        )

    lines.append("")
    lines.append("【测试集参考（历史评估，非本次未来结果）】")
    lines.append("  Prior/Promoted 权重：ungated nonflat overall ≈ 69.7–70.3%")
    lines.append(
        "  Promoted 门控 h=1：coverage≈23%, precision≈63.6%, hit≈66.7%, "
        "gated_nf≈84%（见 optimization_log Phase2）"
    )
    lines.append("  Public：历史评估中短窗方向常塌缩为 FLAT / 弱信号。")
    lines.append("=" * 108)
    return "\n".join(lines)


def main() -> int:
    df = load_csv()
    print(
        f"Loaded {SYMBOL}: {df['timestamps'].iloc[0].date()} -> "
        f"{df['timestamps'].iloc[-1].date()} ({len(df)} bars)"
    )

    names = sorted(set(PROMOTED_MODEL_BY_HORIZON.values()))
    loaded = {}
    for name in names:
        path = model_dir(name)
        print(f"Loading {name} from {path}")
        loaded[name] = load_head_model(path)

    per_model = {name: predict_multihorizon(mod, df) for name, mod in loaded.items()}
    prior = build_prior_ensemble(per_model, df)
    promoted = build_promoted(per_model, df)

    print("Running public Kronos-base...")
    public = predict_public(df)
    recent = recent_context(df, 10)
    report = render_report(public, prior, promoted, recent)
    print(report)

    payload = {
        "symbol": SYMBOL,
        "name": "石头科技",
        "recent_closes": recent,
        "public_kronos_base": public,
        "prior_best_ensemble": prior,
        "phase2_promoted": promoted,
        "per_model_raw": per_model,
        "notes": {
            "prior_best": "R10@h1 + R5@h3/5/10 raw multihorizon heads",
            "promoted": "same weights + h=1 selective gate from promoted_config.py",
            "primary_trading_horizon": 1,
        },
    }
    out_json = ROOT / "outputs" / f"pred_{SYMBOL}_threeway.json"
    out_txt = ROOT / "outputs" / f"pred_{SYMBOL}_threeway.txt"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_txt.write_text(report, encoding="utf-8")
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
