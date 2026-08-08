"""Phase 8 — Round R6: walk-forward model/TTA selection (leakage-safe).

The P5-8 mapping was selected ON the 144-window test anchor itself (Phase 7
quantified the resulting selection bias). This round does it properly:
  * SELECTION set: 688169 windows with 2025-06-01 < context_end_date <=
    2025-12-15 (post-train, pre-test; train_end=2024-12-31 so no train
    overlap; test anchor stays strictly > 2025-12-15).
  * Full 25-model zoo x 9 TTA configs scored on the selection band
    (GPU inference, fixed reference targets ctx=122/dz=0.003/vol=0.5).
  * Pick per-horizon (model, TTA cfg) by selection-band nonflat.
  * Evaluate the chosen mapping on the untouched test anchor (p6 cache),
    vs the P5-8 mapping; gate kept at P5-8 params to isolate the mapping
    effect; dual-bar decision.

Outputs: outputs/p8_valband_store.npz, outputs/eval_p8_r6_walkforward.json
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
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits, predict_per_lookback  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p6_eval_zoo import (  # noqa: E402
    GATE_MAG, LOOKBACK_CONFIGS, P5_BASE_MAP, P5_TTA_BY_H, PRIMARY_W_BY_H,
)
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons  # noqa: E402
from selective_prediction import (  # noqa: E402
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
BAND_LO, BAND_HI = "2025-06-01", "2025-12-15"
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
SYMBOL = "688169"

ZOO = [  # all loadable checkpoints (same list as p6 meta)
    "frozen_cw_consist", "frozen_cw_dz001", "frozen_d025", "frozen_d025_cw",
    "frozen_d025_cw_long", "frozen_d025_cw_v2", "joint_d005", "joint_d025_cw",
    "r1_consist_w3", "r10_joint_splitlr", "r10v2_joint_splitlr_50stocks",
    "r11_joint_splitlr_140stocks", "r13_joint_splitlr_related25",
    "r14_frozen_pool48_688169_only", "r15_warmstart_r10_688169",
    "r16_return_head_only_r10", "r2_cw330", "r3_joint_cw_consist",
    "r4_joint_lr2e6", "r5_frozen_pool48", "r5v2_frozen_pool48_50stocks",
    "r6_frozen_pool64", "r7_frozen_pool32", "r9_joint_pool48", "smoke_frozen",
]


def make_band_windows(df: pd.DataFrame, lookbacks) -> list[dict]:
    """Same as make_tta_windows but for a date BAND (lo, hi]."""
    lo, hi = pd.Timestamp(BAND_LO), pd.Timestamp(BAND_HI)
    max_lb = max(lookbacks)
    needed = max_lb + PREDICT_WINDOW + 1
    out = []
    n = len(df)
    for start in range(n - WINDOW + 1):
        ced = df["timestamps"].iloc[start + LOOKBACK - 1]
        if not (lo < ced <= hi):
            continue
        extra = max_lb - LOOKBACK
        if start - extra < 0:
            continue
        big_start = start - extra
        window = df.iloc[big_start: big_start + needed].copy()
        out.append({"start": start,
                    "context_end_date": str(ced.date()),
                    "features": window[FEATURES].to_numpy(dtype=np.float32),
                    "raw_close": window["close"].to_numpy(dtype=np.float32),
                    "timestamps": window["timestamps"]})
    return out


def main() -> int:
    t0 = time.time()
    df = load_csv(SYMBOL)
    windows = make_band_windows(df, ALL_LOOKBACKS)
    print(f"selection band ({BAND_LO}, {BAND_HI}]: {len(windows)} windows")
    assert len(windows) >= 80

    # ---- 1. zoo inference on the selection band ----
    val = {m: {h: {"per_lb": [[] for _ in ALL_LOOKBACKS]} for h in HORIZONS}
           for m in ZOO}
    td_all = []
    loaded = {}
    for m in ZOO:
        try:
            loaded[m] = load_model(model_dir(m))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {m}: {e}")
    for i, w in enumerate(windows):
        if i % 30 == 0:
            print(f"  window {i}/{len(windows)} ({time.time() - t0:.0f}s)")
        drop = max(ALL_LOOKBACKS) - REF_CTX
        close_ref = w["raw_close"][drop: drop + REF_CTX + 11]
        tgt = make_multihorizon_targets(
            torch.from_numpy(close_ref).unsqueeze(0),
            context_length=REF_CTX, horizons=HORIZONS,
            min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
        td_all.append(tgt["direction"][0].cpu().numpy())
        for m, mod in loaded.items():
            pred = predict_per_lookback(mod, w, ALL_LOOKBACKS)
            for lb_i, lg in enumerate(pred["per_lb_logits"]):
                for hi, h in enumerate(HORIZONS):
                    val[m][h]["per_lb"][lb_i].append(lg[hi])
    td = np.asarray(td_all)

    # ---- 2. per-horizon (model, cfg) selection on the band ----
    selection = {}
    for h in HORIZONS:
        tdh = td[:, HORIZONS.index(h)]
        best = None
        for m in loaded:
            for cfg_name, lbs in LOOKBACK_CONFIGS:
                idx = [ALL_LOOKBACKS.index(x) for x in lbs]
                avg = average_logits([val[m][h]["per_lb"][i] for i in idx], None)
                nf = float(nonflat_accuracy(
                    direction_confidence_from_logits(avg)["hard_pred"], tdh))
                if best is None or nf > best[2]:
                    best = (m, cfg_name, nf, lbs)
        selection[h] = {"model": best[0], "cfg": best[1], "val_nf": best[2],
                        "lbs": list(best[3])}
        # also record where the promoted model ranks on the band
        pm = P5_BASE_MAP[h]
        plbs = P5_TTA_BY_H[h]
        idx = [ALL_LOOKBACKS.index(x) for x in plbs]
        avg = average_logits([val[pm][h]["per_lb"][i] for i in idx], None)
        pnf = float(nonflat_accuracy(
            direction_confidence_from_logits(avg)["hard_pred"], tdh))
        selection[h]["promoted_on_band"] = pnf
        print(f"h={h}: selected {best[0]}/{best[1]} val_nf={best[2]:.2%} "
              f"(promoted {pm} on band: {pnf:.2%})")

    # ---- 3. honest test-anchor evaluation (p6 cache) ----
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")

    def build(mapping, tta_by_h):
        pd_, td_, tr_, pr_, sc_ = {}, {}, {}, {}, {}
        for h in HORIZONS:
            m = mapping[h]
            idx = [ALL_LOOKBACKS.index(x) for x in tta_by_h[h]]
            conf = direction_confidence_from_logits(
                average_logits([tta[m][h]["per_lb_logits"][i] for i in idx], None))
            pd_[h] = conf["hard_pred"]
            sc_[h] = conf["actionable_score"]
            td_[h] = tta[m][h]["td"]
            tr_[h] = tta[m][h]["tr"]
            pr_[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                   PRIMARY_W_BY_H[h])
        s = summarize_horizons(pd_, pr_, td_, tr_, HORIZONS)
        g = apply_consistency_filter(
            apply_consistency_and_magnitude_gate(
                pd_[1], sc_[1], sl[PRIMARY][1]["pret"],
                confidence_threshold=0.45, min_abs_return=GATE_MAG,
                require_sign_agree=False),
            pd_[1], pd_[5], "strict")
        gm = gated_actionable_metrics(g, td_[1])
        bt = absolute_direction_backtest(g, tr_[1], transaction_cost=0.0005)
        s.update(h1_gated_precision=gm["precision_on_calls"],
                 h1_gated_nonflat=gm["gated_nonflat_acc"],
                 h1_gated_coverage=gm["coverage"], h1_gated_backtest=bt)
        return s

    base = build(P5_BASE_MAP, P5_TTA_BY_H)
    cand_map = {h: selection[h]["model"] for h in HORIZONS}
    cand_tta = {h: tuple(selection[h]["lbs"]) for h in HORIZONS}
    cand = build(cand_map, cand_tta)
    for tag, s in (("BASE(P5-8)", base), ("CAND(walk-fwd)", cand)):
        print(f"{tag}: nf={s['nonflat_accuracy_overall']:.2%} "
              f"mae={s['return_mae_overall']:.6f} "
              f"prec={s['h1_gated_precision']:.2%} gnf={s['h1_gated_nonflat']:.2%} "
              f"cov={s['h1_gated_coverage']:.2%} "
              f"bt={s['h1_gated_backtest']['total_return']:.2%}")
    d = dual_bar_decision(cand, base)
    print(f"DECISION: promote={d['promote']} reason={d['reason']} "
          f"d_nf={d['nonflat_delta']:+.4f}")

    out = ROOT / "outputs" / "eval_p8_r6_walkforward.json"
    out.write_text(json.dumps({
        "selection_band": {"lo": BAND_LO, "hi": BAND_HI, "n": len(windows)},
        "selection": {str(h): selection[h] for h in HORIZONS},
        "test_baseline": base, "test_candidate": cand, "decision": d,
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
