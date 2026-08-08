"""P5-8 promoted config — historical backtest report on the 144-window anchor.

Regenerates the h=1 gated backtest from the p6 caches (bit-identical to the
verified baseline: total_return 102.63%) and exports:
  * outputs/backtest_p5_8_promoted.csv  — per-window decisions & pnl
  * outputs/backtest_p5_8_promoted.png  — equity curve vs buy&hold
  * stdout summary (incl. max drawdown, long/short split, ungated reference)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import load_csv, make_windows  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from selective_prediction import (  # noqa: E402
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
)

PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
COST = 0.0005


def tta_conf(store, m, h, lbs):
    idx = [ALL_LOOKBACKS.index(x) for x in lbs]
    avg = average_logits([store[m][h]["per_lb_logits"][i] for i in idx], None)
    return direction_confidence_from_logits(avg)


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    dates = [w["context_end_date"] for w in make_windows(load_csv("688169"))]

    c1 = tta_conf(tta, PRIMARY, 1, (124, 126, 128, 130))
    c5 = tta_conf(tta, SECONDARY, 5, (124, 126, 128, 130, 132, 134))
    hard1, score1, hard5 = c1["hard_pred"], c1["actionable_score"], c5["hard_pred"]
    gate_ret = sl[PRIMARY][1]["pret"]
    tr1 = tta[PRIMARY][1]["tr"]

    gated = apply_consistency_and_magnitude_gate(
        hard1, score1, gate_ret, confidence_threshold=0.45,
        min_abs_return=0.002, require_sign_agree=False)
    gated = apply_consistency_filter(gated, hard1, hard5, "strict")

    # gated: class labels 0=DOWN / 1=FLAT(abstain) / 2=UP.
    # Official semantics (selective_prediction.absolute_direction_backtest):
    # position +1/-1/0; pnl = position * (exp(log_ret)-1) - cost per trade.
    gv = np.asarray(gated, dtype=int)
    sign = np.where(gv == 2, 1.0, np.where(gv == 0, -1.0, 0.0))
    simple_tr1 = np.exp(tr1) - 1.0
    pnl = sign * simple_tr1 - COST * (sign != 0)
    equity = np.cumprod(1 + pnl)

    # buy & hold over the same span
    bh = np.cumprod(1 + simple_tr1)

    # ungated reference: trade every non-flat h=1 call
    ug_sign = np.where(hard1 == 2, 1.0, np.where(hard1 == 0, -1.0, 0.0))
    ug_pnl = ug_sign * simple_tr1 - COST * (ug_sign != 0)
    ug_eq = np.cumprod(1 + ug_pnl)

    df = pd.DataFrame({
        "context_end_date": dates,
        "h1_pred": hard1, "h1_conf": score1, "h5_pred": hard5,
        "gate_call": gv, "true_ret_1d": tr1, "pnl": pnl,
        "equity": equity, "buy_hold": bh,
    })
    csv_out = ROOT / "outputs" / "backtest_p5_8_promoted.csv"
    df.to_csv(csv_out, index=False)

    peak = np.maximum.accumulate(equity)
    mdd = float(np.min(equity / peak - 1))
    n_long = int((sign > 0).sum())
    n_short = int((sign < 0).sum())
    wins = pnl[sign != 0] > 0
    print(f"span: {dates[0]} -> {dates[-1]}  ({len(dates)} windows)")
    print(f"P5-8 gated : ret={equity[-1] - 1:+.2%}  trades={int((sign != 0).sum())} "
          f"(long {n_long} / short {n_short})  hit={wins.mean():.2%}  "
          f"avg_pnl={pnl[sign != 0].mean():+.3%}  maxDD={mdd:.2%}")
    print(f"buy&hold   : ret={bh[-1] - 1:+.2%}")
    print(f"ungated all: ret={ug_eq[-1] - 1:+.2%}  trades={int((ug_sign != 0).sum())}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        x = pd.to_datetime(dates)
        ax.plot(x, equity, label=f"P5-8 gated ({equity[-1] - 1:+.0%})", lw=1.8)
        ax.plot(x, bh, label=f"buy&hold ({bh[-1] - 1:+.0%})", lw=1.2, alpha=0.8)
        ax.plot(x, ug_eq, label=f"ungated ({ug_eq[-1] - 1:+.0%})", lw=1.0,
                alpha=0.6, ls="--")
        ax.set_title("688169 P5-8 gated backtest (2025-12 -> 2026-07)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_out = ROOT / "outputs" / "backtest_p5_8_promoted.png"
        fig.savefig(png_out, dpi=140)
        print(f"Saved {png_out}")
    except Exception as e:  # noqa: BLE001
        print(f"png skipped: {e}")
    print(f"Saved {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
