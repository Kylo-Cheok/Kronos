"""Phase 8 extension — long-only gate optimization for the A-share spot market.

A-share retail cannot short single stocks, so the P5-8 gate (26 of 32 calls
were shorts) must be re-tuned for long/flat trading:
  * position = +1 when the gated call is UP, else 0 (flat);
  * DOWN calls become "stay flat" (avoid), never a short.

Steps:
  1) long-only baseline: P5-8 gate (thr=0.45, mag=0.002, strict_h5), UP only.
  2) sweep: thr(0.35..0.55, 0.01) x mag(0..0.004) x consistency
     {none, strict_h5, soft_h5, strict_h3, strict_h3_and_h5}
     -> rank by total return, report precision / n_trades / maxDD.
  3) recommendation = max return s.t. precision >= 0.75 and n_trades >= 10
     (P5-8 quality bar); relaxed view also shown.

Outputs: outputs/eval_p8_long_only_gate.json,
         outputs/backtest_p5_8_long_only.csv / .png
"""

from __future__ import annotations

import json
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
THR_GRID = np.round(np.arange(0.35, 0.551, 0.01), 2)
MAG_GRID = (0.0, 0.001, 0.002, 0.003, 0.004)
CONSISTENCY = ("none", "strict_h5", "soft_h5", "strict_h3", "strict_h3_and_h5")


def tta_conf(store, m, h, lbs):
    idx = [ALL_LOOKBACKS.index(x) for x in lbs]
    avg = average_logits([store[m][h]["per_lb_logits"][i] for i in idx], None)
    return direction_confidence_from_logits(avg)


def long_only_stats(gated, tr1, td1):
    """Position +1 on UP calls only; DOWN/FLAT -> flat."""
    gv = np.asarray(gated, dtype=int)
    pos = (gv == 2).astype(float)
    simple = np.exp(tr1) - 1.0
    pnl = pos * simple - COST * (pos > 0)
    eq = np.cumprod(1 + pnl)
    n = int((pos > 0).sum())
    # precision on up-calls: true direction is UP (td==2)
    prec = float(np.mean(td1[gv == 2] == 2)) if n else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1))
    return {"total_return": float(eq[-1] - 1), "n_trades": n,
            "hit_rate": float(np.mean(pnl[pos > 0] > 0)) if n else 0.0,
            "precision": prec, "avg_pnl": float(np.mean(pnl[pos > 0])) if n else 0.0,
            "max_drawdown": mdd, "equity": eq, "pos": pos, "pnl": pnl}


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    dates = [w["context_end_date"] for w in make_windows(load_csv("688169"))]

    hard = {h: tta_conf(tta, m, h, lbs)["hard_pred"] for h, m, lbs in [
        (1, PRIMARY, (124, 126, 128, 130)),
        (3, SECONDARY, (124, 126, 128, 130)),
        (5, SECONDARY, (124, 126, 128, 130, 132, 134))]}
    score1 = tta_conf(tta, PRIMARY, 1, (124, 126, 128, 130))["actionable_score"]
    gate_ret = sl[PRIMARY][1]["pret"]
    tr1 = tta[PRIMARY][1]["tr"].astype(np.float64)
    td1 = tta[PRIMARY][1]["td"]

    def build_gate(thr, mag, cons):
        g = apply_consistency_and_magnitude_gate(
            hard[1], score1, gate_ret, confidence_threshold=float(thr),
            min_abs_return=mag, require_sign_agree=False)
        if cons == "strict_h5":
            g = apply_consistency_filter(g, hard[1], hard[5], "strict")
        elif cons == "soft_h5":
            g = apply_consistency_filter(g, hard[1], hard[5], "soft")
        elif cons == "strict_h3":
            g = apply_consistency_filter(g, hard[1], hard[3], "strict")
        elif cons == "strict_h3_and_h5":
            g = apply_consistency_filter(g, hard[1], hard[3], "strict")
            g = apply_consistency_filter(g, hard[1], hard[5], "strict")
        return g

    # 1) long-only baseline (P5-8 params)
    base = long_only_stats(build_gate(0.45, 0.002, "strict_h5"), tr1, td1)
    print(f"LONG-ONLY baseline (P5-8 params): ret={base['total_return']:+.2%} "
          f"trades={base['n_trades']} prec={base['precision']:.2%} "
          f"hit={base['hit_rate']:.2%} avg={base['avg_pnl']:+.3%} "
          f"maxDD={base['max_drawdown']:.2%}")

    # 2) sweep
    rows = []
    for thr in THR_GRID:
        for mag in MAG_GRID:
            for cons in CONSISTENCY:
                s = long_only_stats(build_gate(thr, mag, cons), tr1, td1)
                rows.append({"thr": float(thr), "mag": mag, "consistency": cons,
                             **{k: s[k] for k in ("total_return", "n_trades",
                                                  "hit_rate", "precision",
                                                  "avg_pnl", "max_drawdown")}})
    ok = [r for r in rows if r["n_trades"] >= 10]
    ok.sort(key=lambda r: r["total_return"], reverse=True)
    print("\nTop 10 by return (n_trades>=10):")
    for r in ok[:10]:
        print(f"  thr={r['thr']:.2f} mag={r['mag']:.3f} {r['consistency']:16s} "
              f"ret={r['total_return']:+.2%} n={r['n_trades']} "
              f"prec={r['precision']:.2%} hit={r['hit_rate']:.2%} "
              f"avg={r['avg_pnl']:+.3%} mdd={r['max_drawdown']:.2%}")

    qual = [r for r in rows if r["precision"] >= 0.75 and r["n_trades"] >= 10]
    qual.sort(key=lambda r: r["total_return"], reverse=True)
    print("\nTop 5 with precision>=75% & n>=10:")
    for r in qual[:5]:
        print(f"  thr={r['thr']:.2f} mag={r['mag']:.3f} {r['consistency']:16s} "
              f"ret={r['total_return']:+.2%} n={r['n_trades']} "
              f"prec={r['precision']:.2%} hit={r['hit_rate']:.2%} "
              f"avg={r['avg_pnl']:+.3%} mdd={r['max_drawdown']:.2%}")

    # 3) recommendation
    rec = qual[0] if qual else (ok[0] if ok else None)
    rec_stats = None
    if rec:
        rec_stats = long_only_stats(
            build_gate(rec["thr"], rec["mag"], rec["consistency"]), tr1, td1)
        bh = np.cumprod(np.exp(tr1))
        df = pd.DataFrame({
            "context_end_date": dates, "h1_pred": hard[1], "h1_conf": score1,
            "h5_pred": hard[5], "long_pos": rec_stats["pos"],
            "true_ret_1d": tr1, "pnl": rec_stats["pnl"],
            "equity": rec_stats["equity"], "buy_hold": bh})
        df.to_csv(ROOT / "outputs" / "backtest_p5_8_long_only.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            x = pd.to_datetime(dates)
            ax.plot(x, rec_stats["equity"], lw=1.8,
                    label=f"long-only gated ({rec_stats['equity'][-1] - 1:+.0%})")
            ax.plot(x, bh, lw=1.2, alpha=0.8,
                    label=f"buy&hold ({bh[-1] - 1:+.0%})")
            ax.set_title("688169 long-only gated backtest (2025-12 -> 2026-07)")
            ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(ROOT / "outputs" / "backtest_p5_8_long_only.png", dpi=140)
        except Exception as e:  # noqa: BLE001
            print(f"png skipped: {e}")

    out = ROOT / "outputs" / "eval_p8_long_only_gate.json"
    out.write_text(json.dumps({
        "long_only_baseline_p5_8_params": {k: v for k, v in base.items()
                                           if not isinstance(v, np.ndarray)},
        "recommendation": rec,
        "top10_by_return": ok[:10], "top5_precision_75": qual[:5],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
