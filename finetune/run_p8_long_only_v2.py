"""Long-only optimization v2 — middle tiers + long-horizon-driven variants.

Builds on run_p8_long_only_gate.py (full sweep) and adds:
  A) mid-tier report: prec>=70% n>=8 ; prec>=65% n>=12
  B) h5-driven long: enter when h=5 predicts UP with score5 >= thr
     (multi-day trend signal, traded on chained 1d returns, daily re-eval)
  C) full alignment: h1 & h5 & h10 all UP (+ h1 conf thr)
  D) hold-while-h5-up state machine: enter on gated h1 UP call,
     hold while h=5 keeps predicting UP (max 10d), exit on flip
     -> fewer, longer trades (A-share T+1 friendly)

Output: outputs/eval_p8_long_only_v2.json
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
from run_p8_long_only_gate import (  # noqa: E402
    CONSISTENCY, COST, MAG_GRID, THR_GRID, PRIMARY, SECONDARY,
    long_only_stats, tta_conf,
)
from selective_prediction import (  # noqa: E402
    apply_consistency_and_magnitude_gate,
)


def stats_from_pos(pos, tr1, td1):
    pos = np.asarray(pos, dtype=float)
    simple = np.exp(tr1) - 1.0
    pnl = pos * simple - COST * (pos > 0)
    eq = np.cumprod(1 + pnl)
    n = int((pos > 0).sum())
    peak = np.maximum.accumulate(eq)
    return {"total_return": float(eq[-1] - 1), "n_trades": n,
            "hit_rate": float(np.mean(pnl[pos > 0] > 0)) if n else 0.0,
            "precision": float(np.mean(td1[pos > 0] == 2)) if n else 0.0,
            "avg_pnl": float(np.mean(pnl[pos > 0])) if n else 0.0,
            "max_drawdown": float(np.min(eq / peak - 1)), "equity": eq,
            "pos": pos, "pnl": pnl}


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    dates = [w["context_end_date"] for w in make_windows(load_csv("688169"))]

    c1 = tta_conf(tta, PRIMARY, 1, (124, 126, 128, 130))
    c3 = tta_conf(tta, SECONDARY, 3, (124, 126, 128, 130))
    c5 = tta_conf(tta, SECONDARY, 5, (124, 126, 128, 130, 132, 134))
    c10 = tta_conf(tta, PRIMARY, 10, (126, 128, 130, 132))
    hard1, score1 = c1["hard_pred"], c1["actionable_score"]
    hard3, hard5, hard10 = c3["hard_pred"], c5["hard_pred"], c10["hard_pred"]
    score5 = c5["actionable_score"]
    gate_ret = sl[PRIMARY][1]["pret"]
    tr1 = tta[PRIMARY][1]["tr"].astype(np.float64)
    td1 = tta[PRIMARY][1]["td"]
    result = {}

    # ---- A) mid tiers from the v1 sweep (re-run compactly) ----
    def build_gate(thr, mag, cons):
        g = apply_consistency_and_magnitude_gate(
            hard1, score1, gate_ret, confidence_threshold=float(thr),
            min_abs_return=mag, require_sign_agree=False)
        if cons == "strict_h5":
            g = apply_consistency_filter(g, hard1, hard5, "strict")
        elif cons == "soft_h5":
            g = apply_consistency_filter(g, hard1, hard5, "soft")
        elif cons == "strict_h3":
            g = apply_consistency_filter(g, hard1, hard3, "strict")
        elif cons == "strict_h3_and_h5":
            g = apply_consistency_filter(g, hard1, hard3, "strict")
            g = apply_consistency_filter(g, hard1, hard5, "strict")
        return g

    rows = []
    for thr in THR_GRID:
        for mag in MAG_GRID:
            for cons in CONSISTENCY:
                s = long_only_stats(build_gate(thr, mag, cons), tr1, td1)
                rows.append({"thr": float(thr), "mag": mag, "consistency": cons,
                             **{k: s[k] for k in ("total_return", "n_trades",
                                                  "hit_rate", "precision",
                                                  "avg_pnl", "max_drawdown")}})
    for label, f in (("prec>=70% n>=8",
                      lambda r: r["precision"] >= 0.70 and r["n_trades"] >= 8),
                     ("prec>=65% n>=12",
                      lambda r: r["precision"] >= 0.65 and r["n_trades"] >= 12)):
        tier = sorted([r for r in rows if f(r)],
                      key=lambda r: r["total_return"], reverse=True)
        result[f"tier_{label}"] = tier[:5]
        print(f"\n== tier {label} ==")
        for r in tier[:5]:
            print(f"  thr={r['thr']} mag={r['mag']} {r['consistency']:16s} "
                  f"ret={r['total_return']:+.2%} n={r['n_trades']} "
                  f"prec={r['precision']:.2%} hit={r['hit_rate']:.2%} "
                  f"mdd={r['max_drawdown']:.2%}")

    # ---- B) h5-driven long ----
    b_rows = []
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60):
        pos = ((hard5 == 2) & (score5 >= thr)).astype(float)
        s = stats_from_pos(pos, tr1, td1)
        b_rows.append({"thr": thr, **{k: s[k] for k in (
            "total_return", "n_trades", "hit_rate", "precision", "avg_pnl",
            "max_drawdown")}})
    result["h5_driven"] = b_rows
    print("\n== B) h5-driven long ==")
    for r in b_rows:
        print(f"  score5>={r['thr']}: ret={r['total_return']:+.2%} "
              f"n={r['n_trades']} prec={r['precision']:.2%} "
              f"hit={r['hit_rate']:.2%} mdd={r['max_drawdown']:.2%}")

    # ---- C) full alignment h1&h5&h10 all UP ----
    c_rows = []
    for thr in (0.40, 0.45, 0.50):
        pos = ((hard1 == 2) & (hard5 == 2) & (hard10 == 2)
               & (score1 >= thr)).astype(float)
        s = stats_from_pos(pos, tr1, td1)
        c_rows.append({"thr": thr, **{k: s[k] for k in (
            "total_return", "n_trades", "hit_rate", "precision", "avg_pnl",
            "max_drawdown")}})
    result["full_alignment"] = c_rows
    print("\n== C) h1&h5&h10 all-UP ==")
    for r in c_rows:
        print(f"  thr={r['thr']}: ret={r['total_return']:+.2%} n={r['n_trades']} "
              f"prec={r['precision']:.2%} hit={r['hit_rate']:.2%} "
              f"mdd={r['max_drawdown']:.2%}")

    # ---- D) hold-while-h5-up state machine ----
    d_rows = []
    for thr in (0.40, 0.45, 0.50):
        for max_hold in (5, 10):
            entry = ((hard1 == 2) & (score1 >= thr)
                     & (hard5 == 2))
            pos = np.zeros(len(tr1))
            holding = 0
            for i in range(len(tr1)):
                if holding > 0:
                    holding -= 1
                    pos[i] = 1.0
                    if hard5[i] != 2:  # trend flip -> exit next day
                        holding = 0
                elif entry[i]:
                    pos[i] = 1.0
                    holding = max_hold - 1
            s = stats_from_pos(pos, tr1, td1)
            d_rows.append({"thr": thr, "max_hold": max_hold, **{k: s[k] for k in (
                "total_return", "n_trades", "hit_rate", "precision", "avg_pnl",
                "max_drawdown")}})
    result["hold_while_h5_up"] = d_rows
    print("\n== D) hold-while-h5-up ==")
    for r in d_rows:
        print(f"  thr={r['thr']} hold<={r['max_hold']}: ret={r['total_return']:+.2%} "
              f"n_days={r['n_trades']} hit={r['hit_rate']:.2%} "
              f"mdd={r['max_drawdown']:.2%}")

    out = ROOT / "outputs" / "eval_p8_long_only_v2.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
