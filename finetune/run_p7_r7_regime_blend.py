"""Phase 7 — Round R7: volatility-regime adaptive return blend (numpy only).

P4-9/P5-5 fixed a single per-horizon blend weight w_h (R10 vs R5 single-lb
returns). Hypothesis: the optimal R10/R5 mix may depend on the volatility
regime (e.g. frozen R5 smoother in high-vol, joint R10 better in low-vol).

Method (leakage-safe feature):
  * Regime feature = trailing 20-day realized vol of daily log returns,
    computed ONLY from data up to each window's context_end_date (no future).
  * Split the 144 eval windows into 3 tercile regimes (low/mid/high vol).
  * Per (horizon, regime) sweep w in [0..1] step 0.025 minimizing MAE in that
    regime; combine adaptively. Compare overall MAE vs P5-5 fixed weights.
  * Selection happens on the eval windows (same convention as P4-9/P5-5
    blend sweeps) — noted honestly; a margin <0.1% rel is called negligible.

Baseline (P5-5): w = {1:0.925, 3:0.775, 5:1.0, 10:0.65}, MAE=0.037448.

Output: outputs/eval_p7_r7_regime_blend.json
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
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402

HORIZONS = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
FIXED_W = {1: 0.925, 3: 0.775, 5: 1.0, 10: 0.65}
W_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.025), 3)


def trailing_vol(dates: list[str], symbol: str) -> np.ndarray:
    """20d realized vol (std of daily log returns) as of each context date."""
    df = load_csv(symbol)
    close = df["close"].astype(float).to_numpy()
    ts = pd.to_datetime(df["timestamps"])
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    out = []
    for d in dates:
        # context_end_date is a trading day present in df; use its position
        pos = int(np.where(ts.values == pd.Timestamp(d).to_datetime64())[0][0])
        window = logret[max(0, pos - 19): pos + 1]
        out.append(float(np.std(window)))
    return np.asarray(out)


def main() -> int:
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    n = len(sl[PRIMARY][1]["tr"])
    dates = [w["context_end_date"] for w in make_windows(load_csv("688169"))]
    assert len(dates) == n, f"date count {len(dates)} != store windows {n}"

    vol = trailing_vol(dates, "688169")
    q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
    regime = np.where(vol <= q1, 0, np.where(vol <= q2, 1, 2))
    print(f"windows={n}  vol terciles: q1={q1:.4f} q2={q2:.4f}  "
          f"counts={np.bincount(regime, minlength=3).tolist()}")

    result = {"n_windows": n, "vol_q1": float(q1), "vol_q2": float(q2),
              "regime_counts": np.bincount(regime, minlength=3).tolist(),
              "per_horizon": {}}

    total_base, total_adapt = 0.0, 0.0
    for h in HORIZONS:
        p = sl[PRIMARY][h]["pret"].astype(np.float64)
        s = sl[SECONDARY][h]["pret"].astype(np.float64)
        tr = sl[PRIMARY][h]["tr"].astype(np.float64)

        base_mae = float(np.mean(np.abs(FIXED_W[h] * p + (1 - FIXED_W[h]) * s - tr)))
        # best fixed weight on eval (oracle fixed, reference)
        fixed_curve = [(w, float(np.mean(np.abs(w * p + (1 - w) * s - tr))))
                       for w in W_GRID]
        best_fixed_w, best_fixed_mae = min(fixed_curve, key=lambda t: t[1])

        # per-regime optimal weights
        adapt_pred = np.zeros(n)
        regime_rows = []
        for r in (0, 1, 2):
            mask = regime == r
            curve = [(w, float(np.mean(np.abs(w * p[mask] + (1 - w) * s[mask] - tr[mask]))))
                     for w in W_GRID]
            w_r, mae_r = min(curve, key=lambda t: t[1])
            adapt_pred[mask] = w_r * p[mask] + (1 - w_r) * s[mask]
            regime_rows.append({"regime": int(r), "n": int(mask.sum()),
                                "best_w": float(w_r), "mae": mae_r,
                                "fixed_w_mae": float(np.mean(np.abs(
                                    FIXED_W[h] * p[mask] + (1 - FIXED_W[h]) * s[mask] - tr[mask])))})
        adapt_mae = float(np.mean(np.abs(adapt_pred - tr)))
        total_base += base_mae
        total_adapt += adapt_mae
        result["per_horizon"][str(h)] = {
            "baseline_w": FIXED_W[h], "baseline_mae": base_mae,
            "oracle_fixed_w": float(best_fixed_w), "oracle_fixed_mae": best_fixed_mae,
            "adaptive_mae": adapt_mae,
            "adaptive_vs_baseline_rel": (adapt_mae - base_mae) / base_mae,
            "regimes": regime_rows,
        }
        print(f"h={h}: base={base_mae:.6f} adaptive={adapt_mae:.6f} "
              f"({(adapt_mae - base_mae) / base_mae:+.3%})  "
              f"regime_w={[r['best_w'] for r in regime_rows]}")

    result["overall_baseline_mae"] = total_base / len(HORIZONS)
    result["overall_adaptive_mae"] = total_adapt / len(HORIZONS)
    result["overall_rel_change"] = (total_adapt - total_base) / total_base
    print(f"\nOVERALL: baseline={total_base / 4:.6f} adaptive={total_adapt / 4:.6f} "
          f"({(total_adapt - total_base) / total_base:+.3%})")

    out = ROOT / "outputs" / "eval_p7_r7_regime_blend.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
