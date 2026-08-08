"""Phase 8 — Round R4: market-regime conditional model selection (holdout-validated).

Hypothesis: r10 (joint) vs r5 (frozen) relative edge per horizon may depend
on the market regime. P7-R7 taught us: in-sample regime conditioning
overfits -> this round is built around strict temporal holdout:
  fit per-(horizon, regime) best model on windows[0:72], evaluate on
  windows[72:144]; and the reverse direction. Report both.

Regime feature (past-only): trailing 20d equal-weight market log-return
computed from the local 140-stock universe CSVs, as of each window's
context_end_date. Regimes: up / flat / down by terciles of the trailing
market return over the full sample (tercile edges from fit half only in the
strict variant; we use fit-half edges to avoid leakage).

Baseline: fixed P5-8 mapping (h=1,10->r10; h=3,5->r5), measured per horizon
on the same test half.

Output: outputs/eval_p8_r4_regime_model_select.json
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
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p6_eval_zoo import P5_TTA_BY_H, P5_BASE_MAP  # noqa: E402
from selective_prediction import (  # noqa: E402
    direction_confidence_from_logits,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
CSV_DIR = ROOT / "data" / "a_share_finetune_multiboard" / "csv"


def market_trailing_ret(dates: list[str], trail: int = 20) -> np.ndarray:
    """Equal-weight mean trailing-`trail` log return across the universe."""
    acc = None
    n_stocks = 0
    for path in sorted(CSV_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["timestamps", "close"],
                             parse_dates=["timestamps"])
        except Exception:
            continue
        df = df.sort_values("timestamps").reset_index(drop=True)
        close = df["close"].astype(float).to_numpy()
        lr = np.log(close[trail:] / close[:-trail])
        s = pd.Series(lr, index=df["timestamps"].iloc[trail:])
        acc = s if acc is None else acc.add(s, fill_value=np.nan)
        n_stocks += 1
    mkt = (acc / n_stocks).sort_index()
    out = []
    for d in dates:
        t = pd.Timestamp(d)
        s = mkt.loc[:t]
        out.append(float(s.iloc[-1]) if len(s) else np.nan)
    return np.asarray(out)


def tta_nf(store, model, h, lbs, td):
    idx = [ALL_LOOKBACKS.index(x) for x in lbs]
    avg = average_logits([store[model][h]["per_lb_logits"][i] for i in idx], None)
    return direction_confidence_from_logits(avg)["hard_pred"], td


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    dates = [w["context_end_date"] for w in make_windows(load_csv("688169"))]
    n = len(dates)
    mret = market_trailing_ret(dates)
    print(f"windows={n}, market trailing20d ret: "
          f"min={np.nanmin(mret):.3f} max={np.nanmax(mret):.3f}")

    # per-h hard preds for both models (P5-8 TTA configs per model role)
    hard = {}
    td_all = {}
    for h in HORIZONS:
        td = tta[PRIMARY][h]["td"]
        td_all[h] = td
        for m in (PRIMARY, SECONDARY):
            lbs = P5_TTA_BY_H[h]
            idx = [ALL_LOOKBACKS.index(x) for x in lbs]
            avg = average_logits([tta[m][h]["per_lb_logits"][i] for i in idx], None)
            hard[(m, h)] = direction_confidence_from_logits(avg)["hard_pred"]

    def run_direction(fit_idx, test_idx, label):
        q1, q2 = np.nanquantile(mret[fit_idx], [1 / 3, 2 / 3])
        regime = np.where(mret <= q1, 0, np.where(mret <= q2, 1, 2))
        rows = {}
        tot_base, tot_cond = [], []
        for h in HORIZONS:
            mbase = P5_BASE_MAP[h]
            # fit: best model per regime
            choice = {}
            for r in (0, 1, 2):
                msk = fit_idx[regime[fit_idx] == r]
                if len(msk) == 0:
                    choice[r] = mbase
                    continue
                nf = {m: nonflat_accuracy(hard[(m, h)][msk], td_all[h][msk])
                      for m in (PRIMARY, SECONDARY)}
                choice[r] = max(nf, key=nf.get)
            # test: baseline vs conditional
            b = nonflat_accuracy(hard[(mbase, h)][test_idx], td_all[h][test_idx])
            pred_cond = np.empty(len(test_idx), dtype=int)
            for r in (0, 1, 2):
                msk_t = np.where(regime[test_idx] == r)[0]
                if len(msk_t):
                    pred_cond[msk_t] = hard[(choice[r], h)][test_idx][msk_t]
            c = nonflat_accuracy(pred_cond, td_all[h][test_idx])
            rows[str(h)] = {"baseline_nf": float(b), "conditional_nf": float(c),
                            "choices": {str(r): choice[r] for r in (0, 1, 2)}}
            tot_base.append(b)
            tot_cond.append(c)
            print(f"  [{label}] h={h}: base={b:.2%} cond={c:.2%} choices={choice}")
        print(f"[{label}] overall: base={np.mean(tot_base):.2%} "
              f"cond={np.mean(tot_cond):.2%}")
        return {"per_horizon": rows,
                "overall_base": float(np.mean(tot_base)),
                "overall_cond": float(np.mean(tot_cond))}

    first, second = np.arange(0, 72), np.arange(72, 144)
    res = {
        "fit_first_test_second": run_direction(first, second, "1st->2nd"),
        "fit_second_test_first": run_direction(second, first, "2nd->1st"),
    }
    out = ROOT / "outputs" / "eval_p8_r4_regime_model_select.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
