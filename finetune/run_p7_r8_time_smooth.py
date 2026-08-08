"""Phase 7 — Round R8: causal time-smoothing of direction logits (numpy only).

Adjacent eval windows shift by 1 trading day with ~99% overlapping context,
so per-window logits contain sampling noise. Averaging logits over the last
k window positions (CAUSAL: only current + past windows, no future leakage)
is TTA across time, complementary to TTA across lookback.

Sweep: per (model, horizon) x P5-8 TTA config x time-window k in {1,2,3,4,5}
(k=1 == baseline, no smoothing). Candidate = per-horizon best (k, cfg) on the
P5-8 model mapping; gate re-swept; dual-bar vs baseline from the same cache.

Output: outputs/eval_p7_r8_time_smooth.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from dual_metric_compare import blend_returns, dual_bar_decision, summarize_horizons  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from run_p6_eval_zoo import (  # noqa: E402
    GATE_MAG,
    P5_BASE_MAP,
    P5_TTA_BY_H,
    PRIMARY_W_BY_H,
    gate_sweep,
)
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
K_GRID = (1, 2, 3, 4, 5)
# per-horizon candidate TTA configs: baseline one + its 2 nearest variants
EXTRA_CFGS = {
    1: [(124, 126, 128, 130), (122, 124, 126, 128, 130), (124, 126, 128, 130, 132)],
    3: [(124, 126, 128, 130), (124, 126, 128), (124, 126, 128, 130, 132)],
    5: [(124, 126, 128, 130, 132, 134), (124, 126, 128, 130, 132), (122, 124, 126, 128, 130, 132, 134)],
    10: [(126, 128, 130, 132), (128, 130, 132), (126, 128, 130, 132, 134)],
}


def tta_logits(store, model, h, lookbacks):
    idx = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    return average_logits([store[model][h]["per_lb_logits"][i] for i in idx], None)


def time_smooth(logits: np.ndarray, k: int) -> np.ndarray:
    """Causal trailing mean over window axis (axis 0). k=1 -> identity."""
    if k <= 1:
        return logits
    cum = np.cumsum(logits, axis=0)
    out = logits.copy()
    out[k - 1:] = (cum[k - 1:] - np.concatenate(
        [np.zeros((1,) + logits.shape[1:]), cum[:-k]], axis=0)) / k
    # first k-1 rows: mean of available history (1..i+1)
    for i in range(k - 1):
        out[i] = cum[i] / (i + 1)
    return out


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    result = {}

    # ---- baseline P5-8 (k=1) ----
    bp_dir, bt_dir, bt_ret, bpred_ret, bscore = {}, {}, {}, {}, {}
    for h in HORIZONS:
        conf = direction_confidence_from_logits(
            tta_logits(tta, P5_BASE_MAP[h], h, P5_TTA_BY_H[h]))
        bp_dir[h] = conf["hard_pred"]
        bscore[h] = conf["actionable_score"]
        bt_dir[h] = tta[P5_BASE_MAP[h]][h]["td"]
        bt_ret[h] = tta[P5_BASE_MAP[h]][h]["tr"]
        bpred_ret[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                     PRIMARY_W_BY_H[h])
    base_summary = summarize_horizons(bp_dir, bpred_ret, bt_dir, bt_ret, HORIZONS)
    bg = apply_consistency_filter(
        apply_consistency_and_magnitude_gate(
            bp_dir[1], bscore[1], sl[PRIMARY][1]["pret"],
            confidence_threshold=0.45, min_abs_return=GATE_MAG,
            require_sign_agree=False),
        bp_dir[1], bp_dir[5], "strict")
    bm = gated_actionable_metrics(bg, bt_dir[1])
    bbt = absolute_direction_backtest(bg, bt_ret[1], transaction_cost=0.0005)
    base_summary.update(h1_gated_precision=bm["precision_on_calls"],
                        h1_gated_nonflat=bm["gated_nonflat_acc"],
                        h1_gated_coverage=bm["coverage"], h1_gated_backtest=bbt)
    print(f"BASELINE: nf={base_summary['nonflat_accuracy_overall']:.2%} "
          f"prec={bm['precision_on_calls']:.2%} gnf={bm['gated_nonflat_acc']:.2%} "
          f"bt={bbt['total_return']:.2%}")
    result["baseline_summary"] = base_summary

    # ---- per-horizon (cfg, k) sweep ----
    best_by_h = {}
    sweep_out = {}
    for h in HORIZONS:
        m = P5_BASE_MAP[h]
        td = tta[m][h]["td"]
        rows = []
        for lbs in EXTRA_CFGS[h]:
            raw = tta_logits(tta, m, h, lbs)
            for k in K_GRID:
                sm = time_smooth(raw, k)
                nf = float(nonflat_accuracy(
                    direction_confidence_from_logits(sm)["hard_pred"], td))
                rows.append({"lbs": list(lbs), "k": k, "nf": nf})
        rows.sort(key=lambda r: r["nf"], reverse=True)
        sweep_out[str(h)] = rows[:8]
        best_by_h[h] = rows[0]
        base_nf = [r["nf"] for r in rows
                   if r["k"] == 1 and tuple(r["lbs"]) == P5_TTA_BY_H[h]][0]
        print(f"h={h}: best={rows[0]['lbs']} k={rows[0]['k']} nf={rows[0]['nf']:.2%} "
              f"| baseline cfg k=1 nf={base_nf:.2%} | d={rows[0]['nf'] - base_nf:+.2%}")
    result["sweep_top8_by_horizon"] = sweep_out
    result["best_by_horizon"] = {str(h): best_by_h[h] for h in HORIZONS}

    # ---- candidate from per-h best ----
    cp_dir, ct_dir, ct_ret, cpred_ret, cscore = {}, {}, {}, {}, {}
    for h in HORIZONS:
        b = best_by_h[h]
        m = P5_BASE_MAP[h]
        sm = time_smooth(tta_logits(tta, m, h, tuple(b["lbs"])), b["k"])
        conf = direction_confidence_from_logits(sm)
        cp_dir[h] = conf["hard_pred"]
        cscore[h] = conf["actionable_score"]
        ct_dir[h] = tta[m][h]["td"]
        ct_ret[h] = tta[m][h]["tr"]
        cpred_ret[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                     PRIMARY_W_BY_H[h])
    cand_summary = summarize_horizons(cp_dir, cpred_ret, ct_dir, ct_ret, HORIZONS)
    cb, _ = gate_sweep(cp_dir, cscore, sl[PRIMARY][1]["pret"], ct_dir[1], True)
    if cb:
        cm = gated_actionable_metrics(cb["gated"], ct_dir[1])
        cbt = absolute_direction_backtest(cb["gated"], ct_ret[1], transaction_cost=0.0005)
        cand_summary.update(h1_gated_precision=cm["precision_on_calls"],
                            h1_gated_nonflat=cm["gated_nonflat_acc"],
                            h1_gated_coverage=cm["coverage"], h1_gated_backtest=cbt,
                            h1_gate={"thr": cb["thr"], "mag": GATE_MAG,
                                     "consistency": "strict_h5"})
        print(f"CANDIDATE: nf={cand_summary['nonflat_accuracy_overall']:.2%} "
              f"cov={cm['coverage']:.2%} prec={cm['precision_on_calls']:.2%} "
              f"gnf={cm['gated_nonflat_acc']:.2%} bt={cbt['total_return']:.2%}")
    decision = dual_bar_decision(cand_summary, base_summary)
    print(f"DECISION: promote={decision['promote']} reason={decision['reason']} "
          f"d_nf={decision['nonflat_delta']:+.4f}")
    result.update(candidate_summary=cand_summary, decision_vs_p5_8=decision)

    out = ROOT / "outputs" / "eval_p7_r8_time_smooth.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
