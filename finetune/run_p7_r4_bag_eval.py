"""Phase 7 — Round R4: seed-bagging evaluation (numpy only).

Averages per-lookback direction logits across the 4 seed replicas
(p7_bag_s101..s104, r5 frozen recipe) — optionally including r5 itself —
then per-horizon sweeps the 9 P5-1 TTA configs and compares against the
corresponding single-model baseline from the p6 cache:
  * bagging targets r5's horizons (h=3, h=5);
  * candidate swaps bagged-r5 into the P5-8 mapping for h=3/5, keeps
    everything else identical; gate re-checked; dual-bar vs P5-8.

Output: outputs/eval_p7_r4_bagging.json
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
    LOOKBACK_CONFIGS,
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
BAG_MODELS = ["p7_bag_s101", "p7_bag_s102", "p7_bag_s103", "p7_bag_s104"]


def per_lb_avg(store, models, h, lb_idx):
    """Average logits across models for a single lookback index."""
    arrs = [store[m][h]["per_lb_logits"][lb_idx] for m in models]
    return np.mean(np.stack(arrs, axis=0), axis=0)


def tta_from_members(store, members, h, lookbacks):
    idx = [ALL_LOOKBACKS.index(lb) for lb in lookbacks]
    per_lb = [per_lb_avg(store, members, h, i) for i in idx]
    return average_logits(per_lb, None)


def main() -> int:
    p6 = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    bag = load_per_lb_store(ROOT / "outputs" / "p7_bag_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    # merge: single dict for uniform access (targets identical across stores)
    store = {**p6, **bag}
    result = {}

    # sanity: bag store targets == p6 targets
    for h in HORIZONS:
        assert np.array_equal(bag[BAG_MODELS[0]][h]["td"], p6[SECONDARY][h]["td"]), h

    # ---- 1. per-member and ensemble nonflat on r5 horizons ----
    member_sets = {
        "bag4": BAG_MODELS,
        "bag4+r5": BAG_MODELS + [SECONDARY],
        "r5_alone": [SECONDARY],
    }
    ens_table = {}
    for h in (3, 5):
        td = p6[SECONDARY][h]["td"]
        ens_table[str(h)] = {}
        for tag, members in member_sets.items():
            best = None
            for cfg_name, lbs in LOOKBACK_CONFIGS:
                avg = tta_from_members(store, members, h, lbs)
                nf = float(nonflat_accuracy(
                    direction_confidence_from_logits(avg)["hard_pred"], td))
                if best is None or nf > best[1]:
                    best = (cfg_name, nf, lbs)
            ens_table[str(h)][tag] = {"best_cfg": best[0], "nf": best[1]}
            print(f"h={h} {tag}: best={best[0]} nf={best[1]:.2%}")
        # per-member best for reference
        for m in BAG_MODELS:
            best = None
            for cfg_name, lbs in LOOKBACK_CONFIGS:
                avg = tta_from_members(store, [m], h, lbs)
                nf = float(nonflat_accuracy(
                    direction_confidence_from_logits(avg)["hard_pred"], td))
                if best is None or nf > best[1]:
                    best = (cfg_name, nf)
            print(f"   member {m}: best={best[0]} nf={best[1]:.2%}")
    result["ensemble_nonflat"] = ens_table

    # ---- 2. candidate: P5-8 mapping with bagged-r5 on h=3/5 ----
    # choose ensemble tag with best mean nf over h=3/5 (using each tag's best cfg)
    def tag_score(tag):
        return np.mean([ens_table[str(h)][tag]["nf"] for h in (3, 5)])
    best_tag = max(("bag4", "bag4+r5"), key=tag_score)
    print(f"using ensemble: {best_tag}")

    cfg_by_h = dict(P5_TTA_BY_H)
    members_by_h = {h: ([SECONDARY] if h in (1, 10) else None) for h in HORIZONS}
    # h=1,10 -> r10 (unchanged); h=3,5 -> bagged
    pred_dir, t_dir, t_ret, pred_ret, cand_score = {}, {}, {}, {}, {}
    for h in HORIZONS:
        if h in (3, 5):
            tag_cfg = ens_table[str(h)][best_tag]["best_cfg"]
            lbs = dict(LOOKBACK_CONFIGS)[tag_cfg]
            avg = tta_from_members(store, dict(member_sets)[best_tag], h, lbs)
            cfg_by_h[h] = lbs
        else:
            avg = tta_from_members(store, [PRIMARY], h, P5_TTA_BY_H[h])
        conf = direction_confidence_from_logits(avg)
        pred_dir[h] = conf["hard_pred"]
        cand_score[h] = conf["actionable_score"]
        t_dir[h] = p6[SECONDARY][h]["td"]
        t_ret[h] = p6[SECONDARY][h]["tr"]
        pred_ret[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                    PRIMARY_W_BY_H[h])
    cand_summary = summarize_horizons(pred_dir, pred_ret, t_dir, t_ret, HORIZONS)
    cand_best, _ = gate_sweep(pred_dir, cand_score, sl[PRIMARY][1]["pret"], t_dir[1], True)
    if cand_best:
        cm = gated_actionable_metrics(cand_best["gated"], t_dir[1])
        cbt = absolute_direction_backtest(cand_best["gated"], t_ret[1],
                                          transaction_cost=0.0005)
        cand_summary.update(h1_gated_precision=cm["precision_on_calls"],
                            h1_gated_nonflat=cm["gated_nonflat_acc"],
                            h1_gated_coverage=cm["coverage"],
                            h1_gated_backtest=cbt,
                            h1_gate={"thr": cand_best["thr"], "mag": GATE_MAG,
                                     "consistency": "strict_h5"})

    # ---- 3. baseline P5-8 from same caches ----
    bp_dir, bt_dir, bt_ret, bpred_ret, bscore = {}, {}, {}, {}, {}
    for h in HORIZONS:
        avg = tta_from_members(store, [P5_BASE_MAP[h]], h, P5_TTA_BY_H[h])
        conf = direction_confidence_from_logits(avg)
        bp_dir[h] = conf["hard_pred"]
        bscore[h] = conf["actionable_score"]
        bt_dir[h] = p6[SECONDARY][h]["td"]
        bt_ret[h] = p6[SECONDARY][h]["tr"]
        bpred_ret[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                                     PRIMARY_W_BY_H[h])
    base_summary = summarize_horizons(bp_dir, bpred_ret, bt_dir, bt_ret, HORIZONS)
    bg = apply_consistency_and_magnitude_gate(
        bp_dir[1], bscore[1], sl[PRIMARY][1]["pret"],
        confidence_threshold=0.45, min_abs_return=GATE_MAG, require_sign_agree=False)
    bg = apply_consistency_filter(bg, bp_dir[1], bp_dir[5], "strict")
    bm = gated_actionable_metrics(bg, bt_dir[1])
    bbt = absolute_direction_backtest(bg, bt_ret[1], transaction_cost=0.0005)
    base_summary.update(h1_gated_precision=bm["precision_on_calls"],
                        h1_gated_nonflat=bm["gated_nonflat_acc"],
                        h1_gated_coverage=bm["coverage"], h1_gated_backtest=bbt)
    print(f"CANDIDATE({best_tag}): nf={cand_summary['nonflat_accuracy_overall']:.2%} "
          f"vs BASE nf={base_summary['nonflat_accuracy_overall']:.2%}")
    print(f"CAND gate: cov={cand_summary.get('h1_gated_coverage', 0):.2%} "
          f"prec={cand_summary.get('h1_gated_precision', 0):.2%} "
          f"gnf={cand_summary.get('h1_gated_nonflat', 0):.2%} "
          f"bt={cand_summary.get('h1_gated_backtest', {}).get('total_return', 0):.2%}")
    print(f"BASE gate: cov={bm['coverage']:.2%} prec={bm['precision_on_calls']:.2%} "
          f"gnf={bm['gated_nonflat_acc']:.2%} bt={bbt['total_return']:.2%}")

    decision = dual_bar_decision(cand_summary, base_summary)
    print(f"DECISION: promote={decision['promote']} reason={decision['reason']} "
          f"d_nf={decision['nonflat_delta']:+.4f}")
    result.update(candidate_summary=cand_summary, baseline_summary=base_summary,
                  decision_vs_p5_8=decision, ensemble_used=best_tag,
                  candidate_cfg_by_h={str(h): list(cfg_by_h[h]) for h in HORIZONS})

    out = ROOT / "outputs" / "eval_p7_r4_bagging.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
