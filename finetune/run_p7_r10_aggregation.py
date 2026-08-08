"""Phase 7 — Round R10: final aggregation (h=10 plateau candidate, full dual-bar)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import load_sl_store  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits  # noqa: E402
from run_p6_eval_zoo import P5_TTA_BY_H, P5_BASE_MAP, PRIMARY_W_BY_H, GATE_MAG  # noqa: E402
from run_p5_r8_consistency_gate import apply_consistency_filter  # noqa: E402
from dual_metric_compare import blend_returns, summarize_horizons, dual_bar_decision  # noqa: E402
from selective_prediction import (  # noqa: E402
    absolute_direction_backtest,
    apply_consistency_and_magnitude_gate,
    direction_confidence_from_logits,
    gated_actionable_metrics,
)

H = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"


def build(tta, sl, tta_by_h):
    pd_, td_, tr_, pr_, sc_ = {}, {}, {}, {}, {}
    for h in H:
        m = P5_BASE_MAP[h]
        idx = [ALL_LOOKBACKS.index(x) for x in tta_by_h[h]]
        conf = direction_confidence_from_logits(
            average_logits([tta[m][h]["per_lb_logits"][i] for i in idx], None))
        pd_[h] = conf["hard_pred"]
        sc_[h] = conf["actionable_score"]
        td_[h] = tta[m][h]["td"]
        tr_[h] = tta[m][h]["tr"]
        pr_[h] = blend_returns(sl[PRIMARY][h]["pret"], sl[SECONDARY][h]["pret"],
                               PRIMARY_W_BY_H[h])
    s = summarize_horizons(pd_, pr_, td_, tr_, H)
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


def main() -> int:
    tta = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
    sl = load_sl_store(ROOT / "outputs" / "p6_sl_store.npz")
    cand_tta = dict(P5_TTA_BY_H)
    cand_tta[10] = (128, 130, 132)  # R2/R8 plateau center

    base = build(tta, sl, P5_TTA_BY_H)
    cand = build(tta, sl, cand_tta)
    for tag, s in (("BASE", base), ("CAND", cand)):
        print(f"{tag}: nf={s['nonflat_accuracy_overall']:.4f} "
              f"mae={s['return_mae_overall']:.6f} "
              f"prec={s['h1_gated_precision']:.2%} gnf={s['h1_gated_nonflat']:.2%} "
              f"cov={s['h1_gated_coverage']:.2%} "
              f"bt={s['h1_gated_backtest']['total_return']:.2%}")
    d = dual_bar_decision(cand, base)
    print(f"DECISION: promote={d['promote']} reason={d['reason']} "
          f"d_nf={d['nonflat_delta']:+.4f}")
    out = ROOT / "outputs" / "eval_p7_r10_aggregation.json"
    out.write_text(json.dumps({"baseline": base, "candidate_h10_plateau": cand,
                               "decision": d}, indent=1, default=str), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
