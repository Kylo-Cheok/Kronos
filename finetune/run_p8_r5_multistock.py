"""Phase 8 — Round R5: multi-symbol generalization check.

P7's core critique: the 144-window 688169 anchor is both selection set and
evaluation set. This round asks whether the two promoted choices GENERALIZE:
  (a) per-horizon model mapping (h=1,10->r10; h=3,5->r5) with P5-8 TTA cfgs;
  (b) the R2 h=10 plateau (128,130,132) vs promoted (126,128,130,132).

Protocol: 8 long-history universe symbols; same eval anchor
(context_end_date > 2025-12-15); fixed reference targets (ctx=122,
dz=0.003, vol=0.5) for every symbol; narrow TTA grid 122..134.

Outputs: outputs/p8_ms_stores/<sym>.npz + outputs/eval_p8_r5_multistock.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import load_csv  # noqa: E402
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, average_logits, predict_per_lookback  # noqa: E402
from run_p6_eval_zoo import P5_TTA_BY_H, P5_BASE_MAP  # noqa: E402
from run_tta_eval import make_tta_windows  # noqa: E402
from selective_prediction import (  # noqa: E402
    direction_confidence_from_logits,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
PRIMARY = "r10_joint_splitlr"
SECONDARY = "r5_frozen_pool48"
SYMBOLS = ["000001", "000002", "000063", "000333", "000651", "002415", "600036", "601318"]
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5
OUT_DIR = ROOT / "outputs" / "p8_ms_stores"


def build_symbol(sym: str, loaded: dict) -> dict | None:
    df = load_csv(sym)
    windows = make_tta_windows(df, ALL_LOOKBACKS)
    if len(windows) < 60:
        print(f"  {sym}: only {len(windows)} windows, skip")
        return None
    store = {m: {h: {"per_lb": [[] for _ in ALL_LOOKBACKS]} for h in HORIZONS}
             for m in loaded}
    td_all = []
    for w in windows:
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
                    store[m][h]["per_lb"][lb_i].append(lg[hi])
    return {"store": store, "td": np.asarray(td_all), "n": len(windows)}


def tta_hard(store, m, h, lbs):
    idx = [ALL_LOOKBACKS.index(x) for x in lbs]
    avg = average_logits([store[m][h]["per_lb"][i] for i in idx], None)
    return direction_confidence_from_logits(avg)["hard_pred"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    loaded = {m: load_model(model_dir(m)) for m in (PRIMARY, SECONDARY)}
    per_sym = {}
    for sym in symbols:
        sym_json = OUT_DIR / f"{sym}.json"
        if sym_json.exists():
            print(f"{sym}: cached, loading")
            per_sym[sym] = json.loads(sym_json.read_text(encoding="utf-8"))
            continue
        try:
            r = build_symbol(sym, loaded)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  {sym}: FAILED {e}")
            continue
        if r is None:
            continue
        store, td, n = r["store"], r["td"], r["n"]
        row = {"n_windows": int(n)}
        for h in HORIZONS:
            tdh = td[:, HORIZONS.index(h)]
            nf = {}
            for m in (PRIMARY, SECONDARY):
                nf[m] = float(nonflat_accuracy(tta_hard(store, m, h, P5_TTA_BY_H[h]), tdh))
            promoted_m = P5_BASE_MAP[h]
            other = SECONDARY if promoted_m == PRIMARY else PRIMARY
            row[f"h{h}"] = {"promoted": promoted_m, "promoted_nf": nf[promoted_m],
                            "other_nf": nf[other]}
        # h=10 plateau comparison
        td10 = td[:, 3]
        row["h10_plateau"] = {
            "promoted_4lb": float(nonflat_accuracy(
                tta_hard(store, PRIMARY, 10, (126, 128, 130, 132)), td10)),
            "plateau_3lb": float(nonflat_accuracy(
                tta_hard(store, PRIMARY, 10, (128, 130, 132)), td10)),
        }
        per_sym[sym] = row
        sym_json.write_text(json.dumps(row, indent=1), encoding="utf-8")
        print(f"{sym} (n={n}): " + " ".join(
            f"h{h}={row[f'h{h}']['promoted_nf']:.1%}/vs{row[f'h{h}']['other_nf']:.1%}"
            for h in HORIZONS) +
            f" | h10 plateau {row['h10_plateau']['plateau_3lb']:.1%} "
            f"vs promoted {row['h10_plateau']['promoted_4lb']:.1%}", flush=True)
        del r
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # aggregate
    agg = {"per_symbol": per_sym, "elapsed_sec": round(time.time() - t0, 1)}
    wins = {f"h{h}": 0 for h in HORIZONS}
    plat_wins = 0
    for sym, row in per_sym.items():
        for h in HORIZONS:
            r = row[f"h{h}"]
            if r["promoted_nf"] >= r["other_nf"]:
                wins[f"h{h}"] += 1
        if row["h10_plateau"]["plateau_3lb"] >= row["h10_plateau"]["promoted_4lb"]:
            plat_wins += 1
    agg["mapping_win_count_of"] = {"total": len(per_sym), **wins}
    agg["h10_plateau_win_or_tie"] = {"count": plat_wins, "total": len(per_sym)}
    print("\nmapping promoted>=other wins:", agg["mapping_win_count_of"])
    print("h10 plateau win-or-tie:", agg["h10_plateau_win_or_tie"])
    out = ROOT / "outputs" / "eval_p8_r5_multistock.json"
    out.write_text(json.dumps(agg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
