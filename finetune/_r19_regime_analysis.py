"""R19 regime analysis: does r5 spontaneously condition on market regime?"""
import json
import numpy as np

bh_map = {
    "000001": -0.003, "000002": -0.450, "000063": 0.048, "000333": 0.107,
    "000651": 0.018, "000858": -0.413, "000938": 0.630, "002001": 0.308,
    "002007": -0.108, "002415": 0.243, "002466": -0.078, "002594": -0.017,
    "002714": -0.201, "300003": -0.184, "300015": -0.207, "300059": -0.104,
    "600036": -0.093, "600276": -0.070, "600519": -0.084, "600887": -0.010,
    "601318": -0.177, "601398": 0.009, "603259": 0.374, "603288": 0.028,
    "688169": -0.412, "000568": -0.356, "002129": 0.053, "002230": -0.175,
}

d = json.load(open("outputs/eval_p9_panel_r19_bigpanel.json", encoding="utf-8"))
ps = d["base"]["per_symbol"]
up_cov, dn_cov = [], []
for s in ps:
    sym = s["symbol"]
    bh, cov, ret = bh_map.get(sym, 0), s["cov"], s["ret"]
    (up_cov if bh > 0 else dn_cov).append((sym, bh, cov, ret))

print(f"UP-symbols (bh>0): n={len(up_cov)}  mean cov={np.mean([x[2] for x in up_cov]):.2%}  mean ret={np.mean([x[3] for x in up_cov]):+.3f}")
print(f"DN-symbols (bh<=0): n={len(dn_cov)} mean cov={np.mean([x[2] for x in dn_cov]):.2%} mean ret={np.mean([x[3] for x in dn_cov]):+.3f}")
print()
print("Detail (sorted by bh):")
for sym, bh, cov, ret in sorted(up_cov + dn_cov, key=lambda x: x[1]):
    tag = "UP" if bh > 0 else "DN"
    print(f"  {sym} bh={bh:+.3f} cov={cov:.1%} ret={ret:+.3f} {tag}")
