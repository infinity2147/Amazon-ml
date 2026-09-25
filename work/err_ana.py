"""Tag every OOF error by cause and print examples."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.decode import decode_all
pd.set_option("display.width", 250)
W = "/home/24b4518/ml-projects/work/cache/"; tag = sys.argv[1]
O = pd.read_parquet(W + f"v1/oof_{tag}.parquet")
ids = O.s1_id.unique()
gt = pd.read_parquet(W + "gt.parquet"); gt = gt[gt.source1_entity_id.isin(set(ids))]
g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
s1 = pd.read_parquet(W + "v1/train_s1p.parquet").set_index("entity_id"); cp = pd.read_parquet(W + "v1/train_cp.parquet").set_index("entity_id")
# ids in the OOF frame are only S1s with >=1 candidate; S1s with no candidates are singletons-or-missed
pred = decode_all(O[["s1_id", "cand_id", "p"]], list(g))
P = O.set_index(["s1_id", "cand_id"]).p
cand = O.groupby("s1_id").cand_id.apply(set).to_dict()
rows = []
for s, gs in g.items():
    pr = set(pred[s])
    for x in pr - gs:
        rows.append(("FP_singleton" if not gs else "FP_matched", s, x, P.get((s, x), np.nan)))
    for x in gs - pr:
        rows.append(("FN_not_blocked" if x not in cand.get(s, ()) else "FN_low_p", s, x, P.get((s, x), np.nan)))
E = pd.DataFrame(rows, columns=["tag", "s1", "c", "p"])
E["ctry"] = E.s1.map(s1.country)
print(E.groupby(["ctry", "tag"]).size().unstack().to_string())
def show(tag, n=14, seed=0):
    sub = E[E.tag == tag].sample(min(n, (E.tag == tag).sum()), random_state=seed)
    print(f"\n######## {tag}")
    for r in sub.itertuples():
        a = s1.loc[r.s1]; b = cp.loc[r.c]
        others = ""
        if tag.startswith("FP"):
            # who really owns this record?
            own = [k for k, v in g.items() if r.c in v]
            others = f" | true owner: {own[0] if own else 'none (decoy)'}"
        print(f"p={r.p:.2f} {r.ctry}{others}\n   S1: {a.business_name} | {a.business_address}\n   {r.c[:2]}: {b.business_name} | {b.business_address}")
for t in ["FP_matched", "FP_singleton", "FN_low_p"]:
    show(t)
