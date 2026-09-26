"""Where does the sub5 recipe lose points? X1 OOF predictions on the test-like validation (551,789 S1s).
Per S1 the loss is 1 - F0.5. Gains are computed by fixing one error type at a time (others unchanged)."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as pds, pyarrow as pa
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.decode import decode_all
from er.io_metric import fbeta_entity
W = "cache/v2/"
X = pd.read_parquet(W + "exp_X1.parquet").drop(columns=["p"]).rename(columns={"p1": "p"})   # stage 1 = sub5 recipe
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); cty = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet("cache/gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
owner = {x: s for s, v in g.items() for x in v}
ids = X.s1_id.unique().tolist()
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
allids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0).tolist()
F = pds.dataset(W + "train_feats_tlv3_f2").to_table(columns=["s1_id", "cand_id", "rk", "n_tset", "a_tset", "aw_tset", "house_eq", "r_addr_empty", "r_known_frac"], filter=pds.field("s1_id").isin(pa.array(ids))).to_pandas()
X = X.merge(F, on=["s1_id", "cand_id"])
pred = decode_all(X[["s1_id", "cand_id", "p", "rk"]], allids, tie_col="rk")
cand = X.groupby("s1_id").cand_id.apply(set).to_dict()
N = len(allids); tot = {"singleton_FP": 0, "matched_FP": 0, "FN_in_shortlist": 0, "FN_not_shortlisted": 0}; lost = 0
by = {}
fp_rows = []
for s in allids:
    G = g.get(s, set()); P = set(pred.get(s, [])); f = fbeta_entity(P, G); lost += 1 - f
    c = cty[s]; kind = "singleton" if not G else "matched"; by[(c, kind)] = by.get((c, kind), 0) + 1 - f
    fp = {x for x in P if x not in G}
    if fp:
        tot["singleton_FP" if not G else "matched_FP"] += fbeta_entity(P - fp, G) - f
        fp_rows += [(s, x, kind) for x in fp]
    miss = G - P
    if miss:
        insl = {x for x in miss if x in cand.get(s, ())}
        tot["FN_in_shortlist"] += fbeta_entity(P | insl, G) - f
        tot["FN_not_shortlisted"] += fbeta_entity(P | (miss - insl), G) - f
print(f"S1s {N:,}; macro F0.5 {1 - lost / N:.5f}; total lost {lost / N:.5f}")
print("lost by country x kind:", {f"{a}/{b}": round(v / N, 5) for (a, b), v in sorted(by.items())})
print("gain if this error type were fixed alone (macro points):", {k: round(v / N, 5) for k, v in tot.items()})
E = pd.DataFrame(fp_rows, columns=["s1_id", "cand_id", "kind"]).merge(X, on=["s1_id", "cand_id"])
E["base"] = E.cand_id.str.replace("~c", "", regex=False)
E["type"] = np.where(E.base.map(owner).notna(), "record owned by another S1", "decoy (owned by nobody)")
E["near_twin"] = (E.n_tset >= 85) & (E.aw_tset >= 80) & (E.house_eq != 1)
E["same_addr_diff_name"] = (E.house_eq == 1) & (E.n_tset < 60)
print(f"\nwrong IDs accepted: {len(E):,}"); print(E.groupby(["kind", "type"]).size().unstack(fill_value=0).to_string())
print("near-twin share", round(E.near_twin.mean(), 3), "| same-address-different-name share", round(E.same_addr_diff_name.mean(), 3), "| median p", round(E.p.median(), 3))
T = X[(X.y == 1)].copy(); T["acc"] = [c in set(pred.get(s, [])) for s, c in zip(T.s1_id, T.cand_id)]
M = T[~T.acc]
print(f"\ntrue IDs shortlisted but rejected: {len(M):,}; empty address {M.r_addr_empty.mean():.2f}; pseudo-word name (known_frac<0.5) {(M.r_known_frac < .5).mean():.2f}; house not exact {(M.house_eq != 1).mean():.2f}; median p {M.p.median():.3f}")
print("p of rejected true IDs:", pd.cut(M.p, [0, .1, .3, .5, .7, 1]).value_counts(normalize=True).sort_index().round(3).to_dict())
