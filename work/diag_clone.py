"""Does a model trained/scored on the test-like validation (TLV) exploit exact-duplicate clones?
Cloning is random and independent of the pair's text, so among ORIGINAL decoy pairs with similar text,
those whose clone sits in the same S1 list should get the same probability as those without a clone."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as pds
W = "/home/24b4518/ml-projects/work/cache/v1/"; name = sys.argv[1]
O = pd.read_parquet(W + f"oof_{name}.parquet")
X = pds.dataset(W + "train_feats_tlv_f2").to_table(columns=["s1_id", "cand_id", "na_prod", "house_eq", "n_tset", "a_tset"],
        filter=pds.field("s1_id").isin(pd.unique(O.s1_id).tolist()[:0] or [])).to_pandas() if False else None
X = pds.dataset(W + "train_feats_tlv_f2").to_table(columns=["s1_id", "cand_id", "na_prod", "house_eq", "n_tset", "a_tset"]).to_pandas()
D = O.merge(X, on=["s1_id", "cand_id"], how="left"); del X
D["is_clone"] = D.cand_id.str.endswith("~c"); D["base"] = D.cand_id.str.replace("~c", "", regex=False)
key = D.s1_id + "|" + D.base
n_same = D.groupby(key).cand_id.transform("size")          # 2 = an original and its clone are both listed
D["has_twin_copy"] = n_same >= 2
d = D[(D.y == 0) & (~D.is_clone)]
print(f"{name}: original decoy pairs {len(d):,}; with a clone in the list {d.has_twin_copy.mean():.3f}")
for lab, m in [("all decoys", d.na_prod >= 0), ("near-twin-ish (na_prod>=85, house!=)", (d.na_prod >= 85) & (d.house_eq != 1)),
               ("na_prod>=95 house!=", (d.na_prod >= 95) & (d.house_eq != 1)), ("na_prod 60-85", (d.na_prod >= 60) & (d.na_prod < 85))]:
    s = d[m]; a, b = s[s.has_twin_copy], s[~s.has_twin_copy]
    print(f"  {lab:38s} n={len(s):>9,}  mean p1: clone-in-list {a.p1.mean():.4f} vs none {b.p1.mean():.4f} | mean p(stage2): {a.p.mean():.4f} vs {b.p.mean():.4f} | frac p>0.5: {(a.p>.5).mean():.4f} vs {(b.p>.5).mean():.4f}")
# true matches are never cloned: do they get pushed up because 'no twin copy'?
t = D[D.y == 1]; print(f"  true-match pairs {len(t):,}: share with a twin copy in list {t.has_twin_copy.mean():.4f} (must be ~0)")
