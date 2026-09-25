import sys, numpy as np, pandas as pd, pyarrow.dataset as pds
W = "/home/24b4518/ml-projects/work/cache/v1/"
O = pd.read_parquet(W + "oof_T0.parquet", columns=["s1_id", "cand_id", "y", "p1", "p"])
ids = pd.Series(O.s1_id.unique()).sample(150000, random_state=1); O = O[O.s1_id.isin(set(ids))]
X = pds.dataset(W + "train_feats_tlv_f2").to_table(filter=pds.field("s1_id").isin(ids.tolist())).to_pandas()
D = X.merge(O, on=["s1_id", "cand_id"]); D["is_clone"] = D.cand_id.str.endswith("~c"); D["base"] = D.cand_id.str.replace("~c", "", regex=False)
D["twin"] = D.groupby(["s1_id", "base"]).cand_id.transform("size") >= 2
d = D[(D.y == 0) & (~D.is_clone) & (D.na_prod >= 85) & (D.house_eq != 1)]
feats = [c for c in X.columns if c not in ("s1_id", "cand_id") and X[c].dtype != object]
a, b = d[d.twin], d[~d.twin]
smd = ((a[feats].mean() - b[feats].mean()) / d[feats].std().replace(0, np.nan)).dropna()
top = smd.reindex(smd.abs().sort_values(ascending=False).index).head(12)
print(f"near-twin decoys {len(d):,}; with clone {len(a):,}, without {len(b):,}")
print("largest standardized mean differences (with-clone minus without):"); print(top.round(3).to_string())
print(f"acceptance (p>0.5) with clone {(a.p>.5).mean():.4f}  without {(b.p>.5).mean():.4f}")
# Condition on the strongest differing feature(s): does the gap survive?
f0 = top.index[0]; q = pd.qcut(d[f0].rank(method="first"), 5, labels=False)
print(f"acceptance by quintile of {f0}:")
for k in range(5):
    s = d[q == k]; print(f"  q{k}: with clone {(s[s.twin].p>.5).mean():.4f} (n={s.twin.sum():,})  without {(s[~s.twin].p>.5).mean():.4f} (n={(~s.twin).sum():,})")
