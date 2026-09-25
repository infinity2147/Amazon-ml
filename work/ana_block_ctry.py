"""Recall analysis for one country straight from cached per-view blocking files."""
import sys, numpy as np, pandas as pd
W = "/home/24b4518/ml-projects/work/cache/v1/"
c = sys.argv[1]; tag = "train_b1"
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "ctry"]); cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "ctry"])
a = s1[s1.ctry == c].entity_id.to_numpy(); b = cp[cp.ctry == c].entity_id.to_numpy(); nb = len(b)
ai = pd.Series(np.arange(len(a)), index=a); bi = pd.Series(np.arange(nb), index=b)
gt = pd.read_parquet("/home/24b4518/ml-projects/work/cache/gt.parquet")
gt = gt[gt.source1_entity_id.isin(ai.index)]
gp = [(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x]
gkey = ai[[s for s, _ in gp]].to_numpy() * nb + bi[[x for _, x in gp]].to_numpy()
gsize = pd.Series([s for s, _ in gp]).value_counts()
n_match_ent = len(gsize)
views = ["v_word", "v_name5", "v_addr5"]
cols = {v: pd.read_parquet(W + f"blk_{tag}_{c}_{v}.parquet").set_index("key")[v] for v in views}
V = pd.concat(cols.values(), axis=1).fillna(0)
V["g"] = V.index.isin(gkey)
G = len(gkey)
print(f"{c}: S1 {len(a):,}, gold pairs {G:,}, union {len(V):,} ({len(V)/len(a):.1f}/S1), union recall {V.g.sum()/G:.4f}")
for v in views:
    others = [x for x in views if x != v]
    only = V[(V[v] > 0) & (V[others].sum(axis=1) == 0)]
    print(f"  {v}: recall alone {V.g[V[v] > 0].sum()/G:.4f}; unique gold {only.g.sum():,} ({only.g.sum()/G:.4f}); unique pairs {len(only):,}")
key = V.index.to_numpy(); V["s"] = V[views].sum(axis=1).to_numpy(); V["si"] = key // nb; V["cj"] = key % nb
V["r1"] = V.groupby("si").s.rank(ascending=False, method="first"); V["r2"] = V.groupby("cj").s.rank(ascending=False, method="first")
for n1 in [10, 15, 20, 30, 40, 60]:
    for n2 in [0, 1, 2, 3, 5, 8]:
        k = V[(V.r1 <= n1) | (V.r2 <= n2)]
        h = k[k.g]
        full = (h.groupby("si").size() == gsize.reindex(a[h.si.unique()]).to_numpy()).sum()
        print(f"  cap n_s1={n1:2d} n_c={n2}: {len(k)/len(a):5.1f}/S1  pair recall {len(h)/G:.4f}  full cov {full/n_match_ent:.4f}")
