"""Blocking analysis on train: recall vs cap, per-view unique contribution, per country."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
W = "/home/24b4518/ml-projects/work/cache/"
tag = sys.argv[1] if len(sys.argv) > 1 else "b1"
P = pd.read_parquet(W + f"v1/train_pairs_{tag}.parquet")
s1 = pd.read_parquet(W + "v1/train_s1p.parquet", columns=["entity_id", "ctry"])
gt = pd.read_parquet(W + "gt.parquet")
gold = pd.DataFrame([(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x], columns=["s1_id", "cand_id"])
gold["g"] = 1
P = P.merge(gold, on=["s1_id", "cand_id"], how="left"); P["g"] = P.g.fillna(0).astype(np.int8)
ctry = dict(zip(s1.entity_id, s1.ctry)); P["ctry"] = P.s1_id.map(ctry); gold["ctry"] = gold.s1_id.map(ctry)
vcols = [c for c in P.columns if c.startswith("v_")]
P["_s"] = P[vcols].sum(axis=1)
P["r1"] = P.groupby("s1_id")._s.rank(ascending=False, method="first")
P["r2"] = P.groupby("cand_id")._s.rank(ascending=False, method="first")
nS1 = s1.ctry.value_counts().to_dict()
ng = gold.groupby("ctry").size().to_dict()
gsize = gold.groupby("s1_id").size()
for c in ["india", "us"]:
    Q = P[P.ctry == c]
    print(f"\n=== {c}: gold pairs {ng[c]:,}; uncapped pairs {len(Q):,} ({len(Q)/nS1[c]:.1f}/S1), recall {Q.g.sum()/ng[c]:.4f}")
    for v in vcols:
        only = Q[(Q[v] > 0) & (Q[[x for x in vcols if x != v]].sum(axis=1) == 0)]
        print(f"   {v}: recall alone {Q.g[Q[v]>0].sum()/ng[c]:.4f}, unique gold {only.g.sum():,} ({only.g.sum()/ng[c]:.4f}), unique pairs {len(only):,}")
    for n1 in [10, 15, 20, 30, 40, 60]:
        for n2 in [1, 2, 3, 5, 8]:
            k = Q[(Q.r1 <= n1) | (Q.r2 <= n2)]
            hit = k[k.g == 1]
            full = (hit.groupby("s1_id").size() == gsize.reindex(hit.s1_id.unique())).sum()
            nm = (gsize.index.map(ctry) == c).sum()
            print(f"   cap n_s1={n1:2d} n_c={n2}: {len(k)/nS1[c]:5.1f}/S1  pair recall {len(hit)/ng[c]:.4f}  full cov {full/nm:.4f}")
