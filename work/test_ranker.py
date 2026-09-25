import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import rank_features
W = "cache/v1/"; c = sys.argv[1]; t = time.time()
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "ctry"]); cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "ctry"])
a = s1[s1.ctry == c].entity_id.to_numpy(); b = cp[cp.ctry == c].entity_id.to_numpy(); nb = len(b)
ai = pd.Series(np.arange(len(a)), index=a); bi = pd.Series(np.arange(nb), index=b)
gt = pd.read_parquet("cache/gt.parquet"); gt = gt[gt.source1_entity_id.isin(ai.index)]
gp = [(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x]
gkey = ai[[s for s, _ in gp]].to_numpy() * nb + bi[[x for _, x in gp]].to_numpy()
views = ["v_word", "v_name5", "v_addr5"]
V = pd.concat([pd.read_parquet(W + f"blk_train_b1_{c}_{v}.parquet").set_index("key")[v] for v in views], axis=1).fillna(0).astype(np.float32)
key = V.index.to_numpy(); V = V.reset_index(drop=True)
V["s1_id"] = (key // nb).astype(np.int32); V["cand_id"] = (key % nb).astype(np.int32)
y = np.isin(key, gkey)
X = rank_features(V, views); print("feats", X.shape, round(time.time() - t), flush=True)
rng = np.random.default_rng(0); tr_s1 = rng.random(len(a)) < 0.3
is_tr = tr_s1[V.s1_id.to_numpy()]
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200, verbose=-1, num_threads=16),
              lgb.Dataset(X[is_tr], y[is_tr]), 300)
m.save_model(W + f"ranker_{c}.txt")
V["_s"] = m.predict(X, num_threads=16); V["g"] = y
Q = V[~is_tr].copy()
Q["r1"] = Q.groupby("s1_id")._s.rank(ascending=False, method="first"); Q["r2"] = V.groupby("cand_id")._s.rank(ascending=False, method="first")[~is_tr]
G = Q.g.sum(); nS = (~tr_s1).sum()
gs = pd.Series(key[y] // nb).value_counts()                      # gold size per S1 index
print(f"{c}: held-out S1 {nS:,}; uncapped {len(Q)/nS:.1f}/S1 recall 1.0 of blocked (union recall measured before)")
for n1 in [10, 15, 20, 25, 30, 40]:
    for n2 in [0, 1, 2, 3]:
        k = Q[(Q.r1 <= n1) | (Q.r2 <= n2)]; h = k[k.g]
        cnt = h.groupby("s1_id").size()
        full = (cnt.to_numpy() == gs.reindex(cnt.index).to_numpy()).sum()
        nm = gs.index[~tr_s1[gs.index]].size
        print(f"  n_s1={n1:2d} n_c={n2}: {len(k)/nS:5.1f}/S1  kept {len(h)/G:.4f} of blocked gold  full-cov(of blocked-matchable) {full/nm:.4f}", flush=True)
print("done", round(time.time() - t))
