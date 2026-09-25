"""Union of cached views per country -> stage-1 ranker -> capped candidate set (string ids).
train: fits the ranker on 30% of S1s (pooled countries), saves the set of ranker S1s.
test : applies the saved ranker."""
import sys, time, gc, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import rank_features, group_rank
W = "/home/24b4518/ml-projects/work/cache/v1/"
sp = sys.argv[1]; N1, NC = 20, 1
VIEWS = ["v_word", "v_name5", "v_addr5"]
t = time.time()
log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
s1 = pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["entity_id", "ctry"]); cp = pd.read_parquet(W + f"{sp}_cp.parquet", columns=["entity_id", "ctry"])
ctrys = sorted(set(s1.ctry))

def union(c):
    a = s1.entity_id[s1.ctry == c].to_numpy(); b = cp.entity_id[cp.ctry == c].to_numpy(); nb = len(b)
    V = pd.concat([pd.read_parquet(W + f"blk_{sp}_b1_{c}_{v}.parquet").set_index("key")[v] for v in VIEWS], axis=1).fillna(0).astype(np.float32)
    key = V.index.to_numpy(); V = V.reset_index(drop=True)
    V["si"] = (key // nb).astype(np.int32); V["cj"] = (key % nb).astype(np.int32)
    return V, a, b

if sp == "train":
    gt = pd.read_parquet("/home/24b4518/ml-projects/work/cache/gt.parquet")
    gold = {(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x}
    rng = np.random.default_rng(42)
    rank_s1 = set(s1.entity_id[rng.random(len(s1)) < 0.3])
    pd.Series(sorted(rank_s1)).to_frame("s1_id").to_parquet(W + "ranker_s1.parquet")
    Xs, ys = [], []
    for c in ctrys:
        V, a, b = union(c)
        X = rank_features(V, VIEWS, "si", "cj")
        m = pd.Index(a).isin(rank_s1)[V.si.to_numpy()]
        sid, cid = a[V.si.to_numpy()[m]], b[V.cj.to_numpy()[m]]
        ys.append(np.fromiter(((x, y) in gold for x, y in zip(sid, cid)), bool, m.sum()))
        Xs.append(X[m]); log(f"{c}: union {len(V):,}, ranker rows {m.sum():,}")
        del V, X; gc.collect()
    X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); del Xs
    ranker = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200, verbose=-1, num_threads=24, seed=0),
                       lgb.Dataset(X, y), 300)
    ranker.save_model(W + "ranker.txt"); log(f"ranker trained on {len(X):,} rows, pos {y.mean():.4f}")
    del X; gc.collect()
ranker = lgb.Booster(model_file=W + "ranker.txt")
out = []
for c in ctrys:
    V, a, b = union(c)
    s = ranker.predict(rank_features(V, VIEWS, "si", "cj"), num_threads=24).astype(np.float32)
    r1 = group_rank(V.si.to_numpy(), s)[0]; r2 = group_rank(V.cj.to_numpy(), s)[0]
    k = (r1 <= N1) | (r2 <= NC)
    P = V[k].reset_index(drop=True); P["rk"] = s[k]
    P.insert(0, "cand_id", b[P.pop("cj").to_numpy()]); P.insert(0, "s1_id", a[P.pop("si").to_numpy()])
    out.append(P); log(f"{c}: union {len(V):,} ({len(V)/len(a):.1f}/S1) -> capped {len(P):,} ({len(P)/len(a):.1f}/S1)")
    del V, s; gc.collect()
P = pd.concat(out, ignore_index=True)
P.to_parquet(W + f"{sp}_pairs_r1.parquet"); log(f"saved {len(P):,} pairs")
if sp == "train":
    g = P[[(x, y) in gold for x, y in zip(P.s1_id, P.cand_id)]]
    ev = ~P.s1_id.isin(rank_s1)
    ctry = dict(zip(s1.entity_id, s1.ctry))
    gp = pd.DataFrame(list(gold), columns=["s1_id", "cand_id"]); gp = gp[~gp.s1_id.isin(rank_s1)]; gp["c"] = gp.s1_id.map(ctry)
    g = g[~g.s1_id.isin(rank_s1)].copy(); g["c"] = g.s1_id.map(ctry)
    gsz = gp.groupby("s1_id").size(); hsz = g.groupby("s1_id").size()
    for c in ctrys:
        ids = gsz.index[gsz.index.map(ctry) == c]
        full = (hsz.reindex(ids).fillna(0).to_numpy() == gsz[ids].to_numpy()).mean()
        log(f"{c} (non-ranker S1): pair recall {(g.c == c).sum() / (gp.c == c).sum():.4f}, full coverage {full:.4f}, cands/S1 {(P[ev].s1_id.map(ctry) == c).sum() / ((s1.ctry == c) & ~s1.entity_id.isin(rank_s1)).sum():.1f}")
