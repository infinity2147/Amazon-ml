"""Simulate test-like decoy density on TRAIN: remove a random 19% of S1 businesses from the reference.
Their true S2/S3 records then have no owner, i.e. become decoys, exactly how the generator makes decoys
(copies of businesses absent from S1). Train decoy share 26% -> ~40% like test (0.74*(1-0.19)=0.60 matched).
Reuses cached per-view blocking (pairs of removed S1s are dropped), re-applies the stage-1 ranker cap and
the name+house key view. Writes train_pairs_sim2.parquet and removed_s1.parquet."""
import sys, gc, time, numpy as np, pandas as pd, lightgbm as lgb, pyarrow.dataset as pds
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import rank_features, group_rank
W = "/home/24b4518/ml-projects/work/cache/v1/"; VIEWS = ["v_word", "v_name5", "v_addr5"]; N1, NC, FRAC = 20, 1, 0.19
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "ctry", "name_n", "house"])
cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "ctry", "name_n", "nums"])
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
rng = np.random.default_rng(7)
removed = s1.entity_id[rng.random(len(s1)) < FRAC]
removed.to_frame("s1_id").to_parquet(W + "removed_s1.parquet"); rem = set(removed)
log(f"removed {len(rem):,} of {len(s1):,} S1 ({len(rem)/len(s1):.3f})")
ranker = lgb.Booster(model_file=W + "ranker.txt")
out = []
for c in sorted(s1.ctry.unique()):
    a = s1.entity_id[s1.ctry == c].to_numpy(); b = cp.entity_id[cp.ctry == c].to_numpy(); nb = len(b)
    V = pd.concat([pd.read_parquet(W + f"blk_train_b1_{c}_{v}.parquet").set_index("key")[v] for v in VIEWS], axis=1).fillna(0).astype(np.float32)
    key = V.index.to_numpy(); V = V.reset_index(drop=True)
    V["si"] = (key // nb).astype(np.int32); V["cj"] = (key % nb).astype(np.int32)
    keep_s1 = ~pd.Index(a).isin(rem)
    V = V[keep_s1[V.si.to_numpy()]].reset_index(drop=True)
    s = ranker.predict(rank_features(V, VIEWS, "si", "cj"), num_threads=16).astype(np.float32)
    r1 = group_rank(V.si.to_numpy(), s)[0]; r2 = group_rank(V.cj.to_numpy(), s)[0]
    k = (r1 <= N1) | (r2 <= NC)
    P = V[k].reset_index(drop=True); P["rk"] = s[k]
    P.insert(0, "cand_id", b[P.pop("cj").to_numpy()]); P.insert(0, "s1_id", a[P.pop("si").to_numpy()])
    # name + house key view on the reduced reference
    aa = s1[(s1.ctry == c) & ~s1.entity_id.isin(rem) & (s1.house != "") & (s1.name_n != "")].assign(k=lambda d: d.name_n + "|" + d.house)
    bb = cp[cp.ctry == c].assign(h=lambda d: d.nums.str.split().str[0].fillna(""))
    bb = bb[(bb.h != "") & (bb.name_n != "")].assign(k=lambda d: d.name_n + "|" + d.h)
    bs = aa.groupby("k").size(); aa = aa[aa.k.map(bs) <= 30]
    K = aa[["entity_id", "k"]].rename(columns={"entity_id": "s1_id"}).merge(bb[["entity_id", "k"]].rename(columns={"entity_id": "cand_id"}), on="k")[["s1_id", "cand_id"]].drop_duplicates()
    K["v_key"] = np.float32(1)
    P = P.merge(K, on=["s1_id", "cand_id"], how="outer")
    for col in VIEWS + ["rk", "v_key"]:
        P[col] = P[col].fillna(0).astype(np.float32)
    out.append(P); log(f"{c}: {len(P):,} pairs ({len(P)/keep_s1.sum():.1f}/S1)")
    del V; gc.collect()
P = pd.concat(out, ignore_index=True); P.to_parquet(W + "train_pairs_sim2.parquet"); log(f"saved {len(P):,}")
# how test-like is it? unowned share among S2/S3 records, and near-twin candidates per S1 (compare: train 0.81/0.50, test 1.44/0.83 US/India)
gt = pd.read_parquet("/home/24b4518/ml-projects/work/cache/gt.parquet")
owned = {x for s_, l in zip(gt.source1_entity_id, gt.matched_entity_ids) if s_ not in rem for x in l.split(",") if x}
log(f"S2/S3 records with an owner: {len(owned)/len(cp):.3f}  (train orig 0.74, test est. ~0.60)")
