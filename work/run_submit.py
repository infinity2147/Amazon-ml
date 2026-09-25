"""Memory-lean: train on the stage-2 sample (parquet filter), calibrate with the OOF isotonic fit,
score test in streamed batches, decode, write, validate."""
import sys, os, time, gc, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds, pyarrow as pa
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, train_lgb, calibrate
from er.decode import decode_all
from er.io_metric import write_idlist_tsv, check_outputs
W = "/home/24b4518/ml-projects/work/cache/"; OUT = "/home/24b4518/ml-projects/output/"
tag, frac, rounds = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]); ftag = sys.argv[4] if len(sys.argv) > 4 else tag
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
s1 = pd.read_parquet(W + "v1/train_s1p.parquet", columns=["entity_id"])
gt = pd.read_parquet(W + "gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}; del gt
rank_s1 = set(pd.read_parquet(W + "v1/ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=frac / 0.7, random_state=0); del s1
F = ds.dataset(W + f"v1/train_feats_{ftag}.parquet").to_table(filter=ds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas()
F = add_labels(F, g); feats = feature_cols(F); del g; gc.collect()
oof = pd.read_parquet(W + f"v1/oof_{tag}.parquet", columns=["raw", "y"]); iso = calibrate(oof.raw.to_numpy(), oof.y.to_numpy()); del oof
log(f"training on {len(F):,} pairs, {len(feats)} feats, {rounds} rounds")
X = F[feats].to_numpy(np.float32); y = F.y.to_numpy(); del F; gc.collect()
m = train_lgb(X, y, rounds=rounds, params=dict(num_threads=24)); m.save_model(W + f"v1/model_{tag}.txt"); del X, y; gc.collect()
log("trained")
pf = pq.ParquetFile(W + f"v1/test_feats_{ftag}.parquet")
outs = []
for b in pf.iter_batches(batch_size=2_000_000, columns=["s1_id", "cand_id"] + feats):
    d = b.to_pandas()
    outs.append(pd.DataFrame({"s1_id": d.s1_id, "cand_id": d.cand_id,
                              "p": iso.predict(m.predict(d[feats].to_numpy(np.float32), num_threads=24))}))
Te = pd.concat(outs, ignore_index=True); del outs; log(f"predicted test {len(Te):,}")
Te.to_parquet(W + f"v1/test_pred_{tag}.parquet")
ts1 = pd.read_parquet(W + "v1/test_s1p.parquet", columns=["entity_id", "country"]); tcp = pd.read_parquet(W + "v1/test_cp.parquet", columns=["entity_id"])
test_ids = ts1.entity_id.tolist()
pred = decode_all(Te, test_ids); log("decoded")
os.makedirs(OUT, exist_ok=True)
mp_, cp_ = OUT + "matching_results.tsv", OUT + "candidate_pairs.tsv"
write_idlist_tsv(pred, test_ids, mp_)
write_idlist_tsv(Te.groupby("s1_id").cand_id.apply(list).to_dict(), test_ids, cp_, col="candidate_entity_ids")
log(f"check_outputs: {check_outputs(mp_, cp_, test_ids, tcp.entity_id)}")
c = dict(zip(ts1.entity_id, ts1.country))
mon = pd.DataFrame({"country": [c[s] for s in test_ids], "n": [len(pred[s]) for s in test_ids]})
print(mon.groupby("country").n.agg(empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string())
Te["country"] = Te.s1_id.map(c); best = Te.groupby("s1_id").agg(country=("country", "first"), pmax=("p", "max"))
print("max-p per S1 by country:\n", pd.crosstab(best.country, pd.cut(best.pmax, [0, .1, .3, .5, .7, .9, .99, 1]), normalize="index").round(3).to_string(), flush=True)
