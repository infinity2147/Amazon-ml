"""Candidate submission built from the HONEST test-density recipe (H1 variant C), 5 folds.

Model: stage-1 LightGBM only, on the house-number-fixed data (cache/v2, features r3_f2 with the name+house key view).
Training rows: original train pairs of the 551,789 validation S1s (no clones anywhere in training).
Decoy weighting: a pair whose record is owned by NO S1 stands for (1+q) decoys at test density
  (q = 0.891 US, 0.941 India, from make_tlv.py); its training weight is 1+q.
Calibration: fold k's isotonic map is fit on fold-k S1s' ORIGINAL rows (unseen by model k) with the same weights.
Test: p = mean over folds of iso_k(model_k(x)); each calibrated model is applied with its OWN map.
Yardstick: fold-k models score the TLV rows (clones) of fold-k S1s -> out-of-fold F0.5 at test-like density.
Then decode (one-owner + expected F0.5), write both TSVs, check.
usage: run_cand.py <name>      env: K=5 LR=0.1 THREADS=24"""
import sys, os, gc, time, pickle, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds, pyarrow.parquet as pq, glob
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, evaluate, PARAMS
from er.decode import decode_all
from er.io_metric import write_idlist_tsv, check_outputs

W = "/home/24b4518/ml-projects/work/cache/v2/"; ROOT = "/home/24b4518/ml-projects/"
NAME = sys.argv[1]
E = os.environ.get
K, LR, TH = int(E("K", 5)), float(E("LR", 0.1)), int(E("THREADS", 24))
Q = {"US": 0.891, "India": 0.941}
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)

s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
owned = pd.Index([x for l in gt.matched_entity_ids for x in l.split(",") if x]); del gt
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0)
perm = np.random.default_rng(0).permutation(len(ids)); fold_of = dict(zip(ids.to_numpy()[perm], np.arange(len(ids)) % K))


def load(tag):
    F = pds.dataset(W + f"train_feats_{tag}").to_table(filter=pds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas()
    F = add_labels(F, g); F["fold"] = F.s1_id.map(fold_of).to_numpy(np.int8)
    F["decoy"] = ~F.cand_id.str.replace("~c", "", regex=False).isin(owned)
    F["wq"] = 1.0 + F.s1_id.map(country_of).map(Q).fillna(0.9).to_numpy()
    return F


F0 = load("r3_f2"); feats = [c for c in feature_cols(F0) if c not in ("fold", "decoy", "wq")]
FT = load("tlv3_f2")
log(f"train {len(F0):,} pairs, TLV {len(FT):,} pairs, {len(feats)} features, K={K}")
w0 = np.where(F0.decoy.to_numpy() & (F0.y.to_numpy() == 0), F0.wq.to_numpy(), 1.0)
pT = np.zeros(len(FT), np.float32); isos = []
for k in range(K):
    tr, va = np.flatnonzero(F0.fold.to_numpy() != k), np.flatnonzero(F0.fold.to_numpy() == k)
    tt = np.flatnonzero(FT.fold.to_numpy() == k)
    dtr = lgb.Dataset(F0[feats].iloc[tr], F0.y.iloc[tr], weight=w0[tr], free_raw_data=True)
    dv = lgb.Dataset(F0[feats].iloc[va], F0.y.iloc[va], weight=w0[va], reference=dtr)
    m = lgb.train({**PARAMS, "num_threads": TH, "learning_rate": LR}, dtr, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    rv = m.predict(F0[feats].iloc[va], num_iteration=m.best_iteration, num_threads=TH)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(rv, F0.y.iloc[va].to_numpy(), sample_weight=w0[va])
    pT[tt] = iso.predict(m.predict(FT[feats].iloc[tt], num_iteration=m.best_iteration, num_threads=TH))
    m.save_model(W + f"model_{NAME}_f{k}.txt"); isos.append(iso)
    log(f"fold {k}: best_iter {m.best_iteration}")
    del m, dtr, dv; gc.collect()
pickle.dump({"isos": isos, "feats": feats}, open(W + f"iso_{NAME}.pkl", "wb"))
score, br, _ = evaluate(FT, pT, ids.tolist(), g, country_of)
log(f"YARDSTICK out-of-fold F0.5 on the test-like validation: {score:.5f}   (submission-2-style model: 0.96876; success >= 0.970)")
print(br.round(4)[["n", "mean_f", "lost_share"]].to_string(), flush=True)
del F0, FT; gc.collect()

# ---- test predictions: mean over folds of iso_k(model_k(x)) ----
models = [lgb.Booster(model_file=W + f"model_{NAME}_f{k}.txt") for k in range(K)]
parts = []
for f in sorted(glob.glob(W + "test_feats_r3_f2/*.parquet")):
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=1_000_000, columns=["s1_id", "cand_id"] + feats):
        d = b.to_pandas(); X = d[feats].to_numpy(np.float32); p = np.zeros(len(d), np.float64)
        for m, iso in zip(models, isos):
            p += iso.predict(m.predict(X, num_threads=TH)) / K
        parts.append(pd.DataFrame({"s1_id": d.s1_id.values, "cand_id": d.cand_id.values, "p": p.astype(np.float32)}))
    log(f"predicted {f.split('/')[-1]}")
Te = pd.concat(parts, ignore_index=True); del parts; Te.to_parquet(W + f"test_pred_{NAME}.parquet"); log(f"test pairs {len(Te):,}")
ts1 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"]); test_ids = ts1.entity_id.tolist(); tc = dict(zip(ts1.entity_id, ts1.country))
pred = decode_all(Te, test_ids); log("decoded")
out = ROOT + f"submissions/{NAME}/"; os.makedirs(out, exist_ok=True)
write_idlist_tsv(pred, test_ids, out + "matching_results.tsv")
write_idlist_tsv(Te.groupby("s1_id").cand_id.apply(list).to_dict(), test_ids, out + "candidate_pairs.tsv", col="candidate_entity_ids")
tcp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id"])
log(f"check_outputs: {check_outputs(out + 'matching_results.tsv', out + 'candidate_pairs.tsv', test_ids, tcp.entity_id)}")
mon = pd.DataFrame({"country": [tc[s] for s in test_ids], "n": [len(pred[s]) for s in test_ids]})
print(mon.groupby("country").n.agg(empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string(), flush=True)
log("done")
