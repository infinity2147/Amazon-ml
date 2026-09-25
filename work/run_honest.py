"""Honest test-density experiment: train on ORIGINAL train pairs (no clones), score on the TLV (clones).

Why: a model trained on the TLV sees exact-duplicate clones of decoys, and can learn 'duplicate => decoy',
which cannot transfer to the real test (real second-batch decoys are separate records). Here no model ever sees a
clone. The extra test decoys are represented by a weight on decoy pairs instead:
  a decoy pair (record owned by NO S1) stands for (1+q) decoys at test density, q = 0.891 (US), 0.941 (India)
  (make_tlv.py: test decoys per S1 / train decoys per S1 - 1).
Variants (stage 1 only), all scored on the same TLV rows of the same 551,789 validation S1s, K folds by S1:
  A  train unweighted, isotonic unweighted                      (= what sub2/sub3 did)
  B  train unweighted, isotonic fit with decoy weight 1+q       (post-hoc prior correction)
  C  train with decoy weight 1+q, isotonic with decoy weight    (weighted training + calibration)
The isotonic map of fold k is fit on fold-k S1s' ORIGINAL rows (which model k never saw), applied to fold-k TLV rows.
usage: run_honest.py <orig_feat_tag> <tlv_feat_tag> <name>     env: CACHE, K=3, LR=0.1, THREADS=24"""
import sys, os, gc, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, train_lgb, evaluate

W = os.environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v2/")
tag0, tagT, name = sys.argv[1:4]
E = os.environ.get
K, LR, TH = int(E("K", 3)), float(E("LR", 0.1)), int(E("THREADS", 24))
Q = {"US": 0.891, "India": 0.941}
params = {"num_threads": TH, "learning_rate": LR}
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
    base = F.cand_id.str.replace("~c", "", regex=False)
    F["decoy"] = ~base.isin(owned)                         # record owned by no S1 (clones are decoys too)
    F["wq"] = 1.0 + F.s1_id.map(country_of).map(Q).fillna(0.9).to_numpy()
    return F


F0 = load(tag0); feats = feature_cols(F0)
feats = [c for c in feats if c not in ("fold", "decoy", "wq")]
log(f"orig: {len(F0):,} pairs, {len(feats)} feats; decoy share of negatives {F0[F0.y==0].decoy.mean():.3f}")
FT = load(tagT); log(f"TLV: {len(FT):,} pairs, clone share {FT.cand_id.str.endswith('~c').mean():.3f}")
w_dec0 = np.where(F0.decoy.to_numpy() & (F0.y.to_numpy() == 0), F0.wq.to_numpy(), 1.0)   # 1+q on decoy pairs

pA, pB, pC = (np.zeros(len(FT), np.float32) for _ in range(3)); p0A = np.zeros(len(F0), np.float32)
iso = lambda raw, y, w=None: IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(raw, y, sample_weight=w)
for k in range(K):
    tr, va = np.flatnonzero(F0.fold.to_numpy() != k), np.flatnonzero(F0.fold.to_numpy() == k)
    tt = np.flatnonzero(FT.fold.to_numpy() == k)
    for tagv, weighted in (("U", False), ("W", True)):
        X, y = F0[feats].iloc[tr], F0.y.iloc[tr]
        import lightgbm as lgb
        dtr = lgb.Dataset(X, y, weight=w_dec0[tr] if weighted else None, free_raw_data=True)
        dv = lgb.Dataset(F0[feats].iloc[va], F0.y.iloc[va], weight=w_dec0[va] if weighted else None, reference=dtr)
        from er.model import PARAMS
        m = lgb.train({**PARAMS, **params}, dtr, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        rv = m.predict(F0[feats].iloc[va], num_iteration=m.best_iteration, num_threads=TH)
        rt = m.predict(FT[feats].iloc[tt], num_iteration=m.best_iteration, num_threads=TH)
        yv = F0.y.iloc[va].to_numpy()
        if not weighted:
            pA[tt] = iso(rv, yv).predict(rt); p0A[va] = iso(rv, yv).predict(rv)
            pB[tt] = iso(rv, yv, w_dec0[va]).predict(rt)
        else:
            pC[tt] = iso(rv, yv, w_dec0[va]).predict(rt)
        log(f"fold {k} model {tagv}: best_iter {m.best_iteration}")
        del m, dtr, dv; gc.collect()

s0, _, _ = evaluate(F0, p0A, ids.tolist(), g, country_of); log(f"sanity, train density, variant A: {s0:.5f} (sub3-style stage-1 OOF was ~0.976)")
res = {}
for nm, p in (("A  unweighted train + unweighted calibration", pA), ("B  unweighted train + decoy-weighted calibration", pB), ("C  decoy-weighted train + weighted calibration", pC)):
    s, br, _ = evaluate(FT, p, ids.tolist(), g, country_of); res[nm] = s
    log(f"TLV stage-1 {nm}: {s:.5f}")
    print(br.round(4)[["n", "mean_f", "lost_share"]].to_string(), flush=True)
pd.DataFrame({"s1_id": FT.s1_id, "cand_id": FT.cand_id, "y": FT.y, "pA": pA, "pB": pB, "pC": pC}).to_parquet(W + f"honest_{name}.parquet")
