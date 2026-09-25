"""OOF experiment on a sample of S1 groups: LightGBM -> isotonic -> expected-F decode -> score."""
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, oof_predict, calibrate, evaluate, loco
W = "/home/24b4518/ml-projects/work/cache/"
tag = sys.argv[1]; frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
drop = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else set()
t = time.time()
s1 = pd.read_parquet(W + "v1/train_s1p.parquet", columns=["entity_id", "country"])
gt = pd.read_parquet(W + "gt.parquet")
g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
country_of = dict(zip(s1.entity_id, s1.country))
rank_s1 = set(pd.read_parquet(W + "v1/ranker_s1.parquet").s1_id)   # ranker saw these labels -> exclude
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=frac / 0.7, random_state=0)
import pyarrow.dataset as pds, pyarrow as pa
ftag = os.environ.get("FTAG", tag)
F = pds.dataset(W + f"v1/train_feats_{ftag}").to_table(filter=pds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas() \
    if os.path.isdir(W + f"v1/train_feats_{ftag}") else pd.read_parquet(W + f"v1/train_feats_{ftag}.parquet")
F = F[F.s1_id.isin(set(ids))].reset_index(drop=True)
F = add_labels(F, g)
feats = [c for c in feature_cols(F) if c not in drop]
print(f"{len(F):,} pairs, {len(ids):,} S1, pos rate {F.y.mean():.3f}, {len(feats)} feats", flush=True)
raw, iters = oof_predict(F, feats, params=json.loads(os.environ.get("LGB", "{}")))
iso = calibrate(raw, F.y)
p = iso.predict(raw)
from sklearn.metrics import roc_auc_score, log_loss
print(f"AUC {roc_auc_score(F.y, raw):.5f}  logloss {log_loss(F.y, np.clip(p,1e-6,1-1e-6)):.5f}", flush=True)
f, br, pred = evaluate(F, p, ids.tolist(), g, country_of)
print(f"OOF macro F0.5 = {f:.5f}", flush=True); print(br.round(4).to_string(), flush=True)
# threshold baseline for comparison (decoder must beat it)
from er.io_metric import macro_fbeta
from er.decode import enforce_one_owner
Q = enforce_one_owner(pd.DataFrame({"s1_id": F.s1_id, "cand_id": F.cand_id, "p": p}))
gs = {s: g.get(s, set()) for s in ids}
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    q = Q[Q.p >= th]
    pm = q.groupby("s1_id").cand_id.apply(set).to_dict()
    print(f"  threshold {th}: F0.5 {macro_fbeta(pm, gs):.5f}", flush=True)
pd.DataFrame({"s1_id": F.s1_id, "cand_id": F.cand_id, "y": F.y, "raw": raw, "p": p}).to_parquet(W + f"v1/oof_{tag}.parquet")
ids.to_frame("s1_id").to_parquet(W + f"v1/oof_{tag}_ids.parquet")
if "--loco" in sys.argv:
    loco(F, feats, country_of, g, ids.tolist(), rounds=int(np.mean(iters)))
print("done", round(time.time() - t), "s")
