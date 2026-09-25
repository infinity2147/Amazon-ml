"""E0: score EXISTING two-stage fold models (trained at train decoy density) on the Test-Like Validation.
Same 551,789 validation S1s as sub3; each S1 is scored by the fold model that did not train on it
(folds rebuilt exactly as run_full.py built them). Stage-1 isotonic is refit on the original OOF
(sanity check: must reproduce sub3's logged stage-1 OOF 0.97592); stage-2 isotonic is refit on the
saved OOF (raw2, y). Usage: score_on_tlv.py <model_name> <orig_feat_tag> <tlv_feat_tag>"""
import sys, time, gc, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, calibrate, evaluate
from er.stage2 import sibling_features
W = __import__("os").environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/")
name, tag0, tagT = sys.argv[1:4]
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}; del gt
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0)
load = lambda tag: add_labels(pds.dataset(W + f"train_feats_{tag}").to_table(filter=pds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas(), g)
F0 = load(tag0)
fold_of = {}
for k, (a, b) in enumerate(GroupKFold(5).split(F0, groups=F0.s1_id)):
    fold_of.update(dict.fromkeys(F0.s1_id.iloc[b].unique(), k))
m1 = [lgb.Booster(model_file=W + f"model_{name}_s1_f{k}.txt") for k in range(5)]
m2 = [lgb.Booster(model_file=W + f"model_{name}_s2_f{k}.txt") for k in range(5)]
f1 = m1[0].feature_name(); f2 = m2[0].feature_name()
def predict(models, F, feats):
    fo = F.s1_id.map(fold_of).to_numpy(); out = np.zeros(len(F), np.float32)
    for k, m in enumerate(models):
        idx = np.flatnonzero(fo == k); out[idx] = m.predict(F[feats].iloc[idx], num_threads=24)
    return out
raw1 = predict(m1, F0, f1); iso1 = calibrate(raw1, F0.y)
s, _, _ = evaluate(F0, iso1.predict(raw1), ids.tolist(), g, country_of); log(f"sanity: stage-1 OOF on original validation {s:.5f} (sub3 logged 0.97592)")
O = pd.read_parquet(W + f"oof_{name}.parquet", columns=["raw2", "y"]); iso2 = calibrate(O.raw2.to_numpy(), O.y.to_numpy()); del O, F0; gc.collect()
FT = load(tagT); FT["base_id"] = FT.cand_id.str.replace("~c", "", regex=False)
log(f"TLV sample: {len(FT):,} pairs, {FT.cand_id.str.endswith('~c').mean():.3f} clone pairs")
FT["p1"] = iso1.predict(predict(m1, FT, f1))
s, br, _ = evaluate(FT, FT.p1.to_numpy(), ids.tolist(), g, country_of); log(f"TLV stage-1: {s:.5f}")
cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
FT = sibling_features(FT, cp[cp.entity_id.isin(set(FT.base_id))], base_col="base_id"); del cp; gc.collect()
p = iso2.predict(predict(m2, FT, f2))
s, br, _ = evaluate(FT, p, ids.tolist(), g, country_of); log(f"TLV two-stage: {s:.5f}")
print(br.round(4).to_string(), flush=True)
pd.DataFrame({"s1_id": FT.s1_id, "cand_id": FT.cand_id, "y": FT.y, "p": p}).to_parquet(W + f"tlvscore_{name}.parquet")
