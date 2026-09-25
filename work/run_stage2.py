"""Stage-2 OOF experiment: stage-1 OOF p -> sibling features -> LightGBM OOF -> isotonic -> decode -> score."""
import sys, os, time, json, numpy as np, pandas as pd, pyarrow.dataset as pds, pyarrow as pa
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import feature_cols, oof_predict, calibrate, evaluate
from er.stage2 import sibling_features
W = "/home/24b4518/ml-projects/work/cache/"
tag = sys.argv[1]   # stage-1 tag (oof_{tag}.parquet and train_feats_{tag}/)
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
O = pd.read_parquet(W + f"v1/oof_{tag}.parquet").rename(columns={"p": "p1"})
ids = O.s1_id.unique()
src = W + f"v1/train_feats_{tag}" if os.path.isdir(W + f"v1/train_feats_{tag}") else W + f"v1/train_feats_{tag}.parquet"
F = pds.dataset(src).to_table(filter=pds.field("s1_id").isin(pa.array(ids))).to_pandas()
F = F.merge(O[["s1_id", "cand_id", "y", "p1"]], on=["s1_id", "cand_id"], how="inner"); log(f"{len(F):,} pairs")
cp = pd.read_parquet(W + "v1/train_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
cp = cp[cp.entity_id.isin(set(F.cand_id))]
F = sibling_features(F, cp); log("sibling features")
feats = feature_cols(F) + ["p1"]
feats = [f for f in feats if f != "p1"] + ["p1"]
raw, iters = oof_predict(F, feats, params=json.loads(os.environ.get("LGB", "{}")))
iso = calibrate(raw, F.y); p = iso.predict(raw)
s1 = pd.read_parquet(W + "v1/train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
# evaluate on ALL sample S1 ids (incl. those with no candidates) exactly like stage 1
s1_ids = pd.read_parquet(W + f"v1/oof_{tag}_ids.parquet").s1_id.tolist() if os.path.exists(W + f"v1/oof_{tag}_ids.parquet") else list(ids)
f, br, _ = evaluate(F, p, s1_ids, g, country_of)
log(f"STAGE-2 OOF macro F0.5 = {f:.5f}"); print(br.round(4).to_string())
imp = pd.Series(dict(zip(feats, __import__("lightgbm").train({"objective": "binary", "verbose": -1, "num_threads": 24}, __import__("lightgbm").Dataset(F[feats].iloc[:2_000_000], F.y.iloc[:2_000_000]), 200).feature_importance("gain")))).sort_values(ascending=False)
print(imp.head(25).round(0).to_string())
pd.DataFrame({"s1_id": F.s1_id, "cand_id": F.cand_id, "y": F.y, "raw": raw, "p": p}).to_parquet(W + f"v1/oof_{tag}_s2.parquet")
