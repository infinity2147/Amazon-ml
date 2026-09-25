"""Predict TEST with the saved fold models of a run_tlv.py experiment, then decode, write
submissions/<name>/ and run check_outputs. Isotonic maps are refit on the run's OOF (stage 1 from
raw1, recomputed from the fold models if the run predates saving it; stage 2 from raw2).
usage: predict_test.py <name> <train_feat_tag> <test_feat_tag> [K]   env: TIE, POUT_JSON, CACHE"""
import sys, os, gc, json, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds, lightgbm as lgb
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import calibrate
from er.stage2 import sibling_features
from er.decode import decode_all
from er.io_metric import write_idlist_tsv, check_outputs
W = os.environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/"); ROOT = "/home/24b4518/ml-projects/"
name, tag, ttag = sys.argv[1:4]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
TIE = os.environ.get("TIE", "0") == "1"; pout = json.loads(os.environ.get("POUT_JSON", "null"))
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
M1 = [lgb.Booster(model_file=W + f"model_{name}_s1_f{k}.txt") for k in range(K)]
M2 = [lgb.Booster(model_file=W + f"model_{name}_s2_f{k}.txt") for k in range(K)]
f1, f2 = M1[0].feature_name(), M2[0].feature_name()
O = pd.read_parquet(W + f"oof_{name}.parquet")
if "raw1" not in O.columns:   # recompute stage-1 OOF raw scores: fold k rows by fold-k model
    F = pds.dataset(W + f"train_feats_{tag}").to_table(columns=["s1_id", "cand_id"] + f1, filter=pds.field("s1_id").isin(pa.array(O.s1_id.unique().tolist()))).to_pandas()
    F = O[["s1_id", "cand_id", "fold"]].merge(F, on=["s1_id", "cand_id"], how="left")
    O["raw1"] = np.nan
    for k in range(K):
        b = np.flatnonzero(F.fold.to_numpy() == k); O.loc[b, "raw1"] = M1[k].predict(F[f1].iloc[b], num_threads=24)
    del F; gc.collect(); log("recomputed stage-1 OOF raw scores")
iso1 = calibrate(O.raw1.to_numpy(), O.y.to_numpy()); iso2 = calibrate(O.raw2.to_numpy(), O.y.to_numpy()); del O
def avg(models, X):
    out = np.zeros(len(X))
    for m in models:
        for lo in range(0, len(X), 2_000_000):
            out[lo:lo + 2_000_000] += m.predict(X.iloc[lo:lo + 2_000_000], num_threads=24) / len(models)
    return out.astype(np.float32)
Te = pd.read_parquet(W + f"test_feats_{ttag}"); log(f"test {len(Te):,} pairs")
Te["p1"] = iso1.predict(avg(M1, Te[f1])).astype(np.float32); log("stage 1")
cp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
Te = sibling_features(Te, cp[cp.entity_id.isin(set(Te.cand_id))]); del cp; gc.collect()
Te["raw"] = avg(M2, Te[f2]); Te["p"] = iso2.predict(Te.raw).astype(np.float32); log("stage 2")
Te[["s1_id", "cand_id", "p1", "p", "raw"]].to_parquet(W + f"test_pred_{name}.parquet")
ts1 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"]); ids = ts1.entity_id.tolist(); tc = dict(zip(ids, ts1.country))
po = {s: pout.get(tc[s], max(pout.values())) for s in ids} if pout else 0.0
pred = decode_all(Te[["s1_id", "cand_id", "p", "raw"]], ids, p_outside=po, tie_col="raw" if TIE else None); log("decoded")
out = ROOT + f"submissions/{name}/"; os.makedirs(out, exist_ok=True)
write_idlist_tsv(pred, ids, out + "matching_results.tsv")
write_idlist_tsv(Te.groupby("s1_id").cand_id.apply(list).to_dict(), ids, out + "candidate_pairs.tsv", col="candidate_entity_ids")
tcp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id"])
log(f"check_outputs: {check_outputs(out + 'matching_results.tsv', out + 'candidate_pairs.tsv', ids, tcp.entity_id)}")
mon = pd.DataFrame({"country": [tc[s] for s in ids], "n": [len(pred[s]) for s in ids]})
print(mon.groupby("country").n.agg(empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string(), flush=True)
