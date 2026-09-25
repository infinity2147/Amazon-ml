"""Two-stage train + submit.
stage 1: 5 GroupKFold LightGBM models on the stage-2 sample -> OOF p1 (train) and fold-average p1 (test)
stage 2: sibling features from p1 -> 5 GroupKFold models -> OOF p2 (isotonic fit) and fold-average p2 (test)
decode (one-owner + expected F0.5) -> write -> validate -> monitors."""
import sys, os, gc, time, json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, train_lgb, calibrate, evaluate, PARAMS
from er.stage2 import sibling_features
from er.decode import decode_all
from er.io_metric import write_idlist_tsv, check_outputs

W = "/home/24b4518/ml-projects/work/cache/v1/"; OUT = "/home/24b4518/ml-projects/output/"
tag = sys.argv[1]; frac = float(sys.argv[2]); name = sys.argv[3]
params = {"num_threads": 24, **json.loads(os.environ.get("LGB", "{}"))}
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)

s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}; del gt
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
# simulation mode: S1 businesses removed from the reference -> their records are decoys; never sampled
removed = set(pd.read_parquet(W + "removed_s1.parquet").s1_id) if os.environ.get("SIM") else set()
for r in removed:
    g.pop(r, None)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=frac / 0.7, random_state=0)
ids = ids[~ids.isin(removed)]
F = pds.dataset(W + f"train_feats_{tag}").to_table(filter=pds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas()
F = add_labels(F, g); feats1 = feature_cols(F)
log(f"train sample {len(F):,} pairs / {len(ids):,} S1, {len(feats1)} feats")
test_tag = os.environ.get("TEST_TAG", tag)
Te = pd.read_parquet(W + f"test_feats_{test_tag}"); log(f"test {len(Te):,} pairs (features {test_tag})")
folds = list(GroupKFold(5).split(F, groups=F.s1_id))


def run_stage(F, Te, feats, label):
    oof = np.zeros(len(F), np.float32); te = np.zeros(len(Te), np.float64); its = []
    for k, (a, b) in enumerate(folds):
        m = train_lgb(F[feats].iloc[a], F.y.iloc[a], F[feats].iloc[b], F.y.iloc[b], params=params)
        oof[b] = m.predict(F[feats].iloc[b], num_iteration=m.best_iteration)
        for lo in range(0, len(Te), 2_000_000):
            te[lo:lo + 2_000_000] += m.predict(Te[feats].iloc[lo:lo + 2_000_000], num_iteration=m.best_iteration) / len(folds)
        its.append(m.best_iteration); m.save_model(W + f"model_{name}_{label}_f{k}.txt")
        log(f"{label} fold {k}: best_iter {m.best_iteration}")
    iso = calibrate(oof, F.y)
    return oof, te.astype(np.float32), iso


oof1, te1, iso1 = run_stage(F, Te, feats1, "s1")
F["p1"] = iso1.predict(oof1); Te["p1"] = iso1.predict(te1)
f1, _, _ = evaluate(F, F.p1.to_numpy(), ids.tolist(), g, country_of); log(f"stage-1 OOF F0.5 {f1:.5f}")

cp_tr = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
F = sibling_features(F, cp_tr[cp_tr.entity_id.isin(set(F.cand_id))]); del cp_tr; gc.collect()
cp_te = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
Te = sibling_features(Te, cp_te[cp_te.entity_id.isin(set(Te.cand_id))]); del cp_te; gc.collect()
feats2 = feature_cols(F); feats2 = [c for c in feats2 if c != "p1"] + ["p1"]
log(f"stage-2 feats {len(feats2)}")
oof2, te2, iso2 = run_stage(F, Te, feats2, "s2")
p2 = iso2.predict(oof2)
f2, br, _ = evaluate(F, p2, ids.tolist(), g, country_of); log(f"stage-2 OOF F0.5 {f2:.5f}"); print(br.round(4).to_string(), flush=True)
pd.DataFrame({"s1_id": F.s1_id, "cand_id": F.cand_id, "y": F.y, "p1": F.p1, "raw2": oof2, "p": p2}).to_parquet(W + f"oof_{name}.parquet")

Te["p"] = iso2.predict(te2)
Te[["s1_id", "cand_id", "p1", "p"]].to_parquet(W + f"test_pred_{name}.parquet")
ts1 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"]); tcp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id"])
test_ids = ts1.entity_id.tolist()
pred = decode_all(Te[["s1_id", "cand_id", "p"]], test_ids); log("decoded")
os.makedirs(OUT, exist_ok=True)
mp_, cp_ = OUT + "matching_results.tsv", OUT + "candidate_pairs.tsv"
write_idlist_tsv(pred, test_ids, mp_)
write_idlist_tsv(Te.groupby("s1_id").cand_id.apply(list).to_dict(), test_ids, cp_, col="candidate_entity_ids")
log(f"check_outputs: {check_outputs(mp_, cp_, test_ids, tcp.entity_id)}")
c = dict(zip(ts1.entity_id, ts1.country))
mon = pd.DataFrame({"country": [c[s] for s in test_ids], "n": [len(pred[s]) for s in test_ids]})
print(mon.groupby("country").n.agg(empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string(), flush=True)
