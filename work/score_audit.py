"""Score EXISTING submission models on the untouched audit S1s, exactly as test is scored (every audit S1 is
out-of-sample for every fold model, so each prediction is the fold average), on the audit S1s' test-like rows.

  sub3        two-stage, trained at train density on the 551,789 validation S1s (v1 data, features r2_f2).
              It is submission 2 + the name/house key view; submission 2 itself scored 0.965514 on the
              leaderboard, and sub3 scored within 0.001 of it on the train-density validation.
              Isotonic maps rebuilt from its own OOF, as run_full.py did for test.
  sub4_honest stage-1 only, honest decoy-weighted training (v2 data, features r3_f2); mean of iso_k(model_k(x)).
Decoding for all: one-owner (exactly one owner, tie broken by rk) + expected F0.5, audit population only.
usage: score_audit.py sub3|sub4_honest"""
import sys, gc, pickle, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, calibrate
from er.stage2 import sibling_features
from er.decode import decode_all
from er.io_metric import macro_fbeta, score_breakdown
V1, V2 = "/home/24b4518/ml-projects/work/cache/v1/", "/home/24b4518/ml-projects/work/cache/v2/"
which = sys.argv[1]
s1 = pd.read_parquet(V2 + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(V2 + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}; del gt
audit = sorted(pd.read_parquet(V2 + "audit_s1.parquet").s1_id)
gs = {s: g.get(s, set()) for s in audit}


def report(F, p, label):
    P = pd.DataFrame({"s1_id": F.s1_id.values, "cand_id": F.cand_id.values, "p": p, "rk": F.rk.values})
    pred = {k: set(v) for k, v in decode_all(P, audit, tie_col="rk").items()}
    f = macro_fbeta(pred, gs); br = score_breakdown(pred, gs, country_of)
    print(f"AUDIT {which} {label}: F0.5 {f:.5f}", flush=True)
    print(br.round(4)[["n", "mean_f", "lost_share"]].to_string(), flush=True)


def avg(models, X):
    return np.mean([m.predict(X, num_threads=24) for m in models], axis=0)


if which == "sub4_honest":
    d = pickle.load(open(V2 + "iso_sub4_honest.pkl", "rb")); feats = d["feats"]
    ms = [lgb.Booster(model_file=V2 + f"model_sub4_honest_f{k}.txt") for k in range(5)]
    FA = pds.dataset(V2 + "train_feats_tlv3_f2").to_table(filter=pds.field("s1_id").isin(pa.array(audit))).to_pandas()
    p = np.mean([iso.predict(m.predict(FA[feats], num_threads=24)) for m, iso in zip(ms, d["isos"])], axis=0)
    report(FA, p, "stage1")
else:   # sub3
    rank_s1 = set(pd.read_parquet(V1 + "ranker_s1.parquet").s1_id)
    ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0)
    m1 = [lgb.Booster(model_file=V1 + f"model_sub3_s1_f{k}.txt") for k in range(5)]
    m2 = [lgb.Booster(model_file=V1 + f"model_sub3_s2_f{k}.txt") for k in range(5)]
    f1, f2 = m1[0].feature_name(), m2[0].feature_name()
    # rebuild stage-1 isotonic from the fold models' OOF on the validation sample (as run_full.py fit it)
    F0 = add_labels(pds.dataset(V1 + "train_feats_r2_f2").to_table(filter=pds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas(), g)
    raw = np.zeros(len(F0), np.float32)
    for k, (a, b) in enumerate(GroupKFold(5).split(F0, groups=F0.s1_id)):
        raw[b] = m1[k].predict(F0[f1].iloc[b], num_threads=24)
    iso1 = calibrate(raw, F0.y); del F0, raw; gc.collect()
    O = pd.read_parquet(V1 + "oof_sub3.parquet", columns=["raw2", "y"]); iso2 = calibrate(O.raw2.to_numpy(), O.y.to_numpy()); del O
    FA = pds.dataset(V1 + "train_feats_tlv_f2").to_table(filter=pds.field("s1_id").isin(pa.array(audit))).to_pandas()
    FA["base_id"] = FA.cand_id.str.replace("~c", "", regex=False)
    FA["p1"] = iso1.predict(avg(m1, FA[f1]))
    report(FA, FA.p1.to_numpy(), "stage1")
    cp = pd.read_parquet(V1 + "train_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
    FA = sibling_features(FA, cp[cp.entity_id.isin(set(FA.base_id))], base_col="base_id"); del cp; gc.collect()
    report(FA, iso2.predict(avg(m2, FA[f2])), "two_stage")
