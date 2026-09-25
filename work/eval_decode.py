"""Decode-only variants on a run's saved OOF (no retraining): baseline one-owner (>=, ties keep several
owners), TIE (exactly one owner, ties broken by the raw stage-2 score), POUT (per-country probability
that an S1 has a true match outside its shortlist, measured on train S1s OUTSIDE the validation sample),
and both. usage: eval_decode.py <name> <feat_tag>"""
import sys, os, numpy as np, pandas as pd, pyarrow.dataset as pds
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.decode import decode_all
from er.io_metric import macro_fbeta

W = os.environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/")
name, tag = sys.argv[1:3]
O = pd.read_parquet(W + f"oof_{name}.parquet")
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"])
country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet")
g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0).tolist()
idset = set(ids)

P = pds.dataset(W + f"train_feats_{tag}").to_table(columns=["s1_id", "cand_id"]).to_pandas()
have = pd.MultiIndex.from_frame(P)
del P
rate = {}
for c in ["US", "India"]:
    pool = [s for s in s1.entity_id[(s1.country == c) & ~s1.entity_id.isin(rank_s1)].tolist()
            if s not in idset and g.get(s)][:300000]
    q = pd.MultiIndex.from_tuples([(s, x) for s in pool for x in g[s]])
    rate[c] = float(pd.Series(~q.isin(have), index=q.get_level_values(0)).groupby(level=0).any().mean())
print("p_outside (share of S1s with >=1 true match outside the shortlist):",
      {k: round(v, 4) for k, v in rate.items()}, flush=True)

gsub = {s: g.get(s, set()) for s in ids}


def score(tie, pout, scale=1.0):
    po = {s: rate[country_of[s]] * scale for s in ids} if pout else 0.0
    pred = decode_all(O[["s1_id", "cand_id", "p", "raw2"]], ids, p_outside=po, tie_col="raw2" if tie else None)
    return macro_fbeta({k: set(v) for k, v in pred.items()}, gsub)


x = O[["cand_id", "p"]]
top = x.p == x.groupby("cand_id").p.transform("max")
print(f"records whose best p is shared by 2+ S1s: {(top.groupby(x.cand_id).sum() > 1).sum():,}", flush=True)
for tie, pout, sc in [(0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 1, 0.5), (0, 1, 1.5), (1, 1, 1)]:
    print(f"TIE={tie} POUT={pout} scale={sc}: {score(tie, pout, sc):.5f}", flush=True)
