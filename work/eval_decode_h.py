"""E1: decode-only variants on the honest H1 variant-C predictions for the TLV rows (no retraining).
TIE: one-owner keeps exactly one owner per record (tie on calibrated p broken by `rk`, the ranker score).
POUT: per-country P(an S1 has >=1 true match outside its shortlist), measured on non-ranker train S1s
OUTSIDE the 551,789 validation sample; scaled by {0.5, 1}.
Also reports how many records keep 2+ owners under the old rule."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as pds, pyarrow as pa
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.decode import decode_all, enforce_one_owner
from er.io_metric import macro_fbeta, score_breakdown
W = "/home/24b4518/ml-projects/work/cache/v2/"
s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0).tolist(); idset = set(ids)
H = pd.read_parquet(W + "honest_H1.parquet", columns=["s1_id", "cand_id", "pC"]).rename(columns={"pC": "p"})
R = pds.dataset(W + "train_feats_tlv3_f2").to_table(columns=["s1_id", "cand_id", "rk"], filter=pds.field("s1_id").isin(pa.array(ids))).to_pandas()
H = H.merge(R, on=["s1_id", "cand_id"], how="left"); del R
P = pds.dataset(W + "train_feats_r3_f2").to_table(columns=["s1_id", "cand_id"]).to_pandas(); have = pd.MultiIndex.from_frame(P); del P
rate = {}
for c in ["US", "India"]:
    pool = [s for s in s1.entity_id[(s1.country == c) & ~s1.entity_id.isin(rank_s1)].tolist() if s not in idset and g.get(s)][:300000]
    q = pd.MultiIndex.from_tuples([(s, x) for s in pool for x in g[s]])
    rate[c] = float(pd.Series(~q.isin(have), index=q.get_level_values(0)).groupby(level=0).any().mean())
print("p_outside:", {k: round(v, 4) for k, v in rate.items()}, flush=True)
kept = enforce_one_owner(H[H.p >= 1e-3], "p")
print(f"records kept by 2+ S1s under the old >= rule: {(kept.groupby('cand_id').size() > 1).sum():,} of {kept.cand_id.nunique():,}", flush=True)
gsub = {s: g.get(s, set()) for s in ids}
def run(tie, sc):
    po = {s: rate[country_of[s]] * sc for s in ids} if sc else 0.0
    pred = decode_all(H, ids, p_outside=po, tie_col="rk" if tie else None)
    pred = {k: set(v) for k, v in pred.items()}
    return macro_fbeta(pred, gsub), score_breakdown(pred, gsub, country_of)
for tie, sc in [(0, 0), (1, 0), (1, 0.5), (1, 1.0)]:
    f, br = run(tie, sc); print(f"TIE={tie} POUT scale={sc}: {f:.5f}   singleton F US {br.loc[('US','singleton'),'mean_f']:.4f} India {br.loc[('India','singleton'),'mean_f']:.4f}", flush=True)
