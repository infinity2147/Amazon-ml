"""Label-free test monitor per country (France has no labels): decode the given test predictions exactly as
the submission does (one owner per record, tie broken by rk; expected F0.5) and report per S1:
  empty share, accepted IDs, accepted IDs in two risky buckets
    same_addr_diff_name : exact S1 house number found in the record, name token-set similarity < 60
    near_twin           : name token-set >= 85, street-word token-set >= 80, house number NOT matched exactly
usage: france_monitor.py <pred parquet with s1_id,cand_id,p[,rk]> [label]"""
import sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.decode import decode_all
W = __import__("os").environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v2/")
path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path.split("/")[-1]
P = pd.read_parquet(path)
X = pd.concat([pq.read_table(f, columns=["s1_id", "cand_id", "rk", "house_eq", "n_tset", "aw_tset"]).to_pandas()
               for f in sorted(glob.glob(W + "test_feats_r3_f2/*.parquet"))], ignore_index=True)
if "rk" in P.columns:
    P = P.drop(columns=["rk"])
P = P.merge(X, on=["s1_id", "cand_id"], how="left")
ts1 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"]); ids = ts1.entity_id.tolist()
pred = decode_all(P[["s1_id", "cand_id", "p", "rk"]], ids, tie_col="rk")
acc = pd.DataFrame([(s, c) for s, v in pred.items() for c in v], columns=["s1_id", "cand_id"]).merge(P, on=["s1_id", "cand_id"])
acc["same_addr_diff_name"] = (acc.house_eq == 1) & (acc.n_tset < 60)
acc["near_twin"] = (acc.n_tset >= 85) & (acc.aw_tset >= 80) & (acc.house_eq != 1)
ctry = dict(zip(ts1.entity_id, ts1.country)); acc["country"] = acc.s1_id.map(ctry)
n = ts1.country.value_counts()
empty = pd.Series({c: np.mean([len(pred[s]) == 0 for s in ts1.entity_id[ts1.country == c]]) for c in n.index})
out = pd.DataFrame({"S1": n, "empty_share": empty.round(4),
                    "accepted_per_S1": (acc.groupby("country").size() / n).round(3),
                    "same_addr_diff_name_per_S1": (acc.groupby("country").same_addr_diff_name.sum() / n).round(4),
                    "near_twin_per_S1": (acc.groupby("country").near_twin.sum() / n).round(4)})
print(f"== {label}"); print(out.to_string(), flush=True)
