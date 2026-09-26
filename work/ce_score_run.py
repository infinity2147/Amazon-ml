"""Score (S1, record) pairs with one cross-encoder and write [s1_id, cand_id, ce].
usage: ce_score_run.py <pairs parquet with s1_id,cand_id[,half]> <A|B> <out parquet> [half filter: 0|1|all] [split: train|test]
An S1 in half h (0 = trained model A, 1 = trained model B) must be scored by the OTHER model."""
import os, sys, time, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.crossenc import pair_texts, score_ce
W = "/home/24b4518/ml-projects/work/cache/v2/"
src, model, out = sys.argv[1:4]; hf = sys.argv[4] if len(sys.argv) > 4 else "all"; sp = sys.argv[5] if len(sys.argv) > 5 else "train"
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
P = pd.read_parquet(src)
if hf != "all":
    P = P[P.half == int(hf)]
P = P[["s1_id", "cand_id"]].drop_duplicates().reset_index(drop=True)
s1raw = pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["entity_id", "business_name", "business_address"]).set_index("entity_id")
cpraw = pd.read_parquet(W + f"{sp}_cp.parquet", columns=["entity_id", "business_name", "business_address"])
cpraw = cpraw[cpraw.entity_id.isin(set(P.cand_id))].set_index("entity_id")
ta, tb = pair_texts(P, s1raw, cpraw); del s1raw, cpraw
log(f"scoring {len(P):,} pairs with ce_{model}")
P["ce"] = score_ce(ta, tb, f"/home/24b4518/ml-projects/work/ce/ce_{model}", log)
P.to_parquet(out); log(f"wrote {out}")
