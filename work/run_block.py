import sys, time, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import block, blocking_report
W = "/home/24b4518/ml-projects/work/cache/"
sp = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "b1"
t = time.time()
s1p = pd.read_parquet(W + f"v1/{sp}_s1p.parquet", columns=["entity_id", "ctry", "name_n", "addr"])
cp = pd.read_parquet(W + f"v1/{sp}_cp.parquet", columns=["entity_id", "ctry", "name_n", "addr"])
P = block(s1p, cp, log=lambda m: print(m, flush=True), cache_dir=W + "v1", tag=f"{sp}_{tag}", cap=(60, 8))
P.to_parquet(W + f"v1/{sp}_pairs_{tag}.parquet")
print("blocking done", len(P), round(time.time() - t), "s", flush=True)
