import sys, time, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import block
W = "/home/24b4518/ml-projects/work/cache/v1/"
sp = sys.argv[1]; t = time.time()
s1p = pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["entity_id", "ctry", "name_n", "addr"])
cp = pd.read_parquet(W + f"{sp}_cp.parquet", columns=["entity_id", "ctry", "name_n", "addr"])
block(s1p, cp, log=lambda m: print(m, flush=True), cache_dir=W, tag=f"{sp}_b1", views_only=True)
print("views done", round(time.time() - t), "s", flush=True)
