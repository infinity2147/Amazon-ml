import sys, json, time, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.prep import prepare_split
W = "/home/24b4518/ml-projects/work/cache/"
tl = json.load(open(W + "translit.json"))
for sp in sys.argv[1:]:
    t = time.time()
    raw = (pd.read_parquet(W + f"{sp}_s1.parquet"), pd.concat([pd.read_parquet(W + f"{sp}_s2.parquet"), pd.read_parquet(W + f"{sp}_s3.parquet")], ignore_index=True))
    s1p, cp = prepare_split(None, sp, __import__("os").environ.get("OUT", W + "v1"), tl, n_jobs=32, raw=raw)
    print(sp, "done", len(s1p), len(cp), round(time.time() - t), "s", flush=True)
