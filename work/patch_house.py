"""cache/v2 = cache/v1 with house/postal recomputed by normalize.house_postal (5-digit leading house
numbers; see its docstring). Every other v1 file is symlinked. Equivalent to re-running prep with the
new normalize.py (norm_addr derives nums from the same `addr` tokens), without redoing 20M records."""
import os, sys, time, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.normalize import house_postal, nums_in_order
V1, V2 = "/home/24b4518/ml-projects/work/cache/v1/", "/home/24b4518/ml-projects/work/cache/v2/"
os.makedirs(V2, exist_ok=True)
PATCH = [f"{sp}_{k}.parquet" for sp in ("train", "test") for k in ("s1p", "cp")]
for f in os.listdir(V1):
    if f not in PATCH and not os.path.exists(V2 + f):
        os.symlink(V1 + f, V2 + f)
t = time.time()
for f in PATCH:
    T = pq.read_table(V1 + f)
    hp = [house_postal(nums_in_order(a)) for a in T.column("addr").to_pylist()]
    old = T.column("house").to_pylist()
    new = [h for h, _ in hp]
    T = T.set_column(T.schema.get_field_index("house"), "house", pa.array(new, pa.large_string()))
    T = T.set_column(T.schema.get_field_index("postal"), "postal", pa.array([" ".join(p) for _, p in hp], pa.large_string()))
    pq.write_table(T, V2 + f)
    ch = sum(a != b for a, b in zip(old, new))
    print(f"{f}: {len(new):,} rows, house changed {ch:,} ({ch/len(new):.3f}), empty {sum(1 for h in old if not h)/len(new):.3f} -> {sum(1 for h in new if not h)/len(new):.3f}  [{time.time()-t:.0f}s]", flush=True)
    del T, hp, old, new
