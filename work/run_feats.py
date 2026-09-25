"""Pair features, one country at a time (every feature and context rank is within-country, so this
is exact) -> {sp}_feats_{tag}{SUFFIX}/part_<country>.parquet. Bounded memory."""
import sys, os, time, gc, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.features import build_idf, build_state_map, pair_features, L_COLS, R_COLS
W = __import__("os").environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/")
SUFFIX = "_f2"


def main(sp, tag, limit_s1=None, only=None):
    import pyarrow.dataset as pds
    t = time.time()
    cols = sorted(set(["entity_id", "ctry"] + L_COLS + R_COLS))
    s1cols = [c for c in cols if c not in ("name_alt", "web", "native")]
    out_dir = W + f"{sp}_feats_{tag}{SUFFIX}"
    os.makedirs(out_dir, exist_ok=True)
    ctrys = sorted(pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.unique())
    for c in ctrys:
        if only and c not in only:
            continue
        if os.path.exists(f"{out_dir}/part_{c}.parquet") and not limit_s1:
            print(f"{c}: exists, skip", flush=True); continue
        # load only this country's rows (bounded memory)
        s1p = pds.dataset(W + f"{sp}_s1p.parquet").to_table(columns=s1cols, filter=pds.field("ctry") == c).to_pandas()
        cp = pds.dataset(W + f"{sp}_cp.parquet").to_table(columns=cols, filter=pds.field("ctry") == c).to_pandas()
        import pyarrow as pa
        P = pds.dataset(W + f"{sp}_pairs_{tag}.parquet").to_table(filter=pds.field("s1_id").isin(pa.array(s1p.entity_id.tolist()))).to_pandas()
        if limit_s1:
            P = P[P.s1_id.isin(set(s1p.entity_id.iloc[:limit_s1]))].reset_index(drop=True)
        idf = build_idf(s1p, cp)
        idf["state_map"] = build_state_map(P, s1p, cp)
        print(f"{c}: {len(P):,} pairs; state map {len(idf['state_map'])}", flush=True)
        F = pair_features(P, s1p, cp, idf, n_jobs=24, log=None)
        if not limit_s1:
            F.to_parquet(f"{out_dir}/part_{c}.parquet")
        print(f"{c}: features {F.shape} at {time.time() - t:.0f}s", flush=True)
        del F, P, s1p, cp, idf; gc.collect()
    print("features done", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
