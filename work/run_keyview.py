"""Add an exact-key candidate view to the capped shortlist: same cleaned name + same house number
(record's first number), per country, keys shared by <= MAXB S1s. New pairs get v_key=1 and 0 for
the other view scores; existing pairs get v_key=1 when they also match the key. -> {sp}_pairs_r2."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as pds
W = __import__("os").environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/")
sp = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "r2"; MAXB = 30
P = pd.read_parquet(W + f"{sp}_pairs_r1.parquet")
out = []
for c in sorted(pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.unique()):
    s1 = pds.dataset(W + f"{sp}_s1p.parquet").to_table(columns=["entity_id", "name_n", "house"], filter=pds.field("ctry") == c).to_pandas()
    cp = pds.dataset(W + f"{sp}_cp.parquet").to_table(columns=["entity_id", "name_n", "house"], filter=pds.field("ctry") == c).to_pandas()
    cp["h"] = cp.house   # record's house number by position (was: first of the SORTED number set)
    a = s1[(s1.house != "") & (s1.name_n != "")].assign(k=lambda d: d.name_n + "|" + d.house)
    b = cp[(cp.h != "") & (cp.name_n != "")].assign(k=lambda d: d.name_n + "|" + d.h)
    bs = a.groupby("k").size()
    a = a[a.k.map(bs) <= MAXB]
    J = a[["entity_id", "k"]].rename(columns={"entity_id": "s1_id"}).merge(b[["entity_id", "k"]].rename(columns={"entity_id": "cand_id"}), on="k")[["s1_id", "cand_id"]]
    print(f"{c}: key pairs {len(J):,} ({len(J)/len(s1):.2f}/S1)", flush=True)
    out.append(J)
K = pd.concat(out, ignore_index=True).drop_duplicates(); K["v_key"] = np.float32(1)
P = P.merge(K, on=["s1_id", "cand_id"], how="outer")
for col in [c for c in P.columns if c.startswith("v_") or c == "rk"]:
    P[col] = P[col].fillna(0).astype(np.float32)
print(f"{sp}: pairs r1 -> r2 = {len(P):,}", flush=True)
P.to_parquet(W + f"{sp}_pairs_{OUT}.parquet")
if sp == "train":
    gt = pd.read_parquet("/home/24b4518/ml-projects/work/cache/gt.parquet")
    rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
    gold = pd.DataFrame([(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x], columns=["s1_id", "cand_id"])
    gold = gold[~gold.s1_id.isin(rank_s1)]
    ctry = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "ctry"]).set_index("entity_id").ctry
    gold["c"] = gold.s1_id.map(ctry)
    for tag, Q in [("r1", pd.read_parquet(W + "train_pairs_r1.parquet", columns=["s1_id", "cand_id"])), (OUT, P[["s1_id", "cand_id"]])]:
        h = gold.merge(Q, on=["s1_id", "cand_id"])
        print(tag, {c: round((h.c == c).sum() / (gold.c == c).sum(), 4) for c in ["india", "us"]}, flush=True)
