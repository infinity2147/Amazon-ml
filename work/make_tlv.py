"""Test-Like Validation (TLV) features: clone train decoys up to test decoy density.

Finding A4 (experiments.md): test = train + a second batch of decoys whose per-S1 distribution over
similarity buckets equals train's decoy distribution; true matches per S1 are unchanged. So we clone
each unowned train record (no S1 owns it) with probability q, where
    q = (test decoys per S1) / (train decoys per S1) - 1,  decoys per S1 = records/S1 - owned records/S1,
computed per country (assumes owned records per S1 are the same in test, supported by A4).
A clone gets id '<orig>~c', the same S1 links and pair features. Then the shortlist cap is re-applied
(top 20 by ranker score per S1, the record's own top-1, and key-view pairs), so clones compete for
S1 slots as test decoys do, and all context features (ranks, gaps, margins, list sizes, twin counts)
are recomputed. Output: cache/v1/train_feats_tlv_f2/part_<ctry>.parquet (same schema + nothing new).
Known bias: a clone always shares its original's S1 lists, so decoy errors are more concentrated
than on test and TLV slightly understates the damage."""
import sys, time, gc, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import group_rank
from er.features import add_context
W = __import__("os").environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/"); N1, SEED = 20, 11
SRC, DST = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("train_feats_r2_f2", "train_feats_tlv_f2")
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
import os; os.makedirs(W + DST, exist_ok=True)
gt = pd.read_parquet(W + "../gt.parquet")
owned = pd.Index([x for l in gt.matched_entity_ids for x in l.split(",") if x]); del gt
s1c = {sp: pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.value_counts() for sp in ["train", "test"]}
for c in ["india", "us"]:
    cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "ctry"]); cp = cp.entity_id[cp.ctry == c]
    n_own = cp.isin(owned).sum(); n_tr = s1c["train"][c]
    tcp = pd.read_parquet(W + "test_cp.parquet", columns=["ctry"]).ctry.eq(c).sum()
    dec_tr = (len(cp) - n_own) / n_tr; dec_te = tcp / s1c["test"][c] - n_own / n_tr
    q = dec_te / dec_tr - 1
    log(f"{c}: owned/S1 {n_own/n_tr:.3f}; decoys/S1 train {dec_tr:.3f} test {dec_te:.3f} -> clone prob q={q:.3f}")
    unowned = cp[~cp.isin(owned)].to_numpy(); del cp
    rng = np.random.default_rng(SEED)
    clone_set = pd.Index(unowned[rng.random(len(unowned)) < q])
    F = pq.read_table(W + f"{SRC}/part_{c}.parquet").to_pandas()
    m = F.cand_id.isin(clone_set).to_numpy()
    C = F[m].copy(); C["cand_id"] = C.cand_id + "~c"
    log(f"{c}: {len(F):,} pairs, cloning {len(clone_set):,} records -> {len(C):,} clone pairs")
    F = pd.concat([F, C], ignore_index=True); del C; gc.collect()
    # re-apply the shortlist cap on the augmented lists
    g1 = pd.factorize(F.s1_id)[0]; g2 = pd.factorize(F.cand_id)[0]; rk = F.rk.to_numpy(np.float32)
    r1 = group_rank(g1, rk)[0]; r2 = group_rank(g2, rk)[0]
    keep = (r1 <= N1) | ((r2 <= 1) & (rk > 0)) | (F.v_key.to_numpy() == 1)
    is_clone = F.cand_id.str.endswith("~c").to_numpy()
    log(f"{c}: cap drops {(~keep & ~is_clone).sum():,} original and {(~keep & is_clone).sum():,} clone pairs")
    F = F[keep].reset_index(drop=True); del g1, g2, rk, r1, r2; gc.collect()
    F = add_context(F)
    F.to_parquet(W + f"{DST}/part_{c}.parquet")
    log(f"{c}: wrote {len(F):,} pairs ({F.cand_id.str.endswith('~c').mean():.3f} clone pairs)")
    del F; gc.collect()
log("done")
