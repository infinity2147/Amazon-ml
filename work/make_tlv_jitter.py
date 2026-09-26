"""Tie-free test-like TRAINING rows: same clones as make_tlv.py (seed 11, same q), but every continuous pair feature
of a clone row gets a small Gaussian jitter (1% of that feature's standard deviation), then the shortlist cap and all
context (rank/gap/margin/twin) features are recomputed. A clone then never ties exactly with its original, so a model
trained on these rows cannot learn the synthetic shortcut 'an exact tie with another candidate means decoy' (real test
decoys are distinct records, not copies). Output: train_feats_tlvj3_f2/part_<country>.parquet"""
import os, sys, time, gc, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.blocking import group_rank
from er.features import add_context
W = "/home/24b4518/ml-projects/work/cache/v2/"; SRC, DST = "train_feats_r3_f2", "train_feats_tlvj3_f2"; N1, SEED = 20, 11
CONTEXT = ("_rank_s1", "_gap_s1", "_rank_c", "_gap_c", "_margin_c")
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
os.makedirs(W + DST, exist_ok=True)
gt = pd.read_parquet(W + "../gt.parquet"); owned = pd.Index([x for l in gt.matched_entity_ids for x in l.split(",") if x]); del gt
s1c = {sp: pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.value_counts() for sp in ["train", "test"]}
for c in ["india", "us"]:
    cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "ctry"]); cp = cp.entity_id[cp.ctry == c]
    n_own = cp.isin(owned).sum(); n_tr = s1c["train"][c]
    tcp = pd.read_parquet(W + "test_cp.parquet", columns=["ctry"]).ctry.eq(c).sum()
    dec_tr = (len(cp) - n_own) / n_tr; dec_te = tcp / s1c["test"][c] - n_own / n_tr; q = dec_te / dec_tr - 1
    unowned = cp[~cp.isin(owned)].to_numpy(); del cp
    clone_set = pd.Index(unowned[np.random.default_rng(SEED).random(len(unowned)) < q])
    F = pq.read_table(W + f"{SRC}/part_{c}.parquet").to_pandas()
    C = F[F.cand_id.isin(clone_set).to_numpy()].copy(); C["cand_id"] = C.cand_id + "~c"
    ctx = [k for k in F.columns if k.endswith(CONTEXT) or k in ("n_cands_s1", "n_s1_for_c", "s1_n_twins", "s1_n_twins_heq")]
    cont = [k for k in F.columns if F[k].dtype.kind == "f" and k not in ctx and F[k].iloc[:200000].nunique() > 50]
    rng = np.random.default_rng(99)
    for k in cont:
        sd = float(F[k].iloc[:500000].std()) or 1.0
        C[k] = (C[k].to_numpy() + rng.normal(0, 0.01 * sd, len(C))).astype(F[k].dtype)
    log(f"{c}: q={q:.3f}, {len(C):,} clone pairs, jittered {len(cont)} continuous features: {cont[:8]}...")
    F = pd.concat([F, C], ignore_index=True); del C; gc.collect()
    g1 = pd.factorize(F.s1_id)[0]; g2 = pd.factorize(F.cand_id)[0]; rk = F.rk.to_numpy(np.float32)
    keep = (group_rank(g1, rk)[0] <= N1) | ((group_rank(g2, rk)[0] <= 1) & (rk > 0)) | (F.v_key.to_numpy() == 1)
    F = add_context(F[keep].reset_index(drop=True))
    F.to_parquet(W + f"{DST}/part_{c}.parquet"); log(f"{c}: wrote {len(F):,} pairs"); del F; gc.collect()
log("done")
