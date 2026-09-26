"""Build the cross-encoder scoring lists for the final model from the all-data stage-1 run `s6` (cache v3).
Band = stage-1 p in (0.005, 0.995): train pool rows (out-of-fold p), audit rows (test-like, fold-average p), test rows
(fold-average p). Records are mapped to their base id (validation clones share the original's text).
Model choice: S1s in the 551,789-S1 validation sample whose half is 0 were in model A's training pairs -> scored by B;
every other S1 (rest of the train pool, audit, test) -> model A. Pairs already scored in ce/ce_val.parquet are reused.
usage: ce_final_prep.py train|test"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.crossenc import half_of
V3 = "/home/24b4518/ml-projects/work/cache/v3/"; CE = "/home/24b4518/ml-projects/work/ce/"
which = sys.argv[1]; LO, HI = 0.005, 0.995
if which == "train":
    s1 = pd.read_parquet(V3 + "train_s1p.parquet", columns=["entity_id"])
    rank_s1 = set(pd.read_parquet(V3 + "ranker_s1.parquet").s1_id)
    val = pd.Index(s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0))
    O = pd.read_parquet(V3 + "oof_s6_s1.parquet", columns=["s1_id", "cand_id", "p1"])
    A = pd.read_parquet(V3 + "audit_pred_s6_s1.parquet", columns=["s1_id", "cand_id", "p1"])
    P = pd.concat([O, A], ignore_index=True)
    P = P[(P.p1 > LO) & (P.p1 < HI)].copy()
    P["cand_id"] = P.cand_id.str.replace("~c", "", regex=False)
    P = P[["s1_id", "cand_id"]].drop_duplicates()
    inval = P.s1_id.isin(val).to_numpy()
    P["model"] = np.where(inval & (half_of(P.s1_id.to_numpy()) == 0), "B", "A")
    # no memorisation: a pair is NOT scored (left missing) when its record was in the scoring model's own training
    # pairs (as in experiment X4u). Test records are all new, so this never applies to test.
    Bt = pd.read_parquet(CE + "band_train.parquet")
    seen = {"A": set(Bt[Bt.half == 0].sample(frac=1, random_state=0).head(500000).cand_id),
            "B": set(Bt[Bt.half == 1].sample(frac=1, random_state=1).head(500000).cand_id)}
    blank = np.where(P.model == "A", P.cand_id.isin(seen["A"]), P.cand_id.isin(seen["B"]))
    print(f"band pairs whose record the assigned model saw in training (left unscored): {blank.sum():,} of {len(P):,}")
    P = P[~blank]
    done = pd.read_parquet(CE + "ce_val_unseen.parquet")
    P = P.merge(done, on=["s1_id", "cand_id"], how="left")
    print(f"train+audit band pairs {len(P):,}: model A {(P.model == 'A').sum():,}, model B {(P.model == 'B').sum():,}; already scored {P.ce.notna().sum():,}")
    P[P.ce.notna()][["s1_id", "cand_id", "ce"]].to_parquet(CE + "trainA_done.parquet")
    for m in ["A", "B"]:
        Q = P[P.ce.isna() & (P.model == m)][["s1_id", "cand_id"]]
        Q.to_parquet(CE + f"todo_train_{m}.parquet"); print(f"to score with {m}: {len(Q):,}")
else:
    T = pd.read_parquet(V3 + "test_pred_s6.parquet", columns=["s1_id", "cand_id", "p1"])
    T = T[(T.p1 > LO) & (T.p1 < HI)][["s1_id", "cand_id"]].drop_duplicates()
    T.to_parquet(CE + "todo_test_A.parquet"); print(f"test band pairs to score with A: {len(T):,} ({len(T) / 1732544:.2f}/S1)")
