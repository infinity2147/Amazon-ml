"""Slice-level prior correction (label-free). Per country c and pair-type bucket b, assume the number of TRUE
matches per S1 in b is the same on test as on train (same generator; checked by the obvious-owner proxy), while
the number of CANDIDATES per S1 in b is observed on test. Then the match prior inside b moves from
pi_tr = true_tr/cands_tr to pi_te = true_tr/cands_te, and each pair's odds are multiplied by
[pi_te/(1-pi_te)] / [pi_tr/(1-pi_tr)]."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as pds, pyarrow as pa
W = "/home/24b4518/ml-projects/work/cache/v1/"
cols = ["s1_id", "cand_id", "n_tset", "aw_tset", "house_eq", "house_drop", "r_addr_empty"]
def bucket(d):
    return np.select([d.r_addr_empty == 1, (d.house_eq == 1) & (d.n_tset >= 85), (d.house_eq == 1) & (d.n_tset < 85),
                      d.house_drop == 1, (d.house_eq == 0) & (d.n_tset >= 85) & (d.aw_tset >= 85), d.house_eq == 0, d.house_eq == -1],
                     ["empty_addr", "house_eq_name_close", "house_eq_name_diff", "house_drop", "near_twin", "house_diff_other", "house_missing"], "other")
def multipliers(oof_tag, test_tag, feat_tag, countries=("US", "India"), cap=(0.25, 4.0)):
    O = pd.read_parquet(W + f"oof_{oof_tag}.parquet", columns=["s1_id", "cand_id", "y"])
    T = pd.read_parquet(W + f"test_pred_{test_tag}.parquet", columns=["s1_id", "cand_id", "p"])
    A = O.merge(pds.dataset(W + f"train_feats_{feat_tag}").to_table(columns=cols, filter=pds.field("s1_id").isin(pa.array(O.s1_id.unique()))).to_pandas(), on=["s1_id", "cand_id"])
    B = T.merge(pds.dataset(W + f"test_feats_{feat_tag}").to_table(columns=cols).to_pandas(), on=["s1_id", "cand_id"])
    c1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    c2 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    A["c"], B["c"] = A.s1_id.map(c1), B.s1_id.map(c2)
    A["b"], B["b"] = bucket(A), bucket(B)
    M = {}
    for c in countries:
        a, b = A[A.c == c], B[B.c == c]
        na, nb = a.s1_id.nunique(), b.s1_id.nunique()
        for k in sorted(set(a.b)):
            true_ps = a[(a.b == k)].y.sum() / na
            cand_tr = (a.b == k).sum() / na
            cand_te = (b.b == k).sum() / nb
            pi_tr = true_ps / cand_tr
            pi_te = min(true_ps / max(cand_te, 1e-9), 0.999)
            m = (pi_te / (1 - pi_te)) / (pi_tr / (1 - pi_tr))
            M[(c, k)] = float(np.clip(m, *cap))
            print(f"  {c:6s} {k:22s} true/S1 {true_ps:.3f} cands/S1 train {cand_tr:.3f} test {cand_te:.3f} -> odds x{M[(c, k)]:.3f}", flush=True)
    B["m"] = [M.get((c, k), 1.0) for c, k in zip(B.c, B.b)]
    q = B.p.to_numpy() * B.m.to_numpy()
    B["p_adj"] = q / (q + 1 - B.p.to_numpy())
    return B
if __name__ == "__main__":
    B = multipliers(sys.argv[1], sys.argv[2], sys.argv[3])
    B[["s1_id", "cand_id", "p", "p_adj", "c", "b"]].to_parquet(W + f"test_pred_{sys.argv[2]}_bp.parquet")
    for c in ["US", "India", "France"]:
        b = B[B.c == c]; n = b.s1_id.nunique()
        print(f"{c}: accepted p>.5 per S1 before {(b.p > .5).sum()/n:.3f} after {(b.p_adj > .5).sum()/n:.3f}")
