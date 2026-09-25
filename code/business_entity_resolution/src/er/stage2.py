"""Stage-2 'sibling' features from stage-1 probabilities (OOF on train, model output on test).

For a pair (s, c), the siblings are the OTHER candidates of s that stage 1 already believes in
(p >= conf). True matches of one S1 are noisy copies of the same record, so they agree with each
other; a near-twin decoy (same name and street, different house number) disagrees with them, and a
true match whose name was replaced by a pseudo-word still agrees with them on the address.
Also: p-context within the S1 list and within the record's list of S1s."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .blocking import group_rank


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def sibling_features(F, cp, p_col="p1", conf=0.5, base_col=None, ext=None):
    """F: pairs with s1_id, cand_id, p_col. cp: prepared S2/S3 frame (entity_id, name_n, addr, house, nums).
    base_col: optional column of F holding the record id whose TEXT a row uses (a validation clone
    'X~c' reads the text of 'X'); rows that share a base record are not each other's siblings.
    ext: optional DataFrame [cand_id, p] of the SAME records' pairs with S1s outside F (other S1s of
    the population). They compete in the record-side features (p1_rank_c, p1_gap_c, n_conf_c) exactly
    as every S1 competes on test, but get no rows of their own.
    Returns F with new columns (same row order)."""
    F = F.reset_index(drop=True)
    p = F[p_col].to_numpy(np.float32)
    g1 = pd.factorize(F.s1_id)[0]
    r1, m1, _ = group_rank(g1, p)
    if ext is not None and len(ext):
        pc = np.r_[p, ext.p.to_numpy(np.float32)]
        g2 = pd.factorize(np.r_[F.cand_id.to_numpy(), ext.cand_id.to_numpy()])[0]
        r2, m2, _ = group_rank(g2, pc)
        cm = pc >= conf
        n_conf_c = (np.bincount(g2, weights=cm)[g2] - cm)[:len(F)].astype(np.float32)
        r2, m2 = r2[:len(F)], m2[:len(F)]
    else:
        g2 = pd.factorize(F.cand_id)[0]
        r2, m2, _ = group_rank(g2, p)
        n_conf_c = None
    out = pd.DataFrame(index=F.index)
    out["p1_rank_s1"], out["p1_gap_s1"] = r1, m1 - p
    out["p1_rank_c"], out["p1_gap_c"] = r2, m2 - p
    out["p1_sum_s1"] = np.bincount(g1, weights=p)[g1].astype(np.float32)
    conf_mask = p >= conf
    out["n_conf_s1"] = np.bincount(g1, weights=conf_mask)[g1].astype(np.float32) - conf_mask
    out["n_conf_c"] = n_conf_c if n_conf_c is not None else np.bincount(g2, weights=conf_mask)[g2].astype(np.float32) - conf_mask
    # sibling pairs: every row x every confident row of the same S1 (excluding itself), built in
    # chunks of S1 groups to bound memory (~70 sibling rows per S1)
    C = cp.set_index("entity_id")
    cid = (F[base_col] if base_col else F.cand_id).to_numpy()
    rows_all = pd.DataFrame({"g": g1, "i": np.arange(len(F))})
    n_g = g1.max() + 1
    aggs = []
    for lo in range(0, n_g, 150_000):
        rows = rows_all[(rows_all.g >= lo) & (rows_all.g < lo + 150_000)]
        sib = rows[conf_mask[rows.i.to_numpy()]]
        S = rows.merge(sib, on="g", suffixes=("", "_s"))
        S = S[cid[S.i.to_numpy()] != cid[S.i_s.to_numpy()]]   # not itself, not its own clone
        if len(S) == 0:
            continue
        a = C.loc[cid[S.i.to_numpy()], ["name_n", "addr", "house", "nums"]].to_numpy()
        b = C.loc[cid[S.i_s.to_numpy()], ["name_n", "addr", "house", "nums"]].to_numpy()
        S = S.assign(
            n_ts=_cp(fuzz.token_set_ratio, a[:, 0].tolist(), b[:, 0].tolist()),
            a_ts=_cp(fuzz.token_set_ratio, a[:, 1].tolist(), b[:, 1].tolist()),
            h_eq=np.where((a[:, 2] != "") & (b[:, 2] != ""), (a[:, 2] == b[:, 2]).astype(np.float32), np.nan),
            num_j=[len(set(x.split()) & set(y.split())) / max(len(set(x.split()) | set(y.split())), 1) if x and y else np.nan
                   for x, y in zip(a[:, 3], b[:, 3])],
        )
        aggs.append(S.groupby("i").agg(sib_n_ts_max=("n_ts", "max"), sib_n_ts_mean=("n_ts", "mean"),
                                       sib_a_ts_max=("a_ts", "max"), sib_a_ts_mean=("a_ts", "mean"),
                                       sib_h_eq_mean=("h_eq", "mean"), sib_num_j_max=("num_j", "max")))
        del S, a, b
    agg = pd.concat(aggs)
    out = out.join(agg).fillna({c: -1 for c in agg.columns})
    return pd.concat([F, out.astype(np.float32)], axis=1)
