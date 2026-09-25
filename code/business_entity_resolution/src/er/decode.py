"""Metric-aware decoding: turn calibrated pair probabilities into the per-entity set
that maximises EXPECTED F_beta (singletons included). Exact under independence."""
import numpy as np


def _pb_pmf(ps):
    """Poisson-binomial pmf of sum of Bernoulli(ps)."""
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def expected_fbeta_topk(probs, beta=0.5, p_outside=0.0):
    """probs: calibrated P(match) for one S1 entity's candidates.
    p_outside: P(entity has a true match blocking never surfaced).
    Returns (best_k, expected_F for k=0..n, order). Optimal set is top-k by prob."""
    probs = np.asarray(probs, float)
    order = np.argsort(-probs)
    p = probs[order]
    n = len(p)
    b2 = beta * beta
    suffix = [None] * (n + 1)
    suffix[n] = _pb_pmf([p_outside]) if p_outside > 0 else np.array([1.0])
    for k in range(n - 1, -1, -1):
        suffix[k] = np.convolve(suffix[k + 1], [1 - p[k], p[k]])
    ef = np.zeros(n + 1)
    ef[0] = suffix[0][0]                      # predict empty: right iff no true match at all
    pref = np.array([1.0])
    for k in range(1, n + 1):
        pref = np.convolve(pref, [1 - p[k - 1], p[k - 1]])   # TP count among top-k
        pr = suffix[k]                                        # FN count
        t = np.arange(len(pref))[:, None]
        r = np.arange(len(pr))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(t > 0, (1 + b2) * t / (b2 * (t + r) + k), 0.0)
        ef[k] = pref @ f @ pr
    best_k = int(np.argmax(ef))
    return best_k, ef, order


def decode_entity(cand_ids, probs, beta=0.5, p_outside=0.0, min_p=1e-3, max_n=25):
    keep = sorted((i for i, q in enumerate(probs) if q >= min_p), key=lambda i: -probs[i])[:max_n]
    if not keep:
        return []
    ids = [cand_ids[i] for i in keep]
    ps = [probs[i] for i in keep]
    k, _, order = expected_fbeta_topk(ps, beta, p_outside)
    return [ids[j] for j in order[:k]]


def enforce_one_owner(pairs, prob_col="p", slack=0.0, tie_col=None):
    """Each S2/S3 record belongs to <=1 S1 (verified on train GT: 0 of 7.64M violate).
    tie_col=None: keep a pair if its prob is within `slack` of the record's best (isotonic output is a
    step function, so ties keep SEVERAL owners). tie_col given: keep exactly one owner per record, the
    best by (prob, tie_col) with a final deterministic tie-break on s1_id."""
    if tie_col is None:
        best = pairs.groupby("cand_id")[prob_col].transform("max")
        return pairs[pairs[prob_col] >= best - slack]
    import pandas as pd
    cc = pd.factorize(pairs.cand_id)[0]
    o = np.lexsort((pd.factorize(pairs.s1_id, sort=True)[0], -pairs[tie_col].to_numpy(np.float64),
                    -pairs[prob_col].to_numpy(np.float64), cc))
    c = cc[o]
    first = np.r_[True, c[1:] != c[:-1]]
    return pairs.iloc[np.sort(o[first])]


def decode_all(pairs, s1_ids, beta=0.5, p_outside=0.0, prob_col="p", one_owner=True, min_p=1e-3, tie_col=None):
    """pairs: DataFrame[s1_id, cand_id, p] (calibrated). Returns {s1_id: [ids]} for ALL s1_ids.
    p_outside: a float, or a dict {s1_id: float} (e.g. per country)."""
    if one_owner:
        pairs = enforce_one_owner(pairs, prob_col, tie_col=tie_col)
    pairs = pairs[pairs[prob_col] >= min_p]
    out = {s: [] for s in s1_ids}
    s1 = pairs.s1_id.to_numpy()
    c = pairs.cand_id.to_numpy()
    p = pairs[prob_col].to_numpy()
    order = np.argsort(s1, kind="stable")
    s1, c, p = s1[order], c[order], p[order]
    if len(s1) == 0:
        return out
    bounds = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(s1)]
    for a, b in zip(starts, ends):
        po = p_outside.get(s1[a], 0.0) if isinstance(p_outside, dict) else p_outside
        out[s1[a]] = decode_entity(list(c[a:b]), list(p[a:b]), beta, po, min_p)
    return out
