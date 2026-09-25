"""Multi-view, recall-first blocking at 10M-record scale.

Each view = character n-gram TF-IDF over one text field (hashed, IDF fit transductively on
S1 + S2/S3 of the same split and country). For each view we take
  - reverse top-k: for every S2/S3 record, its k most similar S1 records (each record has <= 1
    true owner, so this is the natural direction), and
  - forward top-k: for every S1, its k most similar S2/S3 records.
The union over views is the candidate set. Everything runs per country bucket (open set:
any label, including an unseen one, is its own bucket)."""
import gc
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

N_FEAT = 2 ** 22
_HV = {}


def _hv(ngram):
    """ngram: 'word' (whitespace tokens, numbers included) or an int n (char_wb n-grams)."""
    if ngram not in _HV:
        if ngram == "word":
            _HV[ngram] = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=N_FEAT,
                                           alternate_sign=False, norm=None, dtype=np.float32)
        else:
            _HV[ngram] = HashingVectorizer(analyzer="char_wb", ngram_range=(ngram, ngram), n_features=N_FEAT,
                                           alternate_sign=False, norm=None, dtype=np.float32)
    return _HV[ngram]


def _hash_chunk(args):
    texts, ngram = args
    return _hv(ngram).transform(texts)


def hash_texts(texts, ngram="word", n_jobs=32, chunk=50000):
    texts = list(texts)
    parts = [(texts[i:i + chunk], ngram) for i in range(0, len(texts), chunk)]
    with Pool(n_jobs) as pool:
        mats = pool.map(_hash_chunk, parts)
    return sp.vstack(mats).tocsr()


def tfidf_pair(A, B, budget=1e11):
    """A, B: raw count matrices (S1 rows, cand rows). Fits IDF on both sides (transductive).
    Cost control: the sparse product costs sum_f dfA(f) * dfB(f); we drop the most common keys
    until that sum is <= budget. Common keys ('rd', 'services', city names) carry little identity;
    rare keys (street names, business words, house numbers) are always kept."""
    n = A.shape[0] + B.shape[0]
    dA = np.bincount(A.indices, minlength=N_FEAT).astype(np.float64)
    dB = np.bincount(B.indices, minlength=N_FEAT).astype(np.float64)
    df = dA + dB
    idf = np.log((1 + n) / (1 + df)).astype(np.float32) + 1
    cost = dA * dB
    tot = cost.sum()
    if tot > budget:
        o = np.argsort(-cost)
        drop = int(np.argmax(tot - np.cumsum(cost[o]) <= budget)) + 1
        idf[o[:drop]] = 0
    out = []
    for M in (A, B):
        M = M.copy()
        M.data = (1 + np.log(M.data)) * idf[M.indices]
        M.eliminate_zeros()
        out.append(l2norm(M, copy=False))
    return out


def topk(Q, X, k, n_threads=32, min_sim=0.1):
    """For each row of Q, top-k rows of X by cosine. Returns (q_idx, x_idx, sim) arrays."""
    C = sp_matmul_topn(Q, X.T.tocsr(), top_n=k, threshold=min_sim, sort=False, n_threads=n_threads)
    C = C.tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)


def view_pairs(a_text, b_text, k_rev, k_fwd, ngram="word", budget=1e11, n_jobs=32, min_sim=0.05):
    A = hash_texts(a_text, ngram, n_jobs)
    B = hash_texts(b_text, ngram, n_jobs)
    A, B = tfidf_pair(A, B, budget)
    parts = []
    if k_rev:
        j, i, s = topk(B, A, k_rev, n_jobs, min_sim)
        parts.append((i, j, s))
    if k_fwd:
        i, j, s = topk(A, B, k_fwd, n_jobs, min_sim)
        parts.append((i, j, s))
    del A, B
    gc.collect()
    i = np.concatenate([p[0] for p in parts])
    j = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    return i, j, s


DEFAULT_VIEWS = {
    # name: (text builder, k_rev, k_fwd, ngram, cost budget)
    "v_word": (lambda d: d.name_n + " " + d.addr, 10, 20, "word", 1e11),
    "v_name5": (lambda d: d.name_n, 5, 10, 5, 1e11),
    "v_addr5": (lambda d: d.addr, 5, 10, 5, 1e11),
}


def block(s1, cands, views=None, n_jobs=32, log=print, cache_dir=None, tag="", cap=(40, 5), views_only=False):
    """s1, cands: prepared frames. Per country and per view, top-k pairs are computed and (if
    cache_dir) saved immediately as int32 index triples, so a crash or rerun resumes and memory
    stays bounded. The per-country union is then capped with cap_candidates(n_s1, n_c).
    Returns DataFrame[s1_id, cand_id, v_<view> sims (0 = not found by that view)]."""
    import os
    views = views or DEFAULT_VIEWS
    out = []
    for c in sorted(set(s1.ctry) | set(cands.ctry)):
        a = s1[(s1.ctry == c) | ((s1.ctry == "unk") & (c != "unk"))]
        b = cands[(cands.ctry == c) | ((cands.ctry == "unk") & (c != "unk"))]
        if len(a) == 0 or len(b) == 0:
            continue
        nb = len(b)
        cols = {}
        for vn, (fn, kr, kf, ng, bud) in views.items():
            path = os.path.join(cache_dir, f"blk_{tag}_{c}_{vn}.parquet") if cache_dir else None
            if path and os.path.exists(path):
                d = pd.read_parquet(path)
            else:
                t = time.time()
                i, j, s = view_pairs(fn(a).tolist(), fn(b).tolist(), kr, kf, ng, bud, n_jobs=n_jobs)
                d = pd.DataFrame({"key": i * nb + j, vn: s}).groupby("key", sort=False)[vn].max().reset_index()
                del i, j, s
                if path:
                    d.to_parquet(path)
                log(f"[block] {c} {vn}: {len(d):,} pairs ({len(d) / len(a):.1f}/S1) in {time.time() - t:.0f}s")
            cols[vn] = None if views_only else d.set_index("key")[vn]
            del d
            gc.collect()
        if views_only:
            continue
        V = pd.concat(cols.values(), axis=1).fillna(0).astype(np.float32)
        del cols
        key = V.index.to_numpy()
        V = V.reset_index(drop=True)
        V.insert(0, "cand_id", b.entity_id.to_numpy()[key % nb])
        V.insert(0, "s1_id", a.entity_id.to_numpy()[key // nb])
        n_union = len(V)
        if cap:
            V = cap_candidates(V, *cap)
        log(f"[block] {c}: union {n_union:,} ({n_union / len(a):.1f}/S1) -> capped {len(V):,} ({len(V) / len(a):.1f}/S1)")
        out.append(V)
        gc.collect()
    if views_only:
        return None
    P = pd.concat(out, ignore_index=True)
    if (s1.ctry == "unk").any() or (cands.ctry == "unk").any():
        P = P.groupby(["s1_id", "cand_id"], as_index=False).max()
    return P


def blocking_report(pairs, gold_map, s1_ids=None, view_cols=None):
    """Pair recall, entity full-coverage and size per view and for the union, over s1_ids."""
    s1_ids = set(gold_map) if s1_ids is None else set(s1_ids)
    gold = {(s, c) for s, cs in gold_map.items() if s in s1_ids for c in cs}
    n_match_ent = sum(1 for s in s1_ids if gold_map.get(s))
    pairs = pairs[pairs.s1_id.isin(s1_ids)]
    view_cols = view_cols or [c for c in pairs.columns if c.startswith("v_")]
    rep = []

    def stats(name, sub):
        got = set(zip(sub.s1_id, sub.cand_id))
        hit = got & gold
        full = sum(1 for s in s1_ids if gold_map.get(s) and all((s, c) in got for c in gold_map[s]))
        rep.append(dict(view=name, pairs=len(got), pair_recall=len(hit) / max(len(gold), 1),
                        entity_full_cov=full / max(n_match_ent, 1), cands_per_s1=len(got) / max(len(s1_ids), 1)))
    for v in view_cols:
        stats(v, pairs[pairs[v] > 0])
    stats("UNION", pairs)
    return pd.DataFrame(rep)


def cap_candidates(P, n_s1=30, n_c=3, score_cols=None):
    """Final candidate set fed to the model. A pair survives if it is among the S1's top n_s1
    by blocking score OR among the record's top n_c S1s (the reverse direction keeps a record's
    true owner even when that S1 has many look-alike records). Blocking score = sum of view sims."""
    score_cols = score_cols or [c for c in P.columns if c.startswith("v_")]
    s = P[score_cols].sum(axis=1).to_numpy()
    P = P.assign(_s=s)
    r1 = P.groupby("s1_id")._s.rank(ascending=False, method="first")
    r2 = P.groupby("cand_id")._s.rank(ascending=False, method="first")
    return P[(r1 <= n_s1) | (r2 <= n_c)].drop(columns="_s").reset_index(drop=True)


def group_rank(g, v):
    """Rank of v (descending, ties broken by position) within groups g, plus the group max and
    group size. Pure numpy (sort once), ~20x faster than pandas groupby-rank on 1e8 rows."""
    order = np.lexsort((-v, g))
    gs = g[order]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    size = np.diff(np.r_[start, len(gs)])
    grp_of_sorted = np.repeat(np.arange(len(start)), size)
    rank_sorted = np.arange(len(gs)) - start[grp_of_sorted] + 1
    vmax_sorted = v[order][start][grp_of_sorted]
    rank = np.empty(len(g), np.float32)
    vmax = np.empty(len(g), np.float32)
    n = np.empty(len(g), np.float32)
    rank[order] = rank_sorted
    vmax[order] = vmax_sorted
    n[order] = size[grp_of_sorted]
    return rank, vmax, n


def rank_features(V, views, s1_col="s1_id", c_col="cand_id"):
    """Cheap per-pair features for the stage-1 ranker, from blocking scores only:
    each view's sim, its rank within the S1's list and within the record's list, the gaps to
    the best, the number of views that found the pair, and the list sizes."""
    g1 = V[s1_col].to_numpy()
    g2 = V[c_col].to_numpy()
    if g1.dtype == object:
        g1 = pd.factorize(g1)[0]
        g2 = pd.factorize(g2)[0]
    X = {}
    for v in views:
        x = V[v].to_numpy(np.float32)
        X[v] = x
        r1, m1, n1 = group_rank(g1, x)
        r2, m2, n2 = group_rank(g2, x)
        X[f"{v}_r1"], X[f"{v}_gap1"] = r1, m1 - x
        X[f"{v}_r2"], X[f"{v}_gap2"] = r2, m2 - x
    X["n_views"] = (V[views].to_numpy() > 0).sum(axis=1).astype(np.int8)
    X["n1"], X["n2"] = n1, n2
    return pd.DataFrame(X)


def cap_by_ranker(V, model, views, n_s1=25, n_c=2):
    """Keep a pair if it is in the S1's top n_s1 or the record's top n_c by the stage-1 ranker."""
    s = model.predict(rank_features(V, views)).astype(np.float32)
    g1, g2 = pd.factorize(V.s1_id)[0], pd.factorize(V.cand_id)[0]
    r1 = group_rank(g1, s)[0]
    r2 = group_rank(g2, s)[0]
    keep = (r1 <= n_s1) | (r2 <= n_c)
    return V[keep].assign(rk=s[keep]).reset_index(drop=True)
