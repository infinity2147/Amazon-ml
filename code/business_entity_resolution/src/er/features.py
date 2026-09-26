"""Pair features. Every feature is country-agnostic in meaning (a similarity, an agree /
conflict / missing state, a rank, a rarity) so it transfers to France. Built in parallel chunks.

Groups (motivated by the data deep dive, NOTES.md):
  name     : fuzzy similarities, no-space ('glued website') similarity, acronym, alt-name (the
             pseudo-word before 'DBA'), skeleton, IDF overlap, pseudo-word flag (tokens unseen in S1)
  address  : fuzzy similarities, IDF overlap of street words, NUMBER SETS compared fuzzily
             (fake extra numbers, digit drops and +-1 are generator noise), postal/unit/state
  record   : empty address, native script, source S3
  joint    : name x address agreement (chains share names, buildings share addresses)
  context  : rank / gap / margin of the pair within the S1's list and within the record's list
"""
import math
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

L_COLS = ["name_n", "legal", "addr", "addr_words", "state", "postal", "house", "units", "nums",
          "name_sk", "name_acr", "name_nums", "landmark", "business_name"]
R_COLS = L_COLS + ["name_alt", "web", "native", "business_address"]

_IDF = {}


def _init(idf):
    global _IDF
    _IDF = idf


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=1).astype(np.float32)


def _tri(a, b):
    out = np.zeros(len(a), np.int8)
    for k, (x, y) in enumerate(zip(a, b)):
        if x and y:
            out[k] = 1 if set(x.split()) & set(y.split()) else -1
    return out


def _idf_overlap(a, b, ctry, field):
    j = np.zeros(len(a), np.float32)
    w = np.zeros(len(a), np.float32)
    mx = np.zeros(len(a), np.float32)
    for k, (x, y, c) in enumerate(zip(a, b, ctry)):
        idf, dflt = _IDF.get((c, field), ({}, 10.0))
        X, Y = set(x.split()), set(y.split())
        if not X or not Y:
            continue
        inter = [idf.get(t, dflt) for t in X & Y]
        uni = sum(idf.get(t, dflt) for t in X | Y)
        j[k] = sum(inter) / uni if uni else 0
        w[k] = sum(inter)
        mx[k] = max(inter) if inter else 0
    return j, w, mx


def _known_frac(names, ctry):
    """Share of a record's name tokens that occur in S1 names of its country (pseudo-word detector)."""
    out = np.zeros(len(names), np.float32)
    for k, (x, c) in enumerate(zip(names, ctry)):
        voc = _IDF.get((c, "s1vocab"), set())
        t = x.split()
        out[k] = sum(1 for u in t if u in voc) / len(t) if t else 0
    return out


def _num_close(a, b):
    """a, b digit strings. 1 exact; .8 one is a prefix/suffix of the other (digit drop);
    .6 |diff| <= 10; else 0."""
    if a == b:
        return 1.0
    if len(a) >= 2 and len(b) >= 2 and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)):
        return 0.8
    if abs(int(a[:9]) - int(b[:9])) <= 10:
        return 0.6
    return 0.0


def _num_feats(la, ra, lh, rh):
    """Number-set agreement. The generator both perturbs house numbers of true matches (digit drop,
    +-1, zero-pad, fake extra number) and creates decoys that are copies of a real business with a
    slightly different house number, so we expose the kind of difference, not just a closeness."""
    n = len(la)
    cover = np.full(n, -1, np.float32)     # mean over S1 numbers of best closeness in record (-1 = missing)
    exact = np.full(n, -1, np.float32)     # share of S1 numbers found exactly
    extra = np.zeros(n, np.float32)        # record numbers with no counterpart in S1
    house = np.full(n, -1, np.float32)     # S1 house number vs best record number
    h_eq = np.full(n, -1, np.int8)         # S1 house number present exactly in record numbers
    h_diff = np.full(n, -1, np.float32)    # log1p min |S1 house - record number|
    h_drop = np.full(n, -1, np.int8)       # some record number = S1 house with one digit dropped
    h_first = np.full(n, -1, np.int8)      # record's house number (first number by position) == S1 house
    for k in range(n):
        A, B = la[k].split(), ra[k].split()
        if A and B:
            best = [max(_num_close(x, y) for y in B) for x in A]
            cover[k] = sum(best) / len(A)
            exact[k] = sum(1 for x in A if x in B) / len(A)
            extra[k] = sum(1 for y in B if max(_num_close(x, y) for x in A) == 0)
            h = lh[k]
            if h:
                house[k] = max(_num_close(h, y) for y in B)
                h_eq[k] = int(h in B)
                hv = int(h[:9])
                h_diff[k] = math.log1p(min(abs(hv - int(y[:9])) for y in B))
                h_drop[k] = int(any(len(y) == len(h) - 1 and any(h[:i] + h[i + 1:] == y for i in range(len(h))) for y in B))
                h_first[k] = int(rh[k] == h) if rh[k] else -1   # record's own house number (by position)
    return cover, exact, extra, house, h_eq, h_diff, h_drop, h_first


def _nospace(s):
    return s.replace(" ", "")


def _chunk_features(args):
    L, R, V = args
    F = {}
    ln, rn = L.name_n.tolist(), R.name_n.tolist()
    la, ra = L.addr.tolist(), R.addr.tolist()
    ctry = L.ctry.tolist()
    F["n_ratio"] = _cp(fuzz.ratio, ln, rn)
    F["n_tset"] = _cp(fuzz.token_set_ratio, ln, rn)
    F["n_tsort"] = _cp(fuzz.token_sort_ratio, ln, rn)
    F["n_partial"] = _cp(fuzz.partial_ratio, ln, rn)
    F["n_jw"] = _cp(JaroWinkler.normalized_similarity, ln, rn)
    lg = [_nospace(x) for x in ln]
    rg = [_nospace(w or x) for x, w in zip(rn, R.web.tolist())]
    F["n_glued_ratio"] = _cp(fuzz.ratio, lg, rg)
    F["n_glued_partial"] = _cp(fuzz.partial_ratio, lg, rg)
    F["n_sk_ratio"] = _cp(fuzz.token_sort_ratio, L.name_sk.tolist(), R.name_sk.tolist())
    alt = R.name_alt.tolist()
    F["n_alt_tset"] = np.where([bool(x) for x in alt], _cp(fuzz.token_set_ratio, ln, alt), -1).astype(np.float32)
    F["has_alt"] = np.array([bool(x) for x in alt], np.int8)
    F["n_acr_hit"] = np.array([int(len(a) > 1 and (a == y.replace(" ", "") or b == x.replace(" ", "")))
                               for a, b, x, y in zip(L.name_acr, R.name_acr, ln, rn)], np.int8)
    F["n_idfj"], F["n_idfw"], F["n_idfmax"] = _idf_overlap(ln, rn, ctry, "name")
    F["r_known_frac"] = _known_frac(rn, ctry)
    F["r_name_ntok"] = np.array([len(x.split()) for x in rn], np.int8)
    F["n_len_ratio"] = np.array([min(len(x), len(y)) / max(len(x), len(y), 1) for x, y in zip(ln, rn)], np.float32)
    # address
    F["a_ratio"] = _cp(fuzz.ratio, la, ra)
    F["a_tset"] = _cp(fuzz.token_set_ratio, la, ra)
    F["a_tsort"] = _cp(fuzz.token_sort_ratio, la, ra)
    F["a_partial"] = _cp(fuzz.partial_ratio, la, ra)
    law, raw_ = L.addr_words.tolist(), R.addr_words.tolist()
    F["aw_tset"] = _cp(fuzz.token_set_ratio, law, raw_)
    F["aw_idfj"], F["aw_idfw"], F["aw_idfmax"] = _idf_overlap(law, raw_, ctry, "addr")
    (F["num_cover"], F["num_exact"], F["num_extra"], F["house_close"], F["house_eq"], F["house_diff"],
     F["house_drop"], F["house_first"]) = _num_feats(L.nums.tolist(), R.nums.tolist(), L.house.tolist(), R.house.tolist())
    F["l_nnums"] = np.array([len(x.split()) for x in L.nums], np.int8)
    F["r_nnums"] = np.array([len(x.split()) for x in R.nums], np.int8)
    # state agreement through the unsupervised alias map (texas->tx, gironde->nouvelle aquitaine)
    smap = _IDF.get("state_map", {})
    F["ts_state"] = _tri(L.state.tolist(), [smap.get((c, x), x) for c, x in zip(ctry, R.state.tolist())])
    for col in ["postal", "units", "legal", "name_nums"]:
        F[f"ts_{col}"] = _tri(L[col].tolist(), R[col].tolist())
    F["r_addr_empty"] = (R.business_address.str.len() == 0).to_numpy().astype(np.int8)
    F["r_native"] = R.native.to_numpy().astype(np.int8)
    F["lm_either"] = ((L.landmark != "") | (R.landmark != "")).to_numpy().astype(np.int8)
    F = pd.DataFrame(F)
    F["na_prod"] = F.n_tset * F.a_tset / 100.0
    F["na_min"] = np.minimum(F.n_tset, F.a_tset)
    F["na_max"] = np.maximum(F.n_tset, F.a_tset)
    return pd.concat([V.reset_index(drop=True), F], axis=1)


def build_state_map(pairs, s1p, cp, min_count=50, min_purity=0.8):
    """Unsupervised alias map for admin areas: among confident blocking pairs (each record's top
    ranker pair with rk >= .9), count (record state spelling, S1 state) and keep majority mappings.
    No labels: works the same on train and on test (France departments -> regions)."""
    if "rk" not in pairs.columns:
        return {}
    P = pairs[pairs.rk >= 0.9]
    P = P.loc[P.groupby("cand_id").rk.idxmax()]
    a = s1p.set_index("entity_id").loc[P.s1_id, ["state", "ctry"]].to_numpy()
    b = cp.set_index("entity_id").loc[P.cand_id, "state"].to_numpy()
    D = pd.DataFrame({"ctry": a[:, 1], "l": a[:, 0], "r": b})
    D = D[(D.l != "") & (D.r != "")]
    cnt = D.groupby(["ctry", "r", "l"]).size().rename("n").reset_index()
    tot = cnt.groupby(["ctry", "r"]).n.transform("sum")
    cnt = cnt[(cnt.n >= min_count) & (cnt.n / tot >= min_purity)]
    return {(c, r): l for c, r, l in zip(cnt.ctry, cnt.r, cnt.l)}


def build_idf(s1p, cp):
    """Transductive IDF tables per country over all records of the split (no labels)."""
    idf = {}
    for c in set(s1p.ctry) | set(cp.ctry):
        a, b = s1p[s1p.ctry == c], cp[cp.ctry == c]
        for field, col in [("name", "name_n"), ("addr", "addr_words")]:
            cnt = Counter()
            for s in pd.concat([a[col], b[col]]).tolist():
                cnt.update(set(s.split()))
            n = len(a) + len(b) + 1
            idf[(c, field)] = ({t: math.log(n / (1 + k)) for t, k in cnt.items()}, math.log(n))
        voc = Counter()
        for s in a.name_n.tolist():
            voc.update(set(s.split()))
        idf[(c, "s1vocab")] = {t for t, k in voc.items()}
    return idf


def pair_features(pairs, s1p, cp, idf, n_jobs=32, chunk=100000, batch=64, log=None):
    """pairs: [s1_id, cand_id, v_* ...]; s1p/cp prepared frames. Returns features (same row order).
    Workers start from a clean forkserver (a forked child would copy the parent's Python string
    objects page by page through refcount writes), receive the IDF tables once, and get chunks
    in bounded batches so only `batch` chunks of strings are in flight at a time."""
    import multiprocessing as mp
    si = pd.Index(s1p.entity_id).get_indexer(pairs.s1_id)
    ci = pd.Index(cp.entity_id).get_indexer(pairs.cand_id)
    assert (si >= 0).all() and (ci >= 0).all(), "pair ids not found in prepared frames"
    V = pairs.reset_index(drop=True)
    V = V.assign(src3=V.cand_id.str.startswith("S3-").to_numpy().astype(np.int8))
    Lc = {c: s1p[c].to_numpy() for c in L_COLS + ["ctry"]}
    Rc = {c: cp[c].to_numpy() for c in R_COLS}

    def make(lo, hi):
        L = pd.DataFrame({c: v[si[lo:hi]] for c, v in Lc.items()})
        R = pd.DataFrame({c: v[ci[lo:hi]] for c, v in Rc.items()})
        return L, R, V.iloc[lo:hi]
    bounds = [(i, min(i + chunk, len(V))) for i in range(0, len(V), chunk)]
    parts = []
    with mp.get_context("forkserver").Pool(n_jobs, initializer=_init, initargs=(idf,)) as pool:
        for k in range(0, len(bounds), batch):
            parts += pool.map(_chunk_features, [make(lo, hi) for lo, hi in bounds[k:k + batch]], chunksize=1)
            if log:
                log(f"[features] {min(k + batch, len(bounds))}/{len(bounds)} chunks")
    return add_context(pd.concat(parts, ignore_index=True))


def _group_second(g, v):
    """Second-largest v in each group (0 if the group has one member), broadcast to rows."""
    order = np.lexsort((-v, g))
    gs = g[order]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    size = np.diff(np.r_[start, len(gs)])
    vs = v[order]
    sec = np.where(size > 1, vs[np.minimum(start + 1, len(vs) - 1)], 0).astype(np.float32)
    out = np.empty(len(g), np.float32)
    out[order] = np.repeat(sec, size)
    return out


def add_context(F, keys=("na_prod", "v_word", "rk")):
    """Competition features: 'is this the best S1 for this record, and by how much?'"""
    from .blocking import group_rank
    g1 = pd.factorize(F.s1_id)[0]
    g2 = pd.factorize(F.cand_id)[0]
    for key in [k for k in keys if k in F.columns]:
        x = F[key].to_numpy(np.float32)
        r1, m1, n1 = group_rank(g1, x)
        r2, m2, n2 = group_rank(g2, x)
        F[f"{key}_rank_s1"], F[f"{key}_gap_s1"] = r1, m1 - x
        F[f"{key}_rank_c"], F[f"{key}_gap_c"] = r2, m2 - x
        F[f"{key}_margin_c"] = x - _group_second(g2, x)
    F["n_cands_s1"], F["n_s1_for_c"] = n1, n2
    # twin context: among this S1's candidates whose name nearly equals the S1 name, how many carry
    # the S1 house number exactly? A name-twin without it, next to twins with it, looks like a decoy.
    if "house_eq" in F.columns:
        twin = (F.n_tset.to_numpy() >= 85)
        heq = (F.house_eq.to_numpy() == 1)
        n_twin = np.bincount(g1, weights=twin, minlength=g1.max() + 1)
        n_twin_heq = np.bincount(g1, weights=twin & heq, minlength=g1.max() + 1)
        F["s1_n_twins"] = n_twin[g1].astype(np.float32)
        F["s1_n_twins_heq"] = n_twin_heq[g1].astype(np.float32)
    return F


def add_density_features(F, s1p, cp):
    """Ambiguity ('density') features, country-neutral in meaning: how many S1 businesses of the same
    country share this name, or this street address. A name shared by 200 S1s (a chain, or France's
    generic 'Bordeaux Club SARL') is weak evidence; an address shared by several S1s (a shared building,
    common in France) makes 'same address' weak evidence. Also a one-digit-substitution house flag
    (a frequent near-twin decoy operator). Text-only: identical for a record and its validation clone.
    F: s1_id, cand_id (clones end in '~c'); s1p: entity_id, ctry, name_n, addr_words, house;
    cp: entity_id, name_n, addr_words, house. Adds dn_* and house_sub1 in place and returns F."""
    base = F.cand_id.str.replace("~c", "", regex=False)
    S = s1p.set_index("entity_id").loc[F.s1_id, ["ctry", "name_n", "addr_words", "house"]].to_numpy()
    C = cp.set_index("entity_id").loc[base, ["name_n", "addr_words", "house"]].to_numpy()
    ctry = S[:, 0]
    k_name = pd.Series(s1p.ctry + "|" + s1p.name_n).value_counts()
    has_a = (s1p.addr_words != "") & (s1p.house != "")
    k_addr = pd.Series((s1p.ctry + "|" + s1p.addr_words + "|" + s1p.house)[has_a]).value_counts()
    k_street = pd.Series((s1p.ctry + "|" + s1p.addr_words)[s1p.addr_words != ""]).value_counts()

    def cnt(series, keys, valid):
        v = pd.Series(keys).map(series).fillna(0).to_numpy(np.float32)
        return np.where(valid, np.log1p(v), -1).astype(np.float32)
    F["dn_s1_name"] = cnt(k_name, ctry + "|" + S[:, 1], np.ones(len(F), bool))
    F["dn_c_name"] = cnt(k_name, ctry + "|" + C[:, 0], C[:, 0] != "")
    F["dn_s1_addr"] = cnt(k_addr, ctry + "|" + S[:, 2] + "|" + S[:, 3], (S[:, 2] != "") & (S[:, 3] != ""))
    F["dn_c_addr"] = cnt(k_addr, ctry + "|" + C[:, 1] + "|" + C[:, 2], (C[:, 1] != "") & (C[:, 2] != ""))
    F["dn_c_street"] = cnt(k_street, ctry + "|" + C[:, 1], C[:, 1] != "")
    h, r = S[:, 3], C[:, 2]
    F["house_sub1"] = np.array([(-1 if not a or not b else int(len(a) == len(b) and a != b and sum(x != y for x, y in zip(a, b)) == 1))
                                for a, b in zip(h, r)], np.int8)
    return F
