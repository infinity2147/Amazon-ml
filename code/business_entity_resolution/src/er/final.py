"""Final model: honest test-density training on all usable train S1s, untouched audit score, test prediction.

Populations (train S1s):
  ranker S1s (30%)   used only to train the shortlist ranker; never trained on here (their candidate lists
                     were cut by a model that saw their labels). They still take part as COMPETITORS.
  audit S1s          10% of all train S1s, drawn once (seeded) from non-ranker S1s outside the historical
                     551,789-S1 validation sample; never used for training, early stopping, calibration or any
                     choice. Scored ONCE, exactly as test is scored.
  pool S1s           every other train S1: the training data (K folds grouped by S1).
Stage 1: LightGBM on ORIGINAL pairs of pool S1s; decoy pairs (record owned by no S1) weighted 1+q, the test/
  train decoy-density ratio minus one (q: US 0.891, India 0.941; France uses the mean). Fold k is early-stopped
  on and calibrated (weighted isotonic) with its own held-out S1s. Prediction for an S1 in the pool = its own
  out-of-fold model; for any other S1 (ranker, audit, test) = mean over k of iso_k(model_k(x)).
Stage 2 (optional): sibling features from stage-1 p, with every OTHER train S1 of the same record present as a
  competitor (record-side features see the whole population, as on test), then the same K-fold training.
Audit: stage-1 and two-stage macro F0.5 on the audit S1s' Test-Like-Validation rows (decoys cloned to test
  density), decoded with one-owner (exactly one owner per record) + expected F0.5.
Artifacts in WORK: audit_s1.parquet, final_{name}_* models/isotonic maps, test_pred_{name}.parquet, report."""
import gc
import json
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from .decode import decode_all
from .features import add_density_features
from .io_metric import macro_fbeta, score_breakdown
from .model import PARAMS, add_labels, feature_cols
from .stage2 import sibling_features

Q = {"US": 0.891, "India": 0.941}
NONF = {"fold", "decoy", "wq", "base_id", "p1", "y", "s1_id", "cand_id"}


def _iso(r, y, w):
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(r, y, sample_weight=w)


def populations(W, gt_path, log, audit_frac=0.10, seed=2026, pool_all=False):
    s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"])
    rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
    nonrank = s1.entity_id[~s1.entity_id.isin(rank_s1)]
    val = set(nonrank.sample(frac=0.25 / 0.7, random_state=0))          # historical validation sample
    ap = W + "audit_s1.parquet"
    if os.path.exists(ap):
        audit = set(pd.read_parquet(ap).s1_id)
    else:
        cand = nonrank[~nonrank.isin(val)].to_numpy()
        n = int(round(audit_frac * len(s1)))
        audit = set(np.random.default_rng(seed).choice(cand, n, replace=False))
        pd.DataFrame({"s1_id": sorted(audit)}).to_parquet(ap)
    # pool_all: shortlists come from the out-of-fold ranker (no S1's list was cut by a model that saw its
    # labels), so the former ranker S1s are valid training data too. The audit set is unchanged.
    pool = (s1.entity_id if pool_all else nonrank)
    pool = pool[~pool.isin(audit)].tolist()
    if pool_all:
        rank_s1 = set()
    log(f"populations: train S1 {len(s1):,} | ranker {len(rank_s1):,} | audit {len(audit):,} | pool {len(pool):,}")
    return s1, rank_s1, audit, pool


def run_final(W, name, gt_path, log, K=3, lr=0.1, threads=24, extra=False, stage2=True, seed=0, pool_all=False, ce_path=None):
    t0 = time.time()
    s1, rank_s1, audit, pool = populations(W, gt_path, log, pool_all=pool_all)
    country_of = dict(zip(s1.entity_id, s1.country))
    gt = pd.read_parquet(gt_path)
    g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    owned = pd.Index([x for l in gt.matched_entity_ids for x in l.split(",") if x]); del gt
    qmean = float(np.mean(list(Q.values())))
    perm = np.random.default_rng(seed).permutation(len(pool))
    fold_of = dict(zip(np.asarray(pool)[perm], np.arange(len(pool)) % K))
    params = {**PARAMS, "num_threads": threads, "learning_rate": lr, "seed": seed, "deterministic": True, "force_row_wise": True}
    s1x = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "ctry", "name_n", "addr_words", "house"]) if extra else None
    # cross-encoder logit per (S1, base record); NaN (missing) outside the uncertain band -> LightGBM missing branch
    CEDF = pd.read_parquet(ce_path, columns=["s1_id", "cand_id", "ce"]).rename(columns={"cand_id": "base_id"}) if ce_path else None
    cpx = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "name_n", "addr_words", "house", "addr", "nums"])

    def prep(F, labelled=True):
        F["base_id"] = F.cand_id.str.replace("~c", "", regex=False)
        if labelled:
            F = add_labels(F, g)
            F["decoy"] = ~F.base_id.isin(owned)
            F["wq"] = 1.0 + F.s1_id.map(country_of).map(Q).fillna(qmean).to_numpy()
        F["fold"] = F.s1_id.map(fold_of).fillna(-1).to_numpy(np.int8)
        if extra:
            add_density_features(F, s1x, cpx[cpx.entity_id.isin(set(F.base_id))])
        if CEDF is not None:
            F = F.merge(CEDF, on=["s1_id", "base_id"], how="left")
        return F

    def load(tag, s1_filter=None, cand_filter=None):
        flt = None
        if s1_filter is not None:
            flt = pds.field("s1_id").isin(pa.array(list(s1_filter)))
        if cand_filter is not None:
            f2 = pds.field("cand_id").isin(pa.array(list(cand_filter)))
            flt = f2 if flt is None else (flt & f2)
        return pds.dataset(W + f"train_feats_{tag}").to_table(filter=flt).to_pandas()

    # ---------------- stage 1 ----------------
    F0 = prep(load("r3_f2", s1_filter=pool))
    w0 = np.where(F0.decoy.to_numpy() & (F0.y.to_numpy() == 0), F0.wq.to_numpy(), 1.0)
    feats1 = [c for c in feature_cols(F0) if c not in NONF]
    log(f"stage-1 training rows {len(F0):,} ({F0.s1_id.nunique():,} S1), {len(feats1)} features, positives {F0.y.mean():.3f}")

    def fit_stage(F, feats, label):
        """K models; returns OOF calibrated p on F and the list of (model, iso)."""
        fo = F.fold.to_numpy(); p = np.zeros(len(F), np.float32); ms = []
        for k in range(K):
            tr, va = np.flatnonzero(fo != k), np.flatnonzero(fo == k)
            dtr = lgb.Dataset(F[feats].iloc[tr], F.y.iloc[tr], weight=w0[tr], free_raw_data=True)
            dv = lgb.Dataset(F[feats].iloc[va], F.y.iloc[va], weight=w0[va], reference=dtr)
            m = lgb.train(params, dtr, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
            r = m.predict(F[feats].iloc[va], num_iteration=m.best_iteration, num_threads=threads)
            iso = _iso(r, F.y.iloc[va].to_numpy(), w0[va]); p[va] = iso.predict(r)
            m.save_model(W + f"final_{name}_{label}_f{k}.txt", num_iteration=m.best_iteration)
            ms.append((m, iso)); log(f"{label} fold {k}: best_iter {m.best_iteration} ({time.time()-t0:.0f}s)")
            del dtr, dv; gc.collect()
        pickle.dump({"isos": [i for _, i in ms], "feats": feats}, open(W + f"final_{name}_{label}_iso.pkl", "wb"))
        return p, ms

    def predict(ms, F, feats, own_fold=False):
        """own_fold: rows of pool S1s use their out-of-fold model; other rows (fold -1) use the average."""
        X = F[feats]; p = np.zeros(len(F), np.float64)
        fo = F.fold.to_numpy() if own_fold else np.full(len(F), -1)
        avg = np.flatnonzero(fo < 0)
        for k, (m, iso) in enumerate(ms):
            for lo in range(0, len(avg), 2_000_000):
                ix = avg[lo:lo + 2_000_000]
                p[ix] += iso.predict(m.predict(X.iloc[ix], num_iteration=m.best_iteration, num_threads=threads)) / len(ms)
            own = np.flatnonzero(fo == k)
            if len(own):
                p[own] = iso.predict(m.predict(X.iloc[own], num_iteration=m.best_iteration, num_threads=threads))
        return p.astype(np.float32)

    p0, M1 = fit_stage(F0, feats1, "s1")
    F0["p1"] = p0
    F0[["s1_id", "cand_id", "p1", "y"]].to_parquet(W + f"oof_{name}_s1.parquet")      # band for the cross-encoder

    # audit rows (test-like: decoys cloned to test density) and their competitors
    FA = prep(load("tlv3_f2", s1_filter=audit))
    FA["p1"] = predict(M1, FA, feats1)
    log(f"audit TLV rows {len(FA):,} ({FA.s1_id.nunique():,} S1)")
    res = {}

    def audit_score(p, label):
        P = pd.DataFrame({"s1_id": FA.s1_id.values, "cand_id": FA.cand_id.values, "p": p, "rk": FA.rk.values})
        aid = sorted(audit)
        pred = {k: set(v) for k, v in decode_all(P, aid, tie_col="rk").items()}
        gs = {s: g.get(s, set()) for s in aid}
        f = macro_fbeta(pred, gs); br = score_breakdown(pred, gs, country_of)
        log(f"AUDIT {name} {label}: F0.5 {f:.5f}")
        print(br.round(4)[["n", "mean_f", "lost_share"]].to_string(), flush=True)
        res[label] = {"f05": round(f, 5), "by": {f"{a}|{b}": round(v, 4) for (a, b), v in br.mean_f.items()}}
        return f

    audit_score(FA.p1.to_numpy(), "stage1")
    FA[["s1_id", "cand_id", "p1", "y"]].to_parquet(W + f"audit_pred_{name}_s1.parquet")
    ext_for = {}
    if stage2:
        # competitors: pairs of OTHER train S1s on the same records (orig rows for training, TLV rows for audit)
        def competitors(tag, cands, exclude):
            C = prep(load(tag, cand_filter=cands), labelled=False)
            C = C[~C.s1_id.isin(exclude)].reset_index(drop=True)
            C["p"] = predict(M1, C, feats1, own_fold=True)
            return C[["cand_id", "p"]]
        e0 = competitors("r3_f2", set(F0.cand_id), set(pool))
        eA = competitors("tlv3_f2", set(FA.cand_id), set(audit))
        log(f"competitor pairs: training {len(e0):,}, audit {len(eA):,}")
        cps = cpx[cpx.entity_id.isin(set(F0.base_id) | set(FA.base_id))][["entity_id", "name_n", "addr", "house", "nums"]]
        F0 = sibling_features(F0, cps, base_col="base_id", ext=e0)
        FA = sibling_features(FA, cps, base_col="base_id", ext=eA); del e0, eA, cps; gc.collect()
        feats2 = [c for c in feature_cols(F0) if c not in NONF] + ["p1"]
        p02, M2 = fit_stage(F0, feats2, "s2")
        FA["p2"] = predict(M2, FA, feats2)
        audit_score(FA.p2.to_numpy(), "two_stage")
    else:
        M2, feats2 = None, None
    del F0; gc.collect()
    json.dump({"name": name, "K": K, "lr": lr, "extra": extra, "stage2": stage2, "audit": res,
               "n_pool": len(pool), "n_audit": len(audit)}, open(W + f"final_{name}_report.json", "w"), indent=1)
    return {"M1": M1, "feats1": feats1, "M2": M2, "feats2": feats2, "cpx": cpx, "s1x": s1x, "res": res}


def load_ce_test(ce_test_path):
    return pd.read_parquet(ce_test_path, columns=["s1_id", "cand_id", "ce"]) if ce_test_path else None


def predict_test(W, name, state, log, threads=24, extra=False, ce_test_path=None):
    """Test: every S1 is out-of-sample -> mean over folds of iso_k(model_k(x)) for both stages; stage-2
    sibling features use all test S1s' stage-1 p (the whole population, as trained)."""
    M1, feats1, M2, feats2 = state["M1"], state["feats1"], state["M2"], state["feats2"]
    parts = []
    tcp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id", "name_n", "addr_words", "house", "addr", "nums"])
    ts1x = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "ctry", "name_n", "addr_words", "house"]) if extra else None
    for f in sorted(os.listdir(W + "test_feats_r3_f2")):
        T = pq.read_table(W + "test_feats_r3_f2/" + f).to_pandas()
        T["base_id"] = T.cand_id
        if extra:
            add_density_features(T, ts1x, tcp[tcp.entity_id.isin(set(T.cand_id))])
        if ce_test_path:
            T = T.merge(load_ce_test(ce_test_path), on=["s1_id", "cand_id"], how="left")
        p = np.zeros(len(T), np.float64)
        for m, iso in M1:
            for lo in range(0, len(T), 2_000_000):
                p[lo:lo + 2_000_000] += iso.predict(m.predict(T[feats1].iloc[lo:lo + 2_000_000], num_iteration=m.best_iteration, num_threads=threads)) / len(M1)
        T["p1"] = p.astype(np.float32)
        if M2 is not None:
            T = sibling_features(T, tcp[tcp.entity_id.isin(set(T.cand_id))][["entity_id", "name_n", "addr", "house", "nums"]])
            p = np.zeros(len(T), np.float64)
            for m, iso in M2:
                for lo in range(0, len(T), 2_000_000):
                    p[lo:lo + 2_000_000] += iso.predict(m.predict(T[feats2].iloc[lo:lo + 2_000_000], num_iteration=m.best_iteration, num_threads=threads)) / len(M2)
            T["p"] = p.astype(np.float32)
        else:
            T["p"] = T.p1
        parts.append(T[["s1_id", "cand_id", "rk", "p1", "p"]]); log(f"test predicted {f}: {len(T):,} pairs")
        del T; gc.collect()
    P = pd.concat(parts, ignore_index=True)
    P.to_parquet(W + f"test_pred_{name}.parquet")
    return P
