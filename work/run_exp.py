"""Unified HONEST experiment driver (one switch per change, same S1s and folds as H1 / sub4_honest).

Training rows : ORIGINAL train pairs only (never the validation clones), decoy pairs (record owned by no
                S1) weighted 1+q (q: US 0.891, India 0.941) so the model sees test decoy density.
Yardstick     : the Test-Like Validation (TLV) rows of the same S1s, each S1 scored by the fold model that
                did not train on it, calibrated by that fold's isotonic map (fit on the fold's ORIGINAL rows,
                weighted), then one-owner (exactly one owner, tie on p broken by `rk`) and expected-F0.5
                decoding. Reported: macro F0.5 over all validation S1s, per country x {matched, singleton}.
LOCO mode     : LOCO=US (or India): train/calibrate on that country's S1s only (K folds), score the OTHER
                country's TLV rows with the fold-AVERAGE of calibrated models -- exactly the path test uses.
Switches (env): K=3 LR=0.1 THREADS=24 NS1=0(all) EXTRA=1 (density features) DROP=a,b MONO=1 STAGE2=1
                W8=1 (decoy weighting) SEED=0
usage: run_exp.py <name>"""
import sys, os, gc, time, json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, PARAMS
from er.features import add_density_features
from er.stage2 import sibling_features
from er.decode import decode_all
from er.io_metric import macro_fbeta, score_breakdown

W = os.environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v2/")
NAME = sys.argv[1]
E = os.environ.get
K, LR, TH, NS1 = int(E("K", 3)), float(E("LR", 0.1)), int(E("THREADS", 24)), int(E("NS1", 0))
EXTRA, MONO, STAGE2, W8, LOCO = E("EXTRA", "0") == "1", E("MONO", "0") == "1", E("STAGE2", "0") == "1", E("W8", "1") == "1", E("LOCO", "")
DROP = [x for x in E("DROP", "").split(",") if x]
CE = E("CE", "")   # parquet [s1_id, cand_id(base record), ce]: cross-encoder logit, NaN (missing) outside the band
Q = {"US": 0.891, "India": 0.941}
NONF = {"fold", "decoy", "wq", "base_id", "p1"}
params = {**PARAMS, "num_threads": TH, "learning_rate": LR, "seed": int(E("SEED", 0)), **json.loads(E("LGB", "{}"))}
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
log(f"config {NAME}: LGB={E('LGB', '{}')} TRAIN_TAG={E('TRAIN_TAG', 'r3_f2')} CE={CE or '-'} K={K} LR={LR} NS1={NS1 or 'all'} EXTRA={EXTRA} MONO={MONO} STAGE2={STAGE2} W8={W8} LOCO={LOCO or '-'} DROP={DROP}")

s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
owned = pd.Index([x for l in gt.matched_entity_ids for x in l.split(",") if x]); del gt
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0)
if NS1:
    ids = ids.iloc[:NS1]
perm = np.random.default_rng(0).permutation(len(ids)); fold_of = dict(zip(ids.to_numpy()[perm], np.arange(len(ids)) % K))
if LOCO:
    tr_ids = [s for s in ids if country_of[s] == LOCO]; ev_ids = [s for s in ids if country_of[s] != LOCO]
else:
    tr_ids, ev_ids = ids.tolist(), None

s1x = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "ctry", "name_n", "addr_words", "house"]) if EXTRA else None
cpx = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "name_n", "addr_words", "house", "addr", "nums"]) if (EXTRA or STAGE2) else None


CEDF = pd.read_parquet(CE, columns=["s1_id", "cand_id", "ce"]) if CE else None


def load(tag, which):
    F = pds.dataset(W + f"train_feats_{tag}").to_table(filter=pds.field("s1_id").isin(pa.array(which))).to_pandas()
    F = add_labels(F, g); F["fold"] = F.s1_id.map(fold_of).to_numpy(np.int8)
    F["base_id"] = F.cand_id.str.replace("~c", "", regex=False)
    F["decoy"] = ~F.base_id.isin(owned)
    F["wq"] = 1.0 + F.s1_id.map(country_of).map(Q).fillna(0.9).to_numpy()
    if EXTRA:
        add_density_features(F, s1x, cpx[cpx.entity_id.isin(set(F.base_id))])
    if CE:
        F = F.merge(CEDF.rename(columns={"cand_id": "base_id"}), on=["s1_id", "base_id"], how="left")
    return F


TRAIN_TAG = E("TRAIN_TAG", "r3_f2")                          # r3_f2 = original rows; tlvj3_f2 = tie-free test-like rows
F0 = load(TRAIN_TAG, tr_ids)                                 # training rows
FT = load("tlv3_f2", ev_ids if LOCO else tr_ids)             # scored rows (test-like)
w0 = np.where(F0.decoy.to_numpy() & (F0.y.to_numpy() == 0), F0.wq.to_numpy(), 1.0) if W8 else np.ones(len(F0))
feats1 = [c for c in feature_cols(F0) if c not in NONF and c not in DROP]
log(f"train rows {len(F0):,} ({F0.s1_id.nunique():,} S1), scored TLV rows {len(FT):,} ({FT.s1_id.nunique():,} S1), {len(feats1)} feats")

MONO_UP = {"n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_glued_ratio", "n_glued_partial", "n_sk_ratio", "n_idfj",
           "a_ratio", "a_tset", "a_tsort", "a_partial", "aw_tset", "aw_idfj", "v_word", "v_name5", "v_addr5", "rk",
           "na_prod", "na_min", "na_prod_margin_c", "rk_margin_c", "v_word_margin_c", "p1"}
MONO_DOWN = {"na_prod_gap_s1", "na_prod_gap_c", "rk_gap_s1", "rk_gap_c", "v_word_gap_s1", "v_word_gap_c", "p1_gap_s1", "p1_gap_c"}


def prm(feats):
    if not MONO:
        return params
    mc = [1 if f in MONO_UP else (-1 if f in MONO_DOWN else 0) for f in feats]
    return {**params, "monotone_constraints": mc, "monotone_constraints_method": "intermediate"}


iso_fit = lambda r, y, w: IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(r, y, sample_weight=w)


def run_stage(feats, label):
    """Returns OOF calibrated p on F0 (train rows) and p on FT (scored rows)."""
    p0 = np.zeros(len(F0), np.float32); pT = np.zeros(len(FT), np.float32)
    fo0, foT = F0.fold.to_numpy(), FT.fold.to_numpy()
    for k in range(K):
        tr, va = np.flatnonzero(fo0 != k), np.flatnonzero(fo0 == k)
        dtr = lgb.Dataset(F0[feats].iloc[tr], F0.y.iloc[tr], weight=w0[tr], free_raw_data=True)
        dv = lgb.Dataset(F0[feats].iloc[va], F0.y.iloc[va], weight=w0[va], reference=dtr)
        m = lgb.train(prm(feats), dtr, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        rv = m.predict(F0[feats].iloc[va], num_iteration=m.best_iteration, num_threads=TH)
        iso = iso_fit(rv, F0.y.iloc[va].to_numpy(), w0[va]); p0[va] = iso.predict(rv)
        tt = np.flatnonzero(foT == k) if not LOCO else np.arange(len(FT))
        pk = iso.predict(m.predict(FT[feats].iloc[tt], num_iteration=m.best_iteration, num_threads=TH))
        if LOCO:
            pT += (pk / K).astype(np.float32)          # other country: fold-average, as on test
        else:
            pT[tt] = pk
        log(f"{label} fold {k}: best_iter {m.best_iteration}")
        del m, dtr, dv; gc.collect()
    return p0, pT


def score(p, label):
    P = pd.DataFrame({"s1_id": FT.s1_id.values, "cand_id": FT.cand_id.values, "p": p, "rk": FT.rk.values})
    sids = ev_ids if LOCO else tr_ids
    pred = decode_all(P, sids, tie_col="rk")
    pred = {k: set(v) for k, v in pred.items()}
    gs = {s: g.get(s, set()) for s in sids}
    f = macro_fbeta(pred, gs); br = score_breakdown(pred, gs, country_of)
    log(f"RESULT {NAME} {label}: TLV F0.5 {f:.5f}")
    print(br.round(4)[["n", "mean_f", "lost_share"]].to_string(), flush=True)
    return f


p0, pT = run_stage(feats1, "s1")
res = {"s1": score(pT, "stage-1")}
if STAGE2:
    F0["p1"] = p0; FT["p1"] = pT
    cps = cpx[cpx.entity_id.isin(set(F0.base_id) | set(FT.base_id))][["entity_id", "name_n", "addr", "house", "nums"]]
    idx0, idxT = F0.index, FT.index
    F0 = sibling_features(F0, cps, base_col="base_id"); FT = sibling_features(FT, cps, base_col="base_id")
    feats2 = [c for c in feature_cols(F0) if c not in NONF and c not in DROP] + ["p1"]
    log(f"stage-2 feats {len(feats2)}")
    p0b, pTb = run_stage(feats2, "s2")
    res["s2"] = score(pTb, "two-stage")
    pd.DataFrame({"s1_id": FT.s1_id, "cand_id": FT.cand_id, "y": FT.y, "p1": pT, "p": pTb}).to_parquet(W + f"exp_{NAME}.parquet")
else:
    pd.DataFrame({"s1_id": FT.s1_id, "cand_id": FT.cand_id, "y": FT.y, "p": pT}).to_parquet(W + f"exp_{NAME}.parquet")
log(f"SUMMARY {NAME}: {json.dumps({k: round(v, 5) for k, v in res.items()})}")
