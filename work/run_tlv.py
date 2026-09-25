"""Two-stage experiment driver on the Test-Like Validation (TLV). One switch per planned change, so
every change is measured alone against the same baseline.

usage: run_tlv.py <feat_tag> <name>          e.g. run_tlv.py tlv_f2 T0
env switches (defaults = the sub3 method, i.e. the baseline):
  K=3 LR=0.1        folds / learning rate (fast experiment mode; final: K=5 LR=0.05)
  HONEST=0|1        1: early stopping on an inner 10% of the TRAINING S1s (not the scored fold) and
                    cross-fitted isotonic (fold k calibrated by a map fit on the other folds' OOF)
  S2POP=sample|full full: the record-side stage-2 features and the one-owner rule also see the pairs of
                    S1s outside the validation sample (stage-1 fold-average p, like test)
  TIE=0|1           1: one-owner keeps exactly one owner (ties broken by the raw stage-2 score)
  POUT=0|ctry       ctry: per-country P(an S1 has a true match outside its shortlist), measured on
                    train S1s outside the validation sample
  DROP=a,b          features to drop;   ONLY_S1=1: stop after stage 1
  TEST=<tag>        also predict test features <tag>, decode, write submissions/<name>/ and validate
Validation S1s: the same 551,789 non-ranker S1s as sub2/sub3 (frac .25/.7, random_state 0); folds are
a seeded random partition of S1s (clones never change an S1's fold)."""
import sys, os, gc, time, json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pds, lightgbm as lgb
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.model import add_labels, feature_cols, train_lgb, calibrate, PARAMS
from er.stage2 import sibling_features
from er.decode import decode_all, enforce_one_owner
from er.io_metric import macro_fbeta, score_breakdown, write_idlist_tsv, check_outputs

W = __import__("os").environ.get("CACHE", "/home/24b4518/ml-projects/work/cache/v1/"); ROOT = "/home/24b4518/ml-projects/"
tag, name = sys.argv[1], sys.argv[2]
E = os.environ.get
K, LR, HONEST, S2POP, TIE, POUT = int(E("K", 3)), float(E("LR", 0.1)), E("HONEST", "0") == "1", E("S2POP", "sample"), E("TIE", "0") == "1", E("POUT", "0")
DROP = [x for x in E("DROP", "").split(",") if x]; TEST = E("TEST", "")
params = {"num_threads": int(E("THREADS", 24)), "learning_rate": LR, **json.loads(E("LGB", "{}"))}
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
log(f"config: tag={tag} K={K} LR={LR} HONEST={HONEST} S2POP={S2POP} TIE={TIE} POUT={POUT} DROP={DROP} TEST={TEST}")

s1 = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "country"]); country_of = dict(zip(s1.entity_id, s1.country))
gt = pd.read_parquet(W + "../gt.parquet"); g = {s: set(x for x in v.split(",") if x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}; del gt
rank_s1 = set(pd.read_parquet(W + "ranker_s1.parquet").s1_id)
ids = s1.entity_id[~s1.entity_id.isin(rank_s1)].sample(frac=0.25 / 0.7, random_state=0)
idset = set(ids)
F = pds.dataset(W + f"train_feats_{tag}").to_table(filter=pds.field("s1_id").isin(pa.array(ids.tolist()))).to_pandas()
F = add_labels(F, g); F["base_id"] = F.cand_id.str.replace("~c", "", regex=False)
feats1 = [c for c in feature_cols(F) if c not in DROP]
perm = np.random.default_rng(0).permutation(len(ids)); fold_of = dict(zip(ids.to_numpy()[perm], np.arange(len(ids)) % K))
fo = F.s1_id.map(fold_of).to_numpy()
inner = pd.util.hash_array(F.s1_id.to_numpy()) % 10 == 0          # 10% of S1s: inner early-stopping set
log(f"validation: {len(F):,} pairs / {len(ids):,} S1, {len(feats1)} feats, clone pairs {F.cand_id.str.endswith('~c').mean():.3f}")


def fit_fold(X, y, k):
    tr = fo != k
    if HONEST:
        a, v = np.flatnonzero(tr & ~inner), np.flatnonzero(tr & inner)
    else:
        a, v = np.flatnonzero(tr), np.flatnonzero(~tr)
    return train_lgb(X.iloc[a], y.iloc[a], X.iloc[v], y.iloc[v], params=params)


def run_stage(F, feats, label):
    raw = np.zeros(len(F), np.float32); models = []
    for k in range(K):
        m = fit_fold(F[feats], F.y, k); b = np.flatnonzero(fo == k)
        raw[b] = m.predict(F[feats].iloc[b], num_iteration=m.best_iteration, num_threads=params["num_threads"])
        models.append(m); m.save_model(W + f"model_{name}_{label}_f{k}.txt")
        log(f"{label} fold {k}: best_iter {m.best_iteration}")
    iso = calibrate(raw, F.y)                          # pooled map: used for test and as the default
    if HONEST:                                         # cross-fitted OOF calibration
        p = np.zeros(len(F), np.float32)
        for k in range(K):
            b = fo == k; p[b] = calibrate(raw[~b], F.y[~b]).predict(raw[b])
    else:
        p = iso.predict(raw).astype(np.float32)
    return raw, p, iso, models


def avg_predict(models, X):
    out = np.zeros(len(X), np.float64)
    for m in models:
        for lo in range(0, len(X), 2_000_000):
            out[lo:lo + 2_000_000] += m.predict(X.iloc[lo:lo + 2_000_000], num_iteration=m.best_iteration, num_threads=params["num_threads"]) / len(models)
    return out.astype(np.float32)


def p_outside_map(s1_ids):
    """Per country: share of S1s with >=1 true match missing from the shortlist, measured on non-ranker
    train S1s OUTSIDE the validation sample (so it is not fit on the scored S1s)."""
    P = pds.dataset(W + f"train_feats_{tag}").to_table(columns=["s1_id", "cand_id"]).to_pandas()
    have = set(zip(P.s1_id, P.cand_id)); del P
    rate = {}
    for c in ["US", "India"]:
        pool = [s for s in s1.entity_id[(s1.country == c) & ~s1.entity_id.isin(rank_s1) & ~s1.entity_id.isin(idset)].tolist() if g.get(s)]
        miss = np.mean([any((s, x) not in have for x in g[s]) for s in pool[:200000]])
        rate[c] = float(miss)
    rate["France"] = max(rate.values())
    log(f"p_outside by country: {rate}")
    return rate


def evaluate(F, p, raw, comp=None, pout=None):
    """Decode with one-owner over F (+ competitor pairs of S1s outside the sample, if given), then score
    the sample S1s only."""
    P = pd.DataFrame({"s1_id": F.s1_id.values, "cand_id": F.cand_id.values, "p": p, "raw": raw})
    if comp is not None:
        P = pd.concat([P, comp[["s1_id", "cand_id", "p", "raw"]]], ignore_index=True)
    P = enforce_one_owner(P, "p", tie_col="raw" if TIE else None)
    P = P[P.s1_id.isin(idset)]
    po = {s: pout[country_of[s]] for s in ids} if pout else 0.0
    pred = decode_all(P, ids.tolist(), p_outside=po, one_owner=False)
    pred = {k: set(v) for k, v in pred.items()}
    gsub = {s: g.get(s, set()) for s in ids}
    return macro_fbeta(pred, gsub), score_breakdown(pred, gsub, country_of)


pout = p_outside_map(ids) if POUT == "ctry" else None
raw1, p1, iso1, M1 = run_stage(F, feats1, "s1")
F["p1"] = p1
comp = None
if S2POP == "full":   # pairs of S1s OUTSIDE the sample that share a record with the sample
    cands = pa.array(F.cand_id.unique().tolist())
    C = pds.dataset(W + f"train_feats_{tag}").to_table(filter=pds.field("cand_id").isin(cands)).to_pandas()
    C = C[~C.s1_id.isin(idset)].reset_index(drop=True)
    r = avg_predict(M1, C[feats1]); comp = pd.DataFrame({"s1_id": C.s1_id, "cand_id": C.cand_id, "raw": r, "p": iso1.predict(r).astype(np.float32)})
    del C; gc.collect(); log(f"competitor pairs (S1s outside the sample): {len(comp):,}")
f1, _ = evaluate(F, p1, raw1, comp, pout); log(f"stage-1 TLV F0.5 {f1:.5f}")
if E("ONLY_S1") == "1":
    sys.exit()
cp_tr = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
F = sibling_features(F, cp_tr[cp_tr.entity_id.isin(set(F.base_id))], base_col="base_id",
                     ext=comp[["cand_id", "p"]] if comp is not None else None); del cp_tr; gc.collect()
feats2 = [c for c in feature_cols(F) if c not in DROP and c != "p1"] + ["p1"]
raw2, p2, iso2, M2 = run_stage(F, feats2, "s2")
f2, br = evaluate(F, p2, raw2, comp, pout); log(f"stage-2 TLV F0.5 {f2:.5f}"); print(br.round(4).to_string(), flush=True)
pd.DataFrame({"s1_id": F.s1_id, "cand_id": F.cand_id, "y": F.y, "fold": fo, "raw1": raw1, "p1": F.p1, "raw2": raw2, "p": p2}).to_parquet(W + f"oof_{name}.parquet")
if comp is not None:
    comp.to_parquet(W + f"comp_{name}.parquet")
if not TEST:
    sys.exit()
del F; gc.collect()
Te = pd.read_parquet(W + f"test_feats_{TEST}")
tr1 = avg_predict(M1, Te[feats1]); Te["p1"] = iso1.predict(tr1).astype(np.float32); log("test stage 1")
cp_te = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"])
Te = sibling_features(Te, cp_te[cp_te.entity_id.isin(set(Te.cand_id))]); del cp_te; gc.collect()
tr2 = avg_predict(M2, Te[feats2]); Te["p"] = iso2.predict(tr2).astype(np.float32); Te["raw"] = tr2; log("test stage 2")
ts1 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"]); test_ids = ts1.entity_id.tolist()
Te[["s1_id", "cand_id", "p1", "p", "raw"]].to_parquet(W + f"test_pred_{name}.parquet")
tc = dict(zip(ts1.entity_id, ts1.country))
po = {s: pout.get(tc[s], max(pout.values())) for s in test_ids} if pout else 0.0
pred = decode_all(Te[["s1_id", "cand_id", "p", "raw"]], test_ids, p_outside=po, tie_col="raw" if TIE else None); log("decoded")
out = ROOT + f"submissions/{name}/"; os.makedirs(out, exist_ok=True)
write_idlist_tsv(pred, test_ids, out + "matching_results.tsv")
write_idlist_tsv(Te.groupby("s1_id").cand_id.apply(list).to_dict(), test_ids, out + "candidate_pairs.tsv", col="candidate_entity_ids")
tcp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id"])
log(f"check_outputs: {check_outputs(out + 'matching_results.tsv', out + 'candidate_pairs.tsv', test_ids, tcp.entity_id)}")
mon = pd.DataFrame({"country": [tc[s] for s in test_ids], "n": [len(pred[s]) for s in test_ids]})
print(mon.groupby("country").n.agg(empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string(), flush=True)
