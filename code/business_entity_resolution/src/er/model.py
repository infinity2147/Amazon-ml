"""Pair model: LightGBM on pair features, out-of-fold by S1 group, isotonic calibration,
expected-F0.5 decoding. Also leave-one-country-out (LOCO) as the France proxy."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .decode import decode_all
from .io_metric import macro_fbeta, score_breakdown

NON_FEATURES = {"s1_id", "cand_id", "y", "p", "raw", "ctry", "fold", "base_id"}

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=32, seed=0)


def feature_cols(F):
    return [c for c in F.columns if c not in NON_FEATURES]


def add_labels(F, gt_map):
    gold = {(s, c) for s, cs in gt_map.items() for c in cs}
    F["y"] = np.fromiter(((s, c) in gold for s, c in zip(F.s1_id, F.cand_id)), np.int8, len(F))
    return F


def train_lgb(X, y, Xv=None, yv=None, rounds=4000, params=None):
    params = {**PARAMS, **(params or {})}
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        return lgb.train(params, dtr, rounds, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(params, dtr, rounds)


def oof_predict(F, feats, n_folds=5, params=None, log=print):
    """GroupKFold over S1 ids. Returns raw OOF probabilities and the best iterations."""
    oof = np.zeros(len(F), np.float32)
    iters = []
    for k, (a, b) in enumerate(GroupKFold(n_folds).split(F, groups=F.s1_id)):
        m = train_lgb(F.iloc[a][feats], F.y.iloc[a], F.iloc[b][feats], F.y.iloc[b], params=params)
        oof[b] = m.predict(F.iloc[b][feats], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"[oof] fold {k}: best_iter={m.best_iteration}")
    return oof, iters


def calibrate(raw, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(raw, y)


def evaluate(F, p, s1_ids, gt_map, country_of, p_outside=0.0, one_owner=True):
    pred = decode_all(pd.DataFrame({"s1_id": F.s1_id.values, "cand_id": F.cand_id.values, "p": p}),
                      list(s1_ids), p_outside=p_outside, one_owner=one_owner)
    pred = {k: set(v) for k, v in pred.items()}
    gsub = {s: gt_map.get(s, set()) for s in s1_ids}
    return macro_fbeta(pred, gsub), score_breakdown(pred, gsub, country_of), pred


def loco(F, feats, country_of, gt_map, s1_ids, params=None, rounds=None, log=print):
    """Train on one country, score the other (France proxy). Calibration is fit on the
    TRAINING country's OOF-free in-sample predictions would be optimistic, so we calibrate on a
    held-out 20% of training-country S1s."""
    ctry = F.s1_id.map(country_of)
    res = {}
    for tr_c in sorted(ctry.unique()):
        for te_c in sorted(ctry.unique()):
            if tr_c == te_c:
                continue
            tr = F[ctry == tr_c]
            te = F[ctry == te_c]
            ids = tr.s1_id.unique()
            rng = np.random.default_rng(0)
            cal_ids = set(rng.choice(ids, len(ids) // 5, replace=False))
            is_cal = tr.s1_id.isin(cal_ids).values
            m = train_lgb(tr[~is_cal][feats], tr.y[~is_cal], rounds=rounds or 600, params=params)
            iso = calibrate(m.predict(tr[is_cal][feats]), tr.y[is_cal])
            p = iso.predict(m.predict(te[feats]))
            te_ids = [s for s in s1_ids if country_of.get(s) == te_c]
            f, _, _ = evaluate(te, p, te_ids, gt_map, country_of)
            res[f"{tr_c}->{te_c}"] = f
            log(f"[loco] {tr_c}->{te_c}: {f:.4f}")
    return res
