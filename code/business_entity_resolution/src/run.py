"""End-to-end pipeline: data -> normalise -> block -> features -> LightGBM -> calibrate -> decode -> output.

    python src/run.py --data <student_resource/dataset> --out output --work work

Every stage caches its result as parquet in --work and is skipped when the file exists
(delete the file, or pass --force, to recompute)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er.blocking import block, blocking_report, cap_candidates  # noqa: E402
from er.decode import decode_all  # noqa: E402
from er.features import build_idf, pair_features  # noqa: E402
from er.io_metric import check_outputs, load_matches, write_idlist_tsv  # noqa: E402
from er.mine import mine_translit  # noqa: E402
from er.model import add_labels, calibrate, evaluate, feature_cols, oof_predict, train_lgb  # noqa: E402
from er.prep import load_split, prepare_split  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def cached(path, fn, force=False):
    if os.path.exists(path) and not force:
        log(f"cache hit {path}")
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--oof-frac", type=float, default=0.25, help="share of train S1s used for OOF calibration")
    ap.add_argument("--n-s1", type=int, default=30)
    ap.add_argument("--n-c", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    W = lambda f: os.path.join(a.work, f)  # noqa: E731

    gt = load_matches(os.path.join(a.data, "train", "train_ground_truth.tsv"))

    # 1. translit rules mined from train labels
    tl_path = W("translit.json")
    raw_train = None
    if not os.path.exists(tl_path) or a.force:
        raw_train = load_split(a.data, "train")
        tl = mine_translit(raw_train[0], raw_train[1], gt)
        json.dump(tl, open(tl_path, "w"))
    tl = json.load(open(tl_path))
    log(f"translit rules: {len(tl)}")

    # 2. normalise
    prepared = {}
    for sp in ("train", "test"):
        p1, p2 = W(f"{sp}_s1p.parquet"), W(f"{sp}_cp.parquet")
        if os.path.exists(p1) and os.path.exists(p2) and not a.force:
            prepared[sp] = (pd.read_parquet(p1), pd.read_parquet(p2))
        else:
            raw = raw_train if sp == "train" else None
            prepared[sp] = prepare_split(a.data, sp, a.work, tl, n_jobs=a.jobs, raw=raw)
        log(f"{sp}: {len(prepared[sp][0]):,} S1, {len(prepared[sp][1]):,} S2/S3")

    # 3. block + cap (the capped set is exactly what the model scores -> candidate_pairs.tsv)
    pairs = {}
    for sp in ("train", "test"):
        s1p, cp = prepared[sp]
        P = cached(W(f"{sp}_pairs.parquet"), lambda: block(s1p, cp, n_jobs=a.jobs, log=log), a.force)
        pairs[sp] = cap_candidates(P, a.n_s1, a.n_c)
        log(f"{sp}: {len(P):,} blocked pairs -> {len(pairs[sp]):,} after cap")
    rep = blocking_report(pairs["train"], gt)
    log("train blocking (capped):\n" + rep.round(4).to_string())

    # 4. features
    feats_df = {}
    for sp in ("train", "test"):
        s1p, cp = prepared[sp]
        feats_df[sp] = cached(W(f"{sp}_feats.parquet"),
                              lambda: pair_features(pairs[sp], s1p, cp, build_idf(s1p, cp), n_jobs=a.jobs), a.force)
    Ftr = add_labels(feats_df["train"], gt)
    feats = feature_cols(Ftr)

    # 5. OOF on a sample of S1 groups -> isotonic calibration + number of boosting rounds
    s1_train = prepared["train"][0]
    country_of = dict(zip(s1_train.entity_id, s1_train.country))
    ids = s1_train.entity_id.sample(frac=a.oof_frac, random_state=0)
    Fs = Ftr[Ftr.s1_id.isin(set(ids))].reset_index(drop=True)
    raw, iters = oof_predict(Fs, feats, log=log)
    iso = calibrate(raw, Fs.y)
    f, br, _ = evaluate(Fs, iso.predict(raw), ids.tolist(), gt, country_of)
    log(f"OOF macro F0.5 on {len(ids):,} S1: {f:.5f}\n{br.round(4).to_string()}")

    # 6. final model on all train pairs, predict test
    m = train_lgb(Ftr[feats], Ftr.y, rounds=int(np.mean(iters) * 1.1))
    Fte = feats_df["test"]
    Fte["p"] = iso.predict(m.predict(Fte[feats]))
    s1_test, cp_test = prepared["test"]
    test_ids = s1_test.entity_id.tolist()
    pred = decode_all(Fte[["s1_id", "cand_id", "p"]], test_ids)

    # 7. write + check
    mp, cpth = os.path.join(a.out, "matching_results.tsv"), os.path.join(a.out, "candidate_pairs.tsv")
    write_idlist_tsv(pred, test_ids, mp)
    write_idlist_tsv(Fte.groupby("s1_id").cand_id.apply(list).to_dict(), test_ids, cpth, col="candidate_entity_ids")
    log(f"check_outputs: {check_outputs(mp, cpth, test_ids, cp_test.entity_id)}")
    ctry = dict(zip(s1_test.entity_id, s1_test.country))
    mon = pd.DataFrame({"country": [ctry[s] for s in test_ids], "n": [len(pred[s]) for s in test_ids]})
    log("test monitor (share empty, mean set size):\n" +
        mon.groupby("country").n.agg(empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string())


if __name__ == "__main__":
    main()
