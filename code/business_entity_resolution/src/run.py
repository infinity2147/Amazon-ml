"""End-to-end pipeline: raw TSVs -> matching_results.tsv + candidate_pairs.tsv.

    python src/run.py --data <student_resource/dataset> --work <cache dir> --out <output dir>
                      [--stages prep,block,rank,keyview,feats,tlv,final,write] [--name final]

Stages cache their outputs in --work and skip work already done (delete a file to recompute it):
  prep     cleaning + transliteration rules          (er/prep.py, er/normalize.py, er/mine.py)
  block    three TF-IDF search views per country      (er/blocking.py)
  rank     learned shortlist cut, 20 per S1           (er/stages.py)
  keyview  exact 'name + house number' candidates     (er/stages.py)
  feats    ~72 pair features                          (er/features.py)
  tlv      test-like validation rows (decoy clones)   (er/stages.py)
  final    honest test-density two-stage LightGBM, audit score, test prediction (er/final.py)
  write    one-owner + expected-F0.5 decoding, both TSVs, all submission checks (er/decode.py, er/io_metric.py)
Seeds are fixed (ranker S1s 42, clones 11, audit 2026, folds 0, LightGBM seed 0, deterministic=True)."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er import stages  # noqa: E402
from er.decode import decode_all  # noqa: E402
from er.final import predict_test, run_final  # noqa: E402
from er.io_metric import check_outputs, write_idlist_tsv  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_and_check(W, out, name):
    P = pd.read_parquet(W + f"test_pred_{name}.parquet")
    ts1 = pd.read_parquet(W + "test_s1p.parquet", columns=["entity_id", "country"])
    test_ids = ts1.entity_id.tolist()
    pred = decode_all(P[["s1_id", "cand_id", "p", "rk"]], test_ids, tie_col="rk")
    os.makedirs(out, exist_ok=True)
    mp, cpth = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    write_idlist_tsv(pred, test_ids, mp)
    write_idlist_tsv(P.groupby("s1_id").cand_id.apply(list).to_dict(), test_ids, cpth, col="candidate_entity_ids")
    tcp = pd.read_parquet(W + "test_cp.parquet", columns=["entity_id"])
    chk = check_outputs(mp, cpth, test_ids, tcp.entity_id)      # header, every S1 once, ids valid, matches in candidates
    owners = pd.Series([x for v in pred.values() for x in v]).value_counts()
    multi = int((owners > 1).sum())
    ctry = dict(zip(ts1.entity_id, ts1.country))
    mon = pd.DataFrame({"country": [ctry[s] for s in test_ids], "n": [len(pred[s]) for s in test_ids]})
    rep = {"check_outputs": chk, "records_with_2plus_owners": multi, "n_s1": len(test_ids),
           "matching_md5": md5(mp), "matching_bytes": os.path.getsize(mp),
           "candidate_md5": md5(cpth), "candidate_bytes": os.path.getsize(cpth),
           "empty_share_by_country": mon.groupby("country").n.apply(lambda x: round((x == 0).mean(), 4)).to_dict(),
           "mean_list_by_country": mon.groupby("country").n.mean().round(3).to_dict()}
    json.dump(rep, open(os.path.join(out, "checks.json"), "w"), indent=1)
    log(f"write: {json.dumps(rep)}")
    assert chk == ["PASS"] and multi == 0, "submission checks failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="student_resource/dataset (train/ and test/ TSVs)")
    ap.add_argument("--work", required=True, help="cache directory")
    ap.add_argument("--out", required=True, help="output directory for the two TSVs")
    ap.add_argument("--stages", default="prep,block,rank,keyview,feats,tlv,final,write")
    ap.add_argument("--name", default="final")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--extra", type=int, default=0, help="1: add density features")
    ap.add_argument("--stage2", type=int, default=1)
    ap.add_argument("--pool-all", type=int, default=0, help="1: train on every non-audit train S1 (needs the rankoof stage)")
    ap.add_argument("--ce-train", default="", help="parquet [s1_id, cand_id, ce]: cross-encoder logits for train pairs")
    ap.add_argument("--ce-test", default="", help="parquet [s1_id, cand_id, ce]: cross-encoder logits for test pairs")
    a = ap.parse_args()
    W = os.path.join(a.work, "")
    os.makedirs(W, exist_ok=True)
    todo = a.stages.split(",")
    gt_path = stages.gt_parquet(a.data, W)
    if "prep" in todo:
        stages.stage_prep(a.data, W, a.jobs, log)
    if "block" in todo:
        stages.stage_block(W, a.jobs, log)
    if "rank" in todo:
        stages.stage_rank(W, gt_path, a.jobs, log)
    if "rankoof" in todo:
        stages.stage_rank_oof(W, gt_path, log, threads=a.threads)
    if "keyview" in todo:
        stages.stage_keyview(W, log)
    if "feats" in todo:
        stages.stage_feats(W, a.jobs, log)
    if "tlv" in todo:
        stages.stage_tlv(W, gt_path, log)
    if "final" in todo:
        st = run_final(W, a.name, gt_path, log, K=a.folds, lr=a.lr, threads=a.threads, extra=bool(a.extra), stage2=bool(a.stage2),
                       pool_all=bool(a.pool_all), ce_path=a.ce_train or None)
        if a.ce_test:                      # test cross-encoder scores may still be computing on the GPU
            while not os.path.exists(a.ce_test):
                time.sleep(30)
        predict_test(W, a.name, st, log, threads=a.threads, extra=bool(a.extra), ce_test_path=a.ce_test or None)
    if "write" in todo:
        write_and_check(W, a.out, a.name)


if __name__ == "__main__":
    main()
