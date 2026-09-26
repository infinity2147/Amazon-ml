"""Pipeline stages from raw TSVs to the pair-feature tables (ported verbatim from the experiment drivers
that produced the submission; every stage caches its output in WORK and is skipped when present).

  prep     translit rules mined from train labels; per-record normalisation -> {sp}_s1p / {sp}_cp
  block    three TF-IDF views per country, top-k both directions -> blk_{sp}_b1_{country}_{view}
  rank     union of views -> LightGBM ranker (trained on a seeded 30% of train S1s = ranker S1s)
           -> keep each S1's top 20 + each record's top 1 -> {sp}_pairs_r1
  keyview  + exact key 'same cleaned name + same house number' (keys shared by <= 30 S1s) -> {sp}_pairs_r3
  feats    ~72 pair features per country -> {sp}_feats_r3_f2/part_<country>
  tlv      test-like validation: clone each unowned train record with prob q (test/train decoys per S1 - 1),
           re-cap and recompute context features -> train_feats_tlv3_f2/part_<country>"""
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from .blocking import block, group_rank, rank_features
from .features import add_context, build_idf, build_state_map, pair_features, L_COLS, R_COLS
from .io_metric import load_matches
from .mine import mine_translit
from .prep import load_split, prepare_split

VIEWS = ["v_word", "v_name5", "v_addr5"]


def gt_parquet(data, W):
    p = W + "gt.parquet"
    if not os.path.exists(p):
        from .io_metric import load_source
        load_source(os.path.join(data, "train", "train_ground_truth.tsv")).to_parquet(p)
    return p


def stage_prep(data, W, jobs, log):
    if all(os.path.exists(W + f"{sp}_{f}.parquet") for sp in ("train", "test") for f in ("s1p", "cp")):
        return log("prep: cached")
    tl_path = W + "translit.json"
    raw_train = load_split(data, "train")
    if not os.path.exists(tl_path):
        gt = load_matches(os.path.join(data, "train", "train_ground_truth.tsv"))
        json.dump(mine_translit(raw_train[0], raw_train[1], gt), open(tl_path, "w"))
    tl = json.load(open(tl_path))
    log(f"prep: {len(tl)} translit rules")
    for sp in ("train", "test"):
        prepare_split(data, sp, W, tl, n_jobs=jobs, raw=raw_train if sp == "train" else None)
        log(f"prep: {sp} done")


def stage_block(W, jobs, log):
    for sp in ("train", "test"):
        s1p = pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["entity_id", "ctry", "name_n", "addr"])
        cp = pd.read_parquet(W + f"{sp}_cp.parquet", columns=["entity_id", "ctry", "name_n", "addr"])
        block(s1p, cp, n_jobs=jobs, log=log, cache_dir=W, tag=f"{sp}_b1", views_only=True)
        del s1p, cp; gc.collect()


def stage_rank(W, gt_path, jobs, log, n1=20, nc=1):
    for sp in ("train", "test"):
        if os.path.exists(W + f"{sp}_pairs_r1.parquet"):
            log(f"rank: {sp} cached"); continue
        s1 = pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["entity_id", "ctry"])
        cp = pd.read_parquet(W + f"{sp}_cp.parquet", columns=["entity_id", "ctry"])
        ctrys = sorted(set(s1.ctry))

        def union(c):
            a = s1.entity_id[s1.ctry == c].to_numpy(); b = cp.entity_id[cp.ctry == c].to_numpy(); nb = len(b)
            V = pd.concat([pd.read_parquet(W + f"blk_{sp}_b1_{c}_{v}.parquet").set_index("key")[v] for v in VIEWS], axis=1).fillna(0).astype(np.float32)
            key = V.index.to_numpy(); V = V.reset_index(drop=True)
            V["si"] = (key // nb).astype(np.int32); V["cj"] = (key % nb).astype(np.int32)
            return V, a, b

        if sp == "train" and not os.path.exists(W + "ranker.txt"):
            gt = pd.read_parquet(gt_path)
            gold = {(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x}
            rank_s1 = set(s1.entity_id[np.random.default_rng(42).random(len(s1)) < 0.3])
            pd.Series(sorted(rank_s1)).to_frame("s1_id").to_parquet(W + "ranker_s1.parquet")
            Xs, ys = [], []
            for c in ctrys:
                V, a, b = union(c)
                X = rank_features(V, VIEWS, "si", "cj")
                m = pd.Index(a).isin(rank_s1)[V.si.to_numpy()]
                sid, cid = a[V.si.to_numpy()[m]], b[V.cj.to_numpy()[m]]
                ys.append(np.fromiter(((x, y) in gold for x, y in zip(sid, cid)), bool, m.sum()))
                Xs.append(X[m]); del V, X; gc.collect()
            X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); del Xs
            lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200, verbose=-1, num_threads=24, seed=0),
                      lgb.Dataset(X, y), 300).save_model(W + "ranker.txt")
            log(f"rank: ranker trained on {len(X):,} rows"); del X; gc.collect()
        ranker = lgb.Booster(model_file=W + "ranker.txt")
        out = []
        for c in ctrys:
            V, a, b = union(c)
            s = ranker.predict(rank_features(V, VIEWS, "si", "cj"), num_threads=24).astype(np.float32)
            r1 = group_rank(V.si.to_numpy(), s)[0]; r2 = group_rank(V.cj.to_numpy(), s)[0]
            k = (r1 <= n1) | (r2 <= nc)
            P = V[k].reset_index(drop=True); P["rk"] = s[k]
            P.insert(0, "cand_id", b[P.pop("cj").to_numpy()]); P.insert(0, "s1_id", a[P.pop("si").to_numpy()])
            out.append(P); log(f"rank: {sp} {c}: union {len(V):,} -> capped {len(P):,}")
            del V, s; gc.collect()
        pd.concat(out, ignore_index=True).to_parquet(W + f"{sp}_pairs_r1.parquet")


def stage_keyview(W, log, maxb=30):
    for sp in ("train", "test"):
        if os.path.exists(W + f"{sp}_pairs_r3.parquet"):
            log(f"keyview: {sp} cached"); continue
        P = pd.read_parquet(W + f"{sp}_pairs_r1.parquet")
        out = []
        for c in sorted(pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.unique()):
            s1 = pds.dataset(W + f"{sp}_s1p.parquet").to_table(columns=["entity_id", "name_n", "house"], filter=pds.field("ctry") == c).to_pandas()
            cp = pds.dataset(W + f"{sp}_cp.parquet").to_table(columns=["entity_id", "name_n", "house"], filter=pds.field("ctry") == c).to_pandas()
            a = s1[(s1.house != "") & (s1.name_n != "")].assign(k=lambda d: d.name_n + "|" + d.house)
            b = cp[(cp.house != "") & (cp.name_n != "")].assign(k=lambda d: d.name_n + "|" + d.house)
            bs = a.groupby("k").size(); a = a[a.k.map(bs) <= maxb]
            out.append(a[["entity_id", "k"]].rename(columns={"entity_id": "s1_id"}).merge(
                b[["entity_id", "k"]].rename(columns={"entity_id": "cand_id"}), on="k")[["s1_id", "cand_id"]])
        Kp = pd.concat(out, ignore_index=True).drop_duplicates(); Kp["v_key"] = np.float32(1)
        P = P.merge(Kp, on=["s1_id", "cand_id"], how="outer")
        for col in [c for c in P.columns if c.startswith("v_") or c == "rk"]:
            P[col] = P[col].fillna(0).astype(np.float32)
        P.to_parquet(W + f"{sp}_pairs_r3.parquet"); log(f"keyview: {sp} {len(P):,} pairs")


def stage_feats(W, jobs, log):
    for sp in ("train", "test"):
        out_dir = W + f"{sp}_feats_r3_f2"
        os.makedirs(out_dir, exist_ok=True)
        cols = sorted(set(["entity_id", "ctry"] + L_COLS + R_COLS))
        s1cols = [c for c in cols if c not in ("name_alt", "web", "native")]
        for c in sorted(pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.unique()):
            if os.path.exists(f"{out_dir}/part_{c}.parquet"):
                continue
            s1p = pds.dataset(W + f"{sp}_s1p.parquet").to_table(columns=s1cols, filter=pds.field("ctry") == c).to_pandas()
            cp = pds.dataset(W + f"{sp}_cp.parquet").to_table(columns=cols, filter=pds.field("ctry") == c).to_pandas()
            import pyarrow as pa
            P = pds.dataset(W + f"{sp}_pairs_r3.parquet").to_table(filter=pds.field("s1_id").isin(pa.array(s1p.entity_id.tolist()))).to_pandas()
            idf = build_idf(s1p, cp); idf["state_map"] = build_state_map(P, s1p, cp)
            F = pair_features(P, s1p, cp, idf, n_jobs=jobs)
            F.to_parquet(f"{out_dir}/part_{c}.parquet"); log(f"feats: {sp} {c} {F.shape}")
            del F, P, s1p, cp, idf; gc.collect()


def stage_tlv(W, gt_path, log, n1=20, seed=11):
    dst = W + "train_feats_tlv3_f2"
    os.makedirs(dst, exist_ok=True)
    gt = pd.read_parquet(gt_path)
    owned = pd.Index([x for l in gt.matched_entity_ids for x in l.split(",") if x]); del gt
    s1c = {sp: pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["ctry"]).ctry.value_counts() for sp in ["train", "test"]}
    for c in sorted(s1c["train"].index):
        if os.path.exists(f"{dst}/part_{c}.parquet"):
            continue
        cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "ctry"]); cp = cp.entity_id[cp.ctry == c]
        n_own = cp.isin(owned).sum(); n_tr = s1c["train"][c]
        tcp = pd.read_parquet(W + "test_cp.parquet", columns=["ctry"]).ctry.eq(c).sum()
        dec_tr = (len(cp) - n_own) / n_tr; dec_te = tcp / s1c["test"][c] - n_own / n_tr
        q = dec_te / dec_tr - 1
        unowned = cp[~cp.isin(owned)].to_numpy(); del cp
        clone_set = pd.Index(unowned[np.random.default_rng(seed).random(len(unowned)) < q])
        F = pq.read_table(W + f"train_feats_r3_f2/part_{c}.parquet").to_pandas()
        C = F[F.cand_id.isin(clone_set).to_numpy()].copy(); C["cand_id"] = C.cand_id + "~c"
        F = pd.concat([F, C], ignore_index=True); del C; gc.collect()
        g1 = pd.factorize(F.s1_id)[0]; g2 = pd.factorize(F.cand_id)[0]; rk = F.rk.to_numpy(np.float32)
        keep = (group_rank(g1, rk)[0] <= n1) | ((group_rank(g2, rk)[0] <= 1) & (rk > 0)) | (F.v_key.to_numpy() == 1)
        F = add_context(F[keep].reset_index(drop=True))
        F.to_parquet(f"{dst}/part_{c}.parquet"); log(f"tlv: {c} q={q:.3f} -> {len(F):,} pairs")
        del F; gc.collect()


def stage_rank_oof(W, gt_path, log, n1=20, nc=1, K=5, frac=0.3, seed=7, threads=32):
    """Out-of-fold shortlist ranker, so that EVERY train S1 can later be used as training data.
    Train S1s are split into K folds (seeded). Ranker k is trained on a random `frac` of all train S1s drawn
    from outside fold k (same size as the original single ranker) and scores fold k's candidate lists, so no
    S1's list is cut by a model that saw its labels. Test lists are cut by the mean of the K rankers.
    Writes {sp}_pairs_r1.parquet (same schema as stage_rank) and rank_folds.parquet."""
    for sp in ("train", "test"):
        if os.path.exists(W + f"{sp}_pairs_r1.parquet"):
            log(f"rank_oof: {sp} cached"); continue
        s1 = pd.read_parquet(W + f"{sp}_s1p.parquet", columns=["entity_id", "ctry"])
        cp = pd.read_parquet(W + f"{sp}_cp.parquet", columns=["entity_id", "ctry"])
        ctrys = sorted(set(s1.ctry))

        def union(c):
            a = s1.entity_id[s1.ctry == c].to_numpy(); b = cp.entity_id[cp.ctry == c].to_numpy(); nb = len(b)
            V = pd.concat([pd.read_parquet(W + f"blk_{sp}_b1_{c}_{v}.parquet").set_index("key")[v] for v in VIEWS], axis=1).fillna(0).astype(np.float32)
            key = V.index.to_numpy(); V = V.reset_index(drop=True)
            V["si"] = (key // nb).astype(np.int32); V["cj"] = (key % nb).astype(np.int32)
            return V, a, b

        if sp == "train":
            gt = pd.read_parquet(gt_path)
            gold = {(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x}; del gt
            rng = np.random.default_rng(seed)
            fold = pd.Series(rng.permutation(len(s1)) % K, index=s1.entity_id.to_numpy())
            pd.DataFrame({"s1_id": fold.index, "fold": fold.to_numpy()}).to_parquet(W + "rank_folds.parquet")
            take = pd.Series(rng.random(len(s1)), index=s1.entity_id.to_numpy())   # per-S1 uniform for sampling
            data = {}
            for c in ctrys:
                V, a, b = union(c)
                X = rank_features(V, VIEWS, "si", "cj").to_numpy(np.float32)
                sid = a[V.si.to_numpy()]
                y = np.fromiter(((x, z) in gold for x, z in zip(sid, b[V.cj.to_numpy()])), bool, len(V))
                data[c] = (X, y, fold.reindex(a).to_numpy()[V.si.to_numpy()].astype(np.int8), take.reindex(a).to_numpy()[V.si.to_numpy()].astype(np.float32))
                log(f"rank_oof: {c} union {len(V):,} rows"); del V; gc.collect()
            cols = [f"{v}{s}" for v in VIEWS for s in ("", "_r1", "_gap1", "_r2", "_gap2")] + ["n_views", "n1", "n2"]
            p_in = frac / (1 - 1 / K)          # share of out-of-fold S1s to sample so each ranker sees ~frac of all S1s
            for k in range(K):
                if os.path.exists(W + f"ranker_oof_{k}.txt"):
                    continue
                Xs, ys = [], []
                for c in ctrys:
                    X, y, fo, tk = data[c]; m = (fo != k) & (tk < p_in)
                    Xs.append(X[m]); ys.append(y[m])
                Xk = np.concatenate(Xs); yk = np.concatenate(ys); del Xs, ys
                lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200, verbose=-1, num_threads=threads, seed=0),
                          lgb.Dataset(pd.DataFrame(Xk, columns=cols), yk), 300).save_model(W + f"ranker_oof_{k}.txt")
                log(f"rank_oof: ranker {k} trained on {len(Xk):,} rows"); del Xk, yk; gc.collect()
            rankers = [lgb.Booster(model_file=W + f"ranker_oof_{k}.txt") for k in range(K)]
            out = []
            for c in ctrys:
                X, y, fo, tk = data[c]
                s = np.zeros(len(X), np.float32)
                for k in range(K):
                    ix = np.flatnonzero(fo == k)
                    s[ix] = rankers[k].predict(pd.DataFrame(X[ix], columns=cols), num_threads=threads)
                V, a, b = union(c)
                r1 = group_rank(V.si.to_numpy(), s)[0]; r2 = group_rank(V.cj.to_numpy(), s)[0]
                keep = (r1 <= n1) | (r2 <= nc)
                P = V[keep].reset_index(drop=True); P["rk"] = s[keep]
                P.insert(0, "cand_id", b[P.pop("cj").to_numpy()]); P.insert(0, "s1_id", a[P.pop("si").to_numpy()])
                hit = y[keep].sum() / max(y.sum(), 1)
                out.append(P); log(f"rank_oof: train {c}: {len(V):,} -> {len(P):,} pairs; share of blocked true pairs kept {hit:.4f}")
                del V, s, data[c]; gc.collect()
            pd.concat(out, ignore_index=True).to_parquet(W + "train_pairs_r1.parquet")
        else:
            rankers = [lgb.Booster(model_file=W + f"ranker_oof_{k}.txt") for k in range(K)]
            out = []
            for c in ctrys:
                V, a, b = union(c)
                Xf = rank_features(V, VIEWS, "si", "cj")
                s = np.mean([r.predict(Xf, num_threads=threads) for r in rankers], axis=0).astype(np.float32); del Xf
                r1 = group_rank(V.si.to_numpy(), s)[0]; r2 = group_rank(V.cj.to_numpy(), s)[0]
                keep = (r1 <= n1) | (r2 <= nc)
                P = V[keep].reset_index(drop=True); P["rk"] = s[keep]
                P.insert(0, "cand_id", b[P.pop("cj").to_numpy()]); P.insert(0, "s1_id", a[P.pop("si").to_numpy()])
                out.append(P); log(f"rank_oof: test {c}: {len(V):,} -> {len(P):,} pairs"); del V, s; gc.collect()
            pd.concat(out, ignore_index=True).to_parquet(W + "test_pairs_r1.parquet")
