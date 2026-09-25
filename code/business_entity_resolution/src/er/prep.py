"""Load one split, normalise every record (multiprocess), cache as parquet."""
import os
from multiprocessing import Pool

import pandas as pd

from .io_metric import load_source
from .normalize import FIELDS, clean_components, fit_admin_vocab, prepare_record, set_translit

_VOCAB = {}


def _init(v, translit):
    global _VOCAB
    _VOCAB = v
    set_translit(translit)


def _prep_chunk(args):
    names, comps, ctrys = args
    out = []
    for n, c, k in zip(names, comps, ctrys):
        sv, nz = _VOCAB.get(k, (frozenset(), frozenset()))
        out.append(prepare_record(n, c, sv, nz))
    return pd.DataFrame(out, columns=FIELDS)


def _comps_chunk(addrs):
    return [clean_components(a) for a in addrs]


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def load_split(data_dir, split):
    s = [load_source(os.path.join(data_dir, split, f"{split}_source{i}.tsv")) for i in (1, 2, 3)]
    return s[0], pd.concat([s[1], s[2]], ignore_index=True)


def prepare_split(data_dir, split, out_dir, translit, n_jobs=32, chunk=20000, raw=None):
    """Writes {out_dir}/{split}_s1p.parquet and {split}_cp.parquet; returns (s1p, cp)."""
    s1, cand = raw if raw is not None else load_split(data_dir, split)
    for df in (s1, cand):
        df["ctry"] = df.country.str.strip().str.lower().replace("", "unk")
    with Pool(n_jobs) as pool:
        c1 = [c for part in pool.map(_comps_chunk, _chunks(s1.business_address.tolist(), chunk)) for c in part]
        c2 = [c for part in pool.map(_comps_chunk, _chunks(cand.business_address.tolist(), chunk)) for c in part]
    vocab = fit_admin_vocab(c1, s1.ctry.tolist(), c2, cand.ctry.tolist())
    for k, (sv, nz) in vocab.items():
        print(f"[prep] {split} {k}: {len(sv)} state comps, {len(nz)} noise comps; e.g. "
              f"{sorted(sv)[:6]} / {sorted(nz)[:6]}", flush=True)
    res = []
    with Pool(n_jobs, initializer=_init, initargs=(vocab, translit)) as pool:
        for df, comps in ((s1, c1), (cand, c2)):
            args = [(n, c, k) for n, c, k in zip(_chunks(df.business_name.tolist(), chunk), _chunks(comps, chunk),
                                                 _chunks(df.ctry.tolist(), chunk))]
            P = pd.concat(pool.map(_prep_chunk, args), ignore_index=True)
            P.index = df.index
            out = pd.concat([df[["entity_id", "business_name", "business_address", "country", "ctry"]], P], axis=1)
            res.append(out)
    os.makedirs(out_dir, exist_ok=True)
    res[0].to_parquet(os.path.join(out_dir, f"{split}_s1p.parquet"))
    res[1].to_parquet(os.path.join(out_dir, f"{split}_cp.parquet"))
    return res[0], res[1]
