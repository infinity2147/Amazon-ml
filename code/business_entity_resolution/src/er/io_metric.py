"""I/O + exact leaderboard metric. Stdlib + pandas only."""
import csv
import pandas as pd


def load_source(path):
    # QUOTE_NONE: addresses can contain stray quotes; default quoting silently merges rows.
    # keep_default_na=False: a business literally named "NA"/"None" must stay a string.
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def parse_idlist(s):
    if s is None:
        return []
    s = str(s).strip()
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def load_matches(path, id_col="matched_entity_ids"):
    """GT or submission -> {s1_id: set(ids)}"""
    df = load_source(path)
    col = id_col if id_col in df.columns else df.columns[1]
    return {s: set(parse_idlist(v)) for s, v in zip(df.source1_entity_id, df[col])}


def write_idlist_tsv(mapping, s1_ids, path, col="matched_entity_ids"):
    """mapping: {s1_id: iterable of ids}. Writes EVERY s1 id (singletons -> empty)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s1, [])))  # dedupe, keep order
            f.write(f"{s1}\t{','.join(ids)}\n")


def fbeta_entity(pred, gold, beta=0.5):
    pred, gold = set(pred), set(gold)
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_fbeta(pred_map, gold_map, beta=0.5, ids=None):
    ids = list(gold_map) if ids is None else list(ids)
    return sum(fbeta_entity(pred_map.get(i, ()), gold_map.get(i, ()), beta) for i in ids) / max(len(ids), 1)


def score_breakdown(pred_map, gold_map, country_of=None, beta=0.5, ids=None):
    """Where are points lost? singleton vs matched, per country."""
    ids = list(gold_map) if ids is None else list(ids)
    rows = []
    for i in ids:
        g = gold_map.get(i, set())
        p = pred_map.get(i, set())
        rows.append(dict(s1=i, country=(country_of or {}).get(i, "?"),
                         kind="singleton" if not g else "matched",
                         n_gold=len(g), n_pred=len(p), f=fbeta_entity(p, g, beta)))
    d = pd.DataFrame(rows)
    out = d.groupby(["country", "kind"]).agg(n=("f", "size"), mean_f=("f", "mean"),
                                             lost_pts=("f", lambda x: (1 - x).sum()))
    out["lost_share"] = out.lost_pts / out.lost_pts.sum()
    return out


def check_outputs(match_path, cand_path, test_s1_ids, test_cand_ids):
    """Local mirror of the official rules + the 'matches subset of candidates' audit.
    Still run utils/validate_submission.py - this is a fast in-pipeline assert."""
    errs = []
    s1_ids = list(test_s1_ids)
    valid = set(test_cand_ids)
    maps = {}
    for path, col in [(match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")]:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        if lines[0] != f"source1_entity_id\t{col}":
            errs.append(f"{path}: bad header {lines[0]!r}")
        rows = [l.split("\t") for l in lines[1:] if l != ""]
        if any(len(r) != 2 for r in rows):
            errs.append(f"{path}: rows without exactly one tab")
        ids = [r[0] for r in rows]
        if len(ids) != len(set(ids)):
            errs.append(f"{path}: duplicate source1 rows")
        if set(ids) != set(s1_ids):
            errs.append(f"{path}: missing {len(set(s1_ids) - set(ids))} / extra {len(set(ids) - set(s1_ids))} S1 ids")
        m = {}
        for r in rows:
            lst = parse_idlist(r[1]) if len(r) > 1 else []
            if len(lst) != len(set(lst)):
                errs.append(f"{path}: duplicate ids in list for {r[0]}")
            bad = [x for x in lst if x not in valid or not x.startswith(("S2-", "S3-"))]
            if bad:
                errs.append(f"{path}: {r[0]} has invalid ids {bad[:3]}")
            m[r[0]] = set(lst)
        maps[col] = m
    mm, cc = maps.get("matched_entity_ids", {}), maps.get("candidate_entity_ids", {})
    leak = sum(len(v - cc.get(k, set())) for k, v in mm.items())
    if leak:
        errs.append(f"{leak} matched ids not present in candidate_pairs (pipeline bug)")
    return errs or ["PASS"]
