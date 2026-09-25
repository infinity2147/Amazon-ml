"""How many true pairs that the current shortlist misses would cheap exact-key lookups recover,
and how many extra candidate pairs would they add? One country at a time, integer keys."""
import sys, numpy as np, pandas as pd
W = "/home/24b4518/ml-projects/work/cache/v1/"
c = sys.argv[1]
import pyarrow.dataset as pds
s1 = pds.dataset(W + "train_s1p.parquet").to_table(columns=["entity_id", "name_n", "house", "addr_words", "addr"], filter=pds.field("ctry") == c).to_pandas()
cp = pds.dataset(W + "train_cp.parquet").to_table(columns=["entity_id", "name_n", "nums", "addr_words", "addr"], filter=pds.field("ctry") == c).to_pandas()
s1["i"] = np.arange(len(s1)); cp["j"] = np.arange(len(cp)); nb = len(cp)
si = pd.Series(s1.i.values, index=s1.entity_id); cj = pd.Series(cp.j.values, index=cp.entity_id)
P = pd.read_parquet(W + "train_pairs_r1.parquet", columns=["s1_id", "cand_id"])
P = P[P.s1_id.isin(si.index)]
have = set((si[P.s1_id].to_numpy() * nb + cj[P.cand_id].to_numpy()).tolist())
gt = pd.read_parquet("/home/24b4518/ml-projects/work/cache/gt.parquet"); gt = gt[gt.source1_entity_id.isin(si.index)]
gp = [(s, x) for s, l in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in l.split(",") if x]
gk = si[[a for a, _ in gp]].to_numpy() * nb + cj[[b for _, b in gp]].to_numpy()
miss = set(gk[~np.isin(gk, list(have))].tolist())
print(f"{c}: S1 {len(s1):,}, gold pairs {len(gk):,}, missed by current shortlist {len(miss):,} ({len(miss)/len(gk):.4f})", flush=True)
cp_first = cp.nums.str.split().str[0].fillna("")
# rare address words: document frequency within the country
from collections import Counter
dfc = Counter(w for s in pd.concat([s1.addr_words, cp.addr_words]) for w in set(s.split()))
def rare_words(s, k=2, maxdf=200):
    ws = sorted((w for w in set(s.split()) if len(w) > 3 and dfc[w] <= maxdf), key=lambda w: dfc[w])
    return ws[:k]
views = {
    "name": (s1.name_n, cp.name_n, s1.name_n != "", cp.name_n != ""),
    "name+house": (s1.name_n + "|" + s1.house, cp.name_n + "|" + cp_first, s1.house != "", cp_first != ""),
}
s1_rw = s1.addr_words.map(rare_words); cp_rw = cp.addr_words.map(rare_words)
hk1 = pd.DataFrame({"i": np.repeat(s1.i.values, s1_rw.str.len()), "k": np.concatenate([[h + "|" + w for w in ws] for h, ws in zip(s1.house, s1_rw)] or [[]])})
hk1 = hk1[~hk1.k.str.startswith("|")]
cn = cp.nums.str.split()
hk2 = pd.DataFrame({"j": np.repeat(cp.j.values, cp_rw.str.len() * cn.str.len().fillna(0).astype(int).clip(upper=2)),
                    "k": np.concatenate([[n + "|" + w for n in (ns or [])[:2] for w in ws] for ns, ws in zip(cn, cp_rw)] or [[]])})

def evaluate(name, A, B, max_bucket):
    ka = pd.factorize(pd.concat([A.k, B.k], ignore_index=True))[0]
    A = A.assign(kk=ka[:len(A)]); B = B.assign(kk=ka[len(A):])
    bs = A.groupby("kk").size()
    A = A[A.kk.map(bs) <= max_bucket]
    J = A.merge(B, on="kk")
    key = J.i.to_numpy().astype(np.int64) * nb + J.j.to_numpy()
    key = np.unique(key)
    new = key[~np.isin(key, list(have))]
    rec = np.isin(new, list(miss)).sum()
    print(f"  {name:15s} max S1 per key {max_bucket:3d}: new pairs {len(new):,} ({len(new)/len(s1):.2f}/S1), "
          f"recovers {rec:,} of {len(miss):,} missed ({rec/len(miss):.3f}), precision of new pairs {rec/max(len(new),1):.3f}", flush=True)
    return set(new.tolist())
for name, (ka, kb, ma, mb) in views.items():
    A = pd.DataFrame({"i": s1.i[ma].values, "k": ka[ma].values}); B = pd.DataFrame({"j": cp.j[mb].values, "k": kb[mb].values})
    for mbk in [3, 10, 30]:
        evaluate(name, A, B, mbk)
for mbk in [3, 10]:
    evaluate("house+rareword", hk1, hk2, mbk)
