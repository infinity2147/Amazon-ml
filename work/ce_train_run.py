"""Train the two cross-encoders (halves A/B of the 551,789 validation S1s) on their uncertain-band pairs and check
each on the OTHER half: AUC of the cross-encoder alone vs the stage-1 LightGBM p (X1 recipe) on the same pairs.
Band = stage-1 p (X1 OOF, test-like rows) in (0.005, 0.995); pairs deduplicated to (S1, base record)."""
import os, sys, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.crossenc import half_of, pair_texts, train_ce, score_ce
W = "/home/24b4518/ml-projects/work/cache/v2/"; OUT = "/home/24b4518/ml-projects/work/ce/"
BB = [os.path.join(r, "") for r, d, f in os.walk("/home/24b4518/ml-projects/work/hf/hub/models--microsoft--mdeberta-v3-base/snapshots")][1]
EPOCHS = int(os.environ.get("EPOCHS", 1)); MAXN = int(os.environ.get("MAXN", 500000))
t = time.time(); log = lambda m: print(f"[{time.time()-t:6.0f}s] {m}", flush=True)
X = pd.read_parquet(W + "exp_X1.parquet", columns=["s1_id", "cand_id", "y", "p1"])
X["cand_id"] = X.cand_id.str.replace("~c", "", regex=False)
B = X[(X.p1 > 0.005) & (X.p1 < 0.995)].groupby(["s1_id", "cand_id"], as_index=False).agg(y=("y", "max"), p1=("p1", "mean"))
B["half"] = half_of(B.s1_id.to_numpy())
log(f"band pairs {len(B):,} from {B.s1_id.nunique():,} S1s; positives {B.y.mean():.3f}; half A {(B.half == 0).sum():,}")
s1raw = pd.read_parquet(W + "train_s1p.parquet", columns=["entity_id", "business_name", "business_address"]).set_index("entity_id")
cpraw = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "business_name", "business_address"])
cpraw = cpraw[cpraw.entity_id.isin(set(B.cand_id))].set_index("entity_id")
B.to_parquet(OUT + "band_train.parquet") if os.path.isdir(OUT) else (os.makedirs(OUT), B.to_parquet(OUT + "band_train.parquet"))
for h, name in [(0, "A"), (1, "B")]:
    d = OUT + f"ce_{name}"
    if not os.path.exists(d + "/config.json"):
        tr = B[B.half == h].sample(frac=1, random_state=h).head(MAXN)
        ta, tb = pair_texts(tr, s1raw, cpraw)
        log(f"training ce_{name} on {len(tr):,} pairs, {EPOCHS} epoch(s)")
        train_ce(ta, tb, tr.y.to_numpy(), d, BB, log, epochs=EPOCHS)
    ev = B[B.half != h].sample(60000, random_state=1)
    ta, tb = pair_texts(ev, s1raw, cpraw)
    s = score_ce(ta, tb, d, log)
    log(f"ce_{name} on the other half (60k band pairs): AUC cross-encoder {roc_auc_score(ev.y, s):.4f} | AUC stage-1 LightGBM p {roc_auc_score(ev.y, ev.p1):.4f} | AUC of their rank average {roc_auc_score(ev.y, pd.Series(s).rank().values + pd.Series(ev.p1.values).rank().values):.4f}")
