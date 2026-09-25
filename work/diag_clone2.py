"""Which stage-2 features differ between originals that have a clone in the list and those that don't?"""
import sys, numpy as np, pandas as pd, pyarrow.dataset as pds
sys.path.insert(0, "/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.stage2 import sibling_features
W = "/home/24b4518/ml-projects/work/cache/v1/"
O = pd.read_parquet(W + "oof_T0.parquet")
ids = pd.Series(O.s1_id.unique()).sample(150000, random_state=1); O = O[O.s1_id.isin(set(ids))].reset_index(drop=True)
X = pds.dataset(W + "train_feats_tlv_f2").to_table(columns=["s1_id", "cand_id", "na_prod", "house_eq"], filter=pds.field("s1_id").isin(ids.tolist())).to_pandas()
O = O.merge(X, on=["s1_id", "cand_id"], how="left"); O["base_id"] = O.cand_id.str.replace("~c", "", regex=False)
cp = pd.read_parquet(W + "train_cp.parquet", columns=["entity_id", "name_n", "addr", "house", "nums"]); cp = cp[cp.entity_id.isin(set(O.base_id))]
F = sibling_features(O[["s1_id", "cand_id", "p1", "base_id"]].copy(), cp, base_col="base_id")
F["y"] = O.y.values; F["na_prod"] = O.na_prod.values; F["house_eq"] = O.house_eq.values; F["is_clone"] = O.cand_id.str.endswith("~c").values
F["twin"] = F.groupby(["s1_id", "base_id"]).cand_id.transform("size") >= 2
d = F[(F.y == 0) & (~F.is_clone) & (F.na_prod >= 85) & (F.house_eq != 1)]
cols = [c for c in F.columns if c.startswith(("p1_", "n_conf", "sib_"))]
r = pd.DataFrame({"twin_in_list": d[d.twin][cols].mean(), "no_twin": d[~d.twin][cols].mean()}); r["ratio"] = r.twin_in_list / r.no_twin
print(f"near-twin decoys: {len(d):,} ({d.twin.mean():.2f} with a clone in the list)"); print(r.round(4).to_string())
