# Deep dive 1: full match groups (S1 + every true match), by country and group size
import pandas as pd, numpy as np
pd.set_option("display.width",260); pd.set_option("display.max_colwidth",90)
s1=pd.read_parquet("cache/train_s1.parquet"); c=pd.concat([pd.read_parquet("cache/train_s2.parquet"),pd.read_parquet("cache/train_s3.parquet")],ignore_index=True)
gt=pd.read_parquet("cache/gt.parquet")
gt["ids"]=gt.matched_entity_ids.map(lambda s:[x for x in s.split(",") if x])
S1=s1.set_index("entity_id"); C=c.set_index("entity_id")
ctry=S1.country
rng=np.random.default_rng(7)
for cn in ["US","India"]:
    sub=gt[gt.source1_entity_id.map(ctry)==cn]
    for i,r in enumerate(sub.sample(7,random_state=11).itertuples()):
        a=S1.loc[r.source1_entity_id]
        print(f"\n##### {cn} {r.source1_entity_id} ({len(r.ids)} matches)\n  S1 | {a.business_name} | {a.business_address}")
        for x in r.ids:
            b=C.loc[x]; print(f"  {x[:2]} | {b.business_name} | {b.business_address}")
