import pandas as pd, numpy as np, random
s1=pd.read_parquet("cache/train_s1.parquet").set_index("entity_id"); gt=pd.read_parquet("cache/gt.parquet")
c=pd.concat([pd.read_parquet("cache/train_s2.parquet"),pd.read_parquet("cache/train_s3.parquet")]).set_index("entity_id")
random.seed(1)
gt=gt[gt.matched_entity_ids!=""]
for ctry in ["US","India"]:
    ids=[i for i in gt.source1_entity_id.sample(4000,random_state=3) if s1.loc[i,"country"]==ctry][:12]
    g=gt.set_index("source1_entity_id")
    for i in ids:
        r=s1.loc[i]; print(f"\n### {i} | {r.business_name} | {r.business_address}")
        for x in g.loc[i,"matched_entity_ids"].split(","):
            q=c.loc[x]; print(f"   {x[:2]} | {q.business_name} | {q.business_address}")
