import pandas as pd, numpy as np
from collections import Counter
s1=pd.read_parquet("cache/train_s1.parquet"); gt=pd.read_parquet("cache/gt.parquet")
s2=pd.read_parquet("cache/train_s2.parquet"); s3=pd.read_parquet("cache/train_s3.parquet")
print("gt ids == s1 ids:", set(gt.source1_entity_id)==set(s1.entity_id))
gt["lst"]=gt.matched_entity_ids.map(lambda s:[x for x in s.split(",") if x])
gt["n"]=gt.lst.str.len()
c1=dict(zip(s1.entity_id,s1.country)); gt["country"]=gt.source1_entity_id.map(c1)
print(s1.country.value_counts(), s2.country.value_counts(), s3.country.value_counts())
print("singleton rate by country", gt.groupby("country").n.apply(lambda x:(x==0).mean()))
print("set size dist\n", gt.groupby("country").n.value_counts().unstack(0).head(20))
gt["n2"]=gt.lst.map(lambda l:sum(x.startswith("S2") for x in l)); gt["n3"]=gt.n-gt.n2
print("S2 per S1 dist", gt.n2.value_counts().sort_index().head(10).to_dict())
print("S3 per S1 dist", gt.n3.value_counts().sort_index().head(10).to_dict())
own=Counter(x for l in gt.lst for x in l)
print("ids with >1 owner", sum(v>1 for v in own.values()), "total matched ids", len(own))
cm={**dict(zip(s2.entity_id,s2.country)),**dict(zip(s3.entity_id,s3.country))}
print("missing from sources", sum(x not in cm for x in own))
print("cross-country", sum(cm.get(x)!=c for l,c in zip(gt.lst,gt.country) for x in l))
for nm,s in [("s2",s2),("s3",s3)]:
    m=s.entity_id.isin(own)
    print(nm,"frac matched", m.mean(), "by country", s.assign(m=m).groupby("country").m.mean().to_dict())
    print(nm,"empty name",(s.business_name=="").mean(),"empty addr",(s.business_address=="").mean(),"empty country",(s.country=="").mean())
print("s1 empty addr",(s1.business_address=="").mean())
