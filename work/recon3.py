import pandas as pd, numpy as np, re, unicodedata
from collections import Counter
gt=pd.read_parquet("cache/gt.parquet")
own=set(x for s in gt.matched_entity_ids for x in s.split(",") if x)
def script(s):
    c=Counter()
    for ch in s:
        if ch.isalpha():
            n=unicodedata.name(ch,"?").split()[0]
            c[n]+=1
    return c.most_common(1)[0][0] if c else "NONE"
for sp in ["train","test"]:
  for i in (2,3):
    d=pd.read_parquet(f"cache/{sp}_s{i}.parquet").sample(200000,random_state=0)
    d["sn"]=d.business_name.map(script); d["sa"]=d.business_address.map(script)
    print(sp,i,"NAME script by country\n",pd.crosstab(d.sn,d.country,normalize="columns").round(4).loc[lambda x:x.max(1)>0.0005])
    print(sp,i,"ADDR script by country\n",pd.crosstab(d.sa,d.country,normalize="columns").round(4).loc[lambda x:x.max(1)>0.0005])
