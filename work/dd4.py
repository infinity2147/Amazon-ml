import pandas as pd, numpy as np, re
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
tr={k:pd.read_parquet(f"cache/train_{k}.parquet") for k in ["s1","s2","s3"]}
te={k:pd.read_parquet(f"cache/test_{k}.parquet") for k in ["s1","s2","s3"]}
def st(df):
    a=df.business_address; n=df.business_name
    return dict(n=len(df), name_empty=(n=="").mean(), addr_empty=(a=="").mean(), NA=a.str.contains(r"\bN/A\b").mean(),
      NULL=a.str.contains("NULL").mean(), upper_addr=(a==a.str.upper()).mean(), native_name=n.map(lambda s:any(ord(ch)>0x24f for ch in s)).mean(),
      ncomp=a.str.count(",").mean()+1, name_len=n.str.len().mean(), addr_len=a.str.len().mean())
rows=[]
for sp,d in [("train",tr),("test",te)]:
    for k,df in d.items():
        for c,g in df.groupby("country"): rows.append(dict(split=sp,src=k,country=c,**st(g)))
print(pd.DataFrame(rows).round(3).to_string())
# leak look (do NOT use): numeric id closeness of matches
gt=pd.read_parquet("cache/gt.parquet").sample(20000,random_state=0)
pairs=[(int(s[3:]),int(x[3:])) for s,l in zip(gt.source1_entity_id,gt.matched_entity_ids) for x in l.split(",") if x]
a=np.array(pairs); print("\nleak check: corr(S1 id, matched id) =",np.corrcoef(a[:,0],a[:,1])[0,1].round(4), "| median |diff|",np.median(np.abs(a[:,0]-a[:,1])))
