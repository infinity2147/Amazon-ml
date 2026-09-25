# label-free proxy: share of S2/S3 records that have an "obvious owner" in S1:
# same country, and (exact normalised name) + share a number token in the address
import pandas as pd, re, numpy as np
nk=lambda s: re.sub(r"\s+"," ",re.sub(r"[^a-z0-9]+"," ",s.lower())).strip()
LEG=r"\b(private|pvt|limited|ltd|llc|inc|incorporated|corp|corporation|co|company|sarl|sas|eurl|llp|pllc|lp)\b"
def nm(s): return re.sub(r"\s+"," ",re.sub(LEG," ",nk(s))).strip()
nums=lambda s: frozenset(t.lstrip("0") for t in re.findall(r"\d+",s))
def proxy(sp):
    s1=pd.read_parquet(f"cache/{sp}_s1.parquet"); c=pd.concat([pd.read_parquet(f"cache/{sp}_s2.parquet"),pd.read_parquet(f"cache/{sp}_s3.parquet")],ignore_index=True)
    s1["k"]=s1.business_name.map(nm)+"|"+s1.country; c["k"]=c.business_name.map(nm)+"|"+c.country
    s1["nu"]=s1.business_address.map(nums); c["nu"]=c.business_address.map(nums)
    idx=s1.groupby("k").nu.apply(list).to_dict()
    hit=[any(u & v for v in idx.get(k,[])) if u else False for k,u in zip(c.k,c.nu)]
    c["hit"]=hit
    return c.groupby("country").hit.mean().round(4).to_dict(), (c.groupby("country").size()/s1.groupby("country").size()).round(2).to_dict()
for sp in ["train","test"]: print(sp, proxy(sp))
