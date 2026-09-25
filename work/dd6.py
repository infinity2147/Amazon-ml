# Are the extra test decoys hard? Per split: fraction of S2/S3 records sharing (house number + a street word)
# with some same-country S1 record, and fraction whose normalised name matches some S1 name.
import pandas as pd, re
LEG=r"\b(private|pvt|limited|ltd|llc|inc|incorporated|corp|corporation|co|company|sarl|sas|eurl|llp|pllc|lp)\b"
nk=lambda s: re.sub(r"\s+"," ",re.sub(LEG," ",re.sub(r"[^a-z0-9]+"," ",s.lower()))).strip()
def akeys(a):
    t=re.sub(r"[^a-z0-9]+"," ",a.lower()).split()
    nums=[x.lstrip("0") for x in t if x.isdigit()]; words=[x for x in t if x.isalpha() and len(x)>3]
    return {(n,w) for n in nums[:2] for w in words[:3]}
for sp in ["train","test"]:
    s1=pd.read_parquet(f"cache/{sp}_s1.parquet"); c=pd.concat([pd.read_parquet(f"cache/{sp}_s2.parquet"),pd.read_parquet(f"cache/{sp}_s3.parquet")],ignore_index=True)
    c=c.sample(600000,random_state=0)
    for ctry in sorted(s1.country.unique()):
        a=s1[s1.country==ctry]; b=c[c.country==ctry]
        K=set().union(*a.business_address.map(akeys)); N=set(a.business_name.map(nk))
        ak=b.business_address.map(lambda x: bool(akeys(x)&K)); nm=b.business_name.map(nk).isin(N)
        r=len(b)/len(a)*len(s1)/len(c)*len(c)/600000*0+ (len(pd.concat([pd.read_parquet(f'cache/{sp}_s2.parquet',columns=['country']),pd.read_parquet(f'cache/{sp}_s3.parquet',columns=['country'])]).query('country==@ctry'))/len(a))
        print(sp,ctry,f"cands/S1={r:.2f}  addr-key-hit={ak.mean():.3f} (per S1 {ak.mean()*r:.2f})  name-hit={nm.mean():.3f} (per S1 {nm.mean()*r:.2f})  neither={(~ak&~nm).mean():.3f} (per S1 {(~ak&~nm).mean()*r:.2f})",flush=True)
