import pandas as pd, re
pd.set_option("display.width",250)
def akeys(a):
    t=re.sub(r"[^a-z0-9]+"," ",a.lower()).split()
    nums=[x.lstrip("0") for x in t if x.isdigit()]; words=[x for x in t if x.isalpha() and len(x)>3]
    return {(n,w) for n in nums[:2] for w in words[:3]}
def show(sp,ctry,n=6,seed=1,gt=None):
    s1=pd.read_parquet(f"cache/{sp}_s1.parquet"); c=pd.concat([pd.read_parquet(f"cache/{sp}_s2.parquet"),pd.read_parquet(f"cache/{sp}_s3.parquet")],ignore_index=True)
    a=s1[s1.country==ctry].sample(n,random_state=seed); b=c[c.country==ctry]
    inv={}
    for i,x in zip(b.entity_id,b.business_address):
        for k in akeys(x): inv.setdefault(k,[]).append(i)
    B=b.set_index("entity_id"); own=gt or {}
    for r in a.itertuples():
        ids=set().union(*[set(inv.get(k,[])) for k in akeys(r.business_address)]) if akeys(r.business_address) else set()
        print(f"\n## {sp} {ctry} S1 | {r.business_name} | {r.business_address}")
        for i in list(ids)[:10]:
            x=B.loc[i]; tag=("MATCH" if own.get(i)==r.entity_id else ("other:"+own[i] if i in own else "decoy")) if gt else ""
            print(f"   {i[:2]} {tag:18s} | {x.business_name} | {x.business_address}")
gt=pd.read_parquet("cache/gt.parquet"); own={x:s for s,l in zip(gt.source1_entity_id,gt.matched_entity_ids) for x in l.split(",") if x}
show("train","US",5,2,own); show("test","US",6,2); show("test","India",4,3)
