import pandas as pd, re
gt=pd.read_parquet("cache/gt.parquet")
own={x:s for s,l in zip(gt.source1_entity_id,gt.matched_entity_ids) for x in l.split(",") if x}
s1=pd.read_parquet("cache/train_s1.parquet")
c=pd.concat([pd.read_parquet("cache/train_s2.parquet"),pd.read_parquet("cache/train_s3.parquet")])
c["owned"]=c.entity_id.isin(own)
k=lambda s: re.sub(r"[^a-z0-9]","",s.lower())
s1["k"]=s1.business_name.map(k); c["k"]=c.business_name.map(k)
s1c=s1.groupby("k").size()
c["s1_same_name"]=c.k.map(s1c).fillna(0)
print("P(exact-norm-name exists in S1) owned vs not:", c.groupby("owned").s1_same_name.apply(lambda x:(x>0).mean()).to_dict())
# addr
ka=lambda s: re.sub(r"[^a-z0-9]","",s.lower())[:25]
s1["ka"]=s1.business_address.map(ka); c["ka"]=c.business_address.map(ka)
s1a=s1.groupby("ka").size(); c["s1_same_addr"]=c.ka.map(s1a).fillna(0)
print("P(addr prefix exists in S1) owned vs not:", c.groupby("owned").s1_same_addr.apply(lambda x:(x>0).mean()).to_dict())
print("\n--- unmatched samples")
for cty in ["US","India"]:
    print(c[(~c.owned)&(c.country==cty)].sample(25,random_state=1)[["entity_id","business_name","business_address"]].to_string())
# singletons
sing=set(gt.source1_entity_id[gt.matched_entity_ids==""])
ss=s1[s1.entity_id.isin(sing)]
ck=c.groupby("k").size()
print("singleton S1 with exact-norm name in S2/S3:", ss.k.map(ck).notna().mean(), " non-singleton:", s1[~s1.entity_id.isin(sing)].k.map(ck).notna().mean())
print(ss.sample(15,random_state=2)[["business_name","business_address","country"]].to_string())
