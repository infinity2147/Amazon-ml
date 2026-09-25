import pandas as pd, re
s1=pd.read_parquet("cache/test_s1.parquet"); 
c=pd.concat([pd.read_parquet("cache/test_s2.parquet"),pd.read_parquet("cache/test_s3.parquet")])
f1=s1[s1.country=="France"]; fc=c[c.country=="France"]
print(len(f1),len(fc), c.country.value_counts().to_dict())
print(f1.sample(20,random_state=0).to_string())
print(fc.sample(30,random_state=0).to_string())
# find matches via street number + first street word
def key(a):
    m=re.match(r"\s*(\d+)\s+(.*?),",a); return None
for _,r in f1.sample(8,random_state=5).iterrows():
    w=[t for t in re.findall(r"[a-zà-ÿ]{5,}",r.business_name.lower())][:1]
    st=re.findall(r"[A-Za-zÀ-ÿ]{5,}",r.business_address)[:1]
    if not w or not st: continue
    m=fc[fc.business_name.str.lower().str.contains(w[0],regex=False)&fc.business_address.str.lower().str.contains(st[0].lower(),regex=False)]
    print(f"\n### {r.business_name} | {r.business_address}")
    print(m.head(8).to_string())
