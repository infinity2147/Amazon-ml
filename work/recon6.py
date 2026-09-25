import pandas as pd, re, numpy as np
from rapidfuzz import fuzz
gt=pd.read_parquet("cache/gt.parquet"); gt=gt[gt.matched_entity_ids!=""].sample(60000,random_state=0)
s1=pd.read_parquet("cache/train_s1.parquet").set_index("entity_id")
c=pd.concat([pd.read_parquet("cache/train_s2.parquet"),pd.read_parquet("cache/train_s3.parquet")]).set_index("entity_id")
P=[(s,x) for s,l in zip(gt.source1_entity_id,gt.matched_entity_ids) for x in l.split(",")]
L=s1.loc[[p[0] for p in P]].reset_index(); R=c.loc[[p[1] for p in P]].reset_index()
M=pd.DataFrame({"ctry":L.country,"src":R.entity_id.str[:2],"ln":L.business_name,"rn":R.business_name,"la":L.business_address,"ra":R.business_address})
lat=lambda s: all(ord(ch)<0x250 for ch in s if ch.isalpha())
low=lambda s: re.sub(r"[^a-z0-9 ]"," ",s.lower())
def nums(s): return set(re.findall(r"\d+",s))
def firstnum(s):
    m=re.search(r"\d+",s); return m.group(0).lstrip("0") if m else ""
M["r_native_name"]=~M.rn.map(lat)
M["r_web"]=M.rn.str.contains(r"\.com\b|^#|\.co\b",case=False,regex=True)
M["name_exact_ci"]=M.ln.str.lower()==M.rn.str.lower()
M["name_tset"]=[fuzz.token_set_ratio(low(a),low(b)) for a,b in zip(M.ln,M.rn)]
M["name_lowsim"]=(M.name_tset<50)&~M.r_native_name&~M.r_web
M["r_addr_empty"]=M.ra==""
M["addr_exact_ci"]=M.la.str.lower()==M.ra.str.lower()
ln=[nums(a) for a in M.la]; rn=[nums(a) for a in M.ra]
M["hn_eq"]=[ (firstnum(a)==firstnum(b)) if firstnum(a) and firstnum(b) else np.nan for a,b in zip(M.la,M.ra)]
M["nums_disjoint"]=[bool(a and b and not {x.lstrip('0') for x in a}&{x.lstrip('0') for x in b}) for a,b in zip(ln,rn)]
M["addr_tset"]=[fuzz.token_set_ratio(low(a),low(b)) if b else np.nan for a,b in zip(M.la,M.ra)]
M["r_native_addr"]=~M.ra.map(lat)
M["pobox"]=M.ra.str.contains(r"PO BOX|PMB",case=False)
M["null"]=M.ra.str.contains("<NULL>|NULL",case=False)|M.rn.str.contains("<NULL>",case=False)
print(M.groupby(["ctry","src"]).mean(numeric_only=True).round(3).T)
print("\nLow-sim (non-native, non-web) name examples:")
print(M[M.name_lowsim].sample(40,random_state=1)[["ln","rn","la","ra"]].to_string())
print("\nnums disjoint examples:")
print(M[M.nums_disjoint].sample(25,random_state=1)[["ln","rn","la","ra"]].to_string())
M.to_parquet("cache/matched_sample.parquet")
