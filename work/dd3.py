# Deep dive 3: France (test) -- formats, and hand-found likely match groups
import pandas as pd, re
pd.set_option("display.width",260); pd.set_option("display.max_colwidth",80)
s1=pd.read_parquet("cache/test_s1.parquet"); c=pd.concat([pd.read_parquet("cache/test_s2.parquet"),pd.read_parquet("cache/test_s3.parquet")],ignore_index=True)
f1=s1[s1.country=="France"].copy(); fc=c[c.country=="France"].copy()
print("France S1",len(f1),"cands",len(fc),"ratio",round(len(fc)/len(f1),2), "| US ratio", round((c.country=="US").sum()/(s1.country=="US").sum(),2), "India ratio", round((c.country=="India").sum()/(s1.country=="India").sum(),2))
print("S2/S3 split France:", fc.entity_id.str[:2].value_counts().to_dict())
last=f1.business_address.str.split(",").str[-1].str.strip()
print("France S1 last component top:", last.value_counts().head(15).to_dict())
lastc=fc.business_address.str.split(",").str[-1].str.strip()
print("France cand last component top:", lastc.value_counts().head(25).to_dict())
print("France S1 # of comps:", f1.business_address.str.count(",").value_counts().head().to_dict())
nk=lambda s: re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
f1["nk"]=f1.business_name.map(nk); print("France S1 name dup rate:", f1.nk.duplicated(keep=False).mean().round(3), f1.nk.value_counts().head(6).to_dict())
# street key: house number + first street word
def key(a):
    m=re.search(r"(\d+)\s*(?:bis|ter)?\s*,?\s*(?:rue|r\.?|avenue|av\.?|bd|boulevard|place|pl|chemin|allee|allée|impasse|quai|cours|route|rte)\s+(?:de\s+la\s+|de\s+l'|du\s+|des\s+|de\s+)?([a-zà-ÿ'-]+)",a.lower())
    return (m.group(1).lstrip("0"),m.group(2)) if m else None
f1["k"]=f1.business_address.map(key); fc["k"]=fc.business_address.map(key)
print("key coverage S1",f1.k.notna().mean().round(3),"cands",fc.k.notna().mean().round(3))
g=fc.dropna(subset=["k"]).groupby("k")
for r in f1.dropna(subset=["k"]).sample(10,random_state=4).itertuples():
    print(f"\n### S1 | {r.business_name} | {r.business_address}")
    if r.k in g.groups:
        for x in fc.loc[g.groups[r.k]].head(8).itertuples(): print(f"   {x.entity_id[:2]} | {x.business_name} | {x.business_address}")
