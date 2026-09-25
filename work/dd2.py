# Deep dive 2: singletons, distractors, chains, duplicates
import pandas as pd, numpy as np, re
pd.set_option("display.width",260); pd.set_option("display.max_colwidth",80)
s1=pd.read_parquet("cache/train_s1.parquet"); c=pd.concat([pd.read_parquet("cache/train_s2.parquet"),pd.read_parquet("cache/train_s3.parquet")],ignore_index=True)
gt=pd.read_parquet("cache/gt.parquet"); gt["ids"]=gt.matched_entity_ids.map(lambda s:[x for x in s.split(",") if x])
owner={x:s for s,l in zip(gt.source1_entity_id,gt.ids) for x in l}
c["owner"]=c.entity_id.map(owner)
nk=lambda s: re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
s1["nk"]=s1.business_name.map(nk); c["nk"]=c.business_name.map(nk)
print("S1 exact-name duplicates (chains):", s1.nk.duplicated(keep=False).mean().round(4), " top:", s1.nk.value_counts().head(8).to_dict())
s1["ak"]=s1.business_address.map(nk)
print("S1 exact-address duplicates:", s1.ak.duplicated(keep=False).mean().round(4))
print("S1 exact name+addr dup:", s1.duplicated(["nk","ak"],keep=False).mean().round(5))
sing=set(gt.source1_entity_id[gt.ids.str.len()==0])
S=s1[s1.entity_id.isin(sing)]
byname=c.groupby("nk")
print("\n### SINGLETONS and cands sharing their exact normalised name")
for r in S.sample(8,random_state=3).itertuples():
    print(f"\nS1 {r.country} | {r.business_name} | {r.business_address}")
    if r.nk in byname.groups:
        g=c.loc[byname.groups[r.nk]].head(4)
        for x in g.itertuples(): print(f"   {x.entity_id[:2]} owner={x.owner} | {x.business_name} | {x.business_address}")
print("\n### DISTRACTORS (no owner) random")
D=c[c.owner.isna()]
print(D.sample(16,random_state=5)[["entity_id","business_name","business_address","country"]].to_string(index=False))
# do distractor names equal some S1 name?
print("\ndistractor name == some S1 name:", D.nk.isin(set(s1.nk)).mean().round(3), " matched cand name == some S1 name:", c[c.owner.notna()].nk.isin(set(s1.nk)).mean().round(3))
# within-cand duplicates for distractors: groups of distractors that look like the same unseen business
print("distractor exact-name dup rate within distractors:", D.nk.duplicated(keep=False).mean().round(3))
