import sys, numpy as np, pandas as pd, pyarrow.dataset as pds, pyarrow as pa
sys.path.insert(0,"/home/24b4518/ml-projects/code/business_entity_resolution/src")
from er.decode import decode_all
pd.set_option("display.width",250)
W="cache/v1/"
O=pd.read_parquet(W+"oof_sub2.parquet",columns=["s1_id","cand_id","y","p1","p"])
ids=pd.read_parquet(W+"oof_r1_f2_ids.parquet").s1_id.tolist()
pred={k:set(v) for k,v in decode_all(O[["s1_id","cand_id","p"]],ids).items()}
O["pred"]=[c in pred[s] for s,c in zip(O.s1_id,O.cand_id)]
# who else claims this record: best p for the same record under a different S1
best_other=O.sort_values("p",ascending=False).groupby("cand_id").head(2)
top=O.loc[O.groupby("cand_id").p.idxmax(),["cand_id","s1_id"]].set_index("cand_id").s1_id
FN=O[(O.y==1)&(~O.pred)].copy()
FN["stolen"]=FN.cand_id.map(top)!=FN.s1_id           # another S1 got this record
cols=["s1_id","cand_id","r_addr_empty","r_known_frac","r_native","house_eq","n_tset","a_tset","has_alt","n_cands_s1","r_nnums","l_nnums"]
F=pds.dataset(W+"train_feats_r1_f2").to_table(columns=cols,filter=pds.field("cand_id").isin(pa.array(FN.cand_id.unique()))).to_pandas()
FN=FN.merge(F,on=["s1_id","cand_id"],how="left")
def tag(r):
    if r.stolen: return "claimed by another S1"
    if r.r_addr_empty: return "record has empty address"
    if r.r_known_frac==0: return "pseudo-word / unknown name"
    if r.house_eq==0: return "house number differs"
    if r.r_native: return "native-script name"
    if r.n_tset<70: return "name changed a lot"
    if r.a_tset<70: return "address changed a lot"
    return "other (both look similar)"
FN["tag"]=FN.apply(tag,axis=1)
print(f"missed-but-shortlisted true IDs: {len(FN):,}")
print(FN.groupby("tag").agg(n=("p","size"),median_p=("p","median"),share_p_gt_03=("p",lambda x:(x>0.3).mean())).sort_values("n",ascending=False).round(3).to_string())
s1=pd.read_parquet(W+"train_s1p.parquet",columns=["entity_id","business_name","business_address"]).set_index("entity_id")
cp=pd.read_parquet(W+"train_cp.parquet",columns=["entity_id","business_name","business_address"]).set_index("entity_id")
for t in FN.tag.value_counts().index[:6]:
    print(f"\n##### {t}")
    for r in FN[FN.tag==t].sample(min(6,(FN.tag==t).sum()),random_state=1).itertuples():
        a=s1.loc[r.s1_id]; b=cp.loc[r.cand_id]
        extra=f" | taken by {top[r.cand_id]}: {s1.loc[top[r.cand_id]].business_name} | {s1.loc[top[r.cand_id]].business_address}" if r.stolen else ""
        print(f"p={r.p:.2f}\n   S1: {a.business_name} | {a.business_address}\n   {r.cand_id[:2]}: {b.business_name} | {b.business_address}{extra}")
