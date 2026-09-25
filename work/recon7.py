import pandas as pd, re
from collections import Counter
from rapidfuzz import fuzz
M=pd.read_parquet("cache/matched_sample.parquet")
bc=lambda s: re.sub(r"\s+"," ",re.sub(r"[^a-z0-9]+"," ",s.lower())).strip()
def mine(a_col,b_col,sub):
    cnt=Counter(); added=Counter(); dropped=Counter()
    for a,b in zip(sub[a_col],sub[b_col]):
        A,B=set(bc(a).split()),set(bc(b).split())
        da,db=A-B,B-A
        for t in db: added[t]+=1
        for t in da: dropped[t]+=1
        if not da or not db or len(da)>3 or len(db)>3: continue
        for x in da:
            y=max(db,key=lambda t:(t[0]==x[0],fuzz.ratio(x,t)))
            if y[0]==x[0] or fuzz.ratio(x,y)>=60: cnt[(x,y)]+=1
    return cnt,added,dropped
for ctry in ["US","India"]:
    sub=M[M.ctry==ctry]
    for col in ["n","a"]:
        cnt,added,dropped=mine("l"+col,"r"+col,sub)
        print(f"\n==== {ctry} {col} rewrites:", [(f"{a}->{b}",c) for (a,b),c in cnt.most_common(45)])
        print(f"---- {ctry} {col} tokens ADDED in S2/S3:", added.most_common(40))
        print(f"---- {ctry} {col} tokens DROPPED:", dropped.most_common(30))
