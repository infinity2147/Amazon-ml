import csv, pandas as pd, sys
D="/home/24b4518/ml-projects/student_resource/dataset/"
for sp in ["train","test"]:
    for i in (1,2,3):
        df=pd.read_csv(f"{D}{sp}/{sp}_source{i}.tsv",sep="\t",dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE,encoding="utf-8")
        print(sp,i,df.shape, list(df.columns), "dup ids", df.entity_id.duplicated().sum(), flush=True)
        df.to_parquet(f"cache/{sp}_s{i}.parquet")
gt=pd.read_csv(D+"train/train_ground_truth.tsv",sep="\t",dtype=str,keep_default_na=False,quoting=csv.QUOTE_NONE)
print(gt.shape); gt.to_parquet("cache/gt.parquet")
