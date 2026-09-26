#!/bin/bash
C=/home/24b4518/ml-projects/work/ce
until [ -f $C/scored_test_A_part0.parquet ] && [ -f $C/scored_test_A_part1.parquet ]; do sleep 20; done
sleep 5
source /home/24b4518/ml-projects/.venv/bin/activate
python -c "
import pandas as pd
C='$C/'
c=pd.concat([pd.read_parquet(C+'scored_test_A_part0.parquet'),pd.read_parquet(C+'scored_test_A_part1.parquet')],ignore_index=True)
c.to_parquet(C+'ce_test_final.tmp.parquet'); print('merged test pairs', len(c))"
mv $C/ce_test_final.tmp.parquet $C/ce_test_final.parquet
