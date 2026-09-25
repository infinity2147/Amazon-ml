set -e
cd /home/24b4518/ml-projects/work
../.venv/bin/python run_keyview.py test > logs_keyview_test.txt 2>&1
../.venv/bin/python run_feats.py train r2 > logs_feats_r2_train.txt 2>&1
../.venv/bin/python run_feats.py test r2 > logs_feats_r2_test.txt 2>&1
../.venv/bin/python run_full.py r2_f2 0.25 sub3 > logs_full_sub3.txt 2>&1
