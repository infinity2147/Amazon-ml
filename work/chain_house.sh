set -e
cd /home/24b4518/ml-projects/work
source ../.venv/bin/activate
export CACHE=/home/24b4518/ml-projects/work/cache/v2/
python run_keyview.py train r3 > logs_keyview_r3_train.txt 2>&1
python run_keyview.py test r3 > logs_keyview_r3_test.txt 2>&1
python run_feats.py train r3 > logs_feats_r3_train.txt 2>&1
python run_feats.py test r3 > logs_feats_r3_test.txt 2>&1
python make_tlv.py train_feats_r3_f2 train_feats_tlv3_f2 > logs_make_tlv3.txt 2>&1
echo CHAIN_DONE >> logs_make_tlv3.txt
