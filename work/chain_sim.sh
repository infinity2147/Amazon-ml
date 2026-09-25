set -e
cd /home/24b4518/ml-projects/work
while pgrep -f "run_sim.py" >/dev/null; do sleep 20; done
grep -q "saved" logs_sim.txt
../.venv/bin/python run_feats.py train sim2 > logs_feats_sim2.txt 2>&1
while pgrep -f "run_full.py r2_f2" >/dev/null; do sleep 30; done
SIM=1 TEST_TAG=r2_f2 ../.venv/bin/python run_full.py sim2_f2 0.25 sub4sim > logs_full_sub4sim.txt 2>&1
