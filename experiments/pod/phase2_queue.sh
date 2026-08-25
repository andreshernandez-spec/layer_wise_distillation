#!/bin/bash
# Phase 2 on the pod: the C stack, then every composed experiment for both stacks.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml

# Two shards: one stage cell reaches only ~54% of an A100, so a sequential stack
# leaves half the card idle for two hours (measured 24 Aug 2026).
echo "== C stack start $(date -u +%H:%M:%S)" >> $LOG
for i in 0 1; do
  setsid nohup python experiments/phase2/train_stages.py $CFG --measure C --q 1e8 --shard $i/2     > /root/cstack_$i.log 2>&1 < /dev/null &
done
while pgrep -f '[t]rain_stages.py' > /dev/null; do sleep 30; done
echo "== C stack done $(date -u +%H:%M:%S) stages=$(ls out/phase2-1.4b/C_mix*_stage*.json 2>/dev/null | wc -l)" >> $LOG

for m in R C; do
  echo "== drift $m $(date -u +%H:%M:%S)" >> $LOG
  python experiments/phase2/drift.py $CFG --measure $m --q 1e8 --rows 16 --batch 2 >> /root/drift_$m.log 2>&1
done
echo "== drift done $(date -u +%H:%M:%S)" >> $LOG

for init in random stagewise oracle; do
  for t in 1e5 1e6 3e6 1e7; do
    m=C; [ $init = oracle ] && m=R
    echo "== heal $init $t $(date -u +%H:%M:%S)" >> $LOG
    python experiments/phase2/heal.py $CFG --init $init --measure $m --q 1e8 --tokens $t >> /root/heal.log 2>&1
  done
done
echo "== heal done $(date -u +%H:%M:%S)" >> $LOG

for st in 1 3 5; do
  echo "== dagger stage$st $(date -u +%H:%M:%S)" >> $LOG
  python experiments/phase2/dagger.py $CFG --stage $st --measure C --q 1e8 --q-retrain 1e7 --on-policy 0.5 >> /root/dagger.log 2>&1
done
echo "PHASE2-POD-DONE $(date -u +%H:%M:%S)" >> $LOG
