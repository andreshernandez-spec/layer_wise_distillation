#!/bin/bash
# The cold-start arm diverged at 1e7 tokens (step 1285) even with warmup 500, and the
# 1e6 probe was too short to see it: probe at the length you will run. Finds the
# longest-surviving lr, then spends it on the equal-FLOPs cell.
# Safe to run beside dagger_stack.py: a random init never loads stage checkpoints.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml
BEST=""
for lr in 3e-5 5e-5; do
  echo "== random 1e7 at lr $lr $(date -u +%H:%M:%S)" >> $LOG
  if python experiments/phase2/heal.py $CFG --init random --measure C --q 1e8 --tokens 1e7 --lr $lr >> /root/rnd.log 2>&1; then
    cp out/phase2-1.4b/heal_random_C_t1e07_s0.json out/phase2-1.4b/rnd_1e7_lr$lr.json
    echo "   lr $lr survived" >> $LOG
    BEST=$lr
  else
    echo "   lr $lr diverged" >> $LOG
  fi
done
if [ -n "$BEST" ]; then
  echo "== equal-flops random 1.74e8 at lr $BEST $(date -u +%H:%M:%S)" >> $LOG
  if python experiments/phase2/heal.py $CFG --init random --measure C --q 1e8 --tokens 1.74e8 --lr $BEST >> /root/heal_eq.log 2>&1; then
    echo "== equal-flops done $(date -u +%H:%M:%S)" >> $LOG
  else
    echo "   equal-flops diverged at $BEST" >> $LOG
  fi
else
  echo "== no stable lr found for the cold start at 1e7" >> $LOG
fi
echo "RANDOM-FIX-DONE $(date -u +%H:%M:%S)" >> $LOG
