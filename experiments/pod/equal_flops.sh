#!/bin/bash
# The cell G2's kill criterion needs, with a store that actually loops.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml
echo "== equal-flops random 1.74e8 at lr 5e-5 (looping store) $(date -u +%H:%M:%S)" >> $LOG
if python experiments/phase2/heal.py $CFG --init random --measure C --q 1e8 --tokens 1.74e8 --lr 5e-5 >> /root/heal_eq2.log 2>&1; then
  echo "== equal-flops done $(date -u +%H:%M:%S)" >> $LOG
else
  echo "== equal-flops FAILED $(date -u +%H:%M:%S)" >> $LOG
fi
echo "EQUAL-FLOPS-DONE $(date -u +%H:%M:%S)" >> $LOG
