#!/bin/bash
# G2's kill criterion: random init healed at the stagewise arm's whole end-to-end FLOPs.
# The budget is read from flops.json, not typed. The first run of this cell was typed
# from the wrong line of the report (1.74e8, the harvest+stages part) and so was 5.3%
# short, in the direction that flatters the stagewise arm.
set -eu
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml

python experiments/phase2/flops.py $CFG --heal-tokens 1e7 > /root/flops.log 2>&1
T=$(python -c "import json; print(f\"{json.load(open('out/phase2-1.4b/flops.json'))['equal_flops_tokens']:.5g}\")")
case "$T" in ''|*[!0-9.e+-]*) echo "bad budget '$T'" >> $LOG; exit 1;; esac

echo "== equal-flops random $T at lr 5e-5 $(date -u +%H:%M:%S)" >> $LOG
if python experiments/phase2/heal.py $CFG --init random --measure C --q 1e8 \
     --tokens "$T" --lr 5e-5 --tag _eqflops >> /root/heal_eq3.log 2>&1; then
  echo "== equal-flops done $(date -u +%H:%M:%S)" >> $LOG
else
  echo "== equal-flops FAILED $(date -u +%H:%M:%S)" >> $LOG
fi
echo "EQUAL-FLOPS-DONE $(date -u +%H:%M:%S)" >> $LOG
