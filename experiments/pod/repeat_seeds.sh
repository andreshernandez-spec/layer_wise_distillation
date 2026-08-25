#!/bin/bash
# After the kill cell: repeat the two decisive 1e7 heals on a second heal trajectory.
# The DAgger gain after healing is 0.085 nats and Phase 1's seed spread on a comparable
# quantity was 0.106, so with one seed the gain is not distinguishable from zero. This
# gives the comparison an error bar. About 40 minutes.
#
# Neither run moves a checkpoint: the plain stack is loaded through --stack-suffix.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml

while pgrep -u root -f '[h]eal.py' >/dev/null; do sleep 60; done

echo "== repeat seed: dagger stack 1e7 h1 $(date -u +%H:%M:%S)" >> $LOG
python experiments/phase2/heal.py $CFG --init stagewise --measure C --q 1e8 \
  --tokens 1e7 --heal-seed 1 --tag _dagger >> /root/heal_rep.log 2>&1 \
  && echo "   dagger h1 ok" >> $LOG || echo "   dagger h1 FAILED" >> $LOG

echo "== repeat seed: plain stack 1e7 h1 $(date -u +%H:%M:%S)" >> $LOG
python experiments/phase2/heal.py $CFG --init stagewise --measure C --q 1e8 \
  --tokens 1e7 --heal-seed 1 --stack-suffix _preDAgger >> /root/heal_rep.log 2>&1 \
  && echo "   plain h1 ok" >> $LOG || echo "   plain h1 FAILED" >> $LOG

echo "REPEAT-SEEDS-DONE $(date -u +%H:%M:%S)" >> $LOG
