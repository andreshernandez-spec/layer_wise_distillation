#!/bin/bash
# Resume Phase 2 after the C stack and drift: the heal curve, DAgger, and the
# equal-FLOPs random cell that G2's kill criterion actually needs.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml

# heal curve: two shards, since one heal reaches only ~half the card
for shard in 0 1; do
  (
    i=0
    for init in random stagewise oracle; do
      for t in 1e5 1e6 3e6 1e7; do
        i=$((i+1)); [ $((i % 2)) -ne $shard ] && continue
        m=C; [ $init = oracle ] && m=R
        echo "== heal $init $t start $(date -u +%H:%M:%S)" >> $LOG
        python experiments/phase2/heal.py $CFG --init $init --measure $m --q 1e8 --tokens $t >> /root/heal_$shard.log 2>&1
      done
    done
  ) &
done
wait
echo "== heal curve done $(date -u +%H:%M:%S) runs=$(ls out/phase2-1.4b/heal_*.json 2>/dev/null | wc -l)" >> $LOG

for st in 1 3 5; do
  echo "== dagger stage$st $(date -u +%H:%M:%S)" >> $LOG
  python experiments/phase2/dagger.py $CFG --stage $st --measure C --q 1e8 --q-retrain 1e7 --on-policy 0.5 >> /root/dagger.log 2>&1
done
echo "== dagger done $(date -u +%H:%M:%S)" >> $LOG

# the cell G2's kill criterion needs: random init at the stagewise arm's total FLOPs
echo "== equal-flops random 1.74e8 start $(date -u +%H:%M:%S)" >> $LOG
python experiments/phase2/heal.py $CFG --init random --measure C --q 1e8 --tokens 1.74e8 >> /root/heal_eq.log 2>&1
echo "== equal-flops random done $(date -u +%H:%M:%S)" >> $LOG
echo "PHASE2-POD-DONE $(date -u +%H:%M:%S)" >> $LOG
