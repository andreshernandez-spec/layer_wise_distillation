#!/bin/bash
# The heal curve, rerun once the top-k store exists, plus the equal-FLOPs random cell
# that G2's kill criterion needs. Two shards: one heal reaches about half the card.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
LOG=/root/phase2.log
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml

# wait for the harvest process to exit, not for a chunk count: the count depends on
# the harvest batch size (5119 rows at batch 8 is 640 chunks, not 1280) and a
# hard-coded threshold waits forever.
while pgrep -u root -f '[e]xperiments/phase0' > /dev/null; do sleep 30; done
if [ "$(ls out/harvest-1.4b/topk/*.npz 2>/dev/null | wc -l)" -lt 1 ]; then
  echo "== NO TOPK STORE, aborting heal $(date -u +%H:%M:%S)" >> $LOG; exit 1
fi
echo "== heal curve (rerun) start $(date -u +%H:%M:%S) chunks=$(ls out/harvest-1.4b/topk/*.npz | wc -l)" >> $LOG
for shard in 0 1; do
  (
    i=0
    for init in random stagewise oracle; do
      for t in 1e5 1e6 3e6 1e7; do
        i=$((i+1)); [ $((i % 2)) -ne $shard ] && continue
        m=C; [ $init = oracle ] && m=R
        python experiments/phase2/heal.py $CFG --init $init --measure $m --q 1e8 --tokens $t >> /root/heal_$shard.log 2>&1 \
          || echo "  heal $init $t FAILED" >> $LOG
      done
    done
  ) &
done
wait
echo "== heal curve done $(date -u +%H:%M:%S) runs=$(ls out/phase2-1.4b/heal_*.json 2>/dev/null | wc -l)" >> $LOG

echo "== equal-flops random 1.74e8 start $(date -u +%H:%M:%S)" >> $LOG
python experiments/phase2/heal.py $CFG --init random --measure C --q 1e8 --tokens 1.74e8 >> /root/heal_eq.log 2>&1 \
  || echo "  equal-flops FAILED" >> $LOG
echo "PHASE2-HEAL-DONE $(date -u +%H:%M:%S)" >> $LOG
