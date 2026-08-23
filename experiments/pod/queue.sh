#!/bin/bash
# Everything Phase 1 needs, in one go, on the pod. Each grid runs as N sharded
# workers; a grid finishes when all its workers exit. Cells resume by skipping
# existing JSON, so a killed worker or a reclaimed pod costs one cell.
#   experiments/pod/queue.sh [n_workers]
set -u
cd /root/lwd
N=${1:-4}
LOG=/root/queue.log
grid() {  # $1 cell config, $2 grid, $3 label
  echo "== $3 start $(date -u +%H:%M:%S)" >> $LOG
  bash experiments/pod/run_sweep.sh "$1" "$2" "$N" "$3" >> $LOG 2>&1
  while pgrep -f "sweep.py .*$2" > /dev/null; do sleep 30; done
  echo "== $3 done $(date -u +%H:%M:%S) cells=$(ls out/phase1-1.4b-a100/*.json 2>/dev/null | wc -l)" >> $LOG
}
# full: the whole design including both seeds, so the separate seeds grid is redundant
grid experiments/pod/cell-1.4b-pod.yaml experiments/phase1/configs/grid-1.4b-full.yaml full
grid experiments/pod/cell-1.4b-a46-pod.yaml experiments/phase1/configs/grid-1.4b-a46.yaml a46
grid experiments/pod/cell-1.4b-pod.yaml experiments/phase1/configs/grid-1.4b-long.yaml long
echo "QUEUE-DONE $(date -u +%H:%M:%S)" >> $LOG
