#!/bin/bash
# Launch N sharded sweep workers over one grid. Each worker is one process; cells
# are skipped if their JSON exists, so workers never collide and a kill costs one cell.
#   experiments/pod/run_sweep.sh <cell-config> <grid> <n_workers> [label]
set -eu
cd /root/lwd
. .venv/bin/activate
CFG=$1; GRID=$2; N=$3; LABEL=${4:-sweep}
NGPU=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) setsid nohup python experiments/phase1/sweep.py "$CFG" "$GRID" \
    --shard "$i/$N" > "/root/${LABEL}_w$i.log" 2>&1 < /dev/null &
done
echo "launched $N workers over $NGPU gpu(s) for $GRID"
exit 0
