#!/bin/bash
# Four matched-schedule replicates, the measurement docs/06 and the paper most need.
#
# The published pairs are not replicates: the second run of each used warmup 50 because
# the config synced to the pod had drifted from the one the first run used, so they bound
# seed plus schedule together. Every cell here runs at the SAME schedule as the published
# one and differs only in --heal-seed.
#
# The control (oracle seed 0) must reproduce 3.4850. A fresh pod rebuilds the top-k store,
# so without it a difference could be the store rather than the seed.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
LOG=/root/seeds.log

# Rotate, do not append: a watchdog grepping this file must not see the previous
# attempt's output. Three false alarms on 25 Aug came from exactly that.
rotate_log () { [ -s "$1" ] && mv "$1" "$1.$(date -u +%Y%m%dT%H%M%SZ)"; :; }
rotate_log "$LOG"
CFG=experiments/phase2/configs/stack-1.4b-pod.yaml
python - <<'PY'
import yaml
c = yaml.safe_load(open("experiments/phase2/configs/stack-1.4b-pod.yaml"))
assert c["heal_warmup"] == 500, f"heal_warmup is {c['heal_warmup']}, the drift that caused this rerun"
assert c["heal_lr"] == 1e-4, c["heal_lr"]
print("config schedule OK")
PY

run () {  # name init measure ckpt-suffix tokens heal-seed extra...
  local name=$1 init=$2 m=$3 tok=$4 hs=$5; shift 5
  echo "== $name start $(date -u +%H:%M:%S)" >> $LOG
  if python experiments/phase2/heal.py $CFG --init $init --measure $m --q 1e8 \
       --tokens $tok --heal-seed $hs "$@" >> /root/$name.log 2>&1; then
    echo "   $name ok $(date -u +%H:%M:%S)" >> $LOG
  else
    echo "   $name FAILED $(date -u +%H:%M:%S)" >> $LOG
  fi
}

# the long pole first: 1.8367e8 tokens is about 2.8 h
run rnd_eq_h1 random C 1.8367e8 1 --lr 5e-5 --tag _eqflops

# these need stage checkpoints, uploaded while the above runs
while [ ! -f /root/CKPT_READY ]; do sleep 60; done
run oracle_h0 oracle R 1e7 0 --tag _control     # must reproduce 3.4850
run oracle_h1 oracle R 1e7 1
run sw_h1     stagewise C 1e7 1

echo "SEEDS-DONE $(date -u +%H:%M:%S)" >> $LOG
