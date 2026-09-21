#!/bin/bash
# docs/08 step 1: the single-stage dose-response, budgets of 49 and 488 rows.
# Resumable: a cell whose JSON exists is skipped, so relaunching after a failure repeats
# nothing. WORKERS cells share the card (about 20 GB each with the full teacher resident).
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
LOG=/root/step1.log
WORKERS=${WORKERS:-3}
BUDGETS=${BUDGETS:-"49 488"}
SEEDS=${SEEDS:-"0 1"}
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%dT%H%M%SZ)"    # rotate, never append

# The schedule is part of the pre-registration. Assert it, because a config that drifted
# between two launches is what made the first replication pass worthless (docs/07).
python - $BUDGETS <<'PY' || { echo "CONFIG DRIFT, not launching" >> /root/step1.log; exit 1; }
import sys, yaml
want = {"cap_steps": 6104, "batch": 8, "seq_len": 2048, "lr": 3e-4, "warmup": 100, "weight_decay": 0.1,
        "val_every": 100, "patience": 10, "cooldown_frac": 0.1, "min_cooldown": 50}
for b in sys.argv[1:]:
    c = yaml.safe_load(open(f"experiments/phase2b/configs/cell-b{b}.yaml"))
    for k, v in want.items():
        assert c["train"][k] == v, f"cell-b{b}: train.{k} is {c['train'][k]}, pre-registered {v}"
    assert c["stage"] == 2 and c["heldout_seqs"] == 64 and c["val_seqs_max"] == 32, b
    assert c["cf"] == {"M": 64, "freqs": [0.5, 1.0, 1.5, 2.0], "seed": 0}, c["cf"]
print("schedule matches the pre-registration")
PY

for B in $BUDGETS; do
  if [ ! -f out/phase2b/harvest-b$B/run.json ]; then
    echo "== harvest b$B start $(date -u +%H:%M:%S)" >> $LOG
    python experiments/phase0/run.py experiments/phase2b/configs/harvest-b$B.yaml > /root/harvest_b$B.log 2>&1 \
      || { echo "== harvest b$B FAILED" >> $LOG; exit 1; }
  fi
  echo "== harvest b$B ready $(date -u +%H:%M:%S)" >> $LOG
done

# budget m lambda wd seed; the long cells first so the tail is not one job on an idle card
JOBS=/root/step1.jobs; : > $JOBS
for B in $BUDGETS; do for S in $SEEDS; do
  for cfg in "32 0 0.1" "8 0 0.1" "8 3 0.1" "8 30 0.1" "2 0 0.1"; do echo "$B $cfg $S" >> $JOBS; done
done; done
for B in $BUDGETS; do for S in $SEEDS; do
  for cfg in "0 0 0.1" "0 3 0.1" "0 30 0.1" "0 0 1.0"; do echo "$B $cfg $S" >> $JOBS; done
done; done
[ -n "${EXTRA_JOBS:-}" ] && printf '%s\n' "$EXTRA_JOBS" >> $JOBS
echo "== $(wc -l < $JOBS) cells queued, $WORKERS workers $(date -u +%H:%M:%S)" >> $LOG

run_one () {
  local B=$1 M=$2 LAM=$3 WD=$4 S=$5
  local name="b${B}_st2_m${M}_cf${LAM}_wd${WD}_s${S}"
  # %g formatting in cell.py drops a trailing .0, so 1.0 is written as 1
  name=$(python -c "print(f'b$B' + '_st2_m' + f'{float($M):g}' + '_cf' + f'{float($LAM):g}' + '_wd' + f'{float($WD):g}' + '_s$S')")
  if [ -f "out/phase2b/cells/$name.json" ]; then echo "   skip $name (exists)" >> /root/step1.log; return 0; fi
  echo "   start $name $(date -u +%H:%M:%S)" >> /root/step1.log
  if python experiments/phase2b/cell.py experiments/phase2b/configs/cell-b$B.yaml \
       --m $M --cf-lambda $LAM --wd $WD --seed $S > /root/cell_$name.log 2>&1; then
    echo "   ok $name $(date -u +%H:%M:%S)" >> /root/step1.log
  else
    echo "   FAILED $name $(date -u +%H:%M:%S)" >> /root/step1.log
  fi
}
export -f run_one
xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _ < $JOBS
echo "STEP1-DONE $(date -u +%H:%M:%S) ok=$(grep -c '   ok ' $LOG) failed=$(grep -c FAILED $LOG)" >> $LOG
