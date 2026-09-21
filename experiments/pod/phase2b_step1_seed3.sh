#!/bin/bash
# docs/08 step 1, the third seed: once the 36 cells are in, the control and the treatment
# that VALIDATION chose at each budget get one more seed, so the gate is read on three.
# Queued behind phase2b_step1.sh; waits for its marker.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
LOG=/root/step1_seed3.log
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%dT%H%M%SZ)"
echo "== waiting for step 1 $(date -u +%H:%M:%S)" >> $LOG
while ! grep -q "STEP1-DONE" /root/step1.log 2>/dev/null; do sleep 60; done

python - > /root/step1_seed3.jobs <<'PY'
import glob, json
from collections import defaultdict
cells = defaultdict(list)
for f in glob.glob("out/phase2b/cells/*.json"):
    r = json.load(open(f))
    cells[(r["budget"]["budget_rows"], r["m"], r["cf_lambda"], r["weight_decay"])].append(r["val_stitch_delta"])
for B in sorted({k[0] for k in cells}):
    val = {k[1:]: sum(v) / len(v) for k, v in cells.items() if k[0] == B}
    ctrl = min((k for k in val if k[0] == 0 and k[1] == 0), key=val.get)
    treat = min((k for k in val if not (k[0] == 0 and k[1] == 0)), key=val.get)
    for m, lam, wd in (treat, ctrl):                   # the long one first
        print(f"{B} {m:g} {lam:g} {wd:g} 2")
PY
echo "== chosen on validation: $(tr '\n' ';' < /root/step1_seed3.jobs) $(date -u +%H:%M:%S)" >> $LOG

run_one () {
  local name=$(python -c "print('b$1_st2_m' + f'{float($2):g}' + '_cf' + f'{float($3):g}' + '_wd' + f'{float($4):g}' + '_s$5')")
  [ -f "out/phase2b/cells/$name.json" ] && { echo "   skip $name" >> /root/step1_seed3.log; return 0; }
  echo "   start $name $(date -u +%H:%M:%S)" >> /root/step1_seed3.log
  python experiments/phase2b/cell.py experiments/phase2b/configs/cell-b$1.yaml --m $2 --cf-lambda $3 --wd $4 --seed $5 \
      > /root/cell_$name.log 2>&1 \
    && echo "   ok $name $(date -u +%H:%M:%S)" >> /root/step1_seed3.log \
    || echo "   FAILED $name $(date -u +%H:%M:%S)" >> /root/step1_seed3.log
}
export -f run_one
xargs -P 4 -L 1 bash -c 'run_one "$@"' _ < /root/step1_seed3.jobs
echo "SEED3-DONE $(date -u +%H:%M:%S) ok=$(grep -c '   ok ' $LOG) failed=$(grep -c FAILED $LOG)" >> $LOG
