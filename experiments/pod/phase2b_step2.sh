#!/bin/bash
# docs/08 step 2 at one budget: stacks for the control and the chosen treatment, composed,
# healed on the budget; plain KD from random init healed the same way. Resumable.
#
#   BUDGET=49 CTRL_WD=0.1 TREAT_M=8 TREAT_CF=0 bash experiments/pod/phase2b_step2.sh
#
# The treatment and the control's weight decay are whatever gate S1 chose ON VALIDATION.
set -u
cd /root/lwd
. .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
B=${BUDGET:?}; CW=${CTRL_WD:?}; TM=${TREAT_M:?}; TC=${TREAT_CF:?}
WORKERS=${WORKERS:-3}
LOG=/root/step2_b$B.log
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%dT%H%M%SZ)"
CELL=experiments/phase2b/configs/cell-b$B.yaml
HEAL=experiments/phase2b/configs/heal-b$B.yaml
HV=out/phase2b/harvest-b$B-full

python - "$B" <<'PY' || { echo "CONFIG DRIFT, not launching" >> $LOG; exit 1; }
import sys, yaml
b = sys.argv[1]
h = yaml.safe_load(open(f"experiments/phase2b/configs/heal-b{b}.yaml"))["heal"]
want = {"cap_tokens": 200000000, "batch": 1, "warmup": 500, "val_every": 200, "patience": 5,
        "cooldown_frac": 0.1, "min_cooldown": 50, "w_kd": 0.9, "weight_decay": 0.1}
for k, v in want.items():
    assert h[k] == v, f"heal-b{b}: heal.{k} is {h[k]}, pre-registered {v}"
print("heal schedule matches the pre-registration")
PY

if [ ! -f $HV/run.json ]; then
  echo "== full harvest b$B start $(date -u +%H:%M:%S)" >> $LOG
  python experiments/phase0/run.py experiments/phase2b/configs/harvest-b$B-full.yaml > /root/harvest_b${B}_full.log 2>&1 \
    || { echo "== full harvest FAILED" >> $LOG; exit 1; }
fi
echo "== full harvest b$B ready $(date -u +%H:%M:%S) topk=$(ls $HV/topk/*.npz | wc -l)" >> $LOG

fmt () { python -c "print(f'{float($1):g}')"; }
stage_one () {   # stage m cf wd
  local n="b${B}_st$1_m$(fmt $2)_cf$(fmt $3)_wd$(fmt $4)_s0"
  [ -f out/phase2b/stacks/$n.json ] && { echo "   skip $n" >> $LOG; return 0; }
  echo "   start $n $(date -u +%H:%M:%S)" >> $LOG
  python experiments/phase2b/cell.py $CELL --stage $1 --m $2 --cf-lambda $3 --wd $4 --seed 0 \
      --harvest $HV --out out/phase2b/stacks > /root/stack_$n.log 2>&1 \
    && echo "   ok $n $(date -u +%H:%M:%S)" >> $LOG || echo "   FAILED $n $(date -u +%H:%M:%S)" >> $LOG
}
heal_one () {    # init m cf wd lr seed
  local tag="$1 m=$2 cf=$3 lr=$5 h$6"
  echo "   start heal $tag $(date -u +%H:%M:%S)" >> $LOG
  python experiments/phase2b/heal.py $HEAL --init $1 --m $2 --cf-lambda $3 --wd $4 --lr $5 --seed $6 \
      > "/root/heal_b${B}_$1_m$2_cf$3_lr$5_h$6.log" 2>&1 \
    && echo "   ok heal $tag $(date -u +%H:%M:%S)" >> $LOG \
    || { grep -q "exists; pass" "/root/heal_b${B}_$1_m$2_cf$3_lr$5_h$6.log" && echo "   skip heal $tag" >> $LOG \
         || echo "   FAILED heal $tag $(date -u +%H:%M:%S)" >> $LOG; }
}
export -f stage_one heal_one fmt; export B LOG CELL HEAL HV

# 1. the two stacks, treatment stages first (they run to the cap, the control stops early)
{ for k in 0 1 2 3 4 5; do echo "$k $TM $TC 0.1"; done
  for k in 0 1 2 3 4 5; do echo "$k 0 0 $CW"; done; } > /root/step2_b$B.stages
xargs -P "$WORKERS" -L 1 bash -c 'stage_one "$@"' _ < /root/step2_b$B.stages
n=$(ls out/phase2b/stacks/b${B}_st*_s0.json 2>/dev/null | wc -l)
[ "$n" -ge 12 ] || { echo "== only $n of 12 stage cells, not healing" >> $LOG; echo "STEP2-FAILED b$B" >> $LOG; exit 1; }
echo "== stacks ready $(date -u +%H:%M:%S)" >> $LOG

# 2. the learning rate of each arm, on heal seed 0
{ for lr in 3e-5 1e-4 3e-4; do
    echo "stack $TM $TC 0.1 $lr 0"; echo "stack 0 0 $CW $lr 0"; echo "random 0 0 0.1 $lr 0"; done; } > /root/step2_b$B.heals0
xargs -P "$WORKERS" -L 1 bash -c 'heal_one "$@"' _ < /root/step2_b$B.heals0

# 3. seeds 1 and 2 at the rate validation chose for each arm
python - "$B" "$TM" "$TC" "$CW" > /root/step2_b$B.heals12 <<'PY'
import glob, json, sys
B, tm, tc, cw = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
best = {}
for f in glob.glob("out/phase2b/heals/heal_*.json"):
    r = json.load(open(f))
    if r["budget"]["budget_rows"] != B or r["heal_seed"] != 0:
        continue
    arm = "random" if r["init"] == "random" else ("treat" if (r["m"], r["cf_lambda"]) == (tm, tc) else "ctrl")
    if arm not in best or r["val_loss"] < best[arm][0]:
        best[arm] = (r["val_loss"], r["lr"])
for arm, (m, c, w) in {"treat": (tm, tc, 0.1), "ctrl": (0, 0, cw), "random": (0, 0, 0.1)}.items():
    for s in (1, 2):
        init = "random" if arm == "random" else "stack"
        print(f"{init} {m:g} {c:g} {w:g} {best[arm][1]:g} {s}")
PY
xargs -P "$WORKERS" -L 1 bash -c 'heal_one "$@"' _ < /root/step2_b$B.heals12
echo "STEP2-DONE b$B $(date -u +%H:%M:%S) ok=$(grep -c '   ok ' $LOG) failed=$(grep -c FAILED $LOG)" >> $LOG
