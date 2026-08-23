#!/bin/bash
# Runs the Phase 0 gate checks on a finished harvest, then the Phase 1 first pass.
# Sequential, each step under the memory watchdog: one heavy job at a time.
#   experiments/phase0/post_harvest.sh out/harvest-1.4b
set -u
cd /home/ahernandez/private/open-source/layer_wise_distillation
H=${1:-out/harvest-1.4b}
G="experiments/tools/memguard.sh 24 6 --"
LOG=$H/post_harvest.log
{
echo "== compare interim vs full stats"; $G python experiments/phase0/compare_stats.py out/harvest-1.4b-interim $H 2>&1
echo "== verify anchors (G0.3)"; $G python experiments/phase0/verify_anchors.py $H --device cuda --batch 4 2>&1 | grep -v "Loading\|Fetching"
echo "== bridge oracle (C0.5)"; $G python experiments/phase0/bridge_oracle.py $H --widths 1024 512 --max-pos 200000 2>&1 | grep -v "Loading\|Fetching"
echo "== attention entropy baseline"; $G python experiments/phase0/attn_entropy.py EleutherAI/pythia-1.4b out/slice/anchor_rows.npy $H/attn_entropy.npz --rows 16 --batch 1 2>&1 | grep -v "Loading\|Fetching"
echo "== phase 1 first pass"; $G python experiments/phase1/sweep.py experiments/phase1/configs/cell-1.4b.yaml experiments/phase1/configs/grid-1.4b-first.yaml 2>&1 | grep -E "cells to run|^stitch|^attn|^eval|final_eval|failed|Error|Traceback|MEMGUARD" | cut -c1-400
echo "POST-HARVEST-DONE"
} > $LOG 2>&1
