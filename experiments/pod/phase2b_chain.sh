#!/bin/bash
# bootstrap -> docs/08 step 1. Nothing waits on a human; every failure leaves a marker
# the watchdog reads.
set -u
cd /root/lwd
LOG=/root/chain.log
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date -u +%Y%m%dT%H%M%SZ)"
echo "== chain start $(date -u +%H:%M:%S)" >> $LOG
HARVEST_CFG=experiments/phase2b/configs/harvest-b49.yaml bash experiments/pod/bootstrap.sh
if ! grep -q "bootstrap done" /root/bootstrap.log 2>/dev/null; then
  echo "CHAIN-FAILED bootstrap $(date -u +%H:%M:%S)" >> $LOG; tail -5 /root/bootstrap.log >> $LOG; exit 1
fi
echo "== bootstrap ok $(date -u +%H:%M:%S)" >> $LOG
bash experiments/pod/phase2b_step1.sh
if grep -q "STEP1-DONE" /root/step1.log 2>/dev/null; then
  echo "CHAIN-DONE $(date -u +%H:%M:%S)" >> $LOG
else
  echo "CHAIN-FAILED step1 $(date -u +%H:%M:%S)" >> $LOG
fi
