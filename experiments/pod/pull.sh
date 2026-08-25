#!/bin/bash
# Pull pod results into the local repo. Run from the project root, repeatedly.
#   experiments/pod/pull.sh <ip> <port>
set -eu
IP=$1; PORT=$2
RS="ssh -i $HOME/.ssh/id_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p $PORT"
mkdir -p out/phase1-1.4b-a100 out/harvest-1.4b-a100
# never pull *.pt: a student stage is ~400 MB and 89 of them is 35 GB, which this
# link (773 kB/s) would take half a day to move. Spectral metrics are computed on
# the pod instead (experiments/pod/spectral.py) and come back as one small JSON.
rsync -az -e "$RS" --exclude="*.pt" root@$IP:/root/lwd/out/phase1-1.4b-a100/ out/phase1-1.4b-a100/
rsync -az -e "$RS" --include="stats_*.npz" --include="run.json" --exclude="*" \
    root@$IP:/root/lwd/out/harvest-1.4b/ out/harvest-1.4b-a100/
rsync -az -e "$RS" root@$IP:/root/queue.log root@$IP:/root/bootstrap.log out/harvest-1.4b-a100/ 2>/dev/null || true
echo "cells: $(ls out/phase1-1.4b-a100/*.json 2>/dev/null | grep -vc fit_ || echo 0)"
