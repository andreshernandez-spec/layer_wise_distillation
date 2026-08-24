#!/bin/bash
# Pull everything Phase 2 produced, check it, and print the gate. Run this the moment
# the pod's queue is empty: a finished pod bills at the full rate until it is deleted,
# and Phase 1 lost $0.90 to exactly that gap.
#
# No --delete on the results pull. Three plain-stack records were rebuilt locally from
# the pod's logs after dagger_stack.py overwrote them, and they do not exist there.
set -eu
cd "$(dirname "$0")/../.."
OUT=out/phase2-1.4b-a100
SSH_OPTS="ssh -i $HOME/.ssh/id_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p ${PORT:?set PORT} -o ConnectTimeout=90"
HOST="root@${IP:?set IP}"

mkdir -p "$OUT/logs"
rsync -az --include='*/' --include='*.json' --exclude='*' -e "$SSH_OPTS" \
      "$HOST:/root/lwd/out/phase2-1.4b/" "$OUT/"
rsync -az -e "$SSH_OPTS" "$HOST:/root/*.log" "$OUT/logs/"

echo "=== result files ==="
ls "$OUT"/*.json | wc -l

echo "=== every heal cell: did it deliver what it asked for? ==="
python - "$OUT" <<'PY'
import glob, json, sys
bad = 0
for f in sorted(glob.glob(sys.argv[1] + "/heal_*.json")):
    d = json.load(open(f))
    want = d["tokens"]
    # tokens_seen is what heal() actually consumed. Older records predate it; their
    # history holds every step, so its last entry is the same number.
    seen = d.get("tokens_seen")
    src = "seen"
    if not seen:
        h = d.get("history") or []
        seen, src = (h[-1]["tokens"] if h else 0), "hist"
    flag = ""
    if want > 0 and seen and seen < want * 0.95:
        flag = f"  *** SHORT: {seen:,} of {want:,.0f} ***"; bad += 1
    elif want > 0 and not seen:
        flag = "  (no token count recorded)"
    print(f"{d['name']:44s} want={want:>12,.0f} {src}={seen:>12,.0f} "
          f"loss {d['loss_before']:.4f} -> {d['loss_after']:.4f}{flag}")
print(f"\n{bad} short run(s)")
PY

echo
echo "=== gate ==="
sed 's|^out: .*|out: '"$OUT"'|' experiments/phase2/configs/stack-1.4b-pod.yaml > /tmp/lwd_gate.yaml
python experiments/phase2/gate.py /tmp/lwd_gate.yaml
