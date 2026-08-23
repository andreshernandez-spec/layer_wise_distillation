#!/bin/bash
# Bootstrap a RunPod pod for Phase 1: env, checkpoint, slice, harvest.
# Run detached; writes /root/bootstrap.log. Idempotent: skips finished steps.
set -eu
cd /root/lwd
LOG=/root/bootstrap.log
exec >>"$LOG" 2>&1
echo "== bootstrap start $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -q -U pip
  pip install -q -e . --no-deps
  pip install -q "transformers>=4.45" safetensors pyyaml numpy scipy requests tokenizers huggingface_hub zstandard
else
  . .venv/bin/activate
fi
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count(), 'gpu(s)')"

# checkpoint, at pod bandwidth
python -c "
from huggingface_hub import snapshot_download
print('model at', snapshot_download('EleutherAI/pythia-1.4b', allow_patterns=['*.json','*.safetensors','*.txt']))"

# the declared slice, by index; the sha must match the laptop's
if [ ! -f out/slice/anchor_rows.npy ]; then
  python experiments/phase0/slice.py experiments/phase0/configs/slice.yaml out/slice
fi
python - <<'PY'
import json
m = json.load(open("out/slice/anchor_rows.json"))
want = "05e6be5eb71d7c9c4756be1dd0d965c074aded84ffeaf54de01a74c5afe8baf3"
assert m["sha256"] == want, f"slice sha mismatch: {m['sha256']}"
print("slice sha OK", m["tokens"], "tokens")
PY

# harvest: statistics for every interface, anchor refs only where Phase 1 needs them
if [ ! -f out/harvest-1.4b/stats_iface6.npz ]; then
  python experiments/phase0/run.py experiments/pod/harvest-1.4b-pod.yaml
fi
echo "== bootstrap done $(date -u +%H:%M:%S)"
