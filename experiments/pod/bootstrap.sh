#!/bin/bash
# Bootstrap a RunPod pod for Phase 1: env, checkpoint, slice, harvest.
# Run detached; writes /root/bootstrap.log. Idempotent: skips finished steps.
set -eu
cd /root/lwd
LOG=/root/bootstrap.log
exec >>"$LOG" 2>&1
# Cap BLAS threads here too. The harvest's f64 covariance, CF sketch and quantile
# accumulation are CPU-heavy, and on a many-core host the default gives them every
# core: load average 23 with the GPU at 8% on a 128-core box (24 Aug 2026).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
echo "== bootstrap start $(date -u +%H:%M:%S) OMP=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Isolated venv, with torch installed explicitly. --system-site-packages looks
# tempting (the image ships torch) but the image's torchvision is built against that
# torch, so installing another version in the venv breaks its ABI and transformers
# dies on import with "operator torchvision::nms does not exist".
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -q -U pip
  # match the laptop's torch so the only difference between platforms is the GPU
  pip install -q "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
  pip install -q -e . --no-deps
  pip install -q "transformers>=4.45" safetensors pyyaml numpy scipy requests tokenizers huggingface_hub zstandard
else
  . .venv/bin/activate
fi
python - <<'PY'
import torch
from transformers.models.gpt_neox import modeling_gpt_neox  # the import that ABI breakage kills
assert torch.cuda.is_available(), "no CUDA in the venv"
assert torch.__version__.startswith("2.13.0"), f"torch {torch.__version__}, expected 2.13.0 to match the laptop"
print("torch", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count(), "gpu(s)")
PY

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

# Inputs this pod cannot regenerate, checked before anything touches the GPU. The
# decontamination record and the held-out set are produced once, on the laptop, from
# downloaded eval suites and a Pile-test shard; a fresh pod has neither, and without
# this check the harvest fails two minutes in with a FileNotFoundError (25 Aug 2026).
missing=""
for f in out/slice/decontam.json out/heldout/heldout_rows.npy; do
  [ -f "$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  echo "== MISSING INPUTS, upload them before bootstrapping:$missing"
  exit 1
fi
echo "== inputs present: decontam.json, heldout_rows.npy"

# harvest: statistics for every interface, anchor refs only where Phase 1 needs them
if [ ! -f out/harvest-1.4b/stats_iface6.npz ]; then
  python experiments/phase0/run.py ${HARVEST_CFG:-experiments/pod/harvest-1.4b-pod.yaml}
fi
echo "== bootstrap done $(date -u +%H:%M:%S)"
