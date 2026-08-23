"""C0.6: the Phase 0 harvester on Kaggle 2x T4, asserting the hardware and exiting
non-zero on any failed check (see .claude/skills/kaggle-notebooks). Pins a SHA.

Edit SHA and CONFIG before pushing. Output lands in /kaggle/working/out.
"""
import json
import os
import subprocess
import sys

REPO = "https://github.com/andreshernandez-spec/layer_wise_distillation.git"
SHA = "PIN-ME"
CONFIG = "experiments/phase0/configs/harvest-1.4b.yaml"


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode}): {cmd}", flush=True)
        sys.exit(1)
    return r.stdout


gpus = sh("nvidia-smi -L")
assert gpus.count("Tesla T4") == 2, f"expected 2x T4, got: {gpus}"
sh(f"git clone {REPO} lwd && cd lwd && git checkout {SHA} && git log --oneline -1")
sh("cd lwd && pip install -q -e . transformers safetensors pyyaml")
import torch  # noqa: E402
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, torch.cuda.device_count()
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)

os.makedirs("/kaggle/working/out", exist_ok=True)
# slice: declared config, fetched by range request (20 MB)
sh("cd lwd && python experiments/phase0/slice.py experiments/phase0/configs/slice.yaml /kaggle/working/out/slice")
meta = json.load(open("/kaggle/working/out/slice/anchor_rows.json"))
assert meta["sha256"] == "05e6be5eb71d7c9c4756be1dd0d965c074aded84ffeaf54de01a74c5afe8baf3", meta["sha256"]
# harvest: rewrite paths into /kaggle/working
import yaml  # noqa: E402
cfg = yaml.safe_load(open(f"lwd/{CONFIG}"))
cfg["slice"] = "/kaggle/working/out/slice/anchor_rows.npy"
cfg["out"] = "/kaggle/working/out/harvest"
yaml.safe_dump(cfg, open("/kaggle/working/harvest.yaml", "w"))
sh("cd lwd && python experiments/phase0/run.py /kaggle/working/harvest.yaml")
sh("cd lwd && python experiments/phase0/verify_anchors.py /kaggle/working/out/harvest --device cuda --batch 4")
sh("cd lwd && python experiments/phase0/bridge_oracle.py /kaggle/working/out/harvest --widths 1024")
# keep the output under Kaggle's size cap: stats, verification, bridge, top-k, and the
# refs of the Phase 1 stage only (interfaces 2 and 3 of 6 stages)
sh("cd /kaggle/working/out/harvest && ls anchors | grep -v -E 'iface(2|3)$' | xargs -I{} rm -rf anchors/{}")
sh("du -sh /kaggle/working/out")
print("KERNEL-OK", flush=True)
