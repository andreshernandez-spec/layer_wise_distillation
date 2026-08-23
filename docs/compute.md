# Compute

What each phase needs, what the free tiers actually allow, and the arithmetic behind
the claims in the source document, corrected. Tier facts are what the `es` project
measured on this account (`es/docs/06-benchmark-runbook.md`,
`.claude/skills/kaggle-notebooks/SKILL.md`); re-verify quotas, they move.

## Tiers

| tier | what | limits | role here |
|---|---|---|---|
| **L** | laptop RTX 3080, 16 GB, Ampere | thermals (package ~95 C under sustained load); single device | Phase 0 harvest, Phase 1 teacher-resident arms, all debugging |
| **K-GPU** | Kaggle 2x T4 (16 GB each, Turing, no bf16 tensor cores) | ~30 GPU-h/week, **max 2 concurrent sessions**, 9 h cap, seconds to queue | Phase 1 sweep tail, Phase 3 stage training |
| **K-TPU** | Kaggle TPU v5e-8 (8 x 16 GB) | ~20 TPU-h/week, **max 1 session**, 2.5-4 h queue, 9 h cap | Phase 2/3 heal (JAX), ES |
| **TRC** | Cloud TPU grant, ~30 days from acceptance | apply ~week 10 | Phase 4 |
| **Spot** | RunPod / Vast H100 or A100 | $2-3/GPU-h | Phase 4 Minitron baseline, Phase 5 benchmarking only |

Effective throughput assumed below, conservative: T4 ≈ 8 TFLOP/s, 3080 Laptop ≈ 15
TFLOP/s (fp16/bf16 with tensor cores, ~25% of peak). **Calibrate these on day one of
Phase 1** with a timing run at real shapes; every wall-clock below scales with them.

## Arithmetic

Sample = position throughout (`docs/00` §2.1). Non-embedding parameter counts:
Pythia-1.4B ≈ 1.2B, 2.8B ≈ 2.5B, 410M ≈ 0.30B.

### Phase 0 harvest

Pythia-1.4B × 10^7 tokens: 2 · 1.2e9 · 1e7 = 2.4e16 FLOPs. **Measured 22 Aug 2026: 0.84
s/row, ~75 min for the 5119 rows on L** (batch 4, fp16, sdpa), with the forward about
half of it and the f64 accumulators (covariance, CF sketch), the quantile reservoir
transfer and the ref/top-k writes the rest. The stats-only pass costs the same
forward. Pythia-2.8B at d=2560 and 9 interfaces: expect ~2.5 h. Both fit an evening on
the laptop; both are a few minutes of T4 queue on Kaggle once the code is pushed.

### Phase 1 sweep

**Measured 22 Aug 2026** (Pythia-1.4B stage 2, 2-block same-width student, batch 8 x
2048, bf16 autocast, fp16 teacher, RTX 3080 Laptop): **0.82 s per 16k-position step**
once past the first evals, so 1e6 positions is 50 s, 1e7 is ~8 min, 1e8 is ~1.4 h per
cell, and the must-exist grid (`grid-1.4b.yaml`, 78 cells, 1.45e8 positions per arm)
is roughly 40 laptop-hours. The per-cell fixed cost (harvest load, contracts,
diagnostics, stitching with the full 1.4B resident) is ~40 s. Cells are one process
each and resumable, so the grid runs as evenings on the laptop or as a Kaggle fan-out.


Teacher stage (4 of 24 blocks) ≈ 200M; student (2 blocks, same width) ≈ 100M.
Per position: teacher forward 4e8 + student fwd+bwd 6e8 = 1e9 FLOPs.
Q grid sums to 1.45e8 positions per cell → 1.45e17 FLOPs per cell.
Must-exist cells: 6 (R, G·iid, I·iid, C·iid, C·ar1, C·mix) plus 2 edge-stage repeats of
C·mix, plus the bridge cell, plus seeds: ~10 cell-equivalents → 1.5e18 FLOPs →
~28 h on L or ~50 T4-hours. Full 13-cell design: ~2e18 → L for two weeks or L + both
Kaggle sessions for one. The R arm needs the teacher's first 12 blocks resident for
live activations; it runs on L.

### Phase 3 Tier 0 (Pythia-2.8B → 410M config)

Teacher stage (4 of 32 blocks) ≈ 310M; student stage (3 blocks of 410M) ≈ 38M.
Per position: 6.2e8 + 2.3e8 ≈ 8.5e8. At Q = 10^8 per stage: 8.5e16 per stage, 6.8e17
for 8 stages → ~3 h per stage on a T4, ~1.6 h on L. With 2 Kaggle sessions (4 T4s) plus
L: **8 stages in about 6 h wall-clock, ~25 GPU-hours**, under the weekly Kaggle quota.
DAgger reruns double it. Heal: 10^7 tokens × 6 × 4.1e8 = 2.5e16, trivial; it goes to
K-TPU for the JAX path, or L.

Total Tier 0 ≈ 5e16 + 7e17 + 2.5e16 ≈ **8e17 FLOPs**, vs from-scratch Pythia-410M
6 · 4.1e8 · 3e11 = **7.4e20**. Claim A's FLOPs clause holds by ~900x under sample =
position. Under sample = sequence it fails (`docs/00` §2.1).

### Phase 4 Tier 1 (OLMo-2-7B → 1B config)

Teacher stage (4 of 32 blocks) ≈ 830M, fp16 1.7 GB, fits anything. Student stage
(2 blocks of OLMo-2-1B) ≈ 130M. Per position ≈ 2.4e9; Q = 10^8 → 2.4e17 per stage,
1.9e18 for 8 stages: ~30 T4-hours or a few v5e-8 hours. Harvest: 2 · 6.5e9 · 1e7 =
1.3e17, ~2.5 h on L with the streaming path (7B fp16 = 14 GB does not leave room for
activations on a 16 GB card; stream). The self-run Minitron baseline at 10-20B tokens on
a 1B student is 6 · 1e9 · 1.5e10 ≈ 1e20 FLOPs: TRC, or ~50-100 H100-hours spot,
$100-300. That is the only paid item before Phase 5.

## Torch and CUDA in the env (22 Aug 2026)

The env carries CUDA 12.9 runtime libraries and JAX's cuda12 plugin (the ES project's
stack), but `torch` was the CPU wheel (`2.13.0+cpu`, pulled in as a dependency). CUDA
libraries next to a CPU torch do nothing: the kernels live in the torch wheel. torch
2.13 exists on the `cu126`, `cu129` and `cu130` indexes, not `cu128`. **`cu129` is the
right pick for a fresh setup**: it reuses the `nvidia-*-cu12 12.9.x` libraries JAX
already installed. `cu130` was used this time (the cu128 attempt failed first and cu130
was the next guess); it works with the 13.2 driver and coexists with JAX's cu12 set, at
the cost of a duplicate ~3 GB of `nvidia-*-cu13` wheels. Install with the env's
interpreter and a fully qualified version, `pip install "torch==2.13.0+cu129"
--index-url https://download.pytorch.org/whl/cu129`: a bare `torch==2.13.0` is
"already satisfied" by the `+cpu` build and pip silently does nothing.

## Network (measured 22 Aug 2026)

The laptop link gave ~0.8 MB/s from Hugging Face and ~0.2 MB/s from download.pytorch.org.
At that rate a 3 GB checkpoint is an hour and the CUDA torch wheel set (~4 GB) is
longer, and two concurrent downloads each get half. `snapshot_download` also stalled
silently twice on a dropped connection (no growth for 20 min, HF reachable); run it with
`HF_HUB_DOWNLOAD_TIMEOUT=30` so it retries, one model at a time, and check the blob size
grows. Plan big downloads (Pythia-2.8B is 5.6 GB, OLMo-2-7B 14 GB) overnight or on the
Kaggle side where the checkpoint can be a dataset input.

## The rented A100 (23 Aug 2026)

**Why**: the laptop's Phase 1 queue was ~11 h and the full 78-cell grid ~40 h more.
One A100 SXM 80GB on RunPod COMMUNITY is $1.39/h and measured **4.5x the laptop**
(harvest 0.19 s/row against 0.84), so the whole of Phase 1 is a few hours and under
$15. Pod `lwd-phase1-a100`, `experiments/pod/`.

**The 8.2 GB of anchors never move.** Laptop upload measured **773 kB/s**, so shipping
them would take ~3 h. The pod regenerates them from the declared slice (21 MB of HTTP
range requests) plus a checkpoint it pulls at its own bandwidth, and asserts the
slice's sha256 against the laptop's before doing anything. That doubles as a
reproducibility check, and `experiments/phase0/compare_stats.py` quantifies it.

**Pod results live in their own directory** (`out/phase1-1.4b-a100/`). Training is
chaotic, so the same config on two GPUs diverges; mixing the two silently into one
table would be a confound. `experiments/phase1/compare_platforms.py` measures the
platform effect on cells run on both.

**Three bootstrap traps, each one a failed launch (23 Aug 2026)**:

1. `python3 -m venv .venv` hides the image's torch, so `pip install -e .` fails on the
   torch dependency and everything after it dies with `No module named 'torch'`.
2. `--system-site-packages` fixes that and breaks something worse: the image's
   `torchvision` is compiled against its own torch, so installing a different torch in
   the venv leaves an ABI mismatch, and transformers dies importing *any* model with
   `RuntimeError: operator torchvision::nms does not exist`. The lazy-import wrapper
   reports it as `Could not import module 'modeling_gpt_neox'`, which points nowhere;
   `spec.loader.exec_module` on the module gives the real traceback.
3. The fix is an **isolated venv with torch installed explicitly**, pinned to the
   laptop's version (`torch==2.13.0+cu130`) so the only difference between platforms
   is the GPU. The bootstrap now imports `modeling_gpt_neox` and asserts the torch
   version before it does any work, so trap 2 cannot recur silently.

**The trap that actually costs money: BLAS thread oversubscription.** The first
4-worker launch sat with the **GPU at 0% for five minutes** while every worker burned
800% CPU. The host has 256 cores, so each process gave its float64 `eigh` (the
contract setup does three, on 2048x2048 covariances) 256 threads: 1024 threads over
256 cores. Capping `OMP_NUM_THREADS=8` per worker (`run_sweep.sh` exports it, with
`MKL`/`OPENBLAS`/`NUMEXPR` to match) took one small cell from "still going after five
minutes" to **14.6 s**, against 43 s for the same cell on the laptop. A rented box
with many cores is not a laptop with a few, and the default is wrong there.

The lesson generalizes: **probe one cell before launching the fleet.** A single
timing cell costs seconds and would have caught this before four workers spent ten
minutes achieving nothing.

Also: `git` refuses a rsync'd repo owned by another uid
(`detected dubious ownership`); `git config --global --add safe.directory /root/lwd`
on the pod, or runs record an empty SHA.

## Storage (local, 864 GB free on 22 Aug 2026)

| item | size |
|---|---|
| Pythia-1.4B + 2.8B fp16 | 8.4 GB |
| 10^7-token slice, two corpora | 80 MB |
| Interface tensors, Pythia-1.4B, 7 interfaces × 10^7 × 2048 × fp16 | 290 GB if kept; keep only the anchors and recompute the rest |
| Anchor store, 10^6 positions per interface, fp8 + scales, 7 interfaces | 15 GB (1.4B), 23 GB (2.8B, 9 interfaces) |
| Top-k store | 2.6 GB |
| Statistics | MBs |

## Calendar

| week | phase | device |
|---|---|---|
| 1-2 | 0 | L, CPU, one K-GPU kernel |
| 3-5 | 1 | L continuously; K-GPU both sessions from week 4 |
| 6-8 | 2 | L + K-GPU; first K-TPU kernel for the heal |
| 8-13 | 3 | K-GPU + L for stages; K-TPU for heal and ES |
| 10 | apply to TRC | |
| 14-18 | 4 | TRC window; one spot booking for the Minitron baseline |
| 19+ | 5 (conditional), 6 | |

## Not wasting the TPU queue

A K-TPU session is one shot per 2.5-4 h wait. Everything that runs there is debugged on
CPU with simulated devices first, asserts the device count in a fresh interpreter, writes
results incrementally, and exits non-zero on any failed check. The `es` project's
`experiments/phase2/kaggle/tpuprep/` is the template.
