# Phase 0: Harvest tooling

**Compute**: laptop RTX 3080 (16 GB) for the harvest; CPU for everything else; one
Kaggle 2xT4 kernel to prove the harvester runs unattended.
**Duration**: 2 weeks.
**Substrate**: Pythia-1.4B (24 layers, d=2048, MHA, fp16 = 2.8 GB). Also dry-run on
Pythia-2.8B so the Phase 3 interfaces exist early.
**Gate**: G0, bottom of this file.

## Goal

From a pre-declared 10^7-token slice of the teacher's corpus, produce everything the
later phases consume, once, reproducibly:

1. per-interface statistics (mean, covariance, CF sketch, per-channel quantiles),
2. an fp8 anchor store at every stage boundary,
3. a top-k logit store for the heal,
4. the bridge-oracle numbers (linear-probe floors) for each candidate width.

And do it with a streaming, stage-resident loop that would also work for a teacher that
does not fit in memory, because Tier 1 needs that and it must not be debugged on rented
time.

## Why first

Every Phase 1 arm except "isotropic noise" is parameterized by harvested statistics.
The fp8 store is where Plan B's rotation meets a real workload. And the token slice has
to be declared and decontaminated before anyone looks at results.

## Definitions (fix these here, use them everywhere)

- **Interface k**: the residual stream entering block `4k` of the teacher (k = 0..S),
  shape `(L, d)` per sequence. Interface 0 is the embedding output; interface S is the
  input to the final norm + head. Stage k is blocks `4k .. 4k+3`.
- **Position**: one `(d,)` vector at one sequence position. Counts of "samples",
  "anchors" and "noise" are in positions. Sequences are reported alongside.
- **Pre-declared slice**: a contiguous range of Pythia's training batches, by index,
  recorded in `experiments/phase0/configs/slice.yaml` with the SHA. 10^7 tokens ≈ 4883
  sequences of 2048 (Pythia's L). Held-out eval text comes from a *different* declared
  range, never from the slice.

## Capabilities delivered

### C0.1: Token slice, declared and decontaminated

**Declared 22 Aug 2026** (`experiments/phase0/configs/slice.yaml`): rows
102,400,000 to 102,405,119 of `EleutherAI/pile-standard-pythia-preshuffled`, i.e. Pythia
training steps 100000-100004, 5120 sequences of 2049 tokens, **10,490,880 tokens**,
sha256 `05e6be5eb71d7c9c4756be1dd0d965c074aded84ffeaf54de01a74c5afe8baf3`. Fetched by
HTTP range request (`experiments/phase0/slice.py`), 20 MB, no bins downloaded. Step
100000 was chosen before any data was seen. Held-out text is the Pile test split, which
no Pythia model trained on.

**Decontaminated 22 Aug 2026** (`experiments/phase0/decontam.py`, 13-gram overlap against
LAMBADA, PIQA, WinoGrande, ARC-e/c, SciQ, LogiQA, HellaSwag): **1 row of 5120** overlaps
(a LAMBADA passage); every other task is clean. The row index is in
`out/slice/decontam.json` and both drivers drop it (`decontam:` config key), leaving
5119 rows, 10,488,831 tokens.

**Held-out text, materialized 22 Aug 2026** (`experiments/phase0/heldout.py`): the first
2M tokens of `monology/pile-uncopyrighted` `test.jsonl.zst`, tokenized with the Pythia
tokenizer, documents joined by EOS, cut into 976 rows of 2049, sha256
`4ba28609909dbe759db7f75f3bf62c5570d7605b39e73b1673960d0eb44ce5fe`. Used for
perplexity and the heal-budget curve; the benchmark suite is its own held-out set.

Reference next-token loss on these rows (`experiments/eval/perplexity.py`, fp16, 22 Aug
2026): Pythia-410M: loss 1.9132, ppl 6.774. Pythia-1.4B: loss 1.7364, ppl 5.677.

`experiments/phase0/slice.py` materializes the slice from the Pile source decided in
`docs/00` §3.6 and writes the token ids to disk with the config that produced them.
Decontamination: 13-gram overlap against the pre-declared eval suite (LAMBADA, PIQA,
WinoGrande, ARC-e/c, SciQ, HellaSwag; the set Pythia reports minus LogiQA, whose HF
dataset is script-based and unloadable under datasets 5 through lm_eval 0.4.12, noted
22 Aug 2026; the slice was decontaminated against it anyway) and against the
held-out perplexity range; overlapping sequences are dropped and the count recorded.

A second, equally sized slice from a different corpus (FineWeb-Edu) is declared at the
same time, for the corpus-swap control. Same decontamination.

### C0.2: Streaming stage-resident harvester

`src/lwd/harvest/`. Loads stage k's blocks only, reads interface k from disk (or, for
k=0, runs the embedding), pushes all sequences through, writes interface k+1, drops the
stage. For a teacher that fits, the same loop runs with all stages resident and no disk
round trip; the streaming path is selected by config, and **both paths are run on
Pythia-1.4B and must produce bitwise-identical interface tensors** (fp16, deterministic
kernels). That equality is the test that the streaming path is correct.

Per-interface accumulators run inside the same pass, on the fp16 activations before any
quantization:

- mean and covariance in f64 accumulation, Ledoit-Wolf shrinkage applied at the end;
  ZCA whitening matrix and its inverse derived and stored;
- CF sketch: M=256 fixed random unit directions (seeded), K=16 frequencies on the grid
  SIGReg uses, cos and sin means accumulated in f64 → `(M, K, 2)` per interface. Also
  the same sketch **in whitened coordinates**, which is the one the contract uses;
- per-channel quantiles at 1024 levels (t-digest or reservoir; exact is fine at 10^7),
  for the marginal Gaussianization maps;
- lag-1 per-channel autocorrelation across positions (for the AR(1) noise arm) and the
  position-averaged channel covariance at a few lags;
- attention entropy per head per block, on real inputs (the Phase 1 diagnostic needs a
  real-input baseline).

**Result, Pythia-1.4B, 22 Aug 2026** (`out/harvest-1.4b/stats_iface*.npz`, 10,483,712
positions per interface, 6 stages of 4 blocks, fp16, stats-only pass after the
finalize incident; `docs/results.md` has the table):

| interface | max / median channel std | lag-1 rho, median | lag-1 rho, max |
|---|---|---|---|
| 0 (embeddings) | 5 | 0.01 | 0.12 |
| 1 | 46 | 0.21 | 0.62 |
| 2 | 48 | 0.34 | 0.76 |
| 3 | 45 | 0.40 | 0.69 |
| 4 | 34 | 0.40 | 0.77 |
| 5 | 26 | 0.42 | 0.78 |
| 6 (pre-norm) | 17 | 0.42 | 0.78 |

Two things the noise design must respect: the massive-activation channels (45x the
median std from the first block on, fading toward the head), and across-position
correlation that is far from i.i.d. (median lag-1 of 0.34 at the Phase 1 stage input,
interface 2, with channels up to 0.76). The AR(1) arm has real parameters to fit.

### C0.3: Anchor store

`src/lwd/harvest/anchors.py`. Per interface, 512 sequences (1.05M positions) chosen by
a seeded permutation of the slice so each interface gets a different subset. The
harvest pass dumps them in fp16 (`ref_*.npy`, the exact reference the gate needs);
the compact store is packed offline once the statistics exist: **divide by the
harvested per-channel std, per-position absmax scale in fp16, int8 codes.** No
rotation.

**Settled 22 Aug 2026 by measurement, overturning the source document's "Hadamard
before fp8".** On a synthetic stream with a 200x massive-activation channel,
per-channel relative error:

| | plain | rotated | per-channel std |
|---|---|---|---|
| fp8 e4m3 | 0.026 | **0.31** | 0.026 |
| int8 | 0.42 | 0.10 | **0.007** |

A rotation spreads the outlier's quantization noise onto every small channel. That is
the right trade for integer codes without a per-channel scale (the QuaRot setting) and
the wrong one for floating-point codes, whose precision is already per-element
relative. Per-channel normalization removes the outlier at the source, and int8 then
beats e4m3 4x at the same byte per element. The raw-norm error metric hides all of
this (the outlier channel is ~99% of the norm and quantizes exactly), so the metric is
`channel_relative_error`, per-channel RMS error over per-channel std. Rotation stays
available as an ablation, not a default; the Plan B synergy is an int4 story, not this
one. Regenerate: `tests/test_anchors.py`.

Layout: one shard per ref file so it uploads as a Kaggle dataset piecewise (verify the
per-dataset size limit before depending on it; design for <= 20 GB shards).

Round trip test: pack, dequantize, feed through teacher stage k, compare to the same
stage applied to the exact refs. The tolerance is **derived**, not chosen: 4x the
quantization floor measured on the same tensors. Record both numbers.

### C0.4: Top-k logit store

10^7 positions × top-64, uint16 ids + fp16 logits, ~2.6 GB. Plus the teacher's
full-vocabulary loss on the slice, so the heal's KL target is checkable.

### C0.5: Bridge oracle

For each candidate student width d_S ∈ {1024, 2048}: ridge regression from the
projected interface k input (top-d_S ZCA directions) to the full interface k+1, and from
the full input to the same target, on 10^6 training positions, evaluated on held-out
real positions. Reported as relative MSE in whitened coordinates. The gap between the
two is the bridge floor for that interface and width. Done for every interface of
Pythia-1.4B and Pythia-2.8B; it is cheap and it decides Tier 0a vs 0b.

### C0.6: Kaggle kernel

Written 22 Aug 2026: `experiments/phase0/kaggle/harvest/`. It clones the repo at a
pinned SHA, so it can only run once the code is on GitHub (feature branch on
`andreshernandez-spec/layer_wise_distillation`; pushing there is authorized by the tree
CLAUDE.md §2.0). Given the laptop's ~1 MB/s link (`docs/compute.md`), the kernel is also
the faster way to run the 1.4B and 2.8B harvests themselves: Kaggle pulls a checkpoint in
a minute. It keeps only interfaces 2 and 3 refs (the Phase 1 stage) to stay under the
output cap.


`experiments/phase0/kaggle/harvest/`: pins the SHA, installs nothing it does not need,
asserts `nvidia-smi -L` shows two T4s, runs the harvester on a 10^5-token sub-slice in
streaming mode, and exits non-zero if any accumulator disagrees with the committed
laptop reference beyond the stated tolerance. The point is not the compute; it is that
the harvester works where Phase 3's stage training will run.

## Environment setup (first day)

1. `conda install -n open-source` a CUDA-enabled torch (or the cu12 pip wheel);
   `python -c "import torch; print(torch.cuda.is_available())"` must print True.
   Record the versions in `docs/compute.md`.
2. `transformers` is 5.15.0 in the env; Pythia is `gpt_neox`, supported. Download
   `EleutherAI/pythia-1.4b` and `pythia-2.8b` (fp16 safetensors, ~8.4 GB total).
3. Pile source decided and a sample of the slice materialized.

## Gate results, Pythia-1.4B (22 Aug 2026)

**G0.3 anchor round trip** (`verify_anchors.py`, int8 + per-channel std, 512 sequences
per interface, tolerance = 4x the measured floor):

| stage | quant. floor | propagated | amplification | pass |
|---|---|---|---|---|
| 0 | 0.0101 | 0.0061 | 0.61 | yes |
| 1 | 0.0119 | 0.0106 | 0.89 | yes |
| 2 | 0.0117 | 0.0101 | 0.87 | yes |
| 3 | 0.0096 | 0.0119 | 1.23 | yes |
| 4 | 0.0088 | 0.0170 | 1.92 | yes |
| 5 | 0.0089 | 0.0364 | **4.09** | **no (4.09 > 4)** |

Passes for stages 0-4, fails by 2% of the margin at the last stage. The floor is flat
(~1% per channel); what grows is the teacher's amplification of input perturbations
with depth, the same profile the composition test showed (`docs/03` C2.2). Recorded as
a marginal fail, not re-tolerated: the Phase 1 stage (2) is comfortably inside, and a
Tier 1 store may need fp16 for the last interface or a per-stage tolerance that
carries the measured amplification. The fp16 refs exist for every interface.

**C0.5 bridge oracle** (`bridge_oracle.py`, ridge from PCA-whitened interface k to raw
interface k+1, 160k train / 40k test positions, relative MSE):

| interface | full input | top-1024 | top-512 | floor 1024 | floor 512 |
|---|---|---|---|---|---|
| 0 | 0.672 | 0.689 | 0.728 | 0.017 | 0.055 |
| 1 | 0.183 | 0.210 | 0.243 | 0.027 | 0.060 |
| 2 | 0.163 | 0.212 | 0.265 | 0.050 | 0.103 |
| 3 | 0.170 | 0.229 | 0.289 | 0.059 | 0.119 |
| 4 | 0.145 | 0.223 | 0.300 | 0.078 | 0.155 |
| 5 | 0.141 | 0.231 | 0.311 | 0.090 | 0.170 |

A linear map explains ~84% of the next interface from the full input (33% at interface
0, where the first blocks are most nonlinear). The 410M-width bridge (top-1024) costs
an extra 2.7-9.0% relative MSE, growing with depth; half width again roughly doubles it.
Read against Phase 1's eps: the bridge floor at the Phase 1 stage input (interface 2)
is 0.050.

**Attention-entropy baseline** (`attn_entropy.py`, fp32 eager, 16 rows x 2048,
`attn_entropy.npz`; the fp16 attempt gave NaN from block 13, softmax overflow on
massive activations). Mean entropy per block in nats, uniform = log 2048 = 7.62:

    4.17 3.34 4.16 3.60 2.81 2.48 2.51 2.61 2.68 3.04 2.64 2.86
    2.44 2.31 2.09 1.96 1.73 1.57 1.63 1.63 1.24 1.13 0.87 0.86

Attention sharpens steadily with depth; the Phase 1 stage (blocks 8-11) sits at
2.6-3.0 nats, 4.6-5.0 below uniform. That is the number a noise arm has to reproduce.

## Incident log

- **22 Aug 2026, first 1.4B harvest.** The 72-minute pass completed (all refs, top-k,
  anchor index on disk, 31 GB) and then crashed in `MeanCov.finalize`: `torch.eye` was
  created on the CPU while the covariance lived on the GPU. The CPU smoke test cannot
  see a device bug. Fixed (eye on `cov.device`); the driver now runs every
  accumulator's `finalize()` after the first batch so a finalize bug fails in seconds.
  Statistics regenerated with a `stats_only` pass (`harvest-1.4b-stats.yaml`); an
  interim set from the 1M-position refs (`stats_from_refs.py`, `out/harvest-1.4b-interim`)
  let the gate scripts run meanwhile and is not cited anywhere.

- **22 Aug 2026, laptop crash (out of memory).** While the stats-only pass ran on the
  GPU, a chain of gate scripts ran beside it on the interim statistics.
  `verify_anchors.py` loaded a whole interface of refs (4.3 GB fp16), dequantized it to
  float32 (8.6 GB) and kept exact and quantized stage outputs in float32 (17 GB more):
  ~30 GB for one interface, plus the bridge oracle's float32 and float64 copies, plus
  the stats pass and the interim job. The machine (61 GB) went down; the stats pass
  (row ~2500 of 5119) was lost and is rerun. Fixes: both scripts now stream one ref
  file at a time with bounded accumulators; background jobs run under a cgroup
  memory cap (`docs/conventions.md`); **one heavy job at a time** on this machine.

## Gate G0

1. Streaming and resident harvester paths produce **bitwise-identical** fp16 interface
   tensors on Pythia-1.4B for the full slice.
2. Every statistic in C0.2 is regenerated from `experiments/phase0/run.py` with a
   committed config and agrees with the stored artifact to `rtol=1e-6` (f64 paths) on a
   re-run. **Sampling check, 22 Aug 2026** (`compare_stats.py`, `out/harvest-1.4b/compare_out.json`):
   the interim set from the 1.05M-position refs agrees with the 10.48M-position pass
   to 0.5-1.9% of channel std on means, 0.6-5.5% Frobenius on covariances (14% at the
   heavy-tailed interface 6), 0.0002-0.004 RMSE on CF sketches and <= 0.012 on lag-1
   rho. A 10% subsample reproduces the contract parameters to about a percent.
3. Anchor round trip (C0.3) passes at the derived tolerance on every interface, and the
   derived tolerance is recorded next to the quantization floor it came from.
4. The top-k store reproduces the teacher's loss on the slice to within the truncation
   error of top-64, and that error is measured and recorded.
5. Bridge-oracle table exists for both teachers, both widths, all interfaces.
6. The Kaggle kernel ran on 2x T4 and its status was COMPLETE with the assertion in
   C0.6 passing (not merely exit 0; see the kaggle-notebooks skill on green runs that
   mean nothing).

Slice declaration (C0.1) has no gate criterion because it is a precondition: Phase 1
does not start until the slice config is committed.
