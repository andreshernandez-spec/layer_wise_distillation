# Conventions

Short and enforceable. Change here, not in code.

## Oracles the tests use

| thing | oracle | test |
|---|---|---|
| streaming harvester | the resident model with hooks, **bitwise** (fp32, eager and sdpa) | `test_harvest.py` |
| stitching machinery | the teacher's own stage stitched back: delta **exactly 0**; a random affine contract pair telescopes to 1e-3 | `test_stitch.py` |
| ZCA whitening | covariance of the output is I to 1e-8; inverse round-trips to 1e-9 | `test_contract.py` |
| marginal Gaussianization | output quantiles at 0.1/0.5/0.9 are those of N(0,1) within 0.05; monotone; inverse exact inside the table | `test_contract.py` |
| CF sketch | standard normal gives exp(-t^2/2) for the cos part and 0 for the sin part, to Monte Carlo error | `test_stats.py` |
| mean / covariance accumulators | `numpy.cov` to 1e-8 | `test_stats.py` |
| AR(1) sampler and lag-1 estimator | each other, rho 0.6 recovered within 0.05 | `test_contract.py`, `test_stats.py` |
| anchor store | per-channel relative error < 1% (int8 + per-channel std); the two traps (int8 plain, fp8 rotated) > 20% | `test_anchors.py` |
| bridge oracle | a map that reads only the top-8 PCA directions gives floor 0; one that reads all gives floor > 0.1 | `test_bridge.py` |
| Jacobian agreement | teacher vs itself through a random affine pair: cosine > 0.999 | `test_diagnostics.py` |
| fit | synthetic eps(Q) with beta 0.35 recovered within the CI | `experiments/phase1/fit.py` self-test |

Where no exact oracle exists, two independent implementations with different failure
modes are compared, not one implementation against itself.

## Precision traps (22 Aug 2026)

- **fp16 eager attention overflows** on the 1.4B's massive activations: NaN attention
  maps from block 13 on. `sdpa` accumulates in fp32 and is fine, so the harvest is
  unaffected; anything that needs the attention weights themselves (entropy
  diagnostics) runs the eager path in fp32.
- **Whitened coordinates round differently from raw ones.** The ZCA eps is relative to
  the top eigenvalue (condition number at most 1/sqrt(eps)); the chained teacher still
  amplifies float-level input perturbations 10-1000x per stage (`docs/03` C2.2).
- **Keep diagnostics in fp32, not fp64**, on GPU: a (8, 16, 2048, 2048) attention map
  is 4 GB in float64.

## Error metrics

- Activation errors are **per-channel normalized** (`channel_relative_error`), never raw
  norm ratios: a massive-activation channel can hold 99% of the norm and hide everything
  else (`docs/01` C0.3).
- eps is relative MSE in phi (whitened) coordinates: sum of squared error over sum of
  squared deviation from the mean, on held-out real positions.

## Tolerances

State them, do not discover them. Derived tolerances (the anchor round trip: 4x the
measured quantization floor) record the number they were derived from.

## Randomness and reproducibility

- Every sampler takes an explicit `torch.Generator`; every cell records its seed.
- Every experiment JSON records SHA, dirty flag, torch/transformers versions, device.
- The token slice is declared by index and SHA before any result exists (`docs/01` C0.1).

## Tests

`pytest --fast` skips the `slow` tier (model loading). `pytest` runs everything on CPU
in about 30 s with pythia-70m cached and the slice fetched; tests that need the slice
skip without it.

## Memory on the laptop (added 22 Aug 2026, after taking it down)

- **One heavy job at a time.** A harvest, a sweep cell, or an eval is a heavy job. Do
  not chain gate scripts beside a running pass.
- **Stream, never materialize an interface.** 512 sequences x 2048 x 2048 is 4.3 GB in
  fp16 and 8.6 GB in float32; a script that holds two or three such copies is most of
  the machine. Accumulate sums per file; keep at most a fixed random subset.
- **Run background jobs under the watchdog**:
  `experiments/tools/memguard.sh 24 6 -- <command>` kills the job when its process
  tree passes 24 GB RSS or the machine's MemAvailable drops under 6 GB. A cgroup cap
  (`systemd-run --user -p MemoryMax=`) is accepted but not enforced on this laptop
  (no memory-controller delegation to the user slice, checked 22 Aug 2026), and
  `ulimit -v` is not an option because CUDA reserves virtual address space far beyond
  physical use.
- Quantile reservoirs are 2^16 rows (0.5 GB per interface at d=2048), not 2^18.

## Reading the Phase 1 cells (added 23 Aug 2026)

Seed-to-seed spread over 42 same-arm pairs: **eps median 0.004, stitching delta median
0.106 nats, max 1.339**. So:

- Draw no conclusion from a stitching difference below Q = 3e6, where the whole
  between-arm spread is inside the seed band; eps separates from 1e6.
- Quote a difference as a result only against the seed spread at that Q, not against
  zero. Two arms 0.002 nats apart are tied.
- The platform effect (laptop vs A100) is 0.46% on eps, the same order as seed noise,
  so cells from both may share a table.
