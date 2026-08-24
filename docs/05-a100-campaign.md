# The A100 campaign, 23 Aug 2026

The Phase 1 grid, run on one rented GPU after the laptop queue proved too slow. This
is the record the tree's rules ask for: platform, commits, cost, and every incident.

## Platform

| | |
|---|---|
| pod | `lwd-phase1-a100`, id `d42gdp1posrxzv`, RunPod COMMUNITY |
| GPU | 1x NVIDIA A100-SXM4-80GB, host CUDA 13.2, 256 CPU cores, 1 TB RAM |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| stack | isolated venv, `torch 2.13.0+cu130` (pinned to match the laptop), transformers 5.15.1, Python 3.12.3 |
| code | rsync of the working tree plus `.git`; SHAs `1e7a6b9` through `d5d28c0`, clean tree at every launch |
| **uptime** | **~13.5 h** |
| **cost** | **$18.95** ($18.67 GPU + $0.28 disk), reconciled against `get-billing` |

## What it produced

114 cells in `out/phase1-1.4b-a100/`, plus spectral metrics for 112 checkpoints.
Grids: `grid-1.4b-full` (89 cells, two seeds), `-a46`, `-a92`, `-a184` (anchor
crossover), `-long` (3e8), `-gauss` (12 cells), and targeted reruns. The findings are
in `docs/02` and the verdict in `docs/04`.

Throughput: **4.5x the laptop per step**, and **concurrency is neutral** once the card
is saturated (4 workers at 0.8 s/step is arithmetically the same as 1 at 0.2). Plan
device-hours, not worker counts. A single cell reaches only ~30% utilisation, so the
tail of a campaign wastes the card unless it is overlapped.

## Why the data never moved

Laptop upload measured **773 kB/s**, so shipping the 8.2 GB anchor set would have taken
~3 h. The pod regenerated it from the 21 MB declared slice plus a checkpoint pulled at
its own bandwidth, asserting the slice sha256 against the laptop's first. Checkpoints
(42 GB) were never pulled either: spectral metrics were computed on the pod and came
back as one small JSON.

## Incidents

Seven, in the order they happened. Each cost something and each is now a test or a
documented practice.

1. **Bootstrap, three failed attempts (~$0.46).** A bare venv hides the image's torch;
   `--system-site-packages` fixes that and breaks the image's torchvision ABI, which
   surfaces as an unrelated-looking `Could not import module 'modeling_gpt_neox'`. Fix:
   isolated venv with torch installed explicitly. The bootstrap now asserts the import
   and the torch version before doing any work.
2. **BLAS thread oversubscription (~$0.25).** 4 workers x 256 default threads on a
   256-core host left the **GPU at 0% for five minutes** while the contract's float64
   eigh calls fought for cores. `OMP_NUM_THREADS=8` took one cell from "still going" to
   14.6 s. Probe one cell before launching a fleet.
3. **The mix arms outside C** built their noise with `structure="mix"`, which the
   samplers do not know; `C_mix` worked only because the runner special-cased it. Cost
   2 cells. Now a test builds every arm named in every grid.
4. **The Gaussianized fit asked for ~100 GB** (whole anchor set cast to float64 in one
   call) and was SIGKILLed on every cell. The 70m smoke test was three orders too
   small. Now chunked, with a test asserting chunked equals one-shot.
5. **One NaN gradient poisoned a 2.7 h run.** `clip_grad_norm_` scales every parameter
   by a coefficient from the total norm, so a single non-finite gradient makes the
   model NaN permanently. One cell in 129. Now skipped and counted, with a test that
   injects an inf mid-run.
6. **The spectral pass died twice at the last line**: LAPACK on the poisoned
   checkpoint, then `json.dump` on numpy scalars, losing ~50 min of completed work
   each time. Now resumable, atomic, and tolerant of both.
7. **39 minutes of idle GPU (~$0.90)**, caught by hand far too late. Now a monitor
   alerts within 3 minutes when there are no jobs and utilisation is under 5%. Its
   first version had a false positive of its own: it matched the interpreter path,
   which misses cells launched as bare `python`. Match the script path, and bracket it
   (`phase1/[r]un.py`) so the pattern does not match the shell running the check.

Roughly **$1.6 of the $18.95 went on incidents**, and three of them (3, 4, 5) were
real bugs in the project's own code that the laptop campaign had not exposed because
they need scale or run length to appear.

## Practices this rental established

- Probe one cell before launching a fleet.
- Count jobs by script path, bracketed; never by interpreter path or by a pattern the
  checking shell contains.
- An idle detector runs for the life of every rental.
- Long jobs write results incrementally and atomically, and resume.
- Pull results continuously; verify local against remote by name before deleting.
- Regenerate large inputs on the pod rather than uploading them.
