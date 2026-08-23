# CLAUDE.md

Instructions for Claude Code in this repository. The tree-level
`~/private/open-source/CLAUDE.md` applies on top of this file (identity, no upstream
pushes, conda env, writing style). Keep this short; detail lives in `docs/`.

## What this project is

Stagewise noise distillation: train student stages of a small LM independently, fed
with noise at the teacher's interfaces plus a small anchor set, then heal end to end.
`PLAN.md` has the phases and gates, `docs/00-plan-review.md` the audit of the source
document, `docs/compute.md` the calendar.

The deliverable through Phase 3 is a **measurement**: β, the noise-query exponent, and
the Tier 0 comparison against from-scratch Pythia-410M. The toolkit and the paper come
after, and only if the measurement says so.

## Ground rules

### 1. Who writes what

No split. Andres removed the "Andres writes the load-bearing code" rule on 22 Aug 2026;
Claude writes code, docs and paper drafts alike. What remains from the tree CLAUDE.md:
no pushes or PRs to upstream projects, `drafts/` stays local, and anything that goes
upstream has to be explained until Andres can defend it line by line.

### 2. Every number has a script

No figure or table enters `docs/` or the paper without a committed, re-runnable script
under `experiments/`, a recorded environment (device, torch/jax version, SHA), and the
seed. Kaggle results cite the kernel directory and the SHA the kernel pinned.

### 3. Gates are not formalities

Phase 1's kill criterion is pre-stated in `docs/02`. If it fires, the project becomes
the measurement paper; do not soften the margin after seeing the data. No paid compute
before G1 closes. No Phase 5 (Mamba) work before G3.

### 4. Pre-registration

The 10^7-token slice, the eval suite, the decontamination procedure and the Tier 0
margin are written down in `docs/` **before** the corresponding run starts, with the
commit SHA as the timestamp.

## Layout (target)

```
layer_wise_distillation/
├── PLAN.md, CLAUDE.md, README.md
├── claude_stagewise-noise-distillation-plan.md   the source document, kept verbatim
├── docs/            00 review, 01-03 phase docs, compute.md, conventions.md (later)
├── src/lwd/         harvest/, contract/, noise/, stage/, compose/, heal/, eval/
├── experiments/     phase0/, phase1/, ...; each with configs/, kaggle/, results/
├── tests/           CPU only, no network, two tiers like `es` (--fast, full)
└── drafts/          local only, never committed
```

## Environment

conda env `open-source` (tree CLAUDE.md §3). `torch` in it is **CPU-only** (2.13.0+cpu);
a CUDA build is a Phase 0 setup step and goes in via `conda install -n open-source`
or the pip CUDA wheel, never into `base`. Local GPU: RTX 3080 Laptop, 16 GB, Ampere.
Kaggle CLI is configured (access token, username in `kaggle config view`, not in git).
