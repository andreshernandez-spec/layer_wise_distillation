# PLAN.md

> The gate criteria in `docs/` are authoritative; the one-liners here are navigation.
> Do not restate a gate here, point at it. (Lesson carried over from `es/PLAN.md`,
> where two abbreviated gates each dropped the load-bearing half.)

Source document: `claude_stagewise-noise-distillation-plan.md` (22 Aug 2026). Literature
verification of its §4: `docs/00-literature.md`. This file
and `docs/` turn it into phases with numbered capabilities, numbered gate criteria, and
a compute calendar that matches what the free tiers actually allow. Where this plan
disagrees with the source document, `docs/00-plan-review.md` says why.

## Thesis

Distill a large pretrained teacher into a small student by training student **stages**
(groups of layers) **independently**, each fed with **random noise at the teacher's
interface** (moment-matched, contract-normalized) plus a small real-activation anchor
set, with a bijective interface normalization and a sketched characteristic-function
contract at each interface, HT-SR spectral metrics as acceptance gates, sharded ES for
the discrete objective terms, and a short end-to-end heal on ~10^7 real tokens. The
stagewise phase never loads the whole model.

Two claims, kept separate (source doc §0): **A**, match a from-scratch small model at a
fraction of the FLOPs and ~10^-4 of the real tokens; **B**, match the incumbent
prune+distill pipeline from the same teacher at a fraction of the real tokens.

**The load-bearing number is β**, the exponent in ε(Q) ≈ c·Q^(-β) + ε_∞ for a
noise-trained stage as a function of noise queries Q, measured against the same stage
trained on real activations. Phase 1 measures it. Everything after Phase 1 is
conditional on it.

## Phases

| # | Phase | Output | Compute | Est. | Gate |
|---|---|---|---|---|---|
| 0 | Harvest tooling | Interface dataset + stats + anchor store, verified | laptop 3080, CPU | 2 wk | G0 |
| 1 | Single-stage measure mismatch | β per arm, budget forecast, go/no-go | laptop 3080 + Kaggle 2xT4 | 3 wk | G1 |
| 2 | Composition and healing | amplification profile, heal-budget curve | laptop + Kaggle | 3 wk | G2 |
| 3 | Tier 0 full pipeline | Pythia-2.8B → 410M-config vs. from-scratch 410M | Kaggle + laptop, ~40 GPU-h | 5 wk | G3 |
| 4 | Tier 1 | OLMo-2-7B → 1B-config vs. OLMo-2-1B; self-run Minitron baseline | TRC window + ~$100-300 spot | 5 wk | G4 |
| 5 | Claim-B headline | Nemotron Nano 2 12B → 9B-class | v5e-8 heal, H100 spot for eval | conditional | - |
| 6 | Artifacts | toolkit + paper | - | 6 wk | - |

Roughly 13 weeks to the Tier 0 decision, ~5 months to a Tier 1 result, ~6-7 months to a
submission. **Apply to TRC around week 10**, not week 1: the grant is ~30 days from
acceptance and has to land on Phase 4 (`es/docs/06-benchmark-runbook.md` §T3 has the
trap written up).

### Phase 0: Harvest tooling → `docs/01-phase0-harvest.md`

Streaming, stage-resident harvester over a pre-declared 10^7-token slice. Interface
statistics (mean, shrinkage covariance, CF sketch, per-channel quantiles), fp8 rotated
anchor store, top-k logit store. Substrate: **Pythia-1.4B** (fits the laptop whole, has an
identical-corpus from-scratch ladder). The streaming path is exercised at Tier 0 even
though the teacher fits, because Tier 1 needs it and it must not be debugged on rented
hardware.

**Gate G0**: six numbered criteria in `docs/01`. The short form: anchors dequantize to
within the fp8 floor and reproduce the next-interface teacher output; every statistic is
reproducible from a committed config and a SHA; the Kaggle kernel for the harvester runs
unattended.

### Phase 1: Single-stage measure mismatch → `docs/02-phase1-mismatch.md`

One teacher stage, one matched student stage, four input measures (real, moment-matched
Gaussian, isotropic, contract-Gaussianized) crossed with sequence structure (i.i.d.,
AR(1), anchor-mixed), evaluated on held-out **real** activations and by **stitching** the
student stage into the teacher. Fit β and ε_∞ per arm. Measure the PCA-bridge floor
separately from the measure-mismatch floor.

**Gate G1**: five numbered criteria in `docs/02`, including the kill criterion and the
written Tier-1/2 budget forecast. No paid compute before G1 is closed.

> **Status 22 Aug 2026, one seed, Pythia-1.4B stage 2.** Phase 0 closed the same day
> (G0.3 passes 5 of 6 stages, marginal fail at the last; `docs/01`). Phase 1 cells
> at Q = 1e5..1e8: the recipe arm (1 real anchor sequence in 11, rest moment-matched
> Gaussian) reaches eps 0.464 and a 0.52-nat stitch delta at 1e8, better than the
> live-real oracle at the full 1e7 real budget (0.535 / 1.00). Pure noise has a floor
> at eps ~0.87 and diverges past ~1e7 (eps 12 at 1e8); AR(1) noise = i.i.d. noise;
> isotropic noise fails; spectral alpha does not separate healthy from degraded
> students. **Control result, 23 Aug 2026**: the same anchors without noise beat
> the mix at 1e8 (eps 0.413 vs 0.464), and ten passes over the 1e7 real budget beat
> both (0.353). Noise adds nothing at ~1M anchor positions per interface; the
> data-frugality result stands (recycled anchors beat 10x more real data seen once).
> **Moved to a rented A100 on 23 Aug 2026** (`docs/compute.md`): the full 89-cell
> design, the scarce-anchor cells and the 3e8 overfitting cells run there for ~$12.
> Two GPUs reproduce eps to a median 0.00%, so laptop and pod cells read as one table.
> **Correction the same day**: the "contract-Gaussianized" arm was affine-only and so
> identical to the Gaussian arm (they agree to 0.01% at every shared Q); marginal
> Gaussianization, a load-bearing claim of the source document's §1.1, is being tested
> separately by `grid-1.4b-gauss.yaml`, read on the stitching delta because eps lives
> in phi coordinates. `docs/02` has every table.

### Phase 2: Composition and healing → `docs/03-phase2-composition.md`

Chain noise-trained stages from Phase 1's best arm. Drift vs. depth, amplification
profile through downstream Lipschitz constants, DAgger-style on-policy mixing, Theseus
swap, heal-budget curve against random-init + equal heal.

**Gate G2**: `docs/03`. Kill: healed stagewise ≤ random-init + equal heal at equal total
FLOPs.

### Phase 3: Tier 0 full pipeline

Pythia-2.8B (32 layers, d=2560) → Pythia-410M config (24 layers, d=1024): 8 stages of 4
teacher / 3 student layers, 2.5x width bridge. Full contract machinery, α gates, ES for
the discrete terms. Comparator: from-scratch Pythia-410M (300B Pile tokens). All four
controls from source doc §0. Pre-registered margin, eval suite and corpus declared in
`docs/04-phase3-tier0.md` **before the run starts** (file to be written at the end of
Phase 2, not before; it depends on β).

Tier 0a, if Phase 1 says the width bridge is the binding floor: Pythia-1.4B → 410M
config (same depth, 2x width only, 3.4x params). Gentler, and the cleanest single-factor
comparison in the family.

**Gate G3**: match the from-scratch comparator within the pre-registered margin using
≤10^8 noise positions per stage and ≤10^7 real tokens. Partial: the paper becomes the
β-measurement + ablation paper, which is still a paper.

### Phase 4: Tier 1

OLMo-2-7B (32 layers, d=4096) → OLMo-2-1B config (16 layers, d=2048): 8 stages of 4/2,
2x width. The self-run token-matched Minitron-style baseline on the same teacher and
corpus is the controlled Claim B. Corpus swap on the 10^7.

### Phase 5: Claim-B headline (conditional)

Nemotron Nano 2, 12B-v2-Base → 9B-class. Hybrid-Mamba interface semantics are a new work
item and stay out of paper one.

### Phase 6: Artifacts

Toolkit (PyTorch/HF for harvest and stage training, JAX for ES and the TPU heal) and
paper.

## What "done" looks like at Phase 3

1. A figure: ε(Q) for each input measure with fitted β, real-activation oracle as the
   floor, and the stitching-loss version of the same plot.
2. A table: Tier 0 student vs. from-scratch Pythia-410M vs. the four controls, on the
   pre-declared suite, with real tokens and noise positions in separate columns.
3. Interface dataset published (Kaggle dataset or HF), with the harvester that made it.
4. A limitations section that says what the noise route cannot do (sequence structure,
   capacity gap), with the measurement that shows it.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| β too small (noise plateaus far above real) | **answered 23 Aug 2026** | β = 0.32 for live real, recycled anchors and the recipe alike, with ε_∞ consistent with zero for all three; only pure noise has a floor (0.417) and it diverges. The exponent is a property of the stage-fitting problem, not the input measure. `docs/02`. |
| Attention stages do not learn from i.i.d. noise | **confirmed 23 Aug 2026** | Sequence-structure arms are in Phase 1 proper, not a side experiment. Hybrid fallback: attention-heavy terms on anchors, MLP-heavy on noise. Say so if it happens. |
| "Stages in concurrent Kaggle sessions" does not exist | **certain** | Kaggle allows 2 concurrent GPU batch sessions and 1 TPU session. Stage parallelism is 2 Kaggle sessions + the laptop. Arithmetic in `docs/compute.md`; Tier 0 still fits in ~1 week. |
| PCA width bridge discards what the next stage needs | medium | Phase 1 measures the bridge floor with a linear-probe oracle before any student is trained on it. Tier 0a exists for the case where it binds. |
| Capacity-gap curse at 6.8x+ | medium | Tier 0a (3.4x) vs Tier 0b (6.8x) is the measurement. |
| Benchmark adjacency of the 10^7 | medium | Pre-declared slice, decontaminated against the suite, corpus-swap ablation. Protocol in `docs/01`, not optional. |
| Spectral metrics as per-stage gates | **high, first test negative (22 Aug 2026)** | On 2-block students at <= 3e7 positions, alpha and stable rank track training length and do not separate a noise-degraded student (eps 1.34) from a healthy one (0.57): `docs/02` "Spectral gates". The stopping signal comes from real anchors. Re-test on Phase 3 stages before dropping alpha; any spectral *loss* ships with an ablation. |
| Pile availability | low-medium | Original hosting is gone; `monology/pile-uncopyrighted` on HF is the fallback, and Pythia's data order is reproducible from their tooling. Settle in Phase 0. |
| TRC window wasted | high if applied early | Apply ~week 10. Have Phase 4 scripts ready the day quota lands. |
| Laptop thermals | medium | Same machine as `es`; package hits 95 C on sustained load. Multi-day sweeps go to Kaggle; the laptop takes the harvest and the debugging. |
| Crowded niche (synthetic-data distillation) | medium | Two named baselines at Tier 0: Puzzle-style blockwise KD on real activations (2411.19146) at matched real tokens, and self-generated-text KD at matched compute. The intro answers "why not just use data" with those numbers plus the depth-parallel and data-egress story. `docs/00-literature.md`. |
| Scope creep into Phase 5 (Mamba interfaces) | medium | Nothing Mamba-related before G3. |

## Decisions Andres owns (see `docs/00-plan-review.md` §3)

1. What a "noise sample" is: a position or a sequence. The FLOPs clause of Claim A
   flips on this.
2. Anchors per interface: 10^6 (storage-driven) or 10^7 (the source doc says both).
3. Substrate for Phase 1: Pythia-1.4B (recommended) or SmolLM2-1.7B.
4. The G1 kill margin (a proposal is in `docs/02`).
