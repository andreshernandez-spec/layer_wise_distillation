# Phase 1: Single-stage measure mismatch (Exp 1)

**Compute**: laptop RTX 3080 for the teacher-resident arms; Kaggle 2xT4 (two sessions)
for the sweep tail. ~50 GPU-hours total at the budget below; see `docs/compute.md`.
**Duration**: 3 weeks.
**Substrate**: Pythia-1.4B, stage k=2 (blocks 8-11) as the primary; stage 0 (blocks
0-3) and stage 5 (blocks 20-23) as the edge checks.
**Gate**: G1, bottom of this file. **Nothing paid before G1.**

## The question

How fast does a student stage trained on **noise** at the teacher's interface approach
the same student trained on **real** activations, as a function of the number of noise
positions Q, and where does it plateau?

The answer is two numbers per arm, β and ε_∞ in

    ε(Q) = c · Q^(-β) + ε_∞

and one derived number: the Q at which the noise arm would reach the real arm's error at
the real arm's budget, extrapolated from the fit. That is the Tier-1/2 budget forecast.

## Design

Two-factor design, every cell trained to the same Q grid.

**Factor 1, input measure** (what the positions are drawn from):

| arm | measure | parameters from |
|---|---|---|
| R | real teacher activations at interface k | the slice, live through the teacher |
| G | N(μ_k, Σ_k), shrinkage covariance | C0.2 |
| I | N(0, σ²I), σ² = tr(Σ_k)/d | C0.2 |
| C | contract-Gaussianized: z ~ N(0,I) in whitened coords, per-channel inverse-CDF to the teacher's marginals, un-whiten | C0.2 quantiles + ZCA |

**Factor 2, sequence structure** (how positions within a sequence relate):

| arm | structure |
|---|---|
| iid | independent across positions |
| ar1 | per-channel AR(1) in whitened coords, ρ from C0.2 lag-1 autocorrelation |
| lr | position-averaged covariance plus a low-rank positional correction fitted from C0.2 |
| mix | iid noise mixed 1:10 with real anchor **sequences** (C0.3), the recipe arm |

R is a single cell (real activations carry their own structure). The other three
measures cross with all four structures: 1 + 12 = 13 cells. Budget cuts go to `lr`
first; G×iid, C×iid, C×ar1, C×mix and I×iid are the cells that must exist.

**Target**: in all cells the target is the teacher's interface k+1 output for the same
input, in transformed coordinates `T̃_k = φ_{k+1} ∘ T_k ∘ φ_k⁻¹` with φ the ZCA
whitening from C0.2. Loss: MSE in those coordinates. No contract term in Phase 1: this
phase measures the measure mismatch, and a CF term would mix in a second effect. The
contract term is a Phase 2 ablation.

**Student**: Pythia-1.4B block architecture (d=2048, 16 heads), **2 blocks** where the
teacher stage has 4, same width. No bridge. This isolates the input measure from the
width question. One additional cell, R with a 1024-wide student through the PCA bridge,
measures what the bridge costs on real data, to be read next to the C0.5 bridge oracle.

**Q grid**: 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8 positions, as sequences of L=2048 (so
49 to 48 828 sequences). R stops at 1e7 (the real token budget; that is the point).
Each cell at each Q is an independent run from the same init seed; two seeds at Q ≤ 1e6
where variance is largest. Optimizer, LR schedule and steps-per-Q fixed across cells and
written in the config before the first run; tune on G×iid at Q=1e6 only, then freeze.

**Length ablation**: C×mix at L ∈ {512, 2048} at Q=1e7, to see whether shorter sequences
(cheaper) change the answer.

## Measurements

Every cell, every Q, on the same held-out set of 10^5 real positions (from the held-out
range, never the slice):

1. **ε**: relative MSE in whitened coordinates, E‖T̃(x) − S(x)‖² / E‖T̃(x) − E T̃‖².
   This is what β is fitted on.
2. **Stitching loss**: swap the student stage into the teacher (with φ folded in),
   measure next-token loss on held-out text, report the delta against the unmodified
   teacher. This is the number that predicts whether the composition will work.
3. **Attention entropy** per head of the *teacher* stage on each arm's inputs vs. on
   real inputs (C0.2 baseline). If the noise arms give near-uniform attention, that is
   the mechanism, and the sequence-structure arms are the fix or they are not.
4. **Jacobian agreement**: cosine between teacher and student stage Jacobian-vector
   products on held-out real inputs, at 256 random directions. Cheap, and the
   Srinivas-Fleuret hook says output matching under input noise ≈ Jacobian matching,
   so this checks whether the theory applies here.
5. Per cell, per Q: wall-clock, device, SHA, seeds.

## Fit

Nonlinear least squares of `ε(Q) = c·Q^(-β) + ε_∞` per cell over the Q grid, with
bootstrap CIs over seeds and held-out resampling. Report β, ε_∞, c, and
`Q*(cell) = (c / (ε_R(1e7) − ε_∞))^(1/β)` when ε_∞ < ε_R(1e7), else "unreachable".
Same fit on the stitching delta.

## Analysis questions, answered in the phase writeup

- Does C beat G beat I, and by how much? (Does the marginal shape matter?)
- Does `ar1` or `mix` recover the attention entropy of real inputs? Does it recover ε?
- Does the edge stage (k=0) behave differently from the middle? The late stage?
- Is the bridge cost (R-1024 vs R) close to the C0.5 oracle floor? If it is far above,
  the student is capacity-limited, not bridge-limited.
- Budget forecast: with `Q*` for the best noise cell, what are the Tier 0, 1 and 2 noise
  budgets and FLOPs, per stage and total? Written out with the arithmetic.

## Gate G1

1. All must-exist cells ran on the full Q grid with the measurements above, from a
   committed config, on recorded hardware.
2. Fits with CIs exist for every cell, for both ε and stitching delta.
3. **Kill criterion, pre-stated (margin to be set by Andres before the first run;
   proposal)**: the best noise cell has
   `ε_∞ ≤ 1.5 × ε_R(1e7)` **and** a stitching delta at Q=1e8 that is `≤ 2 × the R cell's
   delta at 1e7`. If neither the best pure-noise cell nor `mix` meets it, Phases 2-5
   are cancelled and the project becomes the measurement paper (β, ε_∞, the attention
   entropy mechanism, the bridge floor).
4. A written budget forecast for Tier 0, 1 and 2 from the measured `Q*`, with FLOPs and
   wall-clock on the actual tiers in `docs/compute.md`.
5. The length ablation has an answer (does L=512 change β or ε_∞ beyond the CI?).

**Ambiguous** (the kill criterion met by `mix` but not by any pure-noise cell): proceed,
with the recipe defined as anchor-mixed noise and the pure-noise arm demoted to the
ablation it was always meant to be (source doc §0, control 3). Say so in the writeup.

## Implementation status (22 Aug 2026)

Everything below runs on CPU against pythia-70m in `tests/` (21 tests, ~30 s) and was
smoke-run end to end on the 70m harvest. Nothing has run on the 1.4B yet.

| piece | where | note |
|---|---|---|
| contract maps phi (ZCA, PCA top-d_s, marginal Gaussianization), exact inverses | `src/lwd/contract/whiten.py` | `tests/test_contract.py` checks bijection and whitening |
| samplers G, I, C x iid / ar1 / mix | `src/lwd/noise/samplers.py` | ar1 uses the harvested per-channel lag-1 rho; `mix` asserts >= 1 real sequence per batch |
| student stage (fresh GPTNeoX blocks, any width/depth) and trainer | `src/lwd/stage/` | loss = relative MSE in phi coordinates; AdamW, warmup + cosine, bf16 autocast on GPU |
| measurement 1, eps on held-out real activations | `lwd.stage.train.evaluate` | |
| measurement 2, stitching delta | `src/lwd/eval/stitch.py` | hooks replace blocks [a,b) of the full model; teacher's own stage stitched back gives delta = 0 exactly (`tests/test_stitch.py`) |
| measurement 3, teacher attention entropy under the arm's inputs | `src/lwd/eval/diagnostics.py` | eager attention; real-input baseline recorded in the same cell |
| measurement 4, Jacobian agreement | same | forward-mode JVP under eager attention (fused SDPA has no forward AD / double backward) |
| one cell | `experiments/phase1/run.py CONFIG --measure {R,G,I,C} --structure {iid,ar1,mix} --q Q --seed S` | one JSON + log + state dict per cell |
| the grid, resumable | `experiments/phase1/sweep.py CONFIG GRID` | 78 cells in `configs/grid-1.4b.yaml`; skips cells whose JSON exists |
| L arm: live real activations (distinct positions) | `lwd.noise.samplers.LiveReal` + `lwd.harvest.model.Lower` | bitwise-tested against the resident model |
| spectral metrics (alpha, stable rank) | `src/lwd/eval/spectral.py` | gates only; first test negative, see below |
| the table | `experiments/phase1/summarize.py OUT [--md]` | every cell, best and final eps, stitch, Jacobian, entropy |
| the gate | `experiments/phase1/gate.py OUT` | criterion 3 verdicts from the eps_inf CI, Q*, Tier 0/1/2 budgets |
| the fit | `experiments/phase1/fit.py OUT [--metric best_eval]` | `eps = c Q^-beta + eps_inf`, bootstrap CIs, `Q*` against the R arm at 1e7; self-test recovers beta 0.35 as 0.365 [0.32, 0.51] |

Not yet: the `lr` (low-rank positional) structure arm, the bridge cell (R with PCA to
1024: the `pca` contract exists, the cell config does not), the length ablation config.

## First pass, Pythia-1.4B stage 2 (22 Aug 2026)

Six arms at Q = 1e5 and 1e6 (6 and 61 steps), one seed, `out/phase1-1.4b/`. eps is
relative MSE in phi coordinates on 48 held-out sequences (eps at init: 3.10); stitch
is the next-token loss delta of the teacher with the student stitched in (teacher
1.902 nats on those rows); H is the teacher's mean attention entropy on the arm's
inputs vs 2.75 on real inputs (uniform 7.62).

| arm | eps @1e5 | eps @1e6 | stitch @1e6 | Jacobian cos @1e6 | H on arm inputs |
|---|---|---|---|---|---|
| R (real) | 2.930 | **1.621** | 5.42 | 0.205 | 2.44 (held-out real) |
| C_mix (1:10 real) | 2.938 | 1.639 | 5.41 | 0.212 | 3.80 |
| G_iid | 2.941 | 1.653 | 5.39 | 0.214 | 4.81 |
| C_iid (= G, ZCA contract) | 2.941 | 1.653 | 5.39 | 0.214 | 4.81 |
| C_ar1 | 2.941 | 1.655 | 5.40 | 0.213 | 4.81 |
| I_iid (isotropic) | 2.967 | 1.873 | 5.32 | 0.202 | 4.91 |

Reading, with the caveat that at 61 steps every student is still far from trained
(eps > 1 means worse than predicting the mean):

1. Isotropic noise is already behind (1.87 vs 1.65): second moments matter from the
   first steps.
2. Real, anchor-mix and moment-matched Gaussian are within 2% of each other this
   early. The separation, if any, is a Q >= 1e7 question.
3. **AR(1) across positions does nothing**: identical eps to i.i.d., and the
   teacher's attention entropy is identical (4.81). The sampler is verified (lag-1
   0.335 against the harvested 0.337, `tests`/direct check), so this is a result:
   per-channel temporal correlation is not the sequence structure attention reads.
   Only real sequences move the entropy (mix: 3.80 with 1 real in 11). The `lr` arm
   (low-rank positional correction) is likely to share this fate; the question for
   Exp 1b is whether *token-like* structure (e.g. resampling real sequences' token
   identities, or anchors) is needed, not smoother noise.
4. The stitched loss is not yet informative (all ~5.4 nats, Jacobian cosine ~0.21).

Cost: 0.82 s per step of 16k positions; `docs/compute.md`.

## Second pass, Q = 1e7 (22 Aug 2026, 610 steps, one seed)

| arm | eps | stitch delta (nats) | Jacobian cos | H on arm inputs | minutes |
|---|---|---|---|---|---|
| R (anchors, ~10 epochs) | **0.544** | **1.05** | 0.296 | 2.44 | 7.3 |
| C_mix (1 real in 11) | 0.645 | 2.15 | 0.430 | 3.80 | 8.8 |
| C_ar1 | 0.868 | 2.91 | 0.456 | 4.81 | 9.4 |
| G_iid / C_iid | 0.873 | 3.2 | 0.456 | 4.81 | 8.8 |
| I_iid | 1.55 (best 1.37, diverging) | 6.58 | 0.231 | 4.91 | 8.7 |

The separation opens at 1e7: moment-matched noise sits at 1.6x the real arm's eps and
3x its stitched loss delta; one real sequence in eleven halves the gap on eps and
closes most of it on the stitch; isotropic noise diverges. AR(1) across positions buys
nothing on eps and a little on the stitch (2.91 vs 3.2). The noise-trained students
have *higher* Jacobian agreement with the teacher on real inputs (0.456) than the
real-trained one (0.296): output matching under input noise is Jacobian matching
(Srinivas-Fleuret), and the noise arms are doing that rather than fitting the data
manifold.

**Caveat on R, to be fixed before it is called the oracle**: the R arm samples from the
464 anchor sequences of interface 2 (0.95M positions), so at Q = 1e7 it has seen each
~10 times. eps is on 48 held-out sequences, so 0.544 is a real generalization number,
but the plan's oracle is 1e7 *distinct* real positions, which needs live teacher
activations from the slice (`R_live`, embedding + blocks 0-7 resident). Until R_live
runs, the noise/real gap above is a lower bound on the true gap.

A three-point fit (1e5, 1e6, 1e7) of the three-parameter model is exactly determined
and pins eps_inf at 0 for every arm but isotropic; those numbers are not cited. The
fit needs 3e7 and 1e8 (third and fourth passes).

## Third pass: live-real oracle and Q = 3e7 (22 Aug 2026, one seed)

| arm | Q | eps | stitch delta | Jacobian cos | minutes |
|---|---|---|---|---|---|
| L (live real, distinct positions) | 1e6 | 1.616 | 5.40 | 0.206 | 1.6 |
| L | 1e7 | **0.535** | **1.000** | 0.297 | 11.8 |
| R (anchors, recycled) | 1e7 | 0.544 | 1.05 | 0.296 | 7.3 |
| C_mix | 3e7 | **0.571** | **1.13** | 0.480 | 25.0 |
| G_iid | 3e7 | 1.335 (was 0.873 at 1e7) | 3.54 | 0.329 | 25.1 |
| C_ar1 | 3e7 | 1.764 (was 0.868 at 1e7) | 3.92 | 0.252 | 26.9 |

1. **R was a fair oracle.** Distinct real positions (L) at 1e7 give eps 0.535 and a
   1.00-nat stitch delta against 0.544 / 1.05 for the anchor-recycled R; ten passes
   over 1M positions did not overfit at this student size. L costs 60% more per step
   (8 teacher blocks run live).
2. **Anchor-mix keeps improving**: at 3e7 it is within 7% of the real arm's eps at
   1e7 and within 0.13 nat on the stitch, having used 2.7M real positions (1 in 11)
   and 27M noise positions.
3. **Pure-noise arms degrade past 1e7**: both the Gaussian and the AR(1) student get
   markedly worse on real activations between 1e7 and 3e7 while still training on
   noise. That is the measure-mismatch signature the project is about, in its
   sharpest form: the noise-measure optimum is not the real-measure optimum, and more
   optimization toward the former moves away from the latter. Trajectory check below.

**Trajectory check** (held-out eps every 200 steps; training loss is on the arm's own
inputs):

| cell | train loss first / mid / last | best eps (step) | final eps | eps trajectory |
|---|---|---|---|---|
| G_iid 1e7 (610 steps) | 4.83 / 0.357 / 0.326 | 0.869 (400) | 0.873 | 3.10 0.89 0.87 0.87 0.87 |
| G_iid 3e7 (1831) | 4.83 / 0.274 / 0.259 | 0.888 (200) | 1.335 | 3.10 0.89 0.90 0.95 1.00 1.07 1.15 1.25 1.32 1.34 |
| C_ar1 3e7 | 4.80 / 0.278 / 0.262 | 0.883 (200) | 1.764 | 3.10 0.88 0.89 0.95 1.00 1.11 1.31 1.56 1.73 1.76 |
| C_mix 3e7 | 4.52 / 0.344 / 0.325 | 0.571 (1800) | 0.571 | 3.10 0.72 0.65 0.62 0.61 0.59 0.58 0.57 0.57 |
| L 1e7 | 3.46 / 0.585 / 0.513 | 0.535 (609) | 0.535 | 3.10 0.65 0.55 0.54 0.54 |

The noise-measure loss keeps improving while the real-measure error climbs: the two
optima differ, and optimizing past ~1e6 noise positions walks away from the real one.
Consequences for the design:

- **The pure-noise floor at this stage is eps ~0.87**, reached by ~1e6-1e7 positions
  and never improved on; the real arm is at 0.535 and still falling at 1e7. Read as
  a measurement-paper number: beta for pure noise is effectively zero past 1e6.
- **Pure-noise training needs a stopping signal on real data**, which is exactly what
  an independent stage lacks; the anchors supply it even if not trained on. The
  source document's HT-SR gates were meant to substitute for such a signal; whether
  alpha tracks this rise is a cheap thing to test on these checkpoints.
- **Anchor-mix does not have the problem** and is the recipe (source document §0,
  control 3 framing confirmed); pure noise is the ablation.
- For the fit, eps(Q) of a noise arm is the early-stopped (best held-out) value, not
  the final one; `fit.py --metric best_eval`.

**Spectral gates, first test** (`src/lwd/eval/spectral.py`: Hill alpha with KS-chosen
xmin on the ESD of W^T W, and stable rank, per weight matrix of the saved students):

| cell | eps | mean alpha (min-max) | mean stable rank |
|---|---|---|---|
| L 1e7 | 0.535 | 13.1 (8.0-21.9) | 407 |
| C_mix 3e7 | 0.571 | 6.0 (3.8-8.2) | 116 |
| G_iid 1e7 | 0.873 | 8.6 (5.2-17.6) | 240 |
| G_iid 3e7 (degraded) | 1.335 | 5.7 (3.6-7.4) | 137 |
| C_ar1 3e7 (degraded) | 1.764 | 5.8 (3.6-7.1) | 136 |

Alpha and stable rank fall with training length in every arm and do not separate the
healthy 3e7 student (anchor-mix, eps 0.571) from the degraded one (Gaussian, eps
1.335); the one striking collapse (layer-0 MLP input, stable rank 5.4) appears in the
healthy cell too (13.5). These students are young (alpha far above the 2-5 band), so
this is not the regime HT-SR was developed for, but it is the regime a stage trainer
is in, and **the per-stage acceptance gate the source document assigns to alpha
(§1.5) does not work here**. The stopping signal has to come from real anchors.
Revisit with longer-trained stages (Phase 3) before dropping the idea; `PLAN.md` risk
register updated.

**Preliminary fit on the early-stopped metric** (`fit.py --metric best_eval`, 4 points
for the noise arms, 3 for R): C_mix beta 0.29 [0.27, 1.0], eps_inf 0 [0, 0.53]; G_iid
beta 0.33, eps_inf 0.43 [0, 0.88]; R beta 0.31 (3 points, no CI). Not citable until the
1e8 cells land and a second seed exists.

**Gate script** (`experiments/phase1/gate.py`): reads the cells and the early-stopped
fit, states criterion 3 against the reference (L at 1e7: eps 0.535, stitch 1.00 nats;
margins 1.5x and 2x as proposed), and turns Q* into a budget. Verdicts come from the
eps_inf CI: pass only if its upper bound clears the margin, fail only if its lower
bound misses it. Provisional state before the 1e8 cells and a second seed:

| arm | stitch @Qmax | eps verdict | Q* | Tier 0 budget at Q* |
|---|---|---|---|---|
| C_mix | 1.13 (pass) | pass on CI (eps_inf CI [0, 0.53]) | 3.3e7 | 2.3e17 FLOPs, ~8 T4-hours |
| G_iid | 3.54 (fail) | undetermined (CI [0, 0.88]) | 1.1e9 | 7.5e18, ~190 T4-hours |
| C_ar1 | 3.92 (fail) | undetermined | 8e8 | 5.5e18 |
| I_iid | 6.58 (fail) | fail | none | |

Read with the trajectory finding: the pure-noise arms' fitted eps_inf is an artefact
of three or four points; their observed floor is ~0.87 and does not move with Q, so
their Q* is not a budget anyone would spend. The recipe arm is on track to pass G1 on
the proposed margins with a Tier 0 budget far below the 10^8 positions per stage the
source document assumed.

## Fourth pass, Q = 1e8 (22 Aug 2026, 6104 steps, 83 min each, one seed)

| cell | train loss first / mid / last | best eps (step) | final eps | stitch delta | Jacobian cos |
|---|---|---|---|---|---|
| C_mix 1e8 | 4.52 / 0.293 / 0.254 | **0.464** (6000) | **0.464** | **0.519** | 0.470 |
| G_iid 1e8 | 4.83 / 0.227 / 0.213 | 0.886 (200) | **12.24** | 7.65 | 0.278 |

eps trajectory, C_mix: 3.10 0.65 0.60 0.58 0.55 0.53 0.51 0.50 0.49 0.48 0.48 0.47 0.47 0.46 0.46
eps trajectory, G_iid: 3.10 0.90 1.02 1.89 4.21 5.49 7.38 8.18 8.64 9.36 10.26 11.04 11.87 12.24

1. **The recipe arm at 1e8 beats the live-real oracle at the full real budget**:
   eps 0.464 and a 0.52-nat stitch delta against 0.535 / 1.00 for L at 1e7, and it is
   still improving. It has seen 0.95M distinct real positions (the 464 anchor
   sequences, ~10 passes) and 91M noise positions.
2. **Pure Gaussian noise collapses**: the student keeps lowering its loss on noise
   (0.21) while its error on real activations goes from 0.89 to 12.2. Not a plateau,
   a divergence. Whatever the noise-measure optimum is, it is nowhere near the real
   one, and the anchors are what keep the mixed arm attached to it.
3. The comparison in (1) is not yet the controlled one. Two cells are running:
   **R at 1e8** (the same anchors, no noise, same steps: if noise adds nothing, R
   matches C_mix; if noise regularizes, R overfits its 464 sequences and C_mix wins),
   and **L at 1e8** (real, recycled 10x over the 1e7 budget). The headline stands or
   falls on R@1e8.

**Five-point fit, early-stopped eps** (`fit_best_eval.json`, bootstrap 90% CIs over
the points; one seed, so the CIs are about the curve shape, not seed noise):

| arm | points | beta | eps_inf | Q* vs L@1e7 |
|---|---|---|---|---|
| C_mix | 5 (1e5..1e8) | 0.29 [0.14, 1.0] | 0.00 [0, 0.53] | 3.7e7 |
| G_iid | 5 | 0.40 [0.33, 3] | **0.65 [0.38, 0.89]** | none |
| C_ar1 | 4 | 0.32 | 0.41 [0, 0.88] | (meaningless, diverged) |
| R | 3 (1e5..1e7) | 0.31 | 0 | 2.4e7 |

The Gaussian arm's floor is now bounded away from zero on the early-stopped metric,
and the recipe arm's is consistent with zero over the range measured. beta for the
recipe arm (0.29) is close to the real arm's (0.31): mixing does not just lower the
floor, it keeps the exponent. The kill criterion's eps half is therefore about the
recipe arm and reads "pass" on these points (CI upper bound 0.53 < 0.80); the stitch
half passes at 1e8 (0.52 < 2.0 nats). Criterion 3 is met on one seed; criterion 2
(CIs) and the R@1e8 control remain.

## Controls at Q = 1e8 (23 Aug 2026, one seed)

| cell | real positions seen | distinct real | noise positions | eps | stitch delta | Jacobian cos | min |
|---|---|---|---|---|---|---|---|
| L 1e7 (live real, 1 pass) | 1e7 | 1e7 | 0 | 0.535 | 1.000 | 0.297 | 12 |
| **R 1e8** (anchors, ~100 passes) | 1e8 | 0.95M | 0 | **0.413** | **0.502** | 0.328 | 66 |
| C_mix 1e8 (anchors + noise) | 9.1e6 | 0.95M | 9.1e7 | 0.464 | 0.519 | 0.470 | 83 |
| L 1e8 (live real, ~10 passes) | 1e8 | 1e7 | 0 | **0.353** | **0.405** | 0.360 | 102 |

**The headline of the fourth pass was a step-count artefact.** At matched steps the
same anchors without noise beat the mix (0.413 vs 0.464 on eps; stitch equal within
noise), and ten passes over the full 1e7 real budget beat both. Noise inputs are not
destructive inside the mix (pure noise diverges; the mix converges) but they are worse
inputs than re-using a real sequence. What survives:

1. **Data frugality holds, noise is not its carrier.** 0.95M distinct real positions
   recycled ~100x (R@1e8) beat 1e7 distinct positions seen once (L@1e7) by a wide
   margin, and nearly match 1e7 positions seen ten times (L@1e8: 0.353). At this
   student size the limit is optimization steps, not distinct real data, up to 1e8.
2. **Noise volume adds nothing measurable** when ~1M real positions per interface are
   available. Mixing 10 noise sequences per real one costs 12% on eps for no gain.
3. The working thesis (source document: noise supplies volume, anchors pin the
   manifold) is therefore untested in its intended regime and false in this one. The
   intended regime is scarcer anchors: the plan budgets 1e7 real tokens for the
   *whole* model, i.e. ~1e5-1e6 positions per interface when stages do not share
   them, and it is there that noise could still earn its place as a regularizer or
   as the thing that keeps a student training past the point where 1e5 positions
   overfit. That is the next cell set: R vs mix at 1e8 with 46 anchor sequences
   (94k positions), then second seeds, then 3e8 for the overfitting question.

G1 reading: criterion 3's margins are met by the recipe arm, but the control says the
pass belongs to the anchors. If the scarce-anchor cells show no noise benefit either,
Phase 1's answer is the measurement paper: stagewise training on ~1M recycled real
positions per interface, with noise as the ablation that does not help, and the
pure-noise divergence as the mechanism result.

## The campaign moves to a rented A100 (23 Aug 2026)

The laptop queue was stopped mid-grid and everything re-run on one RunPod A100 SXM
(`docs/compute.md` "The rented A100"): 4.5x faster, and the full 89-cell design
(`grid-1.4b-full.yaml`) costs about $2.50 against ~14 laptop-hours. Pod cells live in
`out/phase1-1.4b-a100/` and are never mixed with laptop cells in one table;
`experiments/phase1/compare_platforms.py` measures what the hardware alone is worth.
**The platform effect is nil** (23 Aug 2026, 9 cells run on both, same torch
2.13.0+cu130, different GPUs):

| cell | eps laptop | eps A100 | rel diff |
|---|---|---|---|
| R_iid 1e5 / 1e6 / 1e7 | 2.9296 / 1.6213 / 0.5444 | 2.9296 / 1.6213 / 0.5447 | 0.00 / 0.00 / 0.05% |
| L_iid 1e6 / 1e7 | 1.6164 / 0.5355 | 1.6165 / 0.5355 | 0.00 / 0.01% |
| C_mix 1e5 / 1e6 / 1e7 | 2.9381 / 1.6394 / 0.6446 | 2.9381 / 1.6394 / 0.6416 | 0.00 / 0.00 / 0.46% |
| G_iid 1e5 | 2.9409 | 2.9409 | 0.00% |

Median 0.00%, max 0.46%, and the max is the longest run (1e7), where trajectories have
had the most chance to diverge. Two different GPUs reproduce eps to four decimals, so
laptop and pod cells **may be read in one table**, with the platform named per cell.
That was not safe to assume: it is measured, and `compare_platforms.py` re-measures it
whenever cells are added.

The grid was resized rather than copied. The first passes settled three arms, so
`C_ar1`, `C_iid` and `I_iid` stop at 1e7 and stay as the ablation record, while
`L_iid` (the true oracle, missing from the original grid) and `G_mix` (plain Gaussian
plus anchors: does the contract earn its place inside the mix?) join the four that run
the full ladder to 1e8 with two seeds.

## Correction: the C arm was never the contract arm (23 Aug 2026)

Every cell run before this point used `contract: zca`, an affine map. Under an affine
phi, `ContractGaussianized` (draw z ~ N(0,I) in phi coordinates, push through
phi^-1) is **exactly** the moment-matched Gaussian, because phi^-1 is just
`z @ Winv + mean`. The data says so unambiguously:

| Q | C_iid eps | G_iid eps | diff |
|---|---|---|---|
| 1e5 | 2.9409 | 2.9409 | 0.00% |
| 1e6 (s0 / s1) | 1.6527 / 1.6546 | 1.6527 / 1.6546 | 0.00% |
| 1e7 (s0 / s1) | 0.8725 / 0.8720 | 0.8726 / 0.8721 | 0.01% |
| C_mix vs G_mix, 1e6 | 1.6394 | 1.6394 | 0.00% |

So arm C in the tables above is **not** the contract-Gaussianized arm this document
describes; it is a second copy of the Gaussian arm. Every result stated so far stands
(the recipe arm is anchors plus moment-matched Gaussian noise, which is what it always
was), but the source document's §1.1 claim that marginal Gaussianization is free
because phi is a bijection has never been tested. Marginal Gaussianization only enters
when the cell config sets `contract: gauss`, which builds phi as the per-channel
inverse-CDF map followed by a whitening fitted to the Gaussianized anchors.

`grid-1.4b-gauss.yaml` (tag `gz`) runs L, R, C_mix and G_iid in Gaussianized
coordinates at 1e6, 1e7 and 3e7 to settle it.

**Read that grid on the stitching delta, not on eps.** eps is a relative MSE *in phi
coordinates*, so changing phi changes the metric: the same student scores eps 1.08
under `zca` and 5.40 under `gauss` on the 70m smoke, which says nothing about quality.
The stitching delta is next-token loss in nats through the real model and is
phi-independent, so it is the only number comparable across contract settings.
