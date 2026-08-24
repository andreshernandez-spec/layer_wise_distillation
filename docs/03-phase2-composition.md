# Phase 2: Composition and healing (Exp 2)

**Compute**: laptop + Kaggle 2xT4. Pythia-1.4B → 6 stages of 4 teacher / 2 student
blocks (the Phase 1 student), i.e. a 12-block, d=2048 student, ~700M params.
**Duration**: 3 weeks.
**Gate**: G2, bottom. Written in outline now; the cell list is finalized after G1,
because the recipe arm comes from Phase 1.

## The question

Stages trained independently each carry an error ε_k on the teacher's measure. When
chained, stage k's error becomes stage k+1's input drift, and the realized input of
stage k+1 is no longer the measure it was trained on. How much does that compound, what
does it cost to heal, and does the healed student beat the same heal applied to a random
init?

## Capabilities

### C2.1: Composer

Fold φ maps into neighbouring linear layers, chain student stages, run end to end.
Test: chaining the *teacher's own* stages through the φ/φ⁻¹ pairs reproduces the
teacher bitwise in fp32 (the telescoping claim of source doc §1.1, checked, not
asserted).

### C2.2: Drift and amplification profile

First number, 22 Aug 2026, from `tests/test_compose.py` on pythia-70m: the teacher's
own stages chained through ZCA contract maps agree with the teacher to 2e-11 relative
MSE at the first interface and then drift to 8e-8 and 2e-6 at the next two, while a
bare chain fed a 1e-7 relative input perturbation drifts 7e-13, 2e-9, 5e-9. Float-level
perturbations are amplified 10-1000x per stage in relative MSE, and cutting the
whitening's condition number from 180 to 10 changes that by only ~15x. So the
amplification profile has a floor set by the teacher's own sensitivity, measurable
before any student exists; record it per stage on the 1.4B before reading student drift.


With all six Phase 1 recipe-arm students: per interface, the distance between the
student-propagated interface and the teacher's, in whitened coordinates, on held-out
real text. Per stage, the measured Lipschitz ratio (output drift / input drift) of the
teacher stage on the same inputs. Plot realized drift vs. depth against the product of
the ratios. This is the "measure the amplification, don't assert the mitigation"
item.

### C2.3: Mitigations, as ablations on the same six stages

- **DAgger mixing**: retrain stage k on noise passed through trained stages 1..k−1
  (inference-only, int8), mixed with the original measure at a fixed ratio. Sequential
  across stages by construction; the cost is the parallelism, measure it.
- **Theseus swap**: during stage training, with probability p replace the teacher's
  stage k in the scaffold by the student's and train on the resulting inputs.
- **Contract term**: the teacher-CF sketch loss at the stage output, on and off. This
  is the first place the CF machinery enters, and it enters as an ablation.
- **Gaussian-contract vs teacher-CF** (source doc §1.2 fork): both, same budget.

### C2.4: Heal-budget curve

End-to-end fine-tuning on the 10^7 slice with stored top-k logits (KL + CE), at heal
budgets of 10^5, 10^6, 3·10^6, 10^7 tokens. Three starting points at every budget:
(a) the composed stagewise student, best mitigation; (b) random init, same architecture;
(c) the oracle: stages trained on real activations (Phase 1 arm R). Same optimizer,
same schedule, same data order. Report held-out loss and the pre-declared suite.

### C2.5: FLOPs accounting

Total FLOPs for (a) including harvest, noise phase, any DAgger reruns and the heal,
against (b)'s heal alone and against from-scratch Pythia-410M's 7.4e20. Kept as a
script that reads the run logs, not a spreadsheet.

## Gate G2

1. C2.1 telescoping test passes bitwise in fp32.
2. Amplification profile exists with CIs; the ratio-product prediction is compared to
   the realized drift and the discrepancy is reported, whichever way it goes.
3. Each mitigation has a number on the same axis, same budget.
4. **Kill**: at every heal budget, (a) ≤ (b) on held-out loss at equal total FLOPs.
   If the stagewise init never beats random init plus the same heal, Phase 3 does not
   run. The write-up then is Phase 1's measurement plus this negative.
5. The gap (a) to (c) at 10^7 heal tokens is reported: it is the price of noise at
   Tier 0 scale and the number Phase 3 is trying to keep small.

## What Phase 1 changes here (23 Aug 2026)

Phase 1's results (`docs/02`) redraw parts of this phase before it starts.

**1. There are two recipes, not one, and which applies depends on the anchor budget.**
At ~1M anchor positions per interface, anchors alone beat anchors-plus-noise
(stitching 0.450 against 0.527) and 300 passes do not overfit. At 94k they overfit
from step 600 and noise is what prevents it (1.05 against 0.60 in eps, both seeds).
Phase 2 must therefore chain **two** stacks, not one: an anchors-only stack in the
abundant regime and a mixed stack in the scarce one. Composition may well behave
differently in the two, and that is a result either way. Budget for the pair.

**2. DAgger needs re-thinking before it is built.** C2.3 proposes retraining stage k on
noise passed through the trained stages below it. Phase 1 showed that a student trained
on pure moment-matched noise *diverges* on real activations while its own loss falls,
so noise propagated through k-1 imperfect student stages is further off-manifold than
the noise this phase starts from. Propagate the **anchors** through the trained stages
instead (on-policy real activations, which is what DAgger means in imitation learning)
and keep noise as the volume filler at the current stage. The C2.3 ablation should be
that, not propagated noise.

**3. The stopping signal is not eps, and may not be either metric.** Independent stages
lack a downstream signal; §1.5 of the source document assigned that job to HT-SR alpha
and `docs/02` shows alpha does not separate a healthy student from a degraded one at
this scale. What does separate them is held-out real activations, which every stage has
from its anchors. So the per-stage acceptance gate is: **train on the mix, stop on
held-out anchors.** Cheap, and it is the signal that actually tracks the failure.
There is an open question on top of this (`docs/02`, 3e8 cells): eps improved while the
stitching delta worsened on one seed, so if the two come apart the stopping signal has
to be the stitching delta, which costs a full-model pass. Resolve before Phase 3.

**4. The amplification profile has a floor that is already measured.** C2.2 should be
read against the teacher's own sensitivity: chained teacher stages amplify float-level
input perturbations 10-1000x per stage (`tests/test_compose.py`). Student drift below
that floor is not attributable to the student.

**5. Q per stage is settled well below the plan's assumption.** Q* against the live-real
oracle is 5.1e7 for real, 5.5e7 for recycled anchors and 7.5e7 for the recipe, against
the source document's 1e8 per stage. Phase 2 should use 1e8 for headroom and report
against Q*.

## Machinery built (23 Aug 2026)

| piece | where | state |
|---|---|---|
| composed student LM (teacher edges + distilled stages) | `src/lwd/compose/model.py` | tested: substituting the teacher's own stages recovers its loss |
| all-stages trainer, both recipes, resumable | `experiments/phase2/train_stages.py` | running on the 1.4B |
| drift and Lipschitz profile (C2.2) | `experiments/phase2/drift.py` | exercised end to end on a 3-stage 70m stack |
| end-to-end heal on the stored top-k (C2.4) | `src/lwd/heal/train.py` | KD term verified zero exactly at a teacher match |

### A hypothesis from the smoke stack, to test on the real one

The 70m pipeline test (3 stages, students trained on only 3e4 positions, so barely
trained) gave:

    interface        0        1        2        3
    realized drift   0.000    1.766    1.998    2.276
    teacher stage Lipschitz ratio    0.737    0.480    0.347

Every teacher stage **contracts** a perturbation of the size the student actually
produces, yet the drift still grows with depth. If that survives on the 1.4B, the
compounding-error framing in the source document (§1.6: "per-stage error amplifies
through downstream Lipschitz constants") has the direction wrong at realistic
magnitudes: the problem would be **each stage adding fresh error faster than the next
one damps the inherited error**, not amplification of upstream error.

It would also sit consistently beside the float-level measurement in C2.2 above, where
the same stages amplify 10-1000x. A nonlinear map can be locally expansive near zero
and globally contractive once perturbations are large enough to saturate it, and those
are different regimes, not a contradiction.

**Do not cite this yet.** These students sit at drift ~2, which is worse than
predicting the mean, so their inputs are far off-manifold and the contraction may be
saturation rather than anything about composition. The 1.4B stack, whose stages reach
eps ~0.5, is the test. What the smoke run does establish is that the driver measures
both quantities and that they can be read against each other.

## Stage difficulty by depth (24 Aug 2026, R stack, 1e8 positions each)

| stage | blocks | eps | stitch | Jacobian cos | teacher attn entropy on real inputs |
|---|---|---|---|---|---|
| 0 | 0-3 | **0.524** | **0.724** | **0.038** | 3.92 |
| 1 | 4-7 | 0.477 | 0.415 | 0.273 | 1.97 |
| 2 | 8-11 | 0.413 | 0.506 | 0.328 | 2.75 |
| 3 | 12-15 | 0.348 | **0.760** | 0.347 | 2.48 |
| 4 | 16-19 | 0.309 | 0.543 | 0.291 | 1.41 |

Three things, on one arm and one seed, so read them as the shape of the problem rather
than as settled numbers.

**1. Deeper stages are easier to fit.** eps falls monotonically with depth, 0.524 to
0.309, on identical budgets and identical student capacity. The residual stream gets
more predictable from its own previous layer as depth grows.

**2. Stage 0 is the hard one, and it fails differently.** It has the worst eps, the
second-worst stitching delta, and a **Jacobian cosine of 0.038**: its input-output
Jacobian is very nearly orthogonal to the teacher's. It is matching outputs on the data
it was shown while behaving like a different function locally. Every other stage sits
at 0.27 to 0.35. That is what the source document anticipated structurally when §1.3
said interface 0 is discrete tokens and noise should start after the first block, and
it is now measured: the embedding output is not a residual stream and stage 0 should
not be treated like the others. It also predicts stage 0 will dominate composition
drift, which C2.2 will show directly.

**3. Stitching does not track eps across stages either.** Stage 3 has the second-best
eps and the worst stitching delta. This is the same divergence Phase 1 found across
training length (`docs/02`), now visible across depth, and it is another reason the
per-stage acceptance signal has to be the stitching delta.

## C2.2 on the 1.4B: composition error accumulates, it does not compound (24 Aug 2026)

R stack, six stages at 1e8 positions each, drift measured in whitened coordinates on
held-out real text (`experiments/phase2/drift.py`, `drift_R.json`).

| stage | drift in | teacher Lipschitz | inherited | realized out | fresh |
|---|---|---|---|---|---|
| 0 | 0.000 | **48.5** | 0.000 | 0.510 | 0.510 |
| 1 | 0.510 | 0.454 | 0.232 | 0.776 | 0.544 |
| 2 | 0.776 | 0.448 | 0.348 | 0.935 | 0.587 |
| 3 | 0.935 | 0.500 | 0.467 | 1.097 | 0.630 |
| 4 | 1.097 | 0.841 | 0.923 | 1.301 | 0.378 |
| 5 | 1.301 | 0.753 | 0.980 | 1.535 | 0.556 |

("inherited" is the drift the stage receives times the teacher stage's measured
Lipschitz ratio at that drift magnitude; "fresh" is what the student stage adds on top.)

**Every stage after the first contracts** the error it inherits, by a factor of 0.45 to
0.84. Drift still grows, 0.510 to 1.535, because each stage contributes **fresh error
of about 0.54, essentially constant with depth**. Composition error here is a sum, not
a product.

### This changes what the mitigations are for

Source document §1.6 frames composition as imitation-learning compounding: "per-stage
error amplifies through downstream Lipschitz constants". On this model that is the
wrong direction for five stages out of six. If drift were inherited and amplified,
contracting stages would shrink it; instead the accumulation is additive and dominated
by what each stage adds.

So DAgger and Theseus (C2.3) should be justified and measured as **ways to reduce the
fresh per-stage term under realistic inputs**, not as ways to stop amplification. That
is still a live and sensible motivation: the fresh term includes exposure bias, since
each stage was trained on the teacher's clean interface and is being evaluated on a
drifted one. But the prediction changes. If exposure bias is most of the fresh term,
DAgger should cut it sharply; if the fresh term is mostly irreducible stage error, it
will barely move, and the way to a better composed model is better stages rather than
better propagation. **C2.3 now has a sharp prediction to test rather than an assumed
mechanism to demonstrate.**

### The 48.5x at stage 0 is measured but does not act

The teacher's first stage amplifies a perturbation 48.5x, far out of line with every
other stage. It does not contribute here, because the student uses the teacher's
embedding, so interface 0 has zero drift by construction. What it does explain is the
finding above that stage 0 is the hardest to fit and has a near-orthogonal Jacobian
(0.038): it is approximating a strongly expansive map. It also warns that any variant
which distils the embedding, or which feeds noise at interface 0, inherits a 48x
sensitivity. The source document's instinct in §1.3 (start noise after the first
block) is right, and this is the number behind it.

### C2.3 is built and has a stated prediction

`src/lwd/compose/dagger.py` and `experiments/phase2/dagger.py`. Stage k is retrained on
anchors propagated through the trained stages below it, at a configurable on-policy
fraction, with the target unchanged (the teacher stage applied to the same input): only
the input distribution moves.

It reports **both** numbers, because they answer different questions:

- **eps on the drifted interface**, the one the stage actually meets in the composed
  model, is what DAgger targets;
- **eps on the teacher's clean interface** is the control, and says what the shift cost.

The C2.2 decomposition makes this a test rather than a demonstration. Each stage adds
~0.54 of fresh error. If exposure bias is most of it, retraining on the drifted
interface should cut drifted eps sharply. If it barely moves, the fresh term is
irreducible stage error, DAgger is not the lever, and the route to a better composed
model is better stages. Write the prediction down before running it on the 1.4B.

## C2.3: exposure bias is most of the fresh per-stage error (24 Aug 2026)

The prediction written above, before the run: a large drop in drifted eps means
exposure bias dominates the fresh term and DAgger is the mitigation; a small drop
means the fresh term is irreducible stage error and better stages, not better
propagation, is the route.

First cell, stage 1 of the C_mix stack, retrained at 1e7 positions with half the batch
propagated through the trained stages below it:

| | before | after |
|---|---|---|
| eps on the **drifted** interface it actually meets | 1.3577 | **0.5188** |
| eps on the teacher's clean interface (the control) | 0.5670 | 0.5346 |

Two things worth separating.

**The exposure-bias gap is large and was invisible until measured.** Before retraining,
the stage scores 0.567 on the interface it was trained on and 1.358 on the interface it
actually meets: it is **2.4x worse in deployment than its own training metric says**.
Every per-stage number in Phase 1 and in the depth table above is a clean-interface
number, so all of them flatter the composed model.

**DAgger removes most of that gap, and costs nothing.** Drifted eps falls 62%, and
clean eps *also* improves slightly rather than trading off. After retraining the stage
is better on drifted inputs (0.519) than it ever was on clean ones (0.567).

All three stages, same treatment:

| stage | drifted before | drifted after | cut | clean before | clean after |
|---|---|---|---|---|---|
| 1 | 1.3577 | 0.5188 | 62% | 0.5670 | 0.5346 |
| 3 | 0.7761 | 0.2690 | 65% | 0.5154 | 0.4670 |
| 5 | 0.7635 | 0.1433 | **81%** | 0.3306 | 0.3028 |

**The first branch of the prediction, on every stage.** Drifted error falls 62 to 81%,
the cut grows with depth, and clean error improves as well at every stage: there is no
trade-off to manage. After retraining, each stage is better on the drifted interface
than it ever was on the clean one.

**One caveat, and it is the reason this is not yet the phase's headline.** Drifted eps
is measured against the teacher stage applied to the *drifted* input. If the student's
propagated interface is an easier distribution than the teacher's own, eps can fall
without the composed model improving. The honest test is whether a stack whose stages
were all retrained this way composes to a lower loss, and that is one cheap run: six
retrains at 1e7 positions each, in order, then recompose and measure. Queued.


## C2.4 first attempt, and why its baseline was invalid (24 Aug 2026)

Held-out next-token loss after healing (teacher 1.736 nats), first run:

| heal tokens | random | stagewise | oracle |
|---|---|---|---|
| 1e5 | 7.9352 | 7.5202 | 5.8450 |
| 1e6 | 7.6615 | 6.0165 | 5.2219 |
| 3e6 | **nan** | 5.3214 | 4.5842 |
| 1e7 | **nan** | 3.8309 | 3.5367 |
| (before healing) | 12.9942 | 9.5702 | 6.7196 |

**The random arm diverged**, at step ~380 in both cases, and then skipped 4476 of 4883
steps: 90% of its budget spent on nothing, reported as a NaN loss. The stagewise arm
at the identical learning rate skipped zero steps.

That is not a result about stagewise initialization, it is a result about the
schedule. **Corrected by the probe below: the cause was the warmup, not the peak
learning rate.** With 500 warmup steps instead of 50, lr 1e-4 is stable on the cold
start and is also the best of the three probed:

| lr (cold start, 1e6 tokens, warmup 500) | loss after | skipped |
|---|---|---|
| 1e-5 | 7.5521 | 0/489 |
| 3e-5 | 7.2122 | 0/489 |
| **1e-4** | **6.8339** | 0/489 |

So the two arms can share a peak learning rate after all; what a cold start needs is a
longer ramp to it. Either way the comparison as first run pitted a schedule that
suited one arm against one that did not. G2's kill criterion is precisely the
claim "stagewise beats random init", and it cannot be settled with a baseline that was
never given a working configuration. The equal-FLOPs cell then inherited the same
setting and was killed 12 minutes in, before it wasted 2.2 hours.

Two changes:

1. **The heal aborts after 50 consecutive skipped steps** with a message naming the
   cause. The non-finite guard from Phase 1 stops bad weights getting worse but cannot
   repair them, so without an abort a diverged run silently burns its whole budget.
2. **The cold-start arm gets its own learning rate and warmup**, chosen by a short
   probe (1e6 tokens at 1e-5, 3e-5, 1e-4) rather than assumed. Fair baselines are part
   of the criterion, not an optimization.

The numbers above stand for stagewise and oracle, and the ordering there is already
informative: stagewise starts 3.4 nats better than random and stays ahead of it at
every budget where random survived.

## C2.4 with both arms tuned (24 Aug 2026)

Warmup 500 for both (the probe showed the warm start wanted it too: 5.5625 against
6.0906 at 1e6). Peak lr 1e-4 for stagewise and oracle; the cold start diverges there at
1e7 tokens (step 1285) and was re-probed **at the length that failed**, which is the
lesson from the first probe being run at 1e6 where everything is stable.

| heal tokens | random | stagewise | oracle |
|---|---|---|---|
| 1e5 | 8.5325 | 6.9982 | 5.4803 |
| 1e6 | 6.8356 | 5.5692 | 4.7819 |
| 3e6 | 5.9450 | 5.0123 | 4.2673 |
| 1e7 | 5.2693 @5e-5 | **3.7221** | 3.4850 |
| (no heal) | 12.9942 | 9.5702 | 6.7196 |

Cold start at 1e7: 1e-4 diverges, 3e-5 gives 5.4823, **5e-5 gives 5.2693**.

**Read the 1e7 row with its handicap.** The arms are not on one schedule there: 5e-5
costs the cold start roughly 0.2 nats relative to 1e-4 (interpolating the 1e6 probe),
so the stagewise lead of 1.55 nats is really **about 1.3 to 1.55**. That interval cannot
be tightened without a stable 1e-4 cold-start run, and there is not one.

**Criterion 5, the stagewise-to-oracle gap, is small: 0.23 nats at 1e7** (3.7221 against
3.4850). The anchors-only stack is barely better than anchors-plus-noise once healed,
which matches Phase 1 at this anchor budget and says the recipe choice matters much
less after healing than the per-stage numbers implied. Most of what separated the
stacks before healing (9.57 against 6.72) is closed by the heal.

**None of this is the kill criterion.** Every row compares arms at equal *heal tokens*,
and the stagewise arm has already spent 6.3e17 FLOPs on its stages. The equal-FLOPs
cell (random init, 1.74e8 tokens, 16.6 passes over the same declared slice) is running.
