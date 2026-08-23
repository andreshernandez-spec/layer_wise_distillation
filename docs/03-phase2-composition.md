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
