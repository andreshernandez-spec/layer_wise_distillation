# Interface metrics overstate what survives end-to-end training: a negative result in stagewise distillation

Andres Hernandez

## Abstract

Distilling a language model stage by stage, with each student stage trained
independently against its teacher stage, is attractive because the stages never have to
be trained together. Choosing among the recipes for doing this is usually done with a
metric read at the stage interfaces: the relative error between student and teacher
activations, or a comparable proxy. We report that on Pythia-1.4B this practice is
misleading, and by a large factor.

We measure four interventions in a stagewise pipeline. Each is ranked cleanly and
consistently by an interface-level metric, and every one of those rankings collapses by
roughly an order of magnitude once the composed model is given a modest amount of
end-to-end training. On-policy retraining cuts composed drift at the output by 41% and is
worth 3.34 nats to the unhealed model, and about 0.10 nats after 1e7 tokens of healing.
The choice between training stages on real activations and training them on noise plus a
small anchor set is worth 2.85 nats unhealed and 0.237 healed. Taken to its limit, the
same effect kills the method: at equal total FLOPs, a randomly initialised student of the
same architecture, healed for longer, reaches 3.5430 nats of held-out next-token loss
against the stagewise stack's 3.7560, a gap of 0.213 against a within-arm spread of
0.068. The stagewise construction spends 6.30e17 FLOPs building structure that is worth
less than nothing once the end-to-end budget it displaces is accounted for.

Along the way we report the measurements the pipeline was built to produce, which stand
independently of it: the exponent in eps(Q) = c Q^-beta + eps_inf is about 0.32 for the
input measures that work, the crossover below which synthetic noise is worth adding sits
near 250k real positions per interface, and composition error in this architecture
accumulates additively rather than compounding through downstream Lipschitz constants,
because five of six stages contract the error they inherit.

## 1. Introduction

The appeal of stagewise, or blockwise, distillation is structural. If a student stage can
be trained against its teacher stage in isolation, then the expensive part of
distillation becomes embarrassingly parallel, the memory needed at any moment is the
memory for one stage, and the stages can be trained in any order or on any machine.
NVIDIA's Puzzle [Bercovich et al., 2024] does exactly this at production scale on real
activations. If the activations themselves could be synthesized rather than harvested,
the data requirement would fall as well, which is what Neighbourhood Distillation
[Shao et al., 2020] demonstrated for CNN sub-networks with Gaussian inputs.

A pipeline of that shape has many design decisions: what distribution to feed each stage,
how many real anchors to mix in, how to normalize the interface, whether to retrain
stages on their own drifted inputs. Each of those decisions is cheap to evaluate at the
interface, by comparing the student stage's output against the teacher's on the same
input, and expensive to evaluate end to end, because that requires composing the stack
and training it. So the interface metric is what gets used.

This paper reports that on our substrate the interface metric is not a usable proxy for
the decision it is being used to make. This is not a claim that the two are uncorrelated.
They are correlated, and the interface metric ranks the recipes in the right order most of
the time. The problem is the magnitude: the interface metric says the gap between two
recipes is large, end-to-end training closes almost all of it, and the residual is small
enough to be inside the noise of the comparison. When the same reasoning is applied to
the pipeline as a whole against the obvious baseline, the sign flips.

We state the result as an existence claim rather than a universal one. We measured one
model family at one scale. What we can say is that here is a fully instrumented case
where selecting on interface-level proxies would have led to the wrong conclusion at
every level, including the top-level decision of whether to use the method at all, and
that the failure was not visible from the interface metrics themselves.

### Contributions

1. A four-way demonstration that interface-level rankings in a stagewise LM distillation
   pipeline collapse by roughly 10x under a modest end-to-end budget (Section 4).
2. An equal-FLOPs comparison against random initialisation, with the accounting stated,
   showing the pipeline loses (Section 5).
3. A measurement of how composition error actually behaves in this architecture, which
   contradicts the compounding model usually assumed for it (Section 6).
4. The interface-level measurements themselves, including beta and the anchor crossover,
   reported as what they are: a careful characterisation of a quantity that turns out not
   to decide the question (Section 7).

## 2. Setup

**Teacher and stages.** Pythia-1.4B, 24 blocks, split into six stages of four blocks. Each
student stage is two blocks at the teacher's width, so the student's non-embedding
parameter count is 6.05e8 against the teacher's 1.21e9, a factor of exactly 0.500. The
teacher's embedding, final layer norm and unembedding are transferred and frozen
throughout: they are not what stagewise training produces, and the question here is about
the stages.

**Interfaces and the contract.** Stage boundaries are the residual stream at blocks 0, 4,
8, 12, 16, 20 and 24. All interface errors are measured in whitened coordinates, using a
ZCA transform fitted on the teacher's own activations at that interface with a shrunk
covariance. We write eps for the relative MSE in those coordinates. We considered adding a
marginal Gaussianization on top of the whitening and dropped it: it was worse on 8 of 8
comparable cells.

**Input measures.** A stage can be fed real teacher activations (R, drawn from a fixed
anchor set and recycled), live real activations from a stream (L, each position seen
once), moment-matched Gaussian noise (G), isotropic noise (I), or a mixture of anchors and
noise (C, the recipe). Structure variants (iid across positions, AR(1), or the mix) are
noted as a suffix.

**End-to-end metric.** Held-out next-token loss in nats, on 32 sequences of 2048 tokens
from a decontaminated Pile-test slice never used in training. We refer to composing the
six trained stages into a full model and training it end to end against the teacher's
top-k log-probs as *healing*, and to the token budget spent doing so as the heal budget.

**Compute.** Two rented A100-SXM4-80GB and one RTX 3080 Laptop, $39.63 total. Every figure
in this paper is recomputed from the run records by `experiments/paper/claims.py`, which
fails if any published number drifts from its source file.

## 3. What the interface metrics say

At Q = 1e8 positions per stage, on stage 2 of the teacher, the recipes order themselves
cleanly. Reading the stitching delta, which is next-token loss in nats when a single
trained stage is substituted into the otherwise-real model:

| arm | distinct real positions | noise positions | eps | stitching delta |
|---|---|---|---|---|
| L (live real) | 1e7 | 0 | 0.353 | 0.452 |
| R (anchors, recycled) | 0.95e6 | 0 | 0.412 | 0.450 |
| C_mix (recipe) | 0.95e6 | 9.1e7 | 0.463 | 0.527 |
| G_iid (pure noise) | 0 | 1e8 | 11.07 | 7.92 |

Two things are already visible. Recycling 0.95M distinct real positions about a hundred
times matches the live-real oracle that sees ten times more distinct data (0.450 against
0.452, against a measured seed spread of 0.106 nats over 42 same-arm pairs). And pure
noise, with no anchors at all, does not work.

These numbers are what a practitioner would use to choose. They are also, as the next two
sections show, not the numbers that decide.

## 4. What survives healing

We composed the six stages and healed the composed model against the teacher's top-k
log-probs, sweeping the heal budget. Both the cold start and the warm start were tuned
separately, because using one learning rate for both compares a tuned schedule against an
untuned one rather than comparing the initialisations. Peak 1e-4 with warmup 500 for the
stagewise and oracle arms; the cold start diverges there at 1e7 tokens and runs at 5e-5.

| heal tokens | random init | stagewise (noise recipe) | oracle (real activations) |
|---|---|---|---|
| 0 | 12.9942 | 9.5702 | 6.7196 |
| 1e5 | 8.5325 | 6.9982 | 5.4803 |
| 1e6 | 6.8356 | 5.5692 | 4.7819 |
| 3e6 | 5.9450 | 5.0123 | 4.2673 |
| 1e7 | 5.2693 | 3.7221 | 3.4850 |

The gap between training stages on real activations and training them on the noise recipe
is 2.85 nats before healing and **0.237** after 1e7 tokens. The recipe choice that the
interface metrics separate clearly is worth about a fifth of a nat once the composed model
is trained at all.

The same holds for the strongest intervention we measured. Retraining each stage on its
own drifted inputs, in the manner of DAgger, cuts drifted eps by **53% to 74%** on stages
1 through 5, and by only 7.6% on stage 0, which has no inherited drift to correct because
the student uses the teacher's embedding. Composed, those per-stage gains survive and
compound favourably: drift at the final interface falls from 2.335 to **1.387**, a 41%
cut, which puts the noise-trained stack *below* the stack trained on real activations
(1.473).

Then healing:

| | no heal | 1e6 | 1e7 |
|---|---|---|---|
| stagewise, plain | 9.5702 | 5.5692 | 3.7221 / 3.7900 |
| stagewise, after on-policy retraining | 6.2323 | 5.1555 | 3.6375 / 3.6829 |
| difference | **3.338** | 0.414 | **0.096** |

Two heal trajectories were run for each arm at 1e7 (the two values given), giving spreads
of 0.068 and 0.045. The healed difference of 0.096 is of the same size as those spreads.
Both retrained runs fall below both plain runs, so the ordering is consistent, but with
two runs per arm we cannot place the effect more precisely than "about 0.1 nats". What is
not in doubt is the ratio: an intervention worth 3.34 nats to the unhealed model is worth
of order 0.1 after 1e7 tokens, and the 1e6 column shows the gain being closed rather than
converging to something.

## 5. The equal-FLOPs comparison

Every comparison in Section 4 holds the heal budget fixed, which flatters the stagewise
arm: it has already spent the harvest and all six stage trainings. The comparison that
decides whether the method is worth using has to hold *total* FLOPs fixed.

**Accounting.** Six stage cells at 1e8 positions each, at 2 P_teacher-stage for the
teacher's forward pass plus 6 P_student-stage for the student's forward and backward,
gives 6.050e17. The harvest of teacher activations and top-k log-probs over the declared
1e7-token slice is 2.541e16. The stagewise arm's own 1e7-token heal is 3.630e16. The total
is **6.6671e17**. Dividing by 6 P_student gives **1.8367e8** tokens, which is what the
random-init arm is entitled to spend on healing to match.

| arm | heal tokens | held-out loss |
|---|---|---|
| stagewise, two trajectories | 1e7 | 3.7221 / 3.7900, mean **3.7560** |
| stagewise after on-policy retraining | 1e7 | 3.6375 / 3.6829, mean **3.6602** |
| random init | **1.8367e8** | **3.5430** |

The random arm wins by **0.213 nats** against the stagewise mean, and by 0.179 against its
better trajectory. The within-arm spread is 0.068, so the margin is about three times the
noise. It also beats the on-policy-retrained stack by 0.117, and that stack cost a further
6.05e16 FLOPs to build, so at a properly equal budget it is further behind than that
figure suggests.

**Two handicaps, both favouring the method.** We recorded these before the run finished.
First, the random arm is FLOPs-matched but data-limited: it makes about 17.5 passes over
the same declared 1e7-token slice, while the stagewise arm's heal makes one. A model given
1.8367e8 distinct tokens would do better than this cell does. Second, the two arms are not
on one schedule: 5e-5 against 1e-4, which interpolating from the 1e6 probe costs the random
arm roughly 0.2 nats.

Both handicaps run against the arm that won. Had the comparison gone the other way, they
would have made the margin an upper bound and the conclusion arguable. Because it went
this way, they are reasons to think the true margin is larger.

## 6. Why: composition accumulates, it does not compound

The usual mental model for stagewise composition is borrowed from imitation learning: each
stage's error is amplified by the downstream stages' Lipschitz constants, so error
compounds multiplicatively with depth. On this architecture that model is wrong in the
direction that matters.

| interface | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| realized drift (noise recipe) | 0.637 | 1.171 | 1.541 | 1.549 | 1.748 | 2.335 |
| ratio-product prediction | 0.637 | 46.32 | 20.13 | 8.82 | 4.10 | 2.08 |

The prediction is wrong by a factor of 40 at the second interface. The reason is that
every stage after the first **contracts** the error it inherits: the measured teacher
Lipschitz ratios for stages 1 to 5 lie between 0.45 and 0.84, and agree across two
platforms within 3.1%. Drift still grows with depth, from 0.50 to 1.47 on the
real-activation stack, but as a sum: each stage adds a roughly constant fresh error of
about 0.54, essentially independent of depth.

The first stage is the exception and amplifies by roughly 50 to 70x. We give a range
deliberately. Two platforms measuring the same stack give 48.5 and 72.7, because the
estimator perturbs two sequences with a single noise draw, which is adequate for a
contractive map and not for a strongly expansive one. The order of magnitude is the
result and the digits are not. This amplification does not act on the composed model,
because the student uses the teacher's embedding and interface 0 therefore carries no
drift by construction. What it explains is why stage 0 is by far the hardest to fit: on the real-activation
stack its Jacobian cosine against the teacher's is 0.038, against 0.24 to 0.35 for every
other stage, and it has the highest eps (0.524) and the second-worst stitching delta and why injecting noise at the first
interface is a bad idea.

This changes what the mitigations are for. If drift were inherited and amplified,
contracting stages would shrink it and the fix would be better propagation. Since the
accumulation is additive and dominated by what each stage adds fresh, the fix has to be
better stages, and the fresh term is where exposure bias lives. That is why on-policy
retraining works as well as it does at the interface, and why it still cannot save the
method: it reduces a term that end-to-end training also reduces, and end-to-end training
is cheaper per nat.

## 7. The interface measurements, reported for what they are

The project was built to measure the exponent in eps(Q) = c Q^-beta + eps_inf, the rate at
which a stage trained on synthetic input approaches one trained on real activations as the
synthetic budget Q grows. We report it because it is a careful measurement, and because
the reason it does not settle anything is the point of the paper.

Fitting over 113 single-stage cells on stage 2, with bootstrap 90% intervals:

| arm | cells | beta | eps_inf |
|---|---|---|---|
| L (live real, distinct) | 16 | 0.326 [0.306, 0.480] | 0 |
| R (anchors, recycled) | 22 | 0.336 [0.287, 0.586] | 0.128 |
| C_mix (recipe) | 22 | 0.313 [0.270, 0.560] | 0.106 |
| G_iid (pure noise) | 16 | 0.334 [0.247, 0.774] | 0.410 |
| I_iid (isotropic noise) | 10 | 0.206 [0.184, 0.774] | 0 |

The three arms that work agree inside each other's intervals at about 0.32. Isotropic
noise, whose inputs carry no covariance structure at all, is the exception at 0.206. The
intervals are wide, with upper bounds between 0.48 and 0.77, so this pins the exponent's
scale and not its third digit. What the input measure changes is not the exponent but
eps_inf, the floor: only pure noise has a floor inconsistent with zero.

Two further results from the same sweep are worth stating because they are early warnings
of the paper's main finding, visible before any stage was composed.

**The crossover.** Adding synthetic noise to a fixed anchor budget is worth 0.773 nats of
stitching delta at 94k anchor positions per interface, 0.142 at 188k, and nothing at 377k
or above. Below about 250k real positions per interface, noise buys something; above it,
it does not.

**eps and end-to-end loss diverge with training length.** The R arm trained at 464 anchors
improves its eps from 0.412 to 0.373 going from 1e8 to 3e8 positions, while its stitching
delta *worsens* from 0.450 to 0.710. Both seeds. A stage must not be early-stopped on eps.
This is the same phenomenon as the paper's headline, observed at the level of a single
stage: the interface metric and the end-to-end metric can move in opposite directions.

We also tested whether heavy-tailed self-regularization alpha, estimated per stage with a
Hill estimator and a KS-chosen cutoff, could serve as a training-free gate on stage
quality. Over 112 students it correlates +0.80 with the stitching delta across the whole
pool, which is an artefact of training length rather than a signal: inside a fixed budget
the sign flips (-0.87 at 1e7 positions, +0.84 at 1e8, no signal at 1e6). It is not usable
as a per-stage gate.

## 8. Limitations

**One substrate.** Everything here is Pythia-1.4B, and the single-stage sweep is one stage
(blocks 8 to 11) of it. We claim an existence result, not a universal one. Whether the 10x
overstatement is characteristic of this architecture, this depth, this width ratio, or
transformers generally, we did not measure.

**The equal-FLOPs cell has one seed.** The stagewise arms were run on two heal
trajectories; the random arm at 1.8367e8 tokens was not, because it costs about four hours
of A100 time. The margin is about three times the stagewise spread, so a flip would take a
three-sigma excursion, but the cell is unreplicated and we say so.

**The random arm recycles data.** As stated in Section 5, it makes 17.5 passes over one
1e7-token slice. This biases against it, which is the safe direction here, but it means
the reported margin is not the margin a data-unconstrained baseline would achieve.

**Two named baselines were not run.** A Puzzle-style blockwise distillation on real
activations at matched real-token budget, and self-generated-text distillation at matched
compute, are the two comparisons a reader will want. They were scheduled for the phase this
result cancelled. What we compare against is random initialisation at equal FLOPs, which is
the weaker baseline to beat and which the method nonetheless loses to.

**The Lipschitz estimator is undersampled**, as Section 6 says. It perturbs two sequences.
The contractive stages replicate across platforms within 3.1% and the expansive one does
not.

**We did not test the no-heal regime properly.** If a composed model must ship without any
end-to-end training, the unhealed numbers are the operative ones and they favour the method
strongly (9.57 against 12.99, and 6.23 with on-policy retraining). We did not design for
that setting and do not claim anything about it beyond noting it is where the method's
value would have to live.

## 9. Related work

**Blockwise and neighbourhood distillation.** Shao et al. [2020] distil CNN sub-networks
independently and report that Gaussian noise inputs can substitute for real ones, beating
end-to-end baselines under noise. Puzzle [Bercovich et al., 2024] is the production LLM
version on real activations, with gradients isolated across blocks and roughly 1e9 real
tokens. Our stagewise arm is Puzzle-like construction with the activations partly
synthesized; our oracle arm is Puzzle-like construction with them harvested.

**Noise-driven data-free distillation.** Raikwar and Mishra [NeurIPS 2022] train
end-to-end under Gaussian input noise and identify the shift in hidden-layer activation
distribution as the failure mode. Injecting moment-matched noise at the interface, as we
do, is the direct response to that diagnosis, and it does fix the distributional problem;
the point of this paper is that fixing it is not sufficient.

**Data-free KD for LLMs.** The 2024 to 2026 line substitutes model-generated text for data
and trains end to end [LLM-QAT and successors]. None decomposes into stages or synthesizes
interface activations. At matched compute this is the baseline a reader will propose
instead, and we did not run it.

**Metric proxies in distillation.** Feature-matching objectives are standard, and the
practice of selecting among them on feature-space error is widespread. We are not aware of
a systematic measurement of how much of a feature-space ranking survives end-to-end
finetuning, which is the gap this paper reports into.

## 10. Conclusion

We set out to measure how fast noise-trained stages approach real-activation-trained ones,
and we measured it: the exponent is about 0.32 for input measures that carry the right
second-order structure, the anchor crossover is near 250k real positions per interface, and
composition error in this architecture accumulates rather than compounds.

The pipeline those measurements were meant to support does not work. At equal total FLOPs
it loses to random initialisation trained for longer, by three times the noise, with two
handicaps applied in its own favour. And it loses in a way that none of the interface-level
measurements predicted: every one of them ranked the design choices clearly, and every one
of those rankings shrank by about an order of magnitude the moment the composed model saw
end-to-end gradients.

The transferable claim is the one about proxies. On this substrate, no quantity measured at
a stage interface should be quoted without the end-to-end number beside it, and a design
decision that looks decisive at the interface should be assumed to be worth about a tenth
of what it appears to be worth. We found this at four independent levels, including the
level at which it invalidated the method.

## Reproducibility

Code, run records, the gate scripts and the two rented-node campaign logs including every
incident are in the repository. `experiments/paper/claims.py` recomputes all 37 numbers
quoted here from the raw run records and fails if any of them drifts; it is run as part of
the test suite. Each run record carries the commit SHA, the device and the seed.

Two numbers in earlier drafts of the project documentation did drift and were corrected by
that script: the beta for the recycled-anchor arm, published from 13 points per arm and now
0.336 over all 22 cells, and the stage-0 amplification, quoted at 48.5 from one platform and
now given as a range because a second platform gives 72.7.
