# Interface metrics overstate what survives end-to-end training: a negative result in noise-fed stagewise distillation

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
small anchor set is worth 2.85 nats unhealed and 0.235 healed. Taken to its limit, the
same effect kills the method: at equal total FLOPs, a randomly initialised student of the
same architecture, healed for longer, reaches 3.5430 nats of held-out next-token loss
against the noise-trained stack's 3.7283, a gap of 0.176 against a measured run-to-run
spread of about 0.02.

The comparison also separates two things that are easy to conflate. A stack whose stages
are trained on *harvested* real activations, at the identical FLOPs budget, reaches 3.4931,
ahead of random initialisation rather than behind it, though by less than the schedule
advantage it holds. So stagewise construction itself is not what fails: it is the step of
synthesizing the interface activations, which costs 0.18 nats against the same baseline the
harvested version beats. The 6.05e17 FLOPs of stage training buy structure worth less than
the end-to-end training they displace, but only once the activations are synthetic.

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
NVIDIA's Puzzle [@bercovich2024puzzle] does exactly this at production scale on real
activations. If the activations themselves could be synthesized rather than harvested,
the data requirement would fall as well, which is what Neighbourhood Distillation
[@shao2020neighbourhood] demonstrated for CNN sub-networks with Gaussian inputs.

A pipeline of that shape has many design decisions: what distribution to feed each stage,
how many real anchors to mix in, how to normalize the interface, whether to retrain
stages on their own drifted inputs. Each of those decisions is cheap to evaluate at the
interface, by comparing the student stage's output against the teacher's on the same
input, and expensive to evaluate end to end, because that requires composing the stack
and training it. So the interface metric is what gets used.

This paper reports that on our substrate the interface metric is not a usable proxy for
the decision it is being used to make, and that following it leads to spending a large
compute budget on a step that does not pay for itself. This is not a claim that the two are uncorrelated.
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
   showing that the noise-fed pipeline loses by three times the measured noise while the
   same construction on harvested activations ties (Section 5).
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
| G_iid (pure noise) | 0 | 1e8 | 11.074 | 7.921 |

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
| 1e+05 | 8.5325 | 6.9982 | 5.4803 |
| 1e+06 | 6.8356 | 5.5692 | 4.7819 |
| 3e+06 | 5.9450 | 5.0123 | 4.2673 |
| 1e+07 | 5.2693 | 3.7221 | 3.4850 |

The gap between training stages on real activations and training them on the noise recipe
is 2.85 nats before healing and **0.235** after 1e7 tokens. The recipe choice that the
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
| difference | **3.338** | 0.414 | **0.091** |

The plain arm was run twice on the intended schedule and the retrained arm once, so the
0.091 difference at 1e7 carries the plain arm's 0.012 spread and no error bar on the other
side. It is of the same order as the run-to-run noise measured in Section 5, and we cannot
place it more precisely than "about 0.1 nats". What is
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

The oracle stack, whose stages are trained on harvested real activations, was built from
the same six 1e8-position cells and the same harvest, so it sits at the identical budget
and the comparison applies to it unchanged. We report it here because it separates two
questions that the rest of the paper runs together: whether *stagewise construction* pays,
and whether *synthesizing the interface activations* pays.

| arm | heal tokens | held-out loss | gap to random |
|---|---|---|---|
| random init | **1.8367e8** | **3.5520** (n=2, spread 0.0181) | |
| oracle, real activations | 1e7 | **3.4931** (n=3, spread 0.0135) | **+0.059** |
| noise recipe | 1e7 | **3.7283** (n=2, spread 0.0123) | **-0.176** |
| noise recipe + on-policy retraining | 1e7 | **3.6375** (n=1) | **-0.085** |

The noise recipe loses by **0.176 nats**. On-policy retraining recovers about half of that
and still loses by 0.085, and that stack cost a further 6.05e16 FLOPs to build, so at a
properly equal budget it is further behind than the figure suggests.

**How large is the noise?** Every cell above was repeated on a second heal seed, and the
oracle cell was additionally repeated on a second machine with the top-k store rebuilt from
the declared slice. That gives two separate scales. Run-to-run spread within an arm is
**0.012 to 0.018**. Holding the seed fixed and changing only the machine and the rebuilt
store moves the oracle by **0.0135**, while holding the machine fixed and changing only the
seed moves it by **0.0028**. So almost all of the variability we can see is environmental
rather than stochastic, and the total is about 0.02 nats.

Against that, the 0.176 margin is roughly ten times the noise.

The on-policy arm is the exception: only one of its runs used the intended schedule, so its
0.085 gap has no error bar and should be read as a single measurement.

**The oracle leads, and the lead is smaller than a handicap it holds.** It comes out 0.059
nats ahead of random init, which is about three times the run-to-run noise and reproduces
in both environments (0.058 on the first machine, 0.064 on the second), so we do read it as
a real lead rather than a coin flip. We therefore read the real-activation stack as indistinguishable from
random initialisation at equal FLOPs, not as beating it. This is the sharpest statement the
data supports, and it locates the failure precisely: **stagewise construction on harvested
activations roughly breaks even, and replacing those activations with synthesized ones is
what costs 0.21 nats.**

That distinction matters for what this paper is claiming about prior work. Puzzle-style
blockwise distillation on real activations is not refuted here. What is refuted is the step
this project added on top of it.

**Is FLOPs the right currency?** The case for stagewise construction is parallelism, and
a reader is entitled to ask why we judge it on a serial quantity. Three things, and the
first two go against the method.

Cost is FLOPs. Six stages trained concurrently on six devices bill six device-hours, the
same as one device for six hours, so for anyone paying for compute the comparison above
is the one that matters.

If the constraint is wall-clock rather than cost, stagewise parallelism is real but it is
not exclusive: end-to-end training parallelises too, by data parallelism, and the random
arm would get the same speedup from the same six devices while keeping its FLOPs
advantage. What stagewise has that data parallelism does not is that its concurrency needs
no gradient communication at all, where data parallelism needs an all-reduce per step.
That is a genuine advantage at large device counts or on a poor interconnect. It is also
second order against a 0.213-nat deficit, and we did not measure it.

The memory argument is the one that survives, and only partly. Training a stage in
isolation needs one stage resident rather than the whole model, which is a real benefit if
the teacher does not fit. But the heal does not decompose: it trains the composed student
end to end, and the heal is what makes the stagewise stack competitive at all. So the peak
memory of the full recipe is set by the phase this paper shows you cannot skip.

**Two handicaps, both favouring the method.** We recorded these before the run finished.
First, the random arm is FLOPs-matched but data-limited: it makes about 17.5 passes over
the same declared 1e7-token slice, while the stagewise arm's heal makes one. Second, the
two arms are not on one schedule: 5e-5 against 1e-4, which interpolating from the 1e6 probe
costs the random arm roughly 0.2 nats.

The schedule handicap plainly runs against the random arm, which is the arm that won, so it
is a reason to think the true margin is a little larger.

The data limitation we would rather not lean on, because on inspection it is not clearly a
handicap at all. Both arms are confined to the same slice: the stagewise arm's heal makes
one pass over it, but its six stages were each trained on 1e8 positions drawn from an
anchor set of 0.95M distinct real positions, which is about a hundred passes. In total data
exposure the stagewise arm recycles harder than the random one. And Phase 1 measured what
recycling costs at this scale and found it close to nothing: 0.95M positions recycled about
a hundred times matched a live-real stream carrying ten times more distinct data, 0.450
against 0.452 of stitching delta, well inside the 0.106 seed spread. We therefore read the
data limitation as a small effect of unknown sign, and treat the 0.213 margin as roughly
unbiased on that axis rather than as a lower bound.

They cut the other way for the oracle. That arm ran at 1e-4 while the random arm ran at
5e-5, so roughly 0.2 nats of the oracle's apparent position is schedule rather than method,
which is more than the 0.058 that separates them. An oracle that ties while holding a
handicap worth more than the gap is not a result we would push on, and we do not.

## 6. Why: composition accumulates, it does not compound

The usual mental model for stagewise composition is borrowed from imitation learning: each
stage's error is amplified by the downstream stages' Lipschitz constants, so error
compounds multiplicatively with depth. On this architecture that model is wrong in the
direction that matters.

| interface | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| realized drift (noise recipe) | 0.637 | 1.171 | 1.541 | 1.549 | 1.748 | **2.335** |
| pure propagation, no fresh error | 0.637 | 0.277 | 0.121 | 0.056 | 0.041 | **0.029** |

Take the drift measured at the first interface and propagate it through the stages that
follow, multiplying by each one's measured Lipschitz ratio and adding nothing. Because
every stage after the first **contracts** what it inherits, with ratios between 0.45 and
0.84 that agree across two platforms within 3.1%, that model predicts the drift should
almost vanish: 0.029 by the last interface. It is **2.335**, larger by a factor of 82.

So essentially none of the drift at the output is inherited. All of it is fresh error
introduced by the stages themselves, roughly 0.54 per stage and essentially independent
of depth, minus the part the next stage contracts away. Composition error here is a sum
dominated by its newest term, not a product.

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

**Replication is two runs per cell, three for the oracle.** That is enough to put the
run-to-run spread at 0.012 to 0.018 and to separate its environmental part (0.0135) from
its stochastic part (0.0028), which is what the margins in Section 5 are quoted against.
It is not enough to place the on-policy retraining effect, which has one matched run and
therefore no error bar of its own. A first attempt at these replicates was wasted because
the configuration synced to the machine had drifted to a shorter warmup than the runs it
was meant to replicate; the schedule is now asserted at launch.

**Both arms recycle the same slice.** The random arm makes 17.5 passes over the declared
1e7-token slice; the stagewise arm's stages make roughly a hundred over a 0.95M-position
subset of it. Neither is data-unconstrained, and the reported margin is not the one either
would reach with unlimited distinct data.

**One named baseline was not run.** Self-generated-text distillation at matched compute is
the comparison a reader will still want, and it was scheduled for the phase this result
cancelled. Puzzle-style blockwise distillation on real activations is, in effect, our
oracle arm, though it is our reimplementation at our scale and not their system, and it was
not tuned as a baseline in its own right. Our headline comparison is against random
initialisation at equal FLOPs, which is the weaker baseline to beat and which the noise
recipe nonetheless loses to.

**The Lipschitz estimator is undersampled**, as Section 6 says. It perturbs two sequences.
The contractive stages replicate across platforms within 3.1% and the expansive one does
not.

**We measured cost in FLOPs, not wall-clock or communication volume.** Section 5 argues
that this is the right currency and that stagewise parallelism is not exclusive to
stagewise, but we did not measure a wall-clock comparison against data-parallel end-to-end
training, and on a sufficiently poor interconnect the communication-free property of
stagewise training could matter more than the FLOPs deficit.

**We did not test the no-heal regime properly.** If a composed model must ship without any
end-to-end training, the unhealed numbers are the operative ones and they favour the method
strongly (9.57 against 12.99, and 6.23 with on-policy retraining). We did not design for
that setting and do not claim anything about it beyond noting it is where the method's
value would have to live.

## 9. Related work

**Blockwise and neighbourhood distillation.** Shao et al. [@shao2020neighbourhood] distil CNN sub-networks
independently and report that Gaussian noise inputs can substitute for real ones, beating
end-to-end baselines under noise. Puzzle [@bercovich2024puzzle] is the production LLM
version on real activations, with gradients isolated across blocks and roughly 1e9 real
tokens. Our stagewise arm is Puzzle-like construction with the activations partly
synthesized; our oracle arm is Puzzle-like construction with them harvested.

**Noise-driven data-free distillation.** Raikwar and Mishra [@raikwar2022noise] train
end-to-end under Gaussian input noise and identify the shift in hidden-layer activation
distribution as the failure mode. Injecting moment-matched noise at the interface, as we
do, is the direct response to that diagnosis, and it does fix the distributional problem;
the point of this paper is that fixing it is not sufficient.

**Data-free KD for LLMs.** The 2024 to 2026 line substitutes model-generated text for data
and trains end to end [@liu2023llmqat]. None decomposes into stages or synthesizes
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
it loses to random initialisation trained for longer, by three times the measured noise,
with two handicaps applied in its own favour. The same construction fed harvested
activations instead ties, so what fails is the synthesis, not the decomposition. And it
fails in a way that none of the interface-level measurements predicted: every one of them ranked the design choices clearly, and every one
of those rankings shrank by about an order of magnitude the moment the composed model saw
end-to-end gradients.

The transferable claim is the one about proxies. On this substrate, no quantity measured at
a stage interface should be quoted without the end-to-end number beside it, and a design
decision that looks decisive at the interface should be assumed to be worth about a tenth
of what it appears to be worth. We found this at four independent levels, including the
level at which it invalidated the method.

## A note on the references

`paper/refs.bib` marks each entry VERIFIED or UNVERIFIED. The verified ones were fetched
from arXiv during a literature pass recorded in `docs/00-literature.md`; the unverified
ones were written from memory while drafting and their ids, author lists and years have
not been checked. They are plausible and they are not evidence. Anything dated March 2026
or later rests only on a fetched abstract page and should be re-read before it is cited.
This is the same standard the paper applies to its own numbers, and it should be cleared
before submission rather than at it.

## Reproducibility

Code, run records, the gate scripts and the two rented-node campaign logs including every
incident are in the repository. `experiments/paper/claims.py` recomputes all 37 numbers
quoted here from the raw run records and fails if any of them drifts; it is run as part of
the test suite. Each run record carries the commit SHA, the device and the seed.

Two numbers in earlier drafts of the project documentation did drift and were corrected by
that script: the beta for the recycled-anchor arm, published from 13 points per arm and now
0.336 over all 22 cells, and the stage-0 amplification, quoted at 48.5 from one platform and
now given as a range because a second platform gives 72.7.
