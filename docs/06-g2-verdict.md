# G2 verdict

**Status: final, 24 Aug 2026. The kill criterion fires.** All cells landed, the pod is
deleted. At equal end-to-end FLOPs a randomly initialised student that simply trains
longer reaches **3.5430** against the stagewise stack's **3.7221** (mean of two heal
trajectories, 3.7560). Per the pre-registered gate in `docs/03`, **Phase 3 does not run**
and the write-up becomes Phase 1's measurement plus this negative.

Substrate: Pythia-1.4B, six stages of four teacher blocks each, 2-block same-width
students (6.05e8 non-embedding parameters against the teacher's 1.21e9, a factor 0.500).
Teacher embedding, final norm and head are transferred and frozen throughout. One rented
A100 SXM 80GB. Every figure traces to `out/phase2-1.4b-a100/`, SHAs in each run JSON.

---

## The five criteria

**1. C2.1 telescoping test passes bitwise in fp32.** Met (`docs/03` §C2.1). The composer
reproduces the teacher bitwise when every stage is the identity in whitened coordinates,
which is what makes any later drift number attributable to the students rather than to
the harness.

**2. Amplification profile exists, and the ratio-product prediction is compared against
realized drift.** Met, and the comparison is the phase's main result. See below.

**3. Each mitigation has a number on the same axis, same budget.** Met for DAgger
(six stages, on-policy fraction 0.5, 1e7 positions each, measured per-stage, composed,
and after healing). Theseus swap was not run: C2.2 removed its motivation (see below)
and the budget went to the kill criterion instead. That is a scope cut and is recorded
as one.

**4. Kill: at every heal budget, stagewise ≤ random at equal total FLOPs.** **FIRED.**
The cell is random init healed on **1.8367e8 tokens**, which is the stagewise arm's whole
end-to-end budget: 2.541e16 harvest + 6.050e17 stage training + 3.630e16 for its own 1e7
heal, divided by 6 x 6.05e8 parameters. Two things about it are written down here before
the number exists, because both are easier to state honestly now than afterwards.

*The random arm is FLOPs-matched but data-limited.* It makes about 17.5 passes over the
same declared 1e7-token slice, while the stagewise arm's heal makes one. A model given
1.8367e8 *distinct* tokens would do better than this cell does. So the handicap runs
against the random arm, which means: **if the kill fires, the conclusion is safe, because
it won despite the handicap. If it does not fire, the margin is an upper bound and has to
be quoted as one.** Phase 1's finding that 0.95M recycled real positions matched 10x more
distinct data (0.450 against 0.452) suggests recycling is cheap at that scale, but 17.5
epochs is well past what was tested there, so it is a caveat and not a dismissal.

*The two arms are not on one schedule.* The random arm runs at 5e-5 because 1e-4 diverges
from a cold start at this length; the stagewise arm runs at 1e-4. Interpolating the 1e6
probe, that costs the random arm roughly 0.2 nats, again in the direction that flatters
stagewise.

**The result.** The cell delivered 183,670,784 of 183,670,000 requested tokens over 89,683
steps with zero skipped, and reached **3.5430** on held-out next-token loss.

| arm | heal tokens | held-out loss |
|---|---|---|
| stagewise, seed 0 / seed 1 | 1e7 | 3.7221 / 3.7900 (mean **3.7560**) |
| stagewise after DAgger | 1e7 | 3.6375 / 3.6829 (mean **3.6602**) |
| **random init** | **1.8367e8** | **3.5430** |

The gap is **-0.179 nats** against the stagewise seed-0 cell and **-0.213** against its
two-trajectory mean, on a within-arm spread of 0.068. The kill fires by roughly three
times the noise.

It fires against the best stagewise variant too. The DAgger stack, at 3.6602, is still
0.117 nats behind the random arm, and it *cost 10% more FLOPs* to build (an extra 6.05e16
for the on-policy retraining), so at a properly equal budget it is further behind than
that. There is no version of the stagewise pipeline measured here that survives the
comparison.

And it fires **despite both handicaps written down above**, each of which favoured
stagewise. That is the disposition that makes this verdict safe: had the kill not fired,
the 17.5-epoch recycling and the 0.2-nat schedule penalty would have made the margin an
upper bound and the conclusion arguable. Because it fired, they are only reasons the true
margin is larger.

**5. The stagewise-to-oracle gap at 1e7 heal tokens.** Met: **+0.2371 nats** (stagewise
3.7221, oracle 3.4850).

---

## What was actually learned

### Composition error accumulates, it does not compound

This is the result that reorganized the phase. The source document (§1.6) frames stage
composition as imitation-learning compounding: per-stage error amplified through
downstream Lipschitz constants. Measured, that is the wrong direction for five of six
stages.

| interface | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| realized drift (C) | 0.637 | 1.171 | 1.541 | 1.549 | 1.748 | 2.335 |
| pure propagation, no fresh error | 0.637 | 0.277 | 0.121 | 0.057 | 0.041 | **0.029** |

Propagate the interface-1 drift through the stages that follow, multiplying by each
measured Lipschitz ratio and adding nothing, and it predicts 0.029 at the output. The
realized value is 2.335, larger by 82x, because every stage after the first **contracts**
what it inherits (teacher Lipschitz 0.43 to 0.84) and then adds fresh error of its own.
(Corrected 25 Aug 2026: the first version applied stage 0's 72x amplification a second
time, to a drift that is already its output, and so reported a 40x overshoot instead.)
Drift grows anyway, but as a sum: each stage adds a roughly constant fresh error of about
0.54 regardless of depth.

The discrepancy is reported here whichever way it goes, as criterion 2 requires, and it
goes against the source document. The practical consequence is that the mitigations had
to be re-justified as ways to reduce the *fresh per-stage term* rather than to stop
amplification, which turned C2.3 from a demonstration into a test with a prediction.

The teacher's first stage does amplify, by roughly 50 to 70x (two platforms give 48.5
and 72.7 from an estimator that uses two sequences, so the order of magnitude is the
claim and the digits are not). It does not
act on the composed model, because the student uses the teacher's embedding and interface
0 has zero drift by construction. It explains instead why stage 0 is the hardest to fit
(Jacobian cosine 0.038), and it is the number behind the source document's instinct not
to inject noise at the first interface.

### DAgger works on the metric it was designed for, and not on the one that decides

Exposure bias is most of the fresh per-stage term, as C2.2 predicted it might be.
Retraining each stage on its own drifted inputs cuts drifted ε by 53 to 74% on stages
1 through 5 (stage 0 only 8%, and stage 0 is the one with no inherited drift to correct).

Those gains survive composition, and by more than they had to:

| | plain | after DAgger | real-activation stack |
|---|---|---|---|
| drift at the output | 2.335 | **1.387** (-41%) | 1.473 |

The noise-trained stack ends up *below* the stack trained on real activations. On drift,
DAgger is the largest single effect in Phase 2.

Then the heal:

| | no heal | 1e6 | 1e7 |
|---|---|---|---|
| stagewise, plain | 9.5702 | 5.5692 | 3.7221 |
| stagewise, after DAgger | 6.2323 | 5.1555 | 3.6375 |
| **gain** | **3.338** | 0.414 | **0.085** |

**DAgger is worth 3.34 nats before healing and 0.08 after**, and the 1e6 column shows the
gain being closed rather than converging. The honest claim is conditional: DAgger matters
if the composed model ships without end-to-end training, and is nearly worthless if it
gets even 1e7 tokens.

Say the 0.085 carefully. Both arms were repeated on a second heal trajectory:

| | traj 0 | traj 1 | mean | spread |
|---|---|---|---|---|
| plain | 3.7221 | 3.7900 | 3.7560 | 0.0679 |
| after DAgger | 3.6375 | 3.6829 | 3.6602 | 0.0454 |

so the effect at 1e7 is **0.096 nats**, and both DAgger runs fall below both plain runs.
With two runs per arm that ordering is suggestive, not established: the separation
(0.039 nats between the nearest pair) is smaller than either arm's own spread. The
defensible statement is **"about 0.1 nats, and small enough that two runs per arm cannot
place it confidently"**. What does not depend on the error bar is the comparison that
matters: DAgger is worth 3.34 nats before healing and of order 0.1 after, a factor of
thirty, and the direction of that collapse is not in doubt.

### The pattern across three independent tests

G1 found that anchors, not noise, carried the result. Criterion 5 finds the anchors-only
oracle is only 0.24 nats better than anchors-plus-noise once healed, against 2.85 nats
before healing. Criterion 3 finds DAgger's 41% drift cut is worth 0.08 nats once healed.
Three different interventions, three interface-level metrics that rank the recipes
clearly, and one end-to-end metric that compresses all three rankings by roughly an order
of magnitude as soon as a modest end-to-end budget is allowed.

**No claim in this project should be quoted from ε, drift, or a stitching delta without
the healed number beside it.** On this substrate the interface metrics have consistently
overstated by 10x or more. That is the most transferable thing Phase 2 produced, and it
is a methodological result rather than the systems result the phase set out to find.

---

## What broke, and what it cost

Six incidents, all of them the same species: a step that did less than it was asked and
reported success.

1. **`Edges.embed/head` were decorated `@torch.no_grad()`**, which made the heal a silent
   no-op. Caught because the loss did not move. Test now asserts gradients reach the
   stages and not the frozen edges.
2. **Twelve heal runs trained on nothing.** The pod had been harvested with
   `skip_topk: true` and an empty `TopKStore` yielded no batches without complaining.
   It now raises `FileNotFoundError`.
3. **The random-init arm diverged** and skipped 4476 of 4883 steps while reporting a
   final loss. It now aborts after 50 consecutive non-finite steps. The probe that
   diagnosed it was first run at 1e6 tokens, where nothing diverges; re-probing **at the
   length that failed** corrected the diagnosis from peak lr to warmup.
4. **The equal-FLOPs cell trained on 6% of its budget in 21 minutes and reported
   success.** `TopKStore.batches()` made one pass over the files, capping delivery at one
   epoch (10,483,712 tokens against 174,000,000 requested). It produced a plausible
   number, better than the 1e7 run, because the cosine schedule never annealed. The store
   now loops and `heal()` refuses to return on a materially short run.
5. **The DAgger closing test destroyed three of the plain stack's result files.**
   `dagger_stack.py` promotes stages over the original paths, so the same command later
   loads a different model and writes the same name. I anticipated this and wrote renames
   into the launch script, then wrapped them in `2>/dev/null`; one pattern was wrong and
   failed invisibly. Rebuilt from the logs. `heal.py` and `drift.py` now refuse to
   overwrite a result and record a hash of every checkpoint they loaded.
6. **The equal-FLOPs budget was typed from the wrong line of the FLOPs report.** 1.74e8 is
   the harvest and stages; the stagewise arm also spends 1e7 tokens healing, so the equal
   point is 1.8367e8. The 5.3% shortfall was worth about 0.03 nats and ran in the
   direction that flatters the stagewise arm. `gate.py` had the arithmetic right and would
   have flagged it; the launch script now reads the number from `flops.json` instead of
   carrying a literal.

The common lesson is narrow and worth stating: **on this project, every silent failure ran
in the direction of a more favourable result.** An empty store, a truncated store, a
diverged run, and a short budget all reported success and all made the method look better
than it was. Guards therefore belong at the point where the number is produced, not in the
shell around it.

---

## Recommendation

**Stop the pipeline at Phase 2. Do not run Phase 3.** That is what the pre-registered gate
says, and nothing in the data argues for softening it. The stagewise construction spends
6.30e17 FLOPs building a stack whose composed model, after the same heal, is 0.21 nats
*worse* than the same architecture trained from random init on the same total budget. The
structure it builds is real and measurable at every interface, and it is not worth what it
costs.

Three things follow, in order of how much they are worth.

**1. The measurement paper is the deliverable, and it is a good one.** Phase 1 produced
β ≈ 0.32 across input measures, the anchor crossover at ~250k real positions per interface,
noise as a regularizer against over-fitting the interface objective, and the negative
result on α as a per-stage gate over 112 students. Phase 2 adds composition-accumulates
(pure propagation predicts 0.029 against a realized 2.335), the roughly 50 to 70x first-stage
amplification and what it explains, and exposure bias as most of the fresh per-stage term. Those stand on
their own and none of them depended on the pipeline working.

**2. The headline finding is methodological, and it generalizes past this project.** Four
interventions were each ranked clearly by an interface-level metric, and every one of those
rankings collapsed by roughly an order of magnitude once a modest end-to-end budget was
allowed:

| intervention | interface metric | after 1e7-token heal |
|---|---|---|
| anchors vs noise (G1) | 2.0 nats stitching | recipe choice worth 0.24 |
| anchors-only vs anchors+noise | 2.85 nats unhealed | 0.24 |
| DAgger | 41% composed drift | ~0.1 |
| the whole stagewise stack | eps 0.41 to 0.46 per stage | **-0.21, i.e. negative** |

The last row is the same phenomenon taken to its limit. A paper that reports this honestly
is more useful to the field than one that reports a pipeline that works, because the
practice of selecting distillation recipes on interface-level proxies is widespread and
this is a clean, instrumented case of that practice pointing the wrong way.

**3. What would have to change for the idea to be worth revisiting.** Not more tuning of
this pipeline. The kill margin is not close, it survives the best mitigation measured, and
it was obtained with both handicaps favouring the method. The condition under which
stagewise construction could still pay is the one this design cannot test: a regime where
the end-to-end heal is *not* available, because the composed model has to ship without
end-to-end training, or because the teacher's activations are available but its data is
not. In that regime the unhealed numbers are the operative ones and they are strongly in
favour (9.57 against 12.99, and 6.23 with DAgger). That is a real setting, it is narrower
than the one this project set out to address, and it would need its own pre-registration
rather than a re-reading of these cells.

## The one-paragraph version

Six stages of Pythia-1.4B were distilled independently on noise plus real-activation
anchors, composed, and healed end to end. The composition behaves better than the source
document predicted: error accumulates additively rather than compounding through Lipschitz
constants, because five of six stages contract what they inherit. On-policy retraining cuts
composed drift 41% and puts the noise-trained stack below one trained on real activations.
Every one of those wins is measured at the interfaces, and every one of them nearly
vanishes after a modest end-to-end heal. At equal total FLOPs the whole construction loses
to a randomly initialised student that just trains longer, by 0.21 nats against a
within-arm spread of 0.07, and it loses despite two handicaps applied in its own favour.
The gate fires, Phase 3 does not run, and the result worth publishing is the one about
interface-level metrics overstating by an order of magnitude what survives end-to-end
training.
