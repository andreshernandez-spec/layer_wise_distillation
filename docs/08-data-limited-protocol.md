# 08: The data-limited protocol

**Status: pre-registered 21 Sep 2026.** Andres settled the four open decisions the same
day (last section). The commit that adds this sentence is the timestamp; step 1's
constants are asserted at launch by `experiments/pod/phase2b_step1.sh`, so a config that
drifts from this page stops the run instead of changing the experiment. No cell of this
protocol had been run when this was committed. Two probes were run first, both on Phase 2
artifacts and neither training anything, and both are reported below because they shaped
the design: `cf_probe.py` and a read of the harvested CF sketch.

## Why this exists

G2 judged the method at equal total FLOPs and it lost. That was the source document's own
kill line (§ "Kill: healed stagewise ≤ random-init + equal heal budget at equal total
FLOPs"), implemented literally. Andres's position, stated on 21 Sep after reading the
result, is that FLOPs was never the point:

> The intention of these regularizations is to reduce the need for more samples, to be
> more effective with less data, because for finetuning the main problem is not compute
> but data, and then to a lesser extent compute. If these random samples are required at
> 2x or 8x of regular data I want to know about it, but I would still consider it a
> success if I can produce a better model using these regularizations and random noise
> than without it for the same amount of data.

That is consistent with the rest of the source document ("noise is counted separately from
real tokens in every claim", mandatory control (1) "random-init + identical heal budget",
G3's "≤10^7 real tokens"). It is not consistent with the kill line, and the kill line is
what got measured.

**What this document does not do.** It does not edit G2. G2 fired under the criterion it
was registered with and `docs/06` stands as written. The criterion is being changed after
G2's result was known, which is exactly the move `CLAUDE.md` rule 3 warns about, so the
change is made in the open: a new question, a new protocol, its own margin, declared
before any run. If the new protocol also fails, the project has two negatives and not one
softened one.

## The criterion

1. **The budget is distinct real tokens, D.** Everything real that any arm touches comes
   from the same D-token subset of the declared slice: interface statistics (the whitening
   contract and the noise moments), anchors, the top-k store for the heal, on-policy rows,
   and the validation split used for early stopping and model selection. The source
   document's "real tokens do triple duty" becomes a rule rather than a convenience.
2. **Teacher queries are not data.** Noise positions cost compute only. They are counted
   and reported, never matched.
3. **Compute is reported, not matched, under a common generous cap.** Every arm may spend
   up to the FLOPs of the most expensive method configuration at that D. The baselines get
   the same cap and will stop earlier by validation patience; what they actually used is
   recorded. This is the exact converse of G2: there, the method was charged for its
   compute; here, no baseline can be called compute-starved.
4. **Selection never touches the held-out set.** Early stopping, learning-rate choice and
   checkpoint selection use a validation split carved from D (the last 10% of its rows,
   at least 4), identically for every arm. The 2M-token Pile-test held-out set is read
   once per finished model.
5. **Metric**: held-out next-token loss in nats, as before.

"Identical heal budget, one epoch each" is not the criterion, although the source document
reads that way and the method passes it at every budget (see below). A practitioner short
of data and not of compute runs the baseline for more epochs, and we measured what that
is worth: the random-init arm goes from 5.2693 to 3.5520 on the *same* 1e7 tokens by
recycling them 17.5 times. A data-limited comparison has to let every arm do that.

## What the existing cells already say under this criterion

Numbers from `experiments/paper/claims.py` (`scarce.*`, `long.*`, `heal.*`, `crossover.*`).

**Single stage, scarce anchors (Phase 1, stage 2, Q = 1e8).** Stitching delta, nats:

| real positions | anchors only (R) | anchors + noise (C_mix) | noise worth |
|---|---|---|---|
| 94k | 2.188 / 1.801 | 1.415 / 1.243 | +0.773 / +0.557 |
| 188k | 1.118 | 0.976 | +0.142 |
| 377k | 0.667 | 0.711 | -0.045 |
| 1.05M | 0.450 | 0.527 | -0.078 |
| 1.05M, Q = 3e8 | 0.782 / 0.638 | 0.413 / 0.401 | +0.369 / +0.237 |

This is the hypothesis in miniature: noise helps exactly where real data is scarce or
training is long, and the sign flips near 250k positions.

**It is also not yet evidence, because the baseline was never early-stopped.** At 94k
positions the anchors-only arm reaches its best eps, 0.632, at step 600 of 6104 on both
seeds, a tenth of the way in, and then climbs to 1.061 (1.052 on the second seed). Its
stitching delta was read at the final step. The noise arm peaks at 59% to 66% of the run
and barely moves afterwards (0.593 best, 0.600 final). So "+0.773" compares noise against
a model trained ten times past the point where anyone with a validation split would have
stopped it, and part of what noise is doing in that table is standing in for early
stopping. On eps at each arm's best checkpoint the gap is 0.040, not 0.46. What it is on
stitching delta at the best checkpoint was not recorded. **This is a handicap in the
method's favour of the same species as the ones in `docs/06`, and the new protocol exists
partly to remove it.**

**Composed and healed, D = 1e7.**

| arm | heal | held-out loss |
|---|---|---|
| plain KD from random init (B0) | 1 epoch | 5.2693 |
| plain KD from random init (B0) | 17.5 epochs | 3.5520 (n=2) |
| stagewise, real activations (B1) | 1 epoch | 3.4931 (n=3) |
| stagewise, noise recipe (M) | 1 epoch | 3.7283 (n=2) |

Three readings, none of them settled:

- At an identical one-epoch heal budget the method beats random init at every budget from
  1e5 to 1e7 tokens, by 0.93 to 1.55 nats. That passes the source document's control (1)
  as literally written, and it is the wrong comparison for the reason given above.
- The stagewise pipeline on real activations, with **one** heal epoch, already beats plain
  KD with 17.5 epochs on the same data (3.4931 against 3.5520). Under a data criterion the
  *decomposition* is plausibly a win at D = 1e7. M and B1 were never given a multi-epoch
  heal, so their best on this data is unknown.
- *Noise* does not help at D = 1e7: anchors-only beats the noise recipe by 0.235 after
  healing, which agrees with the single-stage crossover. The hypothesis worth testing is
  that noise's contribution turns positive as D falls below the crossover **and survives
  composition and a D-limited heal**. The "10x collapse under healing" in `docs/06` was
  measured with a heal that had ten times more real data than the anchors did; with the
  heal confined to the same small D, that collapse is not guaranteed.

One fact about scale, since the question mentions 2x and 8x: every noise cell run so far
used about 96 noise positions per distinct real position at the full anchor budget and
about 1000 at the scarce one. Nothing is known yet about small multipliers.

## Design

**Noise is a dose.** The treatment is the noise ratio m, noise positions per real position
in the training stream, and m = 0 is the control. Everything else is identical across
doses: same D, same contract, same selection rule, same cap. That isolates noise from the
stagewise pipeline, which the old C-versus-R comparison did only at m = 10. `DoseMix`
carries the fractional real count from batch to batch, so the ratio is exact over a cycle
and m = 32 is expressible at a batch of 8.

**The CF term is the second treatment** (decision 2). It is the source document's "soft
shape" penalty: 64 fixed unit directions, frequencies {0.5, 1, 1.5, 2}, squared difference
of the two empirical characteristic functions. Two things about it were settled by
measurement before this page was committed.

- *It is a two-sample statistic in whitened coordinates*, student output against teacher
  output on the same batch, so its optimum is the mimic optimum on real and on noise inputs
  alike, which is the condition the source document puts on any regularizer used as a
  loss. It is **not** the sketch the harvest accumulates. That one is in raw coordinates,
  and read back it is unusable as a uniform target: at interface 0 every |cf| is 1.000,
  at interface 6 it is 0.06 by t = 1, and only mid-depth interfaces happen to have unit
  scale.
- *There is something for it to act on.* `experiments/phase2b/cf_probe.py`, on the Phase 2
  stage-2 students and real held-out activations: the variance of the student's output
  along random whitened directions is **0.65** of the teacher's for the anchors-only
  student and **0.51** for the noise recipe. That is the conditional-mean shrinkage an MSE
  objective produces, and it means the next stage reads activations with half the spread
  it was trained on. The CF distance there is 0.017 to 0.037 against a relative MSE of 0.43
  to 0.48, so lambda = 3 puts the term at about a tenth of the MSE and lambda = 30 at about
  parity. Those two values are the grid.

The CF term cannot add information. It can only trade a little MSE for a correct marginal,
and whether the next stage prefers that is the empirical question.

| arm | what it is | answers |
|---|---|---|
| B0 | plain KD from random init on D, multi-epoch, validation-selected, lr grid | is the pipeline worth having at all |
| control | stagewise, m = 0, lambda = 0, weight decay in {0.1, 1.0} | the literal "without it", not left as a strawman |
| M(m) | stagewise, m in {2, 8, 32} | the claim, and how much noise it takes |
| CF | lambda in {3, 30}, at m = 0 and at m = 8 | regularization alone, and with noise |
| M+ | the chosen treatment with on-policy retraining (step 3) | does DAgger survive a D-limited heal |
| ref | KD on D plus teacher-generated text (step 3, reference, not a gate) | the other way to turn teacher compute into data |

**Selection, the same for every arm.** A cosine sized in advance cannot be used: a
checkpoint lifted from the middle of a long cosine run is at near-peak learning rate, and
the arm that peaks early (the control, at scarce data) would be read at its worst. So:
warm up 100 steps to 3e-4, hold the rate, validate every 100 steps, stop after 10
validations without a new best or at the cap of 6104 steps (1e8 positions, the source
document's per-stage budget), then cool the best checkpoint linearly to zero over a tenth
of its step count (at least 50 steps) and keep whichever of the two validates better.
Stages validate on **stitching delta** over the budget's validation rows (at most 32 of
them), not eps: Phase 1 showed eps improve while stitching degrades. Heals validate on
next-token loss over the same rows.

**Choosing among configurations is also selection.** There are seven treatment
configurations against two controls. The best of each side is chosen on validation, and
only then is its held-out number read; choosing on the held-out score would hand the
treatment the best of seven draws.

**Budgets.** D = 49 and 488 rows of the declared slice (100,352 and 999,424 tokens), as
nested prefixes, sha256 per subset in the harvest record; 146 rows (299,008 tokens) only
if the sign changes between them. The last 10% of rows (at least 4) are validation and
everything else, statistics included, comes from the rest. At 49 rows that is 90,112
training positions for a 2048-dimensional covariance, which is what the shrinkage
estimator is for, and whatever that costs the noise arm is part of the method's price. The
already-measured D = 1e7 is the data-rich end.

**Three steps, each gated, cheapest first.**

*Step 1: single-stage dose-response with a fair baseline*, stage 2 (blocks 8 to 11, the
Phase 1 substrate). Nine configurations x two budgets x two seeds = 36 cells, then a third
seed on the chosen control and the chosen treatment at each budget. Three cells share one
A100. About 10 hours, **~$16**.

Gate S1, per budget: the chosen treatment's held-out stitching delta is below the chosen
control's by more than twice the pooled seed standard deviation, and no treatment seed is
above any control seed. If it passes at no budget, the Phase 1 effect was early stopping in
disguise; write that up and stop.

*Step 2: composed, the decisive test*, at each budget that passed S1. Six-stage stacks for
the control and the chosen treatment, every stage selected as above; compose; heal on D
with the same selection rule and an lr grid {3e-5, 1e-4, 3e-4}; B0 on the same D, same
grid, same rule; 3 heal seeds per arm; every arm of one budget on one machine, because the
replication pass put machine-to-machine error at 0.0135 against 0.0028 seed-to-seed. About
**$10 per budget**.

*Step 3: the ends and the references.* Validation-selected multi-epoch heals for M and the
real-activation stack at D = 1e7 on the existing checkpoints (~$6); on-policy retraining
and the self-generated-text reference at the winning budget (~$8).

About **$50** if every step runs; the first gate can stop it at $16.

**Success (S2).** At some budget the chosen treatment beats **both** B0 and the control on
held-out next-token loss by at least 0.10 nats, means over 3 seeds, no overlap between the
arms' ranges, same machine. 0.10 is five times the measured run-to-run noise of 0.02.

**Kill.** The treatment loses to the control at every budget: neither noise nor the CF
term buys anything at the model level even when data is the binding constraint. It beats
the control but loses to B0 at every budget: they help a pipeline that is itself not worth
running.

**Reported whatever happens**, generated not typed: per arm and budget, the noise
positions per distinct real token, the number of passes over the real data, total FLOPs and
their multiple of what the control spent, the step validation chose, whether the run
stopped on patience or hit the cap, the student's output dispersion relative to the
teacher's, and the held-out number. That table is the answer to "2x or 8x".

## The code, all of it tested on CPU before any rental

- `lwd.harvest.budget` and `budget_rows` in the harvest: statistics, anchors and the top-k
  store from the training rows of the budget, validation activations written after the
  statistics are closed. Tested end to end on pythia-70m.
- `lwd.noise.samplers.DoseMix`: exact ratio, m = 0 without a noise sampler, m above the
  batch size.
- `lwd.stage.cf.CFDistance`: zero at the mimic optimum, monotone in under-dispersion, its
  gradient pushes a shrunk student back out, blind to a permutation of positions (which is
  why it is only ever added to the MSE).
- `lwd.stage.select` and `lwd.heal.select`: hold, validate, stop on patience, cool the
  best checkpoint, keep the better; the restored weights are exactly the validated ones.
- `experiments/phase2b/cell.py`: one cell, accounts for every position, refuses to
  overwrite a result, asserts the statistics came from the training rows alone.
- `experiments/phase2b/gate.py`: gate S1, configurations chosen on validation.
- `experiments/pod/phase2b_step1.sh`: resumable queue, schedule asserted at launch, logs
  rotated. The guards from `docs/07` carry over unchanged.

The chain was run end to end twice before this page was committed, neither run being a
cell of the protocol: on pythia-70m (harvest, a control cell, a dosed CF cell, the gate),
and for 35 steps on the real 1.4B on the laptop to exercise fp16 stitching and the
contract at d = 2048 from a 90k-token budget.

## What happens to the paper

Hold it. Its measurements stand, including "at equal FLOPs the noise recipe loses by 0.176
nats". Its framing as a negative result about the method is a statement about a
compute-limited objective the author does not hold. If S2 passes, the paper becomes "loses
at equal compute, wins at equal data below X tokens, and interface metrics overstate
either way", which is a better paper. If S2 fails, the negative is complete instead of
partial. Either way it should not go out before step 2.

## Decisions, settled by Andres on 21 Sep 2026

1. **"Without it" means both.** Success requires beating the same pipeline without the
   treatment and plain KD on the same data.
2. **The CF-regularized arm is in.** Defined above; screened in step 1 alongside the noise
   doses and carried into step 2 if validation chooses it.
3. **Margin 0.10 nats; budget about $50.**
4. **No pretrained start.** Every arm starts from random weights. "Finetuning" in the
   original statement was loose wording for a data-limited setting, not a requirement.
