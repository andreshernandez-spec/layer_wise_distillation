# 08: The data-limited protocol (DRAFT, not yet a pre-registration)

**Status: draft for Andres, 21 Sep 2026.** It becomes a pre-registration when the open
decisions in the last section are settled and the commit SHA is recorded here. Nothing
below has been run.

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
stitching delta at the best checkpoint was not recorded. **This is a handicap in the method's favour of the same
species as the ones in `docs/06`, and the new protocol exists partly to remove it.**

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
stagewise pipeline, which the old C-versus-R comparison did only at m = 10.

| arm | what it is | answers |
|---|---|---|
| B0 | plain KD from random init on D, multi-epoch, validation-stopped, lr grid | is the pipeline worth having at all |
| B1 | stagewise with m = 0 (anchors only), each stage validation-stopped | the literal "without it" |
| M(m) | stagewise with m in {2, 8, 32}, same rule | the claim, and how much noise it takes |
| M+ | M at the best m, with on-policy retraining (step 3 only) | does DAgger survive a D-limited heal |
| ref | KD on D plus teacher-generated text (step 3 only, reference, not a gate) | the other way to turn teacher compute into data |

Stages are selected on **validation stitching delta**, not eps: Phase 1 showed eps improve
while stitching degrades, so eps cannot pick the checkpoint.

**Budgets.** D in {1e5, 3e5, 1e6} tokens as nested row prefixes of the declared slice
(49, 146, 488 rows; sha256 per subset recorded), plus the already-measured 1e7 as the
data-rich end. Statistics are re-harvested per D; at 1e5 that is about 90k training
positions for a 2048-dimensional covariance, which is what the shrinkage estimator is for,
and whatever it costs the noise arm is part of the method's honest price.

**Three steps, each gated, cheapest first.**

*Step 1: single-stage dose-response with a fair baseline* (stage 2, the Phase 1
substrate). D in {1e5, 3e5, 1e6} x m in {0, 2, 8, 32} x 3 seeds = 36 cells, validation
patience, cap 1e8 positions. About 5 A100-hours with four jobs sharing the card, **~$8**.
Gate S1: some m > 0 beats m = 0 on held-out stitching delta by more than twice the pooled
seed spread at some D. If not, the Phase 1 effect was early stopping in disguise; write
that up and stop.

*Step 2: composed, the decisive test*, at each D that passed S1. Full six-stage stacks for
m = 0 and the best m; compose; heal on D with validation stopping and an lr grid
{3e-5, 1e-4, 3e-4}; B0 on the same D with the same grid and rule; 3 heal seeds per arm;
every arm of one D on one machine, because the replication pass put machine-to-machine
error at 0.0135 against 0.0028 seed-to-seed. About 3 A100-hours per D, **~$5 per D**.

*Step 3: the ends and the references.* Multi-epoch validation-stopped heals for M and B1
at D = 1e7 on the existing stacks (~$6); on-policy retraining and the self-generated-text
reference at the winning D (~$8).

All of it is about **$35**; the first gate can stop it at $8.

**Success (S2).** At some D, M(m) beats **both** B0 and B1 on held-out loss by at least
0.10 nats, means over 3 seeds, no overlap between the arms' ranges, same machine. 0.10 is
five times the measured run-to-run noise of 0.02.

**Kill.** M loses to B1 at every D: noise buys nothing at the model level even when data
is the binding constraint, and the Phase 1 effect does not survive composition. M beats B1
but loses to B0 at every D: noise helps a pipeline that is itself not worth running.

**Reported whatever happens**, generated not typed: per arm and D, the noise positions,
teacher queries, total FLOPs, FLOPs as a multiple of what B0 actually spent, the step at
which validation picked the checkpoint, and held-out loss; and the dose-response figure,
held-out loss against m at each D. That table is the answer to "2x or 8x".

## Code it needs

Small, and all of it testable on CPU before any rental.

- **A data budget in the harvest**: restrict statistics, anchors, top-k store and the
  validation split to the first N rows of the declared slice; write the subset's sha.
- **Noise ratio as a parameter**: `AnchorMix` takes m, including 0 and ratios above 7
  (batch 8 cannot hold 1:32, so interleave noise-only batches); record the ratio achieved.
- **Validation in the stage trainer**: periodic stitching delta on the validation rows,
  keep-best state, patience, trajectory in the record.
- **Validation in the heal**: `heal.py` passes `eval_fn=None` today, so no heal has a
  held-out trajectory and nobody knows whether 17.5 epochs was the random arm's optimum.
  Add the validation eval, keep-best, and the lr grid.
- **Accounting in every record**: FLOPs, teacher queries, noise and real positions, so the
  multiplier table comes from `claims.py`.
- The guards from `docs/07` carry over unchanged: schedule asserted at launch, results
  never overwritten, checkpoints hashed, logs rotated, delivered tokens checked.

## What happens to the paper

Hold it. Its measurements stand, including "at equal FLOPs the noise recipe loses by 0.176
nats". Its framing as a negative result about the method is a statement about a
compute-limited objective the author does not hold. If S2 passes, the paper becomes "loses
at equal compute, wins at equal data below X tokens, and interface metrics overstate
either way", which is a better paper. If S2 fails, the negative is complete instead of
partial. Either way it should not go out before step 2.

## Open decisions (Andres)

1. **What "without it" must mean for success.** Recommended: beat both B1 (same pipeline,
   no noise) and B0 (no pipeline). Beating only B1 shows noise helps a pipeline; it does
   not show the result is useful.
2. **Which regularizations.** As run so far, "these regularizations" are the noise mix,
   the whitening contract and weight decay. The sketched-CF distribution term and any alpha
   penalty from the source document were only ever diagnostics; no training loss uses
   them. Recommended: noise only through S1, and add a CF-regularized arm in step 2 only if
   S1 passes, since it is a new capability and not a config change.
3. **The margin**, 0.10 nats, and the budget, about $35 with an $8 first gate.
4. **Whether the student should start pretrained.** "Finetuning" suggests it might. Every
   arm here starts from random weights because the six-by-two-block student has no
   pretrained version. A pretrained small model tuned on D is a fair outside reference and
   a different architecture. Recommended: leave it out of this round and name it as a
   limitation.
