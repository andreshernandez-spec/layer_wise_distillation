# G2 verdict

**Status: draft, 24 Aug 2026. Criterion 4, the kill criterion, is still running** (the
equal-FLOPs random cell, 1.8367e8 tokens, started 20:31 UTC). Everything else is
settled and is written up here as it stands. The recommendation section is deliberately
empty until that cell lands: it is the criterion the phase exists to answer, and
writing a conclusion around a missing number is how the first version of the C2.4
baseline came to be invalid.

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

**4. Kill: at every heal budget, stagewise ≤ random at equal total FLOPs.** *Pending.*

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
| ratio-product prediction | 0.637 | 46.32 | 20.13 | 8.82 | 4.10 | 2.08 |

The prediction is off by a factor of 40 at interface 2 and never recovers, because every
stage after the first **contracts** what it inherits (teacher Lipschitz 0.43 to 0.84).
Drift grows anyway, but as a sum: each stage adds a roughly constant fresh error of about
0.54 regardless of depth.

The discrepancy is reported here whichever way it goes, as criterion 2 requires, and it
goes against the source document. The practical consequence is that the mitigations had
to be re-justified as ways to reduce the *fresh per-stage term* rather than to stop
amplification, which turned C2.3 from a demonstration into a test with a prediction.

The teacher's first stage does amplify, by 48.5x, and that number is real. It does not
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

Say the 0.085 carefully: it is **smaller than the 0.106 nats of seed spread Phase 1
measured** over 42 same-arm pairs, and there is one seed per cell here. So the correct
statement is not "DAgger is worth 0.085 nats after healing" but **"after healing, DAgger's
effect is not distinguishable from zero at this budget"**. That is the stronger version of
the same conclusion, and it does not depend on a number too small to defend. A repeat on a
second heal trajectory is queued to put a real error bar on it; until it lands, 3.338 at
no-heal and 0.414 at 1e6 are the two figures in this table that clear the noise.

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
