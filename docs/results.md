# Results ledger

Every number here has a script and an environment behind it (`CLAUDE.md` §2). Dates are
when the number was produced; SHAs are recorded in the run JSONs under `out/`.

## Reference models (22 Aug 2026, RTX 3080 Laptop, torch 2.13.0+cu130, fp16)

Held-out next-token loss on the declared 2M-token Pile-test rows
(`experiments/eval/perplexity.py`, `docs/01` C0.1):

| model | loss (nats) | ppl |
|---|---|---|
| Pythia-410M (from-scratch comparator, Tier 0) | 1.9132 | 6.774 |
| Pythia-1.4B (teacher, Phase 1) | 1.7364 | 5.677 |

Pre-declared suite, zero-shot, lm_eval 0.4.12 (`python -m lm_eval --model hf
--model_args pretrained=MODEL,dtype=float16 --tasks lambada_openai,piqa,winogrande,arc_easy,arc_challenge,sciq,hellaswag --batch_size 16 --seed 0`;
metric per task as in `experiments/eval/summarize.py`; LogiQA excluded, `docs/01` C0.1):

| model | lambada_openai | piqa | winogrande | arc_easy | arc_challenge | sciq | hellaswag | mean |
|---|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-1.4b | 61.5 | 71.0 | 57.5 | 53.9 | 28.4 | 79.0 | 52.0 | 57.6 |
| EleutherAI/pythia-410m | 51.3 | 67.2 | 53.3 | 45.7 | 24.4 | 72.4 | 40.6 | 50.7 |

Both agree with the published Pythia figures (410M: LAMBADA 51.3 vs 51.6, PIQA 66.9,
ARC-e 52.2 vs 51.9, HellaSwag 33.7; 1.4B: LAMBADA 61.5 vs 61.6, PIQA 70.9, ARC-e 60.6 vs
60.5), so the harness is set up right. The teacher-comparator gap on the suite mean is
the room Tier 0 plays in.

## Phase 0 gates, Pythia-1.4B (22 Aug 2026)

G0.3 anchor round trip: int8 + per-channel std store, floor ~1% per channel, propagated
error <= 1.9x floor through stage 4 and 4.09x at stage 5 (marginal fail against the 4x
derived tolerance). C0.5 bridge floor at interface 2: 0.050 (top-1024), 0.103 (top-512).
Tables and reading: `docs/01` "Gate results".

## Phase 1, first reads (22 Aug 2026)

Pythia-1.4B stage 2, 2-block same-width student, Q = 1e7 positions, one seed: eps
(held-out real, phi coordinates) real 0.544, anchor-mix 0.645, moment-matched noise
0.87, isotropic 1.55; stitched next-token delta 1.05 / 2.15 / 3.2 / 6.6 nats. AR(1)
noise = i.i.d. noise. `docs/02` "Second pass" has the table and the caveat on the R arm.

Third pass (22 Aug 2026): live-real oracle at 1e7 eps 0.535 / stitch 1.00 nats;
anchor-mix at 3e7 eps 0.571 / 1.13 nats; pure Gaussian noise at 3e7 *worsens* to 1.335
(0.873 at 1e7), AR(1) to 1.764. `docs/02` "Third pass".

Fourth pass (22 Aug 2026): anchor-mix at 1e8 eps 0.464 / stitch 0.52 nats, better than
the live-real oracle at the 1e7 real budget (0.535 / 1.00); pure Gaussian noise at 1e8
diverges to eps 12.2. Controls (R and L at 1e8) running. `docs/02` "Fourth pass".

Gate, provisional (22 Aug 2026, one seed): criterion 3 met by the recipe arm (eps_inf
CI upper bound 0.53 < 1.5 x 0.535; stitch 0.52 < 2 x 1.00 nats); pure Gaussian noise
eps_inf 0.65 [0.38, 0.89], stitch fails. Pending: R@1e8 control, second seed, Tier
budgets from Q* = 3.7e7 (Tier 0 ~2.5e17 FLOPs, ~9 T4-hours). `experiments/phase1/gate.py`.

Controls at 1e8 (23 Aug 2026): R (same anchors, no noise) eps 0.413 / stitch 0.50 beats
C_mix (0.464 / 0.52); L recycled 10x over the 1e7 budget 0.353 / 0.41. The fourth-pass
headline was a step-count artefact; noise adds nothing at ~1M anchors per interface.
`docs/02` "Controls at Q = 1e8".

Platform check (23 Aug 2026): 9 Phase 1 cells run on both the RTX 3080 Laptop and a
RunPod A100 SXM, same torch, reproduce eps to a median 0.00% and max 0.46% relative
difference. Laptop and pod cells can be read in one table.

Correction (23 Aug 2026): arm "C" under `contract: zca` is mathematically the same
measure as arm "G" (C and G eps agree to 0.00-0.01% at every shared Q). The marginal
Gaussianization the source document argues for is untested; `grid-1.4b-gauss.yaml`
tests it, and must be read on the stitching delta because eps is phi-dependent.

## Phase 1 complete grid (23 Aug 2026, A100, 89 cells, two seeds, SHA 82ea470)

beta = 0.324 [0.297, 0.506] live real, 0.320 [0.293, 0.532] recycled anchors, 0.305
[0.281, 0.596] recipe: the same exponent whatever the data. eps_inf consistent with
zero for all three; only pure noise has a floor (0.417) and it diverges.

At Q = 1e8, stitching delta in nats: live real (1e7 distinct positions) **0.452**,
recycled anchors (0.95e6 distinct, ~100 passes) **0.450**, a difference of a twentieth of the seed spread, i.e. indistinguishable, anchors + 91M noise
positions 0.527, pure noise 7.92. Ten times fewer distinct tokens, same quality; added
noise makes it worse. `docs/02` "The complete grid".

Scarce anchors (23 Aug 2026): at 46 anchor sequences (94k positions) and Q = 1e8,
anchors-only overfits (eps 0.632 at step 600 to 1.061; stitch 2.188) while
anchors+noise converges (0.600; stitch 1.415). Noise flips from liability at 0.95M
anchor positions to necessity at 94k. Crossover being located. `docs/02` "Scarce anchors".

Scarce-anchor reversal replicates on both seeds (R final eps 1.061/1.052 against
C_mix 0.600/0.595). At 464 anchors there is no overfitting even at 300 passes
(eps 0.374, plateau). Open: R's stitching worsened 0.450 to 0.782 between 1e8 and 3e8
while eps improved; one seed only, needs a second before it can be cited.

Marginal Gaussianization (23 Aug 2026, 12 cells): worse than affine whitening on 8 of
8 readable cells, by 0.33 to 2.33 nats of stitching delta. The source document's
"normalization is free because phi is a bijection" holds for the optimum and not for
finite-step training. Default stays `contract: zca`. `docs/02`.

Anchor-budget crossover (23 Aug 2026, Q=1e8, stitching delta): noise is worth +0.77
nats at 94k anchor positions, +0.14 at 188k, and nothing at 0.95M. The crossover sits
between 188k and 950k; the 377k cell is running. `docs/02`.

Settled 23 Aug 2026: (a) the crossover is at ~250k real positions per interface
(noise worth +0.773 nats at 94k, +0.142 at 188k, -0.045 at 377k, -0.078 at 0.95M);
(b) eps and end-to-end loss diverge with training length: R at 464 anchors improves
eps 0.412 to 0.373 from 1e8 to 3e8 while its stitching delta worsens 0.450 to 0.710,
both seeds. A stage must not be stopped on eps. `docs/02`.

Corrected 23 Aug 2026: at the full anchor budget, "noise buys nothing" holds at 1e8
and fails at 3e8. Anchors-only improves eps 0.412 to 0.373 while stitching degrades
0.450 to 0.710; anchors+noise improves both (0.463 to 0.393 eps, 0.527 to 0.413
stitching) and wins by 0.297 nats at 3e8. One mechanism explains the phase: noise is a
regularizer against over-fitting the interface objective, which arises from too few
anchors or too much training. `docs/02`.

Spectral gates, 112 students (23 Aug 2026): alpha correlates +0.80 with stitching
across all students, which is a training-length artefact; within a fixed budget the
sign flips (-0.87 at 1e7, +0.84 at 1e8, no signal at 1e6). Not usable as a per-stage
gate. `docs/02`.

## Phase 2, composition and healing (24 Aug 2026, A100, SHAs 6b8f60c..33773eb)

Substrate: Pythia-1.4B, six stages of four teacher blocks, 2-block same-width students
(6.05e8 non-embedding params against 1.21e9). Teacher embedding, final norm and head
transferred and frozen.

Composition error accumulates, it does not compound. Every stage after the first
contracts what it inherits (teacher Lipschitz 0.43 to 0.84) and adds a roughly constant
fresh error of about 0.54. Realized drift at the output is 2.335 (C stack); the
ratio-product prediction from stage 0's drift is 46.3 at interface 2 alone. The source
document's compounding frame is the wrong direction for five stages out of six.
`docs/03`, `docs/06`.

Stage 0 is the hardest to fit (Jacobian cosine 0.038) because the teacher's first stage
amplifies a perturbation 48.5x. It does not act on the composed model, since the student
uses the teacher's embedding and interface 0 has zero drift by construction. `docs/03`.

Exposure bias is most of the fresh per-stage term: on-policy retraining (DAgger, p=0.5,
1e7 positions) cuts drifted eps 53 to 74% on stages 1 to 5, and only 8% on stage 0,
which has no inherited drift to correct. `docs/03`.

DAgger survives composition and does not survive healing (24 Aug 2026). Composed drift
at the output falls 2.335 to 1.387, a 41% cut, which puts the noise-trained stack below
the real-activation stack (1.473). The same stacks after healing: 9.5702 to 6.2323
unhealed (worth 3.34 nats), 5.5692 to 5.1555 at 1e6, and 3.7221 to 3.6375 at 1e7 (worth
0.085). The gain is being closed, not converging. The 0.085 is inside Phase 1's 0.106-nat
seed spread and there is one seed per cell, so the defensible statement is that after
healing DAgger's effect is not distinguishable from zero. `docs/06`.

The heal-budget curve, both arms tuned, warmup 500, teacher 1.736 nats:

| heal tokens | random | stagewise | oracle |
|---|---|---|---|
| 1e5 | 8.5325 | 6.9982 | 5.4803 |
| 1e6 | 6.8356 | 5.5692 | 4.7819 |
| 3e6 | 5.9450 | 5.0123 | 4.2673 |
| 1e7 | 5.2693 @5e-5 | 3.7221 | 3.4850 |
| (no heal) | 12.9942 | 9.5702 | 6.7196 |

Criterion 5, the price of noise at Tier 0 scale: **+0.2371 nats** at 1e7 heal tokens
(stagewise 3.7221, oracle 3.4850), against 2.85 nats before healing. `docs/06`.

The pattern across three independent interventions (G1's anchors-vs-noise, criterion 5's
anchors-vs-anchors+noise, criterion 3's DAgger): each is ranked clearly by an
interface-level metric and each collapses by roughly an order of magnitude once a modest
end-to-end budget is allowed. No claim in this project should be quoted from eps, drift
or a stitching delta without the healed number beside it. `docs/06`.

G2's kill criterion, 24 Aug 2026: at equal end-to-end FLOPs (6.667e17: harvest, six stage
cells, and the stagewise arm's own 1e7 heal) a random-init student healed on 1.8367e8
tokens reaches **3.5430** against the stagewise stack's 3.7221/3.7900 over two heal
trajectories (mean 3.7560, spread 0.0679). Gap **-0.213 nats**, about 3x the within-arm
spread. It also beats the DAgger stack (3.6602 mean), which cost 10% more FLOPs to build.
The kill fired despite two handicaps applied in the stagewise arm's favour: the random arm
recycles one 1e7-token slice 17.5 times rather than seeing distinct tokens, and runs at
5e-5 against 1e-4 (worth ~0.2 nats). **Phase 3 does not run.** `docs/06`.
