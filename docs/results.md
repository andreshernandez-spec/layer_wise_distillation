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
recycled anchors (0.95e6 distinct, ~100 passes) **0.450**, anchors + 91M noise
positions 0.527, pure noise 7.92. Ten times fewer distinct tokens, same quality; added
noise makes it worse. `docs/02` "The complete grid".

Scarce anchors (23 Aug 2026): at 46 anchor sequences (94k positions) and Q = 1e8,
anchors-only overfits (eps 0.632 at step 600 to 1.061; stitch 2.188) while
anchors+noise converges (0.600; stitch 1.415). Noise flips from liability at 0.95M
anchor positions to necessity at 94k. Crossover being located. `docs/02` "Scarce anchors".
