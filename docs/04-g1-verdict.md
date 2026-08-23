# G1 verdict

**Status: provisional, 23 Aug 2026.** Three cells still running (a184, R@3e8 seed 1,
C_mix@3e8 rerun); none can overturn what follows, they sharpen one number each. Every
figure traces to `out/phase1-1.4b-a100/` and `out/phase1-1.4b/`, SHAs in each run JSON.

Substrate: Pythia-1.4B, stage 2 (blocks 8-11), 2-block same-width student, 109 cells
across an RTX 3080 Laptop and a rented A100, two seeds on everything decisive.

---

## The five criteria

**1. All must-exist cells ran, from a committed config, on recorded hardware.** Met.
89-cell design plus 20 targeted cells. Zero unexplained failures; the three that did
fail are diagnosed in this document's "What broke" section and were re-run.

**2. Fits with CIs for eps and the stitching delta.** Met. Bootstrap CIs on 13 points
per arm, and, more usefully, **measured error bars from repeated seeds**: eps 0.004,
stitching delta 0.106 nats over 42 same-arm pairs. Every claim below is quoted against
those.

**3. The kill criterion.** Pre-registered as: best noise cell has
`eps_inf <= 1.5 x eps_real(1e7)` and a stitching delta at its largest Q within
`2 x` the real arm's. **Met by the recipe arm** (eps_inf CI upper bound 0.382 against a
0.798 margin; stitching 0.527 against a 1.995 margin). Every pure-noise arm fails on
stitching.

**4. A written Tier-1/2 budget forecast from the measured Q\*.** Met, below.

**5. The length ablation.** Not run, and it should be: the whole campaign is L=2048.
Carry it into Phase 2 rather than reopening Phase 1.

**Formally, G1 passes.** The rest of this document is why that sentence would be
misleading on its own.

---

## What was actually learned

### beta, the number the phase existed to produce

`eps(Q) = c Q^-beta + eps_inf`, early-stopped eps, 13 points per arm:

| arm | beta | eps_inf |
|---|---|---|
| live real, distinct positions | 0.324 [0.297, 0.506] | 0 [0, 0.186] |
| recycled anchors | 0.320 [0.293, 0.532] | 0 [0, 0.253] |
| recipe (anchors + noise) | 0.305 [0.281, 0.596] | 0 [0, 0.382] |
| pure moment-matched noise | 0.339 | **0.417**, and it diverges |

**beta is ~0.32 whatever the data is.** The three working arms agree inside each
other's CIs. The exponent is a property of the stage-fitting problem, not of the input
measure. What the input measure changes is `eps_inf`, and only pure noise has one that
is not consistent with zero.

### The result the phase produced, which is not the one it set out to find

At Q=1e8, stitching delta in nats (seed spread 0.106):

| arm | distinct real positions | noise positions | stitch |
|---|---|---|---|
| live real | 1e7 | 0 | 0.452 |
| recycled anchors | 0.95e6 | 0 | **0.450** |
| recipe | 0.95e6 | 9.1e7 | 0.527 |
| pure noise | 0 | 1e8 | 7.92 |

**0.95M real positions recycled ~100 times are indistinguishable from ten times more
distinct data.** 0.450 against 0.452 is a twentieth of the seed spread. That is a real
and useful result about data frugality, and **noise is not what delivers it**.

### Where noise does earn its place

| anchor positions | R stitch | mix stitch | noise is worth |
|---|---|---|---|
| 94k | 2.188 | 1.415 | +0.77 nats |
| 188k | 1.118 | 0.976 | +0.14 nats |
| 0.95M | 0.450 | 0.527 | tied |

Below ~200k positions per interface the anchors-only student **overfits** (peaks at
step 600, then degrades monotonically) and noise prevents it; both seeds agree. Above
~1M, noise buys nothing. The crossover is between 188k and 950k.

This matters because one teacher pass over the token slice yields activations at every
interface at once, so with a **resident** teacher each interface has ~1e7 distinct
positions and noise is pointless. Scarcity binds when activations must be **stored**:
`docs/00` §2.2 caps the store at 1e6 positions per interface, and at Tier 2 (d=5120,
16 interfaces) even that is 80 GB.

**So the defensible claim is not "noise replaces data". It is "noise buys back the
difference between storing activations and recomputing them."** That is a systems
claim, it is measured, and it fits the depth-parallel story the project already has to
tell. It is a smaller claim than the source document's, and it is the one the data
supports.

### Four design decisions the data settled

| decision | source document | measured |
|---|---|---|
| sequence structure of the noise | AR(1) or fitted process may fix attention | **AR(1) is i.i.d.**, both seeds. Only real sequences move the teacher's attention entropy (2.44 real, 4.81 noise, 3.80 at 1:10 mix) |
| marginal Gaussianization | "unbounded and free of mismatch", §1.1 | **worse on 8 of 8 readable cells**, by 0.33 to 2.33 nats. phi cannot move the optimum but does reweight the finite-step objective |
| HT-SR alpha as a per-stage gate | §1.5, the substitute for a downstream signal | **does not separate** a healthy student (eps 0.57) from a degraded one (1.34). Use held-out anchors |
| Hadamard rotation before fp8 anchors | §2, Plan-B synergy | **wrong trick**: rotation makes fp8 worse (0.026 to 0.31). int8 + per-channel std gives 0.007 |

---

## Budget forecast (criterion 4)

Q* against the live-real oracle at 1e7: 5.1e7 (real), 5.5e7 (recycled anchors), 7.5e7
(recipe), against the source document's assumed 1e8 per stage.

| tier | per-stage FLOPs | stages | total at Q*=7.5e7 |
|---|---|---|---|
| Tier 0, 2.8B to 410M | 8.5e8 | 8 | 5.1e17 (~18 T4-hours) |
| Tier 1, OLMo-7B to 1B | 2.4e9 | 8 | 1.4e18 (~50 T4-hours) |
| Tier 2, 32B to 2B | 6e9 | 16 | 7.2e18 (~250 T4-hours) |

All three are inside free-tier reach for the stage phase, which was the point of the
architecture. Measured throughput: 4.5x the laptop on one A100, and **concurrency is
neutral** once the card is saturated, so plan device-hours, not worker counts.

---

## What broke, and what it cost

Three bugs, none of which the laptop campaign would have surfaced:

1. **Mix arms outside C** built their noise component with `structure="mix"`, which the
   samplers do not know. `C_mix` worked only because the runner special-cased it. Cost:
   2 cells. Now covered by a test that builds every arm named in every grid.
2. **The Gaussianized fit** cast the whole anchor set to float64 in one call: ~100 GB
   at d=2048, SIGKILL on every cell. The 70m smoke test was three orders too small.
   Cost: 6 cell-starts. Now chunked, with a test asserting chunked equals one-shot.
3. **One NaN gradient poisoned a whole run.** `clip_grad_norm_` scales every parameter
   by a coefficient built from the total norm, so a single non-finite gradient makes
   the model NaN permanently. `C_mix@3e8` went from loss 0.2246 to NaN at step 13741 of
   18311 with no warning. One cell in 129, and the risk grows with run length. Cost:
   2.7 GPU-hours. Now skipped and counted, with a test that injects an inf mid-run.

Also worth carrying forward: BLAS thread oversubscription left the GPU at 0% for five
minutes on a 256-core host (`docs/compute.md`), and the platform effect between two
GPUs is 0.46% on eps, the same order as seed noise.

---

## Recommendation

**Proceed to Phase 2, with the claim rewritten.** The thesis that survives contact with
the data is narrower than the source document's and still worth a paper:

- stagewise training reaches the same beta as end-to-end-quality real data;
- ~1M recycled real positions per interface match ten times more distinct data;
- below ~200k positions noise is what makes stagewise training work at all, and that
  is exactly the regime storage forces at Tier 1 and 2;
- the interface contract, in its strong form, is not the mechanism: affine whitening
  beats marginal Gaussianization, and moment matching beats every richer noise model
  tried.

Phase 2 changes are in `docs/03`: chain two stacks rather than one, propagate anchors
rather than noise for the DAgger ablation, and stop each stage on held-out anchors.

**If the project needs the strong claim** ("noise replaces data at scale"), Phase 1
does not support it and no amount of Phase 2 will rescue it. Say so in the intro and
claim the systems result instead.
