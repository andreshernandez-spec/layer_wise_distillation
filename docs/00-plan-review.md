# 00: Review of the source document

Audit of `claude_stagewise-noise-distillation-plan.md` (22 Aug 2026), done 22 Aug 2026
before any code. Three parts: what holds, what is inconsistent or wrong, and the
decisions only Andres can take. Numbers checked against fetched HF configs and against
what the `es` project measured on the same free tiers.

## 1. What holds

> **Measured and partly overturned, 23 Aug 2026.** The bijection argument is correct
> about the optimum and does not survive as advice: marginal Gaussianization, the
> strongest normalization §1.1 licenses, is **worse on 8 of 8 readable cells** by 0.33
> to 2.33 nats of stitching delta. phi is a bijection, so it cannot move the optimum,
> but it does reweight the finite-step objective, and squashing the massive-activation
> channels down-weights exactly the channels the downstream model needs. `docs/02`
> "Marginal Gaussianization makes it worse". The rest of the section stands, and
> `contract: zca` is the default.

**The bias analysis (§1.1) is the strongest part of the document.** "A penalty tilts the
optimum; a bijection applied to both sides moves nothing" is the right dividing line,
and it is what lets normalization be as aggressive as needed (ZCA, marginal
Gaussianization) without a mismatch term. Retargeting the CF sketch to the teacher's
empirical CF rather than N(0,I) follows from the same line. Keep it verbatim.

**Separating Claim A from Claim B** and insisting on a from-scratch comparator with
identical data (Pythia) is correct and rare. The four mandatory controls are the right
four. The confound protocol (dominated resources, pre-declared corpus, decontamination,
corpus swap) is what a skeptical reviewer would ask for.

**β as the single load-bearing number** is the right framing and the right place for the
kill criterion. Phase 1 is cheap and it is where the project lives or dies.

**The verified literature (§4)** is consistent with what I know: whole-network noise KD
fails, decomposition makes noise viable (Neighbourhood Distillation, Haroush), the
interface-contract use of SIGReg appears open. The verification pass is done: `docs/00-literature.md`. All 22 ids resolve; nothing
found preempts the idea; two baselines (Puzzle BLD, self-generated-text KD) and two
citation corrections (Minitron figures are in 2408.11796; Nano 2 Base used ~480B
tokens) come out of it.

**The arithmetic in §2 checks** under one reading of "sample" (see 2.1). Harvest
2·30e9·1e7 = 6e17. Noise 2·30e9·1e8 = 6e18. Stage memory (2 GB int8 teacher stage +
2 GB student with AdamW) checks for 16 stages of a 32B model. Top-k store checks and is
smaller than stated: Pythia's vocab (50304) fits uint16, so 10^7 × 64 × (2+2) B = 2.6 GB.

## 2. What is inconsistent, wrong, or missing

### 2.1 "Noise sample" is undefined, and Claim A's FLOPs clause flips on it

§1.3 says ~10^8 noise samples per stage. §2 costs the noise phase as 6e18 FLOPs, which is
only right if a sample is **one position** (one d-vector). But every stage in a
transformer contains attention, and attention needs a **sequence**. If a sample is a
sequence of L=2048 positions, the noise phase is 2048x larger, ~1.2e21 FLOPs, which
**exceeds** the from-scratch budget of Pythia-410M (6 · 4.1e8 · 3e11 ≈ 7.4e20). Claim A's
"comparable-or-less total FLOPs" is then false by construction.

Under sample = position, 10^8 positions is ~5·10^4 sequences of 2048. The arithmetic
holds and the FLOPs clause is satisfied by three orders of magnitude (Tier 0 total
≈ 8e17 vs 7.4e20; `docs/compute.md`). **Recommendation: define sample = position, count
sequences separately, and report both.** Phase 1 should also vary L, because shortening L
to save compute changes the attention statistics the stage sees.

### 2.2 Anchors per interface: 10^6 or 10^7?

§1.3: "10M real activation anchors at every interface". §2 storage table: "~1M
positions/interface, ~5 GB/interface". Both cannot be true. At d=5120, 10^7 anchors per
interface is 51 GB per interface, ~800 GB for 16 interfaces. **Recommendation: 10^6 per
interface, and a different 10^6 subset of the 10^7 at each interface** so that, across
interfaces, all real tokens are used somewhere. For Tier 0 (Pythia-1.4B, 2.8B) the
teacher fits the laptop and real activations can be recomputed live, so the store is
exercised for its gate but not relied on until Tier 1.

### 2.3 "Stages run in separate concurrent Kaggle sessions" is not available

Measured on this account (`.claude/skills/kaggle-notebooks`, `es/docs/06`): **at most 2
concurrent GPU batch sessions** (each 2x T4), ~30 GPU-h/week, **at most 1 TPU session**,
~20 TPU-h/week, 2.5-4 h TPU queue, 9 h cap. Sixteen concurrent stage sessions do not
exist. Parallelism is 2 Kaggle sessions + the laptop. The document also never mentions
the **local RTX 3080 Laptop (16 GB, Ampere)**, which is the best Phase 0-1 device
available: no cap, no queue, bf16 tensor cores the T4s lack, and the `es` project
already ran a 13 h sweep on it. The corrected calendar is in `docs/compute.md`; Tier 0
still fits in about a week of wall-clock.

### 2.4 Hadamard rotation before fp8 is the wrong trick for this store

§2 rotates anchors with a Hadamard transform before fp8 quantization. Two problems.
Pythia-2.8B has d=2560 = 128*20 and OLMo-2 has 5120, neither a power of two
(QuaRot-style code needs H_128 ⊗ H_20), which put Plan B on Phase 0's critical path.
And, **measured 22 Aug 2026** (`docs/01` C0.3, `tests/test_anchors.py`): on a stream
with a 200x massive-activation channel, rotation makes fp8 *worse* (0.026 → 0.31
per-channel relative error) because it spreads the outlier's quantization noise onto
every small channel, a trade that only pays for integer codes with no per-channel
scale. Per-channel std normalization plus int8 gives 0.007 at the same byte per
element. The store is int8 + per-channel std, no rotation; Plan B's FWHT has no
workload here. The synergy claim in §6 should go.

### 2.5 Substrate: SmolLM2 is the wrong first substrate

§3 offers "SmolLM2-1.7B or Pythia-2.8B" for Phase 0. Fetched configs:

| model | layers | d | heads | kv heads | note |
|---|---|---|---|---|---|
| Pythia-410M | 24 | 1024 | 16 | MHA | from-scratch comparator, 300B Pile |
| Pythia-1.4B | 24 | 2048 | 16 | MHA | same depth as 410M: width-only bridge |
| Pythia-2.8B | 32 | 2560 | 32 | MHA | 8 stages of 4/3 vs 410M |
| Pythia-6.9B | 32 | 4096 | 32 | MHA | |
| SmolLM2-1.7B | 24 | 2048 | 32 | MHA | |
| SmolLM2-360M | 32 | 960 | 15 | 5 (GQA) | **deeper** than the 1.7B |
| OLMo-2-1B | 16 | 2048 | 16 | MHA | comparator, 4T tokens |
| OLMo-2-7B | 32 | 4096 | 32 | MHA | 8 stages of 4/2 vs 1B |
| OLMo-2-13B | 40 | 5120 | 40 | MHA | 8 stages of 5/2 vs 1B |
| OLMo-2-32B | 64 | 5120 | 40 | 8 (GQA) | 16 stages of 4/1 vs 1B |

SmolLM2's small models are deeper than its large one and trained on different mixes,
so there is no clean from-scratch comparator and the depth grouping runs the wrong way.
**Pythia-1.4B → 410M-config is the cleanest Phase 1 substrate in existence**: identical
depth, identical corpus and order, width bridge only, 3.4x. Phase 3 then adds the depth
grouping with 2.8B → 410M (6.8x). Both teachers fit the laptop in fp16 (2.8 GB and
5.6 GB), so Phases 0-3 run without the streaming path being on the critical path,
while the streaming path is still built and gated in Phase 0 for Tier 1.

### 2.6 The width bridge has a floor that nobody measures

§1.4 projects the student onto the teacher's top-d_S principal subspace. The student's
*input* is then the projected interface, and whatever the next teacher stage reads from
the discarded directions is unlearnable by the student. That floor is independent of
noise and must be measured before attributing any gap to the input measure. Phase 1 adds
a **bridge oracle**: a linear probe from the projected input to the teacher's next
interface, and the same probe from the full input. The difference is the bridge floor.

### 2.7 Exp 1b is not a sub-experiment

Every Pythia stage contains attention, so "sequence structure" is not a follow-up to the
measure-mismatch question, it is a factor of it. Phase 1 crosses input measure with
sequence structure from the start (`docs/02`). The document's own risk 2 says as much.

### 2.8 Power-law fit needs a floor term

§3 fits ε(Q) ≈ c·Q^(-β). Both the real arm (student capacity) and the noise arms
(measure mismatch) have a nonzero asymptote, and a pure power law fitted through a
plateau reports a β that drifts toward zero as Q grows. Fit ε(Q) = c·Q^(-β) + ε_∞ and
report both. The kill criterion is then about ε_∞ of the noise arm relative to the real
arm, which is what it should be about (`docs/02` G1).

### 2.9 Minor

- The Pile is no longer hosted by EleutherAI. `monology/pile-uncopyrighted` (HF) is the
  usual substitute; Pythia's exact batch order is reproducible from the `pythia` repo.
  Decide which in Phase 0 and pre-declare the slice by index.
- "Overnight T4" for a 30B harvest: a 30B teacher does not fit a T4 in any precision
  without streaming, and Kaggle's 9 h cap with ~60 GB of scratch makes the per-stage
  dump-and-reload awkward. Tier 1 harvests belong on the laptop (streaming from disk,
  no cap) or in the TRC window.
- The repo remote was `github.com`; the tree CLAUDE.md §1.1 requires the
  `specgithub.com` alias so the right key is used. Fixed 22 Aug 2026 (local config only).
- `torch` in the env is CPU-only. The CUDA build is a Phase 0 setup item.

## 3. Decisions for Andres

1. **Sample = position** (recommended) or sequence. Changes Claim A's FLOPs clause.
2. **Anchors per interface**: 10^6 (recommended) or 10^7.
3. **Phase 1 substrate**: Pythia-1.4B (recommended) or SmolLM2-1.7B.
4. **G1 kill margin**: `docs/02` proposes ε_∞(best noise arm) ≤ 1.5 × ε_real(10^7) in
   whitened relative MSE **and** stitching-loss delta ≤ 2x the real arm's. Set the
   numbers before the data exists.
6. **Pile source**: `monology/pile-uncopyrighted` vs. Pythia's own data tooling.
7. **Second new baseline**: self-generated-text KD at matched compute (`docs/00-literature.md`
   C4). It is what a reviewer will propose instead; run it at Tier 0 or explain why not.
