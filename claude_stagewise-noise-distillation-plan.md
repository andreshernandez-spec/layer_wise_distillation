# Plan C — Stagewise Noise Distillation with Interface Contracts

**Prepared:** 22 Aug 2026 · **For:** Andres · **Status:** scoped, pre-Phase-0 · **Companion plans:** A (Chebyshev sparse-attention scoring), B (FWHT JAX library), ES library

Working thesis: distill a large pretrained teacher into a small student by training student **stages independently and in parallel**, driven by **per-stage random noise** (moment-matched, contract-normalized) plus a **small real-token anchor set**, with (i) bijective interface normalization, (ii) a sketched characteristic-function contract at each interface (SIGReg machinery retargeted to the teacher's empirical CF), (iii) HT-SR spectral metrics as per-stage acceptance gates, (iv) sharded ES for the genuinely non-differentiable objective terms, and (v) a short end-to-end heal. The stagewise phase never loads the whole model; only the heal does.

---

## 0. Headline claims and their comparators

Two claim types, deliberately separated:

**Claim A (vs. from-scratch).** Student matches a small model trained from scratch on the full corpus, using ≤10⁷ real tokens + N teacher noise queries (N reported separately), at R× compression, at comparable-or-less total FLOPs. Requires a family where the from-scratch small baseline *exists*:

| Family | Teacher | From-scratch comparator | Data | Notes |
|---|---|---|---|---|
| Pythia | 2.8B / 6.9B / 12B | 410M / 1.4B / 2.8B | identical corpus, identical order, all sizes from scratch, 154 ckpts each | airtight controls; dated absolute quality |
| OLMo 2 | 7B / 13B / 32B | OLMo-2-1B (4T tok) | open (Dolma/OLMo-mix); token counts differ mildly by size (4T/4T/5T/6T) — state honestly | the modern option |

**Claim B (vs. incumbent prune+distill pipeline).** Student matches the lab's own compressed model from the *same teacher* at a fraction of the real tokens.

| Pair | Ratio | Why / why not |
|---|---|---|
| **Nemotron Nano 2: 12B-v2-Base → 9B-v2-Base** | 1.33× | **Primary.** Same teacher literally; recipe documented (extract exact distillation token count from the Nano 2 report); teacher's data partially open → the 10M anchors can come from the teacher's own distribution |
| Llama-3.1-8B → Llama-3.2-1B | 8× | Recognition value; distillation token budget undisclosed → efficiency claim goes qualitative |
| Llama-3.1-Minitron-4B (≈94B tok), Mistral-NeMo-Minitron-8B (≈380B tok, from 12B) | 2× / 1.5× | Citable token denominators (verify exact numbers in the Minitron reports) |
| Qwen3 14B/32B → 1.7B-config | 8–19× | **Excluded for now**: corpus closed (36T, synthetic-heavy, described in TR arXiv:2505.09388 but not released), distillation denominator undocumented, and their 1.7B was distilled from the *flagship* teachers, not the ones I'd use. Revisit only after strong results; the dense ladder + MaxText support keep it attractive as substrate, not comparator. |

**The controlled version of Claim B (scientific core):** run the incumbent pipeline *myself* at Tier 0/1 — token-matched Minitron-style prune+distill on the same Pythia/OLMo teacher and corpus vs. the stagewise route. Data held fixed by construction (~1–2×10²⁰ FLOPs for a 7B→1B baseline at 10–20B tokens ≈ one TRC window or ~$100-300 spot H100).

**Confound protocol (data-quality hole).** At 10M real tokens the data budget cannot carry capability; capability flows through the shared teacher. The confound is asymmetric: stay on the dominated-resources side (same teacher, fewer tokens, no-better corpus) and a win is strengthened, not undermined. Mandatory: pre-declare the anchor/heal corpus, decontaminate against the eval suite, and run a corpus-swap ablation on the 10M (cheap) to show insensitivity. The residual risk at this scale is *benchmark adjacency*, not general quality.

**Mandatory controls:** (1) random-init + identical heal budget (the "better than nothing" bar — if not beaten, no paper); (2) stages trained on real activations (oracle ceiling); (3) pure-noise, no anchors (ablation — anchors are the recipe, pure-noise is the ablation, not vice versa); (4) corpus swap on the 10M.

**Capacity-gap risk:** distillation scaling-law work (Busbridge et al. 2025, verify arXiv:2502.08606) finds over-strong teachers hurt students past a gap threshold; 8–35× is untested territory. Stagewise decomposition (each student stage matches a *group* of teacher layers, not the whole function) is the claimed mitigation — measure it, don't assert it.

---

## 1. Settled design decisions

### 1.1 Transform, don't tilt (bias analysis)

The dividing line: **a penalty tilts the optimum; a bijection applied to both sides moves nothing.**

- Student stage k's target is defined in transformed coordinates: `T̃_k = φ_{k+1} ∘ T_k ∘ φ_k⁻¹` with fixed invertible `φ`. Matching `T̃_k` exactly is matching `T_k` exactly. Composition telescopes (`φ⁻¹∘φ` cancels at interfaces); affine `φ` folds into neighboring linear layers at deployment.
- Normalization strength is therefore unbounded and free of mismatch: whitening (ZCA from harvested stats) and even per-channel marginal Gaussianization (RBIG-style monotone CDF maps from anchors) are bijections. This preserves outlier/massive-activation channels (load-bearing: attention sinks) — squashed monotonically, never suppressed.
- Regularizers as *losses* must share the mimic optimum. SIGReg-toward-N(0,I) does **not** (teacher's whitened interfaces aren't exactly Gaussian → bias floor). Fix: retarget the sketched-CF statistic to the **teacher's empirical interface CF** (see 1.2). A fixed Gaussian prior may be used only annealed-to-zero for early-training stabilization.
- Gates and acceptance criteria (α-band, CF-distance thresholds) never bias an optimum — rejecting a run isn't tilting it. Spectral/HT-SR machinery lives on this side of the line.

### 1.2 Interface contract

- **Hard moments:** explicit whitening adapter per interface from harvested mean/cov (shrinkage-regularized; ~2M numbers at d≈2048 from ~1M positions).
- **Soft shape:** sketched-CF penalty. Fix M projection directions × K frequencies per interface; during the harvest pass accumulate the teacher's cos/sin means → an M×K×2 sketch, kilobytes per interface, free. Same statistic doubles as (a) training loss term with the teacher as common optimum, and (b) live per-stage contract-violation gauge.
- Design fork (ablation, not a decision): Gaussian-contract stages (self-consistent with each other, biased vs. teacher) vs. teacher-CF stages (consistent with teacher, small realized input drift into the next stage). Default: teacher-CF + DAgger-style on-policy mixing to absorb drift; the heal erases residual interface bias (low-dimensional, smooth error — cheap for fine-tuning to fix).

### 1.3 Noise and anchors

- **Budget:** ~10⁷ real tokens + ~10⁸ noise samples per stage; noise is counted separately from real tokens in every claim. Per-stage independent noise costs the same total FLOPs as shared noise (per-stage FLOPs × per-stage samples sums to one full-model pass) → effectively fresh samples every batch, no epochs, nothing to overfit.
- **Real tokens do triple duty:** ~1M positions → interface statistics; the same single teacher pass → 10M real activation anchors *at every interface*, mixed ~1:10 real:noise during stage training to pin the student on-manifold while noise supplies volume; remainder + stored logits → the heal.
- **Sequence structure is the crux** i.i.d.-per-position noise satisfies the per-token contract but hands attention degenerate statistics (near-uniform attention entropy, no induction structure). Exp 1b races: (i) i.i.d., (ii) fitted sequence process (position-averaged channel covariance + low-rank positional correction, or AR(1) across positions), (iii) anchors-only carrying sequence structure.
- **Edge stages:** interface 0 is discrete tokens — noise starts *after* the first block; head stage trains against stored top-k logits, not activations. Interior interfaces host the contract machinery.
- Theory hooks: density-ratio bound (error on true measure ≤ error on noise measure × sup density ratio); Srinivas–Fleuret: output-matching under input noise ≈ Jacobian matching.

### 1.4 Architecture matching and width bridging

- What must match: **interface semantics and stage count.** Internals need not (blockwise distillation is architecture-agnostic — Arch-Net); block types kept identical in paper one as a confound-killer, not a requirement.
- Width bridge: the per-interface whitening/PCA projection doubles as the dimension bridge — student targets the teacher's **top-d_S principal subspace** (basis free from harvested covariance). Alternative: trainable linear adapter, discarded at deployment, healed after.
- Depth mismatch is handled by grouping: teacher layers per stage ≠ student layers per stage; stage *count* equal.

### 1.5 Spectral gates and ES

- HT-SR usage pattern follows the successful precedents (allocation/diagnosis, never steady-state loss): α-in-band (SETOL: α≈2 ideal; 2–5 healthy) + correlation-trap checks as **per-stage acceptance gates** — the substitute for the downstream validation signal independent stages lack. Optionally TempBalance/AlphaDecay-style per-stage LR/decay allocation.
- Differentiable spectral surrogates exist (stable rank, spectral entropy, fixed-k Hill) — mind the JAX `eigh` JVP degeneracy landmine (see Plan-A research doc appendix).
- ES earns its place for the genuinely discrete terms (xmin selection, KS-fit quality, rank counts) and because stages are small enough for population-based ES: EGGROLL low-rank perturbations per stage, populations across stages, scalar-fitness all-reduce. **This is the ES library's showcase application.**

### 1.6 Composition and healing

- Frame: imitation-learning compounding error (exposure bias). Per-stage error amplifies through downstream Lipschitz constants; measure the amplification profile.
- Mitigations to ablate: DAgger mixing (later stages trained on student-propagated noise; stages 1..k−1 run inference-only, quantized, ~+4GB); BERT-of-Theseus-style stochastic stage swap inside the teacher scaffold; final end-to-end heal on 10M real tokens + stored top-k logits (teacher never resident).

---

## 2. Compute and storage model (verified arithmetic)

| Item | Estimate | Hardware |
|---|---|---|
| Harvest pass: 30B teacher × 10M tokens, stage-resident streaming (GPTQ-style: load stage, push all tokens, dump interface, drop) | ~6×10¹⁷ FLOPs | overnight T4 / minutes v5e-8 |
| Noise phase: 10⁸ samples through all stages (≡ one full-model pass worth per sample set) | ~6×10¹⁸ FLOPs | ≈4 H100-h equivalent, but runs stage-resident on free hardware |
| Stage-training peak memory (32B teacher, 16 stages): 4-layer teacher stage int8 (~2GB) + 125M student stage w/ AdamW (~2GB) + activations | ~8GB | half a T4; stages run in **separate concurrent Kaggle sessions** |
| Anchor storage: ~1M positions/interface, fp8 + per-channel scales, **Hadamard-rotated before quantization** (Plan-B synergy; activations quantize better post-rotation) | ~5GB/interface, ~80GB total | packageable as a public Kaggle dataset |
| Top-k logit store (10M × top-64) | ~4GB | — |
| End-to-end heal: 2B student, AdamW state ~35GB sharded | fits v5e-8 (128GB) | teacher not resident |
| Self-run Minitron baseline (7B→1B, 10–20B tok) | ~1–2×10²⁰ FLOPs | TRC window or ~$100–300 spot H100 |
| Full pipeline through Tier 1 | free hardware | paid H100 only for cross-platform benchmarking (consistent with standing compute strategy) |

TRC timing constraint (standing): ~30-day grants — apply around week 10, not week 1, so the window lands on Phase 4.

---

## 3. Phases, budgets, gates, kill criteria

### Phase 0 — Harvest tooling (1–2 wks, CPU + Kaggle T4)
Streaming stage-resident harvester; interface stats (mean/cov + shrinkage, CF sketches, per-channel quantiles for Gaussianization maps); fp8+rotation anchor store; top-k logit store. Substrate: SmolLM2-1.7B or Pythia-2.8B.
**Gate:** dequantized anchors reproduce teacher next-stage outputs within tolerance (set tolerance from fp8 noise floor).
**Deliverable:** harvested interface dataset (publishable).

### Phase 1 — Exp 1: single-stage measure mismatch (1–2 wks, T4/TPU)
Matched student stages trained on (a) real activations [oracle], (b) moment-matched Gaussian, (c) isotropic noise, (d) contract-Gaussianized noise; all evaluated on held-out **real** activations. Fit ε(Q) ≈ c·Q^(−β) per arm — **β is the load-bearing number of the whole project**: it forecasts Tier-1/2 query budgets and how much of the 4–5-order token gap noise can close.
Exp 1b: sequence-structure race (1.3).
**Kill:** (b) plateaus far above (a) at matched capacity with β too small to close the gap at feasible Q → route requires generative interface models → collapses into existing DFKD → kill or pivot to a measurement paper.
**Gate:** written Tier-1/2 budget forecast from measured β before any paid compute.

### Phase 2 — Exp 2: composition & healing (2–3 wks, Kaggle)
Chain noise-trained stages; drift vs. depth; amplification profile; DAgger and Theseus ablations; heal-budget curve.
**Kill:** healed stagewise ≤ random-init + equal heal budget at equal total FLOPs.

### Phase 3 — Tier 0 full pipeline (3–5 wks, free Kaggle, stages parallel across sessions)
Pythia-2.8B → 410M-config (or SmolLM2-1.7B → ~300M): full contract machinery, α gates, ES for discrete terms. Comparator: from-scratch Pythia-410M. All four controls (§0).
**Decision gate:** match the from-scratch comparator within a pre-registered margin using ≤10⁸ noise + ≤10⁷ real → Phase 4. Partial → the paper becomes "how far does noise get you" (β-measurement + ablations), still publishable.

### Phase 4 — Tier 1 (TRC window + ~$100–300 spot)
OLMo-2-7B or 13B → 1B-config vs. real OLMo-2-1B; **self-run token-matched Minitron baseline on the same teacher/corpus** (the controlled Claim B). Corpus-swap ablation on the 10M.

### Phase 5 — Claim-B headline (conditional on Phases 3–4 extrapolation)
Nemotron Nano 2: 12B-v2-Base → 9B-class vs. NVIDIA's 9B-v2-Base. New work item: hybrid-Mamba stage semantics (Mamba2 state + conv at interfaces — define the interface object before committing). Heal on v5e-8; H100 spot for benchmarking only.

### Phase 6 — Artifacts
OSS toolkit (harvest + stagewise-train; PyTorch/HF hooks for harvest & stage training where friction is lowest, JAX for ES + TPU heal — framework-by-purpose, per standing principle) + paper. Both written by Andres.

---

## 4. Literature status (from the 22 Aug research pass)

**Verified this session (fetched/searched):**
- Blockwise distillation with teacher-fed inputs is established and parallelizes: Pipe-BD (arXiv:2301.12443); Arch-Net (arXiv:2111.01135, explicitly architecture-agnostic, block-coordinate style); DNA/DONNA lineage (blockwise NAS-KD).
- Whole-network noise KD fails; field consensus: generators struggle to map input noise to the data distribution (NAYER, CVPR 2024, arXiv:2310.00258).
- **Closest prior art:** Neighbourhood Distillation (arXiv:2010.01189) — non-end-to-end distillation of sub-networks; Gaussian-noise inputs succeed at small/shallow sub-network scope; cites Haroush et al. 2019 (noise works when the student–teacher gap is small). Decomposition is what makes noise viable. No transformer-scale instance found.
- Mechanism of the failure: hidden-activation distribution shift under Gaussian inputs (OpenReview K8JngctQ2Tu, 2022) — supports the interface-measure framing.
- LeJEPA / SIGReg (arXiv:2511.08544): random 1D projections + CF matching (Epps–Pulley), linear time/memory, provably prevents collapse. Already being repurposed: Weak-SIGReg (arXiv:2603.05924, Mar 2026) as a general supervised-training stabilizer. The **interface-contract use appears open** — window moving.
- HT-SR/RG formalization: SETOL (arXiv:2507.17912) — α≈2 optimal, TRACE-LOG/ERG condition. Usage precedents are allocation/diagnosis, not loss: AlphaDecay (arXiv:2506.14562, per-module weight decay), AlphaPruning (arXiv:2410.10912, layerwise pruning ratios), TempBalance (per-layer LR); anti-grokking detection (arXiv:2602.02859).
- Model/data facts: Nemotron 3 Nano 30B-A3B — 31.6B total / 3.2B active, **trained from scratch on 25T tokens**, 23 Mamba-2+MoE layers + 6 attention layers (arXiv:2512.20848; family report 2512.20856). **No small Nemotron is from scratch at any generation** (9B←12B, 47B←56B by compression). Small Nemotron-3 *configs* (1B-A315M / 2B-A500M / 4B-A770M) exist in Megatron-LM and were trained by third parties only to 88–215B-token subsets (arXiv:2606.00371) — the from-scratch comparator cannot be bought or found in this family. Qwen3: 36T tokens, 119 languages, synthetic-heavy pipeline, **not released** (arXiv:2505.09388). Nemotron open data: 10T+ tokens cumulative; the open subset ≠ the full training mix (recipes note results differ); licenses mixed (NVIDIA Data Agreement gating most, CC-BY-4.0 on e.g. Pretraining-Code-v3).

**From model knowledge — verify IDs and numbers before citing in the paper:**
Minitron (2407.14679) and the ~94B / ~380B distillation-token figures; Sheared-LLaMA (2310.06694); Distillation Scaling Laws / capacity-gap curse (Busbridge et al., 2502.08606); Srinivas & Fleuret, Jacobian matching (1803.00443); DeepInversion (1912.08795); ZeroQ (2001.00281); DAFL (1904.01186); ZSKD (1905.08114); Haroush et al., "The Knowledge Within" (1912.01274); BERT-of-Theseus (2002.02925); model stitching (Lenc & Vedaldi 2015; Bansal et al. 2106.07682); boomerang distillation / layer patching (Kangaslahti et al. 2026); Blockwise SSL at scale (2302.01647); NoProp (2503.24322); greedy layerwise (Belilovsky 1812.11446); Pythia (2304.01373); OLMo 2 (2501.00656); Llama 3.2 1B/3B pruned+distilled from 8B (Meta blog); massive activations / attention sinks (Sun et al. 2024; Xiao et al. 2023).

**Open prior-art checks (assign to a deep-research pass; per the standing "literature closure" principle, verify before Phase 3):**
1. Any transformer/LLM-scale noise-fed blockwise distillation (2023–2026)?
2. Sliced/sketched-CF distribution-matching used as a *feature distillation* loss (vs. sliced-Wasserstein feature KD)?
3. Any attempt to use HT-SR α directly as a training loss, and its failure mode?
4. LeJEPA follow-ups post-Feb-2026 that touch distillation or interface priors?
5. Exact Nano 2 12B→9B distillation token count; exact Minitron token figures.

---

## 5. Risks

1. **β too small (Exp 1).** The kill criterion exists for this; the fallback measurement paper retains value.
2. **Sequence structure.** If neither fitted noise processes nor anchor-mixing fix attention-stage matching, attention stages may need real-activation training while MLP stages use noise — a weaker but honest hybrid claim.
3. **Capacity-gap curse** at 8×+; measure the stagewise-mitigation claim explicitly.
4. **Benchmark adjacency of the 10M** — handled by pre-declaration + decontamination + corpus swap; non-negotiable protocol.
5. **Goodhart on spectral metrics** — gates only; any spectral *loss* must be a differentiable surrogate with an ablation showing it isn't optimized into meaninglessness.
6. **Crowded adjacent niche** (synthetic-data distillation dominates LLM practice). Surviving motivations: data-frugality/privacy (no data egress), depth-parallel training systems story (stages across free sessions is itself a result), interface-contract theory, cheap-init reducing synthetic-token budgets. The "why not just use data" reviewer question gets answered in the intro, not the rebuttal.
7. **Hybrid-Mamba/MoE stage semantics** (Phase 5 only) — do not let it leak into paper one.

## 6. Cross-plan synergies

- **ES library:** per-stage EGGROLL populations for discrete objective terms; showcase application.
- **Plan B (FWHT):** Hadamard rotation before fp8 anchor quantization; the anchor store is a real workload for the kernel.
- **Plan A:** shared model substrates (Qwen3 ladder if it re-enters; OLMo otherwise); shared harvesting code.

## 7. Constraints (standing)

- Andres writes all load-bearing code and upstream-facing prose; AI assistance for research, prototyping, benchmarking only.
- Compute: Kaggle free (dual T4, TPU v5e-8) primary; TRC applied ~week 10; Vast/RunPod/Lambda H100 spot for benchmarking only.
- Every external claim in the paper traces to a verified source or a script in this repo.
