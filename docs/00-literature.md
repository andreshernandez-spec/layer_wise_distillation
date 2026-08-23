# 00: Literature verification (22 Aug 2026)

Background pass over the source document's §4, run by a research agent with web
access on 22 Aug 2026. Every arXiv id was fetched from `arxiv.org/abs/<id>`; every
number below has the URL it came from. Items marked *unverified* were not read from a
PDF. Re-check anything from March 2026 onward before citing: it is past the agent's
training data and rests only on the fetched pages.

## A. arXiv ids in the source document

All 22 ids in §4 resolve to the claimed papers. Also verified: 2603.05924 Weak-SIGReg,
2602.02859 anti-grokking (Prakash, Martin), 2506.14562 AlphaDecay, 2410.10912
AlphaPruning. Cite OLMo 2 as "Team OLMo et al."

## B. Token counts (Claim B denominators)

| pipeline | distillation tokens | source |
|---|---|---|
| Llama-3.1-Minitron-4B (from 8B) | 94B (+ ~100B teacher correction) | model card; 2408.11796 Table 1 |
| Mistral-NeMo-Minitron-8B (from 12B) | 380B (+0.1T teacher fine-tune) | model card; 2408.11796 Table 1 |
| **Nemotron Nano 2, 12B-v2-Base → 9B-v2-Base** | **~480B** (120B depth KD + 360B width KD + 2.5B long-context) | 2508.14444 §4.3; 9B-v2-Base model card |
| Nemotron Nano 2 aligned 9B-v2 (not Base) | ~136B across five stages | 2508.14444 §4.3 |

Two corrections to the source document. The 94B/380B figures come from **2408.11796**
("LLM Pruning and Distillation in Practice"), not 2407.14679 (Nemotron-4 15B → 8B/4B);
swap the citation. And the Nano 2 **Base** denominator is ~480B, *larger* than
Minitron-8B's 380B, so the Claim B ratio vs. Nano 2 is more favourable than the source
document assumed, but only if the comparison is Base-to-Base; the aligned 9B used ~136B
and the two pipelines must not be mixed.

Sources: https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base,
https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base,
https://arxiv.org/html/2408.11796, https://arxiv.org/html/2508.14444v2,
https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base

## C. Prior art

### C1. Noise-fed blockwise distillation of transformers: **not found**

arXiv title/abstract sweeps and web searches found no paper that trains transformer or
LLM blocks by distillation with random or moment-matched noise at the block input.

- **2010.01189 Neighbourhood Distillation** (Shao et al., 2020) is the origin: CNN
  sub-networks distilled independently, "can be successfully adapted to use gaussian
  noise inputs" and beats end-to-end baselines under noise. No transformer, no moment
  matching, no distribution term. OpenAlex reports **0 citations**; Semantic Scholar
  rate-limited (429); Google Scholar not queried. Nobody appears to have followed it
  up. Position paper one as its transformer/LLM extension.
- **Raikwar & Mishra, NeurIPS 2022** ("Discovering and Overcoming Limitations of
  Noise-engineered Data-free KD", no arXiv id; code `Piyush-555/GaussianDistillation`).
  End-to-end KD under Gaussian input noise; "the shift in the distribution of hidden
  layer activation" is the failure. This is the source document's "OpenReview
  K8JngctQ2Tu" item. It is the direct motivation for injecting noise *at the interface*,
  moment-matched. Their mitigation not read (*unverified*).
- **2411.19146 Puzzle** (Bercovich et al., NVIDIA, 2024). "Blockwise local knowledge
  distillation": each child block trained independently against its parent block,
  normalized MSE, gradients isolated across blocks, pipeline-parallel, **~1B real tokens**.
  This is the production version of independent per-block distillation for LLMs, on
  real activations. **It is the baseline paper one must name and beat at matched
  budget**: BLD on real activations at the same real-token budget is the source
  document's control (2), and Puzzle gives it a citation and a recipe.
- Adjacent only: 2310.04550 Module-wise Adaptive Distillation, 2402.16918 m2mKD
  (module-wise, real data); 2506.14202 DiffusionBlocks (independent blocks, no
  teacher); 2403.19135 LLM-Streamline, 2605.15491 Ghosted Layers (replacement layers
  on real calibration activations); 2012.03096 and 2301.12443 (blockwise KD, CNNs, real
  data); PSAQ-ViT family (2203.02250 and successors: synthesize *inputs* from noise for
  ViT quantization, not interface noise). NAYER's "noisy layer" is a generator
  component, unrelated to the mechanism despite the keyword.

### C2. Sketched-CF feature distillation: **not found**

arXiv abstract search for "characteristic function" with "knowledge distillation" or
"feature distillation": 0 results. Nearest:

- 2412.08139 WKD-F: per-sample diagonal-Gaussian 2-Wasserstein feature KD (parametric
  second moments; the CF sketch is its non-parametric generalization).
- 2504.01757 KD²M: distribution-matching framework for feature KD (vision); metric
  list *unverified*.
- 2502.20653 NCFD: neural CF discrepancy as a loss for *dataset* distillation.
- 1909.07425 OCFGAN and 2006.08413: CF distance for generative models, linear time.

Framing note: SIGReg in LeJEPA matches to a **fixed isotropic Gaussian**. The plan's
teacher-vs-student term is a **two-sample** CF test, closer to OCFGAN's CFD than to
SIGReg proper. Describe it that way in the paper; "SIGReg retargeted" undersells the
difference and invites the wrong comparison.

LeJEPA follow-ups (40 abstracts through 2026-08-18): none touches distillation or
interface priors. Relevant: **2607.17019 "Regularize or Localize"** applies SIGReg to
LM hidden states and KV cache during (continued) pretraining and reports reduced
anisotropy, which is evidence that a sketched-CF term on LLM hidden states is trainable
at scale. 2606.02572 VISReg (sliced-Wasserstein sketch), 2605.26900 / 2606.17603
SPHERE-JEPA (analytic sketch replacements), 2602.01456 (non-Gaussian target): alternative
statistics, not KD.

### C3. HT-SR alpha as a training loss: one toy-scale attempt

- **2304.02911** (Xiao, Li, Xie, Zhou, 2023): differentiable alpha via a Hill estimator
  at k = n/2, penalties on weighted alpha and stable rank, FC3/LeNet5/ResNet18 on
  KMNIST/CIFAR10. Reported that penalizing throughout training "will be
  over-regularized"; needed decayed or thresholded variants. No transformer follow-up
  found.
- Everything else in the Martin/Mahoney line uses alpha as a controller, never a loss:
  2312.00359 TempBalance (LR), 2410.10912 AlphaPruning (sparsity), 2506.14562 AlphaDecay
  (weight decay), 2602.02859 (diagnostic).

The honest sentence for the paper: "direct alpha penalties are untested beyond
CIFAR scale, and the one attempt needed gating to avoid over-regularization". Not
"known to fail". This supports the source document's gates-only stance (§1.5, risk 5).

### C4. Data-free KD for LLMs, 2024-2026: self-generated text, end to end

2305.17888 LLM-QAT, 2311.01689 DFKD-T³, 2410.09982 Self-Data Distillation (Cerebras,
pruned Llama-3.1-8B recovery), 2510.08600 / 2606.04238 Recover-LoRA. All substitute
model-generated text for data and train end to end. None decomposes into stages, none
synthesizes interface activations. Surveys 2402.13116 and the 2025/26 AI Review survey
list neither. The second baseline paper one must name: **self-generated-text
distillation at matched compute** (LLM-QAT style), because that is what a reviewer
will say "just do" instead.

## What this changes in the plan

1. Two named baselines for paper one, both to be run at Tier 0: **Puzzle-style BLD on
   real activations** (matched real tokens) and **self-generated-text KD** (matched
   compute). Source document control (2) becomes the first; the second is new.
2. Claim B vs. Nano 2 uses ~480B as the denominator, Base-to-Base.
3. Minitron token figures cite 2408.11796.
4. The CF term is described as a two-sample CF test (OCFGAN lineage), with SIGReg as the
   sketching precedent, and 2607.17019 as evidence of trainability on LM hidden states.
5. Neighbourhood Distillation and Raikwar & Mishra go in the intro as the origin and the
   failure mode the interface injection answers.
