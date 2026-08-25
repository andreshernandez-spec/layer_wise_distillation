# layer_wise_distillation

Stagewise noise distillation: train the stages of a small language model independently,
each fed with noise at the teacher's interface plus a small anchor set, then heal the
composed model end to end. The idea is that stages never have to see real data together,
so the expensive part of distillation becomes embarrassingly parallel.

**It does not pay.** At equal total FLOPs the composed, healed student is worse than the
same architecture trained from random init on the same budget, by 0.213 nats of held-out
next-token loss against a 0.068 within-arm spread. Gate G2 fired on 24 Aug 2026 and the
pipeline stops there. `docs/06-g2-verdict.md` is the verdict and the reasoning.

What the project produced instead is a measurement, and it is the reason the repo is
worth reading.

## The result

Four interventions were tried. Each is ranked cleanly by a metric taken at the stage
interfaces, and every one of those rankings collapses by roughly an order of magnitude
once the composed model gets a modest amount of end-to-end training:

| intervention | measured at the interface | after a 1e7-token heal |
|---|---|---|
| anchors against noise (G1) | 2.0 nats of stitching delta | recipe choice worth 0.24 |
| anchors only against anchors+noise | 2.85 nats unhealed | 0.24 |
| DAgger, on-policy retraining | 41% less composed drift | about 0.1 |
| the stagewise stack itself | eps 0.41 to 0.46 per stage | **-0.21, it loses** |

**No claim here should be quoted from eps, drift or a stitching delta without the healed
number beside it.** Selecting distillation recipes on interface-level proxies is common
practice, and this is an instrumented case of that practice pointing the wrong way.

Supporting measurements, each with a committed script, a recorded environment and a seed:

- **β ≈ 0.32** in ε(Q) = c·Q^(−β) + ε_∞, and the arms that work agree inside each other's
  confidence intervals: live real 0.326, recycled anchors 0.336, the noise recipe 0.313.
  Isotropic noise, whose inputs carry no covariance structure, is the exception at 0.206.
  The CIs are wide (upper bounds 0.48 to 0.58), so this pins the exponent's scale, not
  its third digit. `docs/02`.
- **The anchor crossover is at about 250k real positions per interface.** Below it noise
  is worth up to 0.77 nats; at 0.95M anchors it is worth nothing. `docs/02`.
- **Noise is a regularizer against over-fitting the interface objective**, not a
  substitute for data: it does nothing at 1e8 training positions and wins by 0.297 nats
  at 3e8, where anchors-only starts degrading end to end while ε still improves.
  A stage must not be stopped on ε. `docs/02`.
- **HT-SR α does not work as a per-stage gate.** Over 112 students it correlates +0.80
  with stitching across the pool, which is a training-length artefact; inside a fixed
  budget the sign flips. `docs/02`.
- **Composition error accumulates, it does not compound.** Five of six stages contract
  what they inherit (teacher Lipschitz 0.43 to 0.84) and each adds a roughly constant
  0.54 of fresh error. The ratio-product prediction is wrong by 40x at the second
  interface. `docs/03`.
- **The teacher's first stage amplifies by roughly 50 to 70x**, which is why stage 0 is
  the hardest to fit (Jacobian cosine 0.038) and why noise should not be injected at
  interface 0. Two platforms give 48.5 and 72.7: the estimator perturbs two sequences,
  which is enough for the contractive stages (they agree within 3.1%) and not for this
  one. The order of magnitude is the result. `docs/03`.
- **Exposure bias is most of the fresh per-stage error**: on-policy retraining cuts
  drifted ε by 53 to 74% on stages 1 to 5, and 8% on stage 0, which has none to correct.
  `docs/03`.

Substrate throughout: Pythia-1.4B, six stages of four teacher blocks, 2-block same-width
students. Two rented A100s and an RTX 3080 Laptop, $39 of compute in total.

## Layout

- `PLAN.md`: phases, gates, risk register, and every settled decision with how it was
  settled. Phase 3 onward is kept as written but did not run.
- `docs/`: `00` audit of the source document and the literature check, `01`-`03` the
  phase records, `04` the G1 verdict, `05` and `07` the two rented-A100 campaigns
  including every incident, `06` the G2 verdict, `results.md` the ledger.
- `src/lwd/`: harvest (interface statistics, int8 anchors, top-k log-probs), contract
  (whitening and marginal Gaussianization), noise samplers, stage trainer, composer,
  heal, eval.
- `experiments/`: `phase0/` to `phase2/` drivers and configs, `pod/` the rented-node
  scripts, `tools/` the memory guard.
- `claude_stagewise-noise-distillation-plan.md`: the source document, kept verbatim.
  `docs/00-plan-review.md` records which of its claims survived measurement and which
  did not.

## Reproducing

Environment: conda env `open-source` (see the tree-level `CLAUDE.md`),
`pip install -e ".[harvest,dev]"`. Tests: `pytest --fast` (structural) or `pytest`
(39 tests, CPU only, no network).

Every figure in `docs/` traces to a JSON under `out/` carrying the commit SHA, the device
and the seed. The `out/` tree is gitignored; the two A100 campaigns are described in
`docs/05` and `docs/07` with the pod, image, and package versions.

A note on the incident logs in those two documents: they are long on purpose. Six of the
failures in Phase 2 were runs that did less work than they were asked for and reported
success, and every one of them biased toward a more favourable result. The guards that
came out of it (a store that refuses to under-deliver, a result file that will not be
silently overwritten, a record that hashes the checkpoints it loaded) are the part of
this repo most likely to be useful elsewhere.
