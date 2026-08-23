# layer_wise_distillation

Stagewise noise distillation: train the stages of a small language model
independently, each fed with noise at the teacher's interface plus a small anchor set,
then heal end to end. The question the project answers first is how fast noise-trained
stages approach real-activation-trained ones (the exponent β in `docs/02`).

- `PLAN.md`: phases, gates, risks.
- `docs/`: review of the source document, literature check, phase docs, compute.
- `src/lwd/`: harvest (interface statistics, fp8 anchors, top-k logits), contract,
  noise, eval.
- `experiments/phase0/`: the harvest drivers and their configs.

Environment: conda env `open-source` (see the tree-level `CLAUDE.md`),
`pip install -e ".[harvest,dev]"`. Tests: `pytest --fast` (structural) or `pytest`.
