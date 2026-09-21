# The fourth pod: step 1 of the data-limited protocol

Record of the rental that runs `docs/08` step 1. Same shape as `docs/05` and `docs/07`.

| | |
|---|---|
| pod | `lwd-step1-a100`, id `vm499pfn1f5b8w`, RunPod SECURE, US-MD-1 |
| GPU | 1x NVIDIA A100-SXM4-80GB, driver 595.91.07, host CUDA 13.2, 250 GB RAM, 16 vCPU |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| stack | isolated venv, `torch 2.13.0+cu130`, **`transformers 5.17.0`**, Python 3.12.3 |
| code | rsync of the working tree at `6ca230c`; the pre-registration is `7f7643c` |
| rate | $1.59/h |
| started | 2026-09-21 17:12:31 UTC |

`transformers` is 5.17.0 here and was 5.15.1 on the August pods: the bootstrap did not pin
it. Every arm of this protocol runs on this pod, so no comparison inside it is affected,
and the bootstrap pins 5.15.1 from now on. Any number from this pod that is set beside an
August number differs in the library as well as the machine.

## Timeline

- 17:12:15 a first pod (`sqc7aw447gsro4`) came up on a host reporting CUDA **12.8**. The
  bootstrap pins `torch 2.13.0+cu130` to match every earlier run, which needs a 13.0
  driver, so it would have failed at the CUDA assert about five minutes in. Deleted after
  16 seconds and recreated with `gpu.minCudaVersion: "13.0"`, which the create call
  supports and which should be in every future request.
- 17:13:41 chain launched. Bootstrap, slice (sha `05e6be5e`, checked) and the 49-row
  budgeted harvest done by 17:18:24: under five minutes, because a budgeted harvest reads
  49 rows and not 5119.
- 17:20:42 488-row harvest done, 36 cells queued on 3 workers.
- 17:24 each cell holds about 10 GB with the full teacher resident, half of what was
  budgeted, and three of them left the card at 80%. GNU xargs takes `SIGUSR1` to add a
  worker without a restart; a fourth took the card to 100% and total throughput from 94k
  to 99k positions/s. A fifth would add nothing. (Two signals sent, one took effect.)
- 17:33 the third-seed pass queued behind the 36 cells, so the pod never waits for a
  decision: it reads which control and which treatment validation chose at each budget and
  runs seed 2 for those.

- 18:00 the first cell finished and showed the selection rule was flawed (`docs/08`,
  Amendment 1). Queue stopped at 18:01, the three finished treatment cells quarantined as
  `cells-rule-v1`, trainer rewritten and tested, relaunched from zero at 18:07:38. Cost of
  the false start: about $1.35.

- Provenance gap, noticed at 18:15: the pod holds an rsync of the tree without `.git`, so
  `git rev-parse` returns nothing there and the step 1 cell records carry an empty `sha`.
  The tree synced for the relaunch is the content of `ff1e9f8` (the amendment commit, made
  minutes later with no further change to the trainer). `cell.py` and `heal.py` now fall
  back to a `SHA` file written at sync time, which step 2 will carry.

- 19:12 the four 49-row control cells moved ahead of the queue's order in a side runner.
  The queue had the long cells first, so the controls would have arrived five hours later,
  and the m = 0 path had only ever run on the 70m rehearsal.
- 20:36 **the first cell to reach the cap died there**, after 85 minutes: the wd 1.0
  control, which is the arm a fair comparison most depends on. The cap is 6104 steps on a
  validation grid of 100, so with the best at the cap the rewind point is 5504, off the
  grid, and the pruning deleted the snapshot at 5500 that the rewind had to fall back on
  (`max()` of an empty sequence). The unit test had used a cap on the grid. Fixed in
  `c8c2a84` with a test whose cap is not.
- 20:39 every arm that stops on patience was unaffected, so only the cells that could still
  reach the cap on the old code were replaced: three 488-row cells (killed, which the queue
  logs as FAILED; those three lines are deliberate) and both wd 1.0 controls, rerun beside
  the queue on the fix. The third-seed pass now waits for them as well, since validation
  cannot choose the control before the strongest one exists. Waste: about 20 GPU-minutes
  of the killed cells, plus the 85 minutes of the one that died.

## Cost

*pending: read the bill the day after deletion, not at deletion (`docs/05`, `docs/07`).*
