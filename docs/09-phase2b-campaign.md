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

## Cost

*pending: read the bill the day after deletion, not at deletion (`docs/05`, `docs/07`).*
