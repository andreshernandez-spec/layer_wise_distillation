# The second rented A100: Phase 2

Companion to `docs/05`. Same shape of record: platform, commits, cost, and every
incident, so that a number in `docs/03` or `docs/06` can be traced to the machine that
produced it.

| | |
|---|---|
| pod | `lwd-phase2-a100`, id `ah6v60cvzktkg0`, RunPod **SECURE**, US-MD-1 |
| GPU | 1x NVIDIA A100-SXM4-80GB, host CUDA 13.0 |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| stack | isolated venv, `torch 2.13.0+cu130`, Python 3.12.3 |
| code | rsync of the working tree; SHAs `6b8f60c` through `33773eb` |
| rate | $1.59/h |
| started | 2026-08-24 11:42:51 UTC |

SECURE rather than COMMUNITY this time: `docs/05`'s pod was community and the campaign
ran long enough that a host reclaim would have cost more than the price difference.

## What it produced

`out/phase2-1.4b-a100/`: 34 result JSONs, the pod logs, and 18 stage checkpoints
(the C_mix stack before and after DAgger, and the R_iid stack). The `dagger_C_stage*.pt`
files are byte-identical to the promoted `C_mix_*` ones and were not pulled.

- 6 stage cells, C_mix at 1e8 positions each (the stack Phase 2 composes)
- drift profiles for R, C, and C after DAgger
- 6 DAgger cells, on-policy 0.5, 1e7 positions each
- the C2.4 heal curve, 12 cells, three inits x four budgets
- 5 schedule probes (peak lr and warmup, both arms)
- the DAgger stack's re-heal at 1e6 and 1e7
- the equal-FLOPs cell at 1.8367e8 tokens

## Incidents

Six, listed with their consequences in `docs/06` under "What broke". In short:

1. `Edges.embed/head` were `@torch.no_grad()`, making the heal a no-op.
2. Twelve heal runs trained on nothing (empty `TopKStore`, harvested with `skip_topk`).
3. The random arm diverged, skipping 4476 of 4883 steps, and still reported a loss.
4. The equal-FLOPs cell delivered one epoch instead of 17 and reported success.
5. The DAgger closing test overwrote three of the plain stack's result files.
6. The equal-FLOPs budget was typed from the wrong line of the FLOPs report (5.3% short).

Plus four watchdog bugs, none of them costing GPU time but all of them costing attention:
`pgrep -c || echo 0` shifting a field, a pattern matching its own shell, a pattern built
from the launch command rather than the actual (relative-path) cmdline, and the subtlest
one, worth writing out.

A progress monitor ran a remote command whose last statement was a `grep` for failure
signatures, and took the ssh exit status as "was the poll reachable":

    pgrep -f heal.py && echo ALIVE || echo DEAD
    tail -1 log | grep -oE "'tokens': [0-9]+"
    grep -lE 'Traceback|under-delivered' log        # <- last statement, sets the status

An ssh command exits with the status of its last remote statement. That statement exits 1
when it finds no failure signature, which is the healthy case. So the poll registered an
ssh failure exactly when the run was fine, and the watchdog counted down toward declaring
the pod unreachable while the pod sat at 97% GPU.

It also produced a wrong diagnosis before the right one. The failures coincided with a
7 GB rsync, so the first explanation was bandwidth contention and the transfer was
throttled on that theory. Timing three ssh connections (2.2 s each, consistently) killed
that explanation and pointed at the exit status. **Measure the thing you are blaming
before acting on the blame.** End every remote branch with `|| true` and the script with
`exit 0`.

## What it cost

Reconciled the day after, per `docs/05`'s correction: the last bucket lands late and
reading the bill at deletion time understated Phase 1 by 7%.

| | |
|---|---|
| uptime | **12.0 h** (11:42:51 to 23:41 UTC, deleted straight after the last cell) |
| cost | **~$19.03** at $1.59/h, to be reconciled against `get-billing` on 25 Aug |

Reconcile tomorrow, not now: `docs/05` recorded $18.95 for the Phase 1 pod by reading the
bill minutes after deletion and the final figure was $20.31.

## What to do differently next time

**The idle detector is not the thing that saves money here; the under-delivery guards
are.** Phase 1's lesson was 39 minutes of idle GPU. Phase 2 had almost no idle time and
still wasted more than that on runs that finished early and reported success: the
truncated equal-FLOPs cell alone burned 21 minutes producing a number that had to be
thrown away, and the twelve empty heal runs burned more. A watchdog that asks "is the GPU
busy?" cannot see any of that. The guard has to be at the point where the number is
produced, and it has to assert the work actually happened.

**Probe at the length you will run.** The lr probe at 1e6 tokens found nothing wrong with
a schedule that diverges at step 1285 of a 1e7 run. A probe shorter than the run tests a
different question.

**Queue anything that promotes checkpoints in place strictly last, and give it its own
output names.** `dagger_stack.py` was correctly queued after the heal curve and still
destroyed three of its results, because the renames meant to separate them were wrapped
in `2>/dev/null` and one pattern was wrong.
