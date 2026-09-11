# Fence-free completion: the barrier job (S4 residual)

The attached kernel was built without `CONFIG_ROCKCHIP_RKNPU_FENCE`, so `JOB_FENCE_OUT`
returns `-EINVAL` (`tests/board_async.c`, `board_results.json`) and there is no pollable
completion fd. Jobs still run **in order per core**, so a small blocking submission after
a queued one completes only after it:

```c
ornpu_submit_flags(model, ORNPU_JOB_NONBLOCK, NULL);   /* queue the inference */
ornpu_run(barrier, barrier_in, ..., barrier_out, ...); /* blocking: after it */
ornpu_sync_outputs(model);                             /* outputs are final */
```

That is a single-instance stream with **lag 0** - each output is read as soon as it is
produced - instead of the lag-1 queue-then-drain pipeline that needs a second full
inference. The barrier costs one small job; the probe uses
`conv_geometry_suite/model000.bin` (one 126-word Conv task, 24 us median).

## Measured (`tests/board_barrier.c`, 64 iterations, 0 mismatches everywhere)

| container | tasks | synchronous | barrier-completed | barrier job | delta |
| --- | --- | --- | --- | --- | --- |
| `mixed_head_suite/model001` (serial, mixed engines) | 7 | 183.5 us | **134.5 us** | 23.9 us | **-27%** |
| `deep_chain_suite/batched/model002` (one job) | 16 | 107.3 us | 134.5 us | 24.8 us | +25% |

Reading the two rows together:

* for a **serial** container the barrier pattern is *faster* as well as lag-0: queueing
  the whole list non-blocking and paying one barrier wait replaces N blocking ioctl
  round trips (183.5 -> 134.5 us);
* for a container already submitted as **one job** the barrier is pure overhead
  (107.3 -> 134.5 us, i.e. the ~25 us barrier job), and the lag-1 drain pipeline - which
  overlaps the next real inference instead of a no-op - is the better throughput choice
  (`board_async.c` measures 1.13x on a 6-task v5 container here, 1.5-2.25x in the S4
  suite).

So the *functional* gap the residual described ("read each output as it completes") is
closed without a kernel change; a driver-level fence fd stays the prerequisite only for
infrastructure the barrier cannot provide - sharing the NPU between processes and
dma-buf style producer/consumer sync.

## Evidence

* `board_results.json` - parsed measurements per case plus the raw board output;
* `board_results.txt` - the same, human readable;
* the fence re-check: `fence_out rc=-22`, i.e. still `-EINVAL` on this kernel.

Reproduction:

```sh
PYTHONPATH=src python3 research/run_barrier_probe.py --iterations 64
```
