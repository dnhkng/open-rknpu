# Cross-job pipelining: non-blocking submission (S4)

The driver's `SUBMIT` ioctl takes job flags (the upstream `rknpu_ioctl.h` (`research/vendor/README.md`)):
`JOB_PC`, `JOB_NONBLOCK`, `JOB_PINGPONG`, `JOB_FENCE_IN`, `JOB_FENCE_OUT`. The runtime
previously only used the blocking form (optionally one task per ioctl), so the CPU sat
in the ioctl for the whole job. This probe measures what the flags actually allow on
the attached RV1103.

## Primitives added

`runtime/open_rknpu.h`:

* `ornpu_submit_flags(model, extra_flags, *fence_fd)` - submits the current model with
  extra driver job flags (`0x2` NONBLOCK, `0x8` FENCE_IN, `0x10` FENCE_OUT); an input
  fence fd is read from `*fence_fd`, the driver's output fence fd is written back.
* `ornpu_sync_outputs(model)` - cache maintenance before reading results of a queued job.
* `ornpu_wait_fence(fd, timeout_ms)` - `poll()` on a driver fence fd.
* `ornpu_set_input(model, tensor_index, data, size)` - pack one external input so the
  next job can be prepared while the NPU runs.

## Measured (tests/board_async.c, 48 interleaved rounds; one round is two inferences)

`board_async <model> <input> <rounds> <expected>` alternates a synchronous round
(A then B blocking) with a pipelined round (queue A with NONBLOCK, run B blocking -
which drains A, since the driver runs jobs in order - then read A's outputs). Drift
from board load hits both modes equally.

| container | synchronous pair | pipelined pair | ratio | outputs |
| --- | --- | --- | --- | --- |
| 3-conv, one batched job | 195 us | 87 us | **2.25x** | exact |
| 3-conv, three serial jobs | 220 us | 148 us | **1.49x** | exact |

`NONBLOCK` was accepted (`rc=0`) and the queued job's output matched the reference and
the expected bytes once drained, for both containers.

## Prerequisite: completion fences

`JOB_FENCE_OUT` (and therefore `FENCE_IN`) returns **-EINVAL**: the running kernel was
built without `CONFIG_ROCKCHIP_RKNPU_FENCE`, so the driver has no fence object to hand
back (`rknpu_job.c` returns -EINVAL for both fence branches under `#else`). Without a
pollable completion fd a pipeline must drain with a following blocking job, which is
what the measurement above does; a single-instance stream that reads each output as it
completes needs either that kernel config or a double-buffered output arena. That is
the documented prerequisite, not an assumption.

## Reproduction

```sh
# cross-compile tests/board_async.c with runtime/open_rknpu.c, then:
adb push board_async /userdata/open-npu-research/async_probe/
adb shell /userdata/open-npu-research/async_probe/board_async model.bin input.u8 48 expected.i8
PYTHONPATH=src python3 -m pytest tests/test_async.py -q
```
