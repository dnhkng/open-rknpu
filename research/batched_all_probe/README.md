# Every container can be one job (S10 residual)

Emitters that build their containers by hand or one task at a time never learned about
`submission='batched'`, so `compile_sequence(..., submission='batched')` used to refuse
them with "the container is serial". Since S10 the scheduler runs the container through
`sequence.relink_for_batched(data)` instead: only the task tails (next-program link plus
the successor's fetch amount, `compose.amount_control`) and header flag bit 0 change, and
the programs are untouched. A legacy `ORNPUBIN` container has no task table and no
submission flag, so it is a no-op there - one program is one ioctl either way.

`run_batched_all_probe.py` takes retained, board-verified serial containers from suites
whose emitters never handled the mode, checks that the emitter path and the post-pass
agree where the ONNX still compiles to the retained bytes, and runs both modes over the
suite's own expected files with a clean boot per model.

## Board evidence (16 runs per case, `tests/board_bench.c`)

| container | tasks | serial ioctls | one-job ioctls | serial min | one-job min | exact |
| --- | --- | --- | --- | --- | --- | --- |
| `add_suite` 000 (`Conv,Conv,Mul` graph) | 3 | 3 | **1** | 64.8 us | 29.5 us | yes |
| `mul_batch_suite` 001 | 12 | 12 | **1** | 192.8 us | 99.2 us | yes |
| `mul_batch_suite` 002 (loader depth) | 48 | 48 | **1** | 889.0 us | 445.4 us | yes |
| `mul_batch_broadcast_suite` 003 | 32 | 32 | **1** | 409.8 us | 128.6 us | yes |
| `elementwise_deep_suite` 002 | 5 | 5 | **1** | 73.5 us | 37.9 us | yes |
| `add_geometry_suite` 000 | 3 | 3 | **1** | 49.0 us | 26.0 us | yes |
| `conv_geometry_suite` 000 (single task) | 1 | 1 | 1 | 22.5 us | 21.9 us | yes |

`engine_runs` is what the runtime reports, so the job count is measured. The 32- and
48-task cases cross program-size boundaries (126-word Conv and 78-word elementwise
tasks) 16 times, which is exactly where the old fixed-control table failed.

A sweep over every `research/*/model000.onnx` confirms the mode is never the reason a
profile is refused: **156 of 161 compile** (and the five that do not fail identically
without `submission`, i.e. for profile reasons), while before this change 139 were
refused.

Reproduction:

```sh
PYTHONPATH=src python3 research/run_batched_all_probe.py --iterations 16
```
