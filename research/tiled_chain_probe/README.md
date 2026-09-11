# Height-strip tiled Conv chain: implemented and exact

`open_rknpu.tiled_chain.compile_tiled_chain(model, tiles=T, serial=...)` emits an
`[Conv, Relu]*(N-1) + [Conv]` chain whose 8x8 height is split into `T` strips. For every
(layer, strip) pair it writes a native16 program with strip geometry (`tih`/`toh` rows)
through the extracted `native.native_fields(...)` builder, reads the strip's rows of the
previous surface and writes its rows of the next one, and double-buffers the
intermediates. Quantization and activation come from the untiled `chain_n` emitter, so
both containers share one integer reference and one expected file. `K > 1` strips (3x3
kernels) are covered too - see `../tiled_k3_probe/`.

## Board evidence

`tests/board_bench.c`, every one of the 16 inputs in the file (not just the first),
8 runs each, identical input/expected bytes as the untiled
`deep_chain_suite/model002` container (`board_results.txt`, `board_results.json`):

| variant | tasks | mode | min | exact |
| --- | --- | --- | --- | --- |
| `tiles1` | 16 | serial | 200.4 us | yes |
| `tiles1` | 16 | one job | 70.3 us | yes |
| `tiles2` | 32 | serial | 382.4 us | yes |
| `tiles2` | 32 | one job | 93.6 us | yes |
| `tiles4` | 64 | serial | 847.6 us | yes |
| `tiles4` | 64 | one job | 150.8 us | yes |

All six reproduce the untiled chain's expected bytes exactly, so the strip geometry, the
per-strip double buffering, the 16-lane weight/bias packing, the quantization chain and
the linked one-job tails are correct.

`pre-s7` is the container this directory held before the S7 pad/activation fix
(`model_tiles2_pre_s7.bin`): it applied the graph's hidden Relu at a time when the
untiled `chain_n` emitter did not (`docs/plans/pipelining-plan.md` S9), so it is retained as the
counter-example for that finding. After the S9 fix the current containers apply the
hidden Relu too (on a 1x1 chain the clamp registers are inert on this IP, so the
retained pre-s7 container is still exact on the board).

## What it shows

* Tiling is **available and exact** for 1x1 and 3x3 kernels, serially or as one job.
* At 8x8 it is **slower** per inference than an untiled chain when submitted serially
  (roughly linear in the task count), because each task carries a fixed program/DMA cost
  that dominates a few microseconds of arithmetic. Submitted as one linked job the tiled
  forms are faster than their own serial submission but still slower than an untiled
  single job (151 us for 64 tasks against 70 us for 16).
* It does **not** save arena at this size either: every task program is 1088 bytes, so
  the payload grows faster than the strip surfaces shrink (28,672 / 45,056 / 81,920 B
  against 40,960 B untiled). The memory win would need maps where surface bytes dominate
  program bytes, or strips whose input fetch must stay under the 6144-atom limit.

So tiled chains stay an available option rather than a default.

## Bounds

`chain_n`'s padded 1x1/3x3 kernels, hidden channels 3..16, tiles dividing 8, and the
loader's 64-task table (16 layers x 8 strips is refused). `tiles=1` is the single-strip
control; `K > 1` uses one shared double-buffered surface per layer instead of isolated
strip surfaces (`../tiled_k3_probe/README.md`).

Reproduction:

```sh
PYTHONPATH=src python3 research/run_tiled_chain_probe.py --iterations 8 --cases 16
PYTHONPATH=src python3 -m pytest tests/test_tiled_chain.py -q
```
