# Join-chain runtime scale — a per-channel gain on a DAG result

A shared 1×1 stem fans out to 3..5 dense/depthwise heads, chained joins fold them,
and a final `Mul(result, scale)` applies a **runtime per-channel scale** whose
operand is the **second external input** (three INT8 codes on the `[1,3,1,1]`
tensor, passed as `code + 128`). This is the first profile with a runtime input
consumed *inside* a fan-out rather than at the graph boundary, and it composes the
P1 chain with the P4 runtime-scale elementwise mode.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Heads | Kernels | Joins | Codes |
| --- | --- | --- | --- | --- |
| 000 | dense ×3 | 1,3,1 | Add, Mul | 127, 127, 127 |
| 001 | dense, depthwise, dense | 3,1,3 | Mul, Add | 64, 0, −64 |
| 002 | depthwise, dense, depthwise | 3,1,3 | Add, Mul | −128, 127, 32 |
| 003 | dense, depthwise, dense, depthwise | 1,3,1,5 | Add, Sub, Add | 96, −48, 24 |
| 004 | dense ×4 | 3,3,1,3 | Mul, Mul, Add | 127, 127, 127 |
| 005 | depthwise, depthwise, dense, dense | 3,1,1,3 | Add, Add, Mul | 64, 0, −64 |
| 006 | dense, depthwise, dense, dense, dense | 1,3,1,3,1 | Max, Add, Mul, Sub | −128, 127, 32 |
| 007 | dense ×3 (no stem Relu) | 1,1,1 | Mul, Mul | 96, −48, 24 |
| 008 | dense, depthwise, dense | 1,5,3 | Add, Add | 127, 127, 127 |
| 009 | dense ×5 | 1,3,1,3,1 | Add, Sub, Max, Mul | 64, 0, −64 |
| 010 | dense, depthwise, depthwise, dense | 3,3,5,1 | Mul, Add, Add | −128, 127, 32 |
| 011 | dense ×3 | 3,3,3 | Sub, Sub | 96, −48, 24 |

Every model runs 32 input cases including all-zero, all-255 and 128; at least 26%
of every model's expected outputs are nonzero (the deepest Mul chains are the
lowest), asserted by `tests/test_join_scale.py` so a passing board run cannot be
vacuous.

## Emission

The scale step reuses the verified elementwise program from
`compile_standalone_mul`: the join result is the accumulator operand (`0x5018`),
the declared `[1,3,1,1]` input is the ERDMA per-channel operand row
(`0x5038`, `0x5034 = 4`, `0x5040 = 16`), and the output conversion is
`mul_output_conversion(join_scale · operand_scale)`. No extra conversion task is
needed because the join grid is already native16.

## Two hardware findings (retained)

1. **Fresh slot required for the scaled primary.** When the last join reused a
   buffer that an earlier Conv head had written, the board returned stale data for
   every run (`board_dump` showed all-zero outputs). Giving every internal of this
   profile its own slot fixes it, so `compile_join_chain` places internals
   sequentially when a runtime scale follows. The reuse itself is legal for other
   profiles (the diamond reuses the stem slot), so this is a conservative rule for
   the elementwise-scale composition, not a general aliasing claim. The failing and
   passing containers were compared register by register; the only difference was
   the primary's offset.
2. **`rint(a·b/128)` is not the hardware model.** The folded scale product is not
   exactly `1/128` in float32, so the hardware requantization differs from
   `rint(a·b/128)` by one on ~1% of boundary values (71 of 6144 bytes in the
   probe). The suite's reference therefore uses
   `elementwise.mul_requant_reference` with the container's own multiplier and
   shift.

## Reproduction

```sh
PYTHONPATH=src python research/build_join_scale_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py join_scale_suite
```

Each input record is `image (192 B) ++ codes + 128 (3 B)`; `manifest.json` records
the geometry, codes and join structure of every model.
