# Using the runtime from C

`open-rknpu` ships a small **libc-only** board runtime: two files, a public header and an
implementation, with no vendor libraries and no dependency beyond `sys/ioctl.h`, `mmap`
and `poll` ([runtime/Makefile](../runtime/Makefile)). The runtime sources also ship inside
the wheel under `share/open-rknpu/runtime/`, so a board binary can be built without a
checkout ([pyproject.toml](../pyproject.toml)).

| File | What it is |
| --- | --- |
| [`runtime/open_rknpu.h`](../runtime/open_rknpu.h) | the public API: the structs, the constants, and every function prototype |
| [`runtime/open_rknpu.c`](../runtime/open_rknpu.c) | loader/validator, device allocation, packing, submission |
| [`runtime/main.c`](../runtime/main.c) | the reference command-line runner (`open-rknpu-run MODEL.bin INPUT.u8 OUTPUT.i8`) |
| [`runtime/sequence_format.md`](../runtime/sequence_format.md) | the byte-level container layout this API loads |

This page is a narrative guide to that header. Where a statement is a code behaviour it is
annotated with the function that implements it; **if this page and the header ever
disagree, the header wins.**

## The model lifecycle

```c
ornpu_info info;
ornpu_model *model = NULL;

if (ornpu_inspect(path, &info)) return 1;   /* validate, no device access   */
if (ornpu_open(path, &model)) return 1;     /* validate, allocate, open NPU */
/* ... ornpu_get_info / ornpu_get_tensor / ornpu_run ... */
ornpu_close(model);
```

| Call | What it does | Returns |
| --- | --- | --- |
| `ornpu_inspect(path, &info)` | reads and validates the whole container (header, tables, quantization, payload, checksum) and fills `info`; never touches `/dev/rknpu` | `0`, or negative errno |
| `ornpu_open(path, &model)` | validates again, opens `/dev/rknpu`, allocates the 4 KiB task buffer and the `arena_size` payload/arena buffer, copies the payload, builds the kernel task records and derives the engine runs | `0`, or negative errno; `*model` is set to `NULL` on failure |
| `ornpu_get_info(model, &info)` | copies the cached `ornpu_info` (includes `task_count`, `submission_serial`, `engine_runs`) | `-EINVAL` on `NULL` |
| `ornpu_get_tensor(model, index, &ti)` | describes tensor table entry `index` (0..`tensor_count-1`) | `-EINVAL` if out of range |
| `ornpu_get_input_tensor(model, index, &ti)` | describes external input with **role index** `index`; for a legacy container it synthesizes a per-input descriptor | `-EINVAL` if out of range |
| `ornpu_get_constant(model, index, &ci)` | describes constant descriptor `index` (v4 containers) | `-EINVAL` if out of range |
| `ornpu_run(model, in, in_size, out, out_size)` | single-input/single-output inference (the legacy entry point; on v5 it requires exactly one external input and one output) | `0`, or negative errno |
| `ornpu_run_io(model, ins, n_in, outs, n_out)` | named-tensor inference: one packed buffer per external tensor | `0`, or negative errno |
| `ornpu_set_input(model, tensor_index, data, size)` | packs one input into the arena without submitting (prep for the async path) | `0`, or negative errno |
| `ornpu_set_constant(model, index, data, size)` | replaces one complete constant region in the mapped arena | `-EINVAL` unless `size` matches the descriptor exactly |
| `ornpu_submit_flags(model, extra_flags, &fence_fd)` | experimental: submits with extra driver job flags | `0`, or negative errno |
| `ornpu_sync_outputs(model)` | cache maintenance so a queued job's outputs are readable | `0`, or negative errno |
| `ornpu_wait_fence(fd, timeout_ms)` | `poll()`s a fence fd | `0`, `-ETIMEDOUT`, or negative errno |
| `ornpu_set_submit_core(model, core_mask, subcore, pairs)` | experimental job-field probe; inert on RV1106 | `-EINVAL` on bad arguments |
| `ornpu_close(model)` | unmaps and destroys both device buffers, closes the fd, frees the model; safe on `NULL` | void |

`ornpu_open` validates the container **before** it opens `/dev/rknpu`, so a malformed file
cannot reach the device ([runtime/open_rknpu.c](../runtime/open_rknpu.c), `ornpu_open`).
Each model instance is used synchronously: the runtime keeps one input/output arena per
model, so overlapping calls on the *same* model are not supported. The two-instance
pipeline in [`tests/board_async.c`](../tests/board_async.c) opens two models instead.

## The structures

### `ornpu_info`

```c
uint32_t height, width, input_channels, output_channels;
uint32_t input_bytes, output_bytes;   /* flat packed NHWC API buffer sizes */
float    output_scale;  int32_t output_zero_point;
uint32_t output_height, output_width;
float    input_scale;   uint32_t input_zero_point;
uint32_t batch;
uint32_t input_tensor_count, constant_count, tensor_count, output_tensor_count;
uint32_t task_count;          /* submission descriptors in the container */
uint32_t submission_serial;   /* 1: one descriptor per submission */
uint32_t engine_runs;         /* ioctls a non-serial container is submitted as */
```

* `height`/`width`/`input_channels` and `output_height`/`output_width`/`output_channels`
  are the **primary** input and output (role index 0) — for a legacy executable the fixed
  header geometry.
* `input_bytes = batch × H × W × C` and `output_bytes = batch × H × W × C` of the primary
  tensors. These are the sizes `ornpu_run` requires, with **no row or pixel padding**.
* `input_scale`/`input_zero_point` dequantize the UINT8 input:
  `real = (byte − input_zero_point) × input_scale`; `input_zero_point` is `uint32_t`
  (0..255). `output_scale`/`output_zero_point` dequantize the INT8 output:
  `real = (int8 − output_zero_point) × output_scale`; `output_zero_point` is a **signed**
  `int32_t` in −128..127.
* `batch` is 1..16 (v5).
* `task_count` and `submission_serial` come from the container; `engine_runs` is derived at
  open time from the task tails. Note that `ornpu_inspect` never runs the derivation, so it
  leaves `engine_runs` at 0 ([runtime/open_rknpu.c](../runtime/open_rknpu.c),
  `compute_runs`, `ornpu_inspect`).
* For v5, `constant_count` is 0 and `tensor_count` is the number of table entries.

### `ornpu_tensor_info`

```c
char     name[24];            /* NUL-terminated UTF-8 */
uint32_t role, layout, index; /* role: 0 in, 1 out, 2 internal; index per role */
uint32_t batch, height, width, channels;
uint32_t arena_offset, arena_bytes;   /* storage inside the arena */
uint32_t api_offset, api_bytes;       /* placement in a flat per-role buffer */
```

* `role` is one of `ORNPU_TENSOR_INPUT` (0), `ORNPU_TENSOR_OUTPUT` (1),
  `ORNPU_TENSOR_INTERNAL` (2).
* `layout` is one of `ORNPU_LAYOUT_PACKED` (0), `ORNPU_LAYOUT_NATIVE16` (1),
  `ORNPU_LAYOUT_PACKED_INT8` (2). It describes the **arena** storage, not the buffer you
  pass in.
* `index` orders the tensors **within a role**, contiguous from zero. It is *not* the
  table index that `ornpu_get_tensor` and `ornpu_run_io.tensor_index` use.
* `arena_offset`/`arena_bytes` place the tensor in the arena; `arena_bytes` includes layout
  padding, so it can exceed `api_bytes`.
* `api_offset`/`api_bytes` place the tensor in a **flat packed NHWC** per-role buffer:
  `api_bytes = batch × H × W × C`, and `api_offset` is the cumulative `api_bytes` of the
  lower-indexed tensors of the same role ([runtime/open_rknpu.c](../runtime/open_rknpu.c),
  `describe_tensor`). This is how `tests/board_io.c` builds one concatenated input buffer
  and one concatenated output buffer.

### `ornpu_constant_info`

```c
char     name[24];
uint32_t byte_offset, bytes, kind;   /* kinds 1 packed params, 2 packed bias, 3 factor */
```

### `ornpu_io`

```c
typedef struct { uint32_t tensor_index; void *data; size_t size; } ornpu_io;
```

`tensor_index` indexes the **tensor table**; `size` must equal that tensor's `api_bytes`.

## Packing rules that matter

1. **Inputs are packed NHWC UINT8; outputs are packed NHWC INT8.**
   `ornpu_run` documents "Input: packed NHWC UINT8; Output: packed NHWC INT8. No padding in
   API buffers" ([runtime/open_rknpu.h](../runtime/open_rknpu.h)). Nothing else is exposed:
   there is no NCHW path and no float control API.
2. **Row padding and channel planes live only in the arena.** Packed tensors have their row
   stride aligned to 16 (`ceil(W/16)×16`), and native16 tensors are 16-lane signed-byte
   planes. The runtime packs and unpacks for you; a caller never sees either layout
   ([runtime/open_rknpu.c](../runtime/open_rknpu.c), `pack_tensor_input`,
   `unpack_tensor_output`).
3. **native16 staging is internal.** The runtime converts a packed UINT8 buffer to signed
   bytes by subtracting 128, arranges 16 channels per plane, fills unused lanes with
   `input_zero_point − 128`, and reverses the mapping on output. A native16 tensor's API
   buffer is still the flat `batch×H×W×C` bytes ([runtime/sequence_format.md](../runtime/sequence_format.md),
   "Version 5 named tensors").
4. **`input_bytes`/`output_bytes` are API buffer sizes, not arena sizes.** Size your
   `malloc`s from them (or from `api_bytes` per tensor); the arena is allocated by
   `ornpu_open` and reported by `ornpu_tensor_info.arena_bytes`.
5. **The output zero point is signed.** `output_zero_point` of −128..127 is legal; casting
   it to an unsigned type is a bug. `ornpu_info.output_zero_point` and the corrected
   `int8` domain are the dequantization contract.
6. **Buffer sizes are checked, not clamped.** `ornpu_run` rejects an input or output whose
   size differs from the expected flat size by even one byte (`-EINVAL`); `ornpu_run_io`
   does the same per tensor ([tests/board_api.c](../tests/board_api.c) exercises both).

## `ornpu_run` versus `ornpu_run_io`

`ornpu_run` is the simple path: one input, one output, both flat packed NHWC. For a v5
container it accepts the file only when `input_tensor_count == 1` **and**
`output_tensor_count == 1`; otherwise it returns `-EINVAL` and you must use `ornpu_run_io`
([runtime/open_rknpu.c](../runtime/open_rknpu.c), `ornpu_run`).

`ornpu_run_io` is the named-tensor path. Enumerate the table once, bind one buffer per
external tensor by table index, and call it every inference:

```c
ornpu_info info;
ornpu_get_info(model, &info);
size_t in_total = 0, out_total = 0;
for (uint32_t t = 0; t < info.tensor_count; t++) {
    ornpu_tensor_info ti;
    ornpu_get_tensor(model, t, &ti);
    if (ti.role == ORNPU_TENSOR_INPUT)  in_total  += ti.api_bytes;
    if (ti.role == ORNPU_TENSOR_OUTPUT) out_total += ti.api_bytes;
}
uint8_t *input = malloc(in_total);
int8_t  *output = malloc(out_total), *expected = malloc(out_total);
ornpu_io ins[8], outs[8];
for (uint32_t t = 0; t < info.tensor_count; t++) {
    ornpu_tensor_info ti;
    ornpu_get_tensor(model, t, &ti);
    if (ti.role == ORNPU_TENSOR_INPUT)
        ins[ti.index]  = (ornpu_io){ t, input  + ti.api_offset, ti.api_bytes };
    else if (ti.role == ORNPU_TENSOR_OUTPUT)
        outs[ti.index] = (ornpu_io){ t, output + ti.api_offset, ti.api_bytes };
}
int rc = ornpu_run_io(model, ins, info.input_tensor_count, outs, info.output_tensor_count);
```

| Named tensor | `role` | `layout` | `api_bytes` | Arena bytes |
| --- | --- | --- | ---: | ---: |
| `input0` | 0 input | 0 packed | 192 | 384 |
| `conv0` | 2 internal | 1 native16 | (not bindable) | 1,024 |
| `maxpool1` | 2 internal | 1 native16 | (not bindable) | 256 |
| `output` | 1 output | 1 native16 | 48 | 256 |

The table above is `research/walk_chain_suite/model000.bin` (see
[container-example.md](container-example.md) for the full walkthrough); internals are
enumerated but never bound. `tests/board_io.c` is the reference binder, and it also asserts
that a wrong external-buffer count is rejected before any submit.

Two behaviours worth knowing:

* the runtime checks the **count** of bound tensors, each entry's role, and each entry's
  size, but it does not detect the same `tensor_index` supplied twice (the remaining input
  would simply keep its previous arena contents). "Every external tensor supplied exactly
  once" is the caller's contract, stated in the header, not a checked invariant
  ([runtime/open_rknpu.c](../runtime/open_rknpu.c), `ornpu_run_io`).
* for a **legacy** (v1–v4) container, `ornpu_run_io` returns `-EINVAL` (there is no tensor
  table); use `ornpu_run`, or `ornpu_get_input_tensor` for a synthesized descriptor
  ([tests/board_api.c](../tests/board_api.c)).

## Replacing constants

A v4 container may describe named constant regions in its payload (v5 does not). Enumerate
them and replace the whole region in the mapped arena:

```c
ornpu_constant_info ci;
for (uint32_t i = 0; i < info.constant_count; i++) {
    ornpu_get_constant(model, i, &ci);              /* name, byte_offset, bytes, kind */
    if (!strcmp(ci.name, "conv.parameters"))
        ornpu_set_constant(model, i, new_region, ci.bytes);   /* size must match */
}
```

`ornpu_set_constant` copies exactly `ci.bytes` bytes into the arena at the descriptor's
offset and returns `-EINVAL` for any other size, a `NULL` buffer, or an out-of-range index.
Replacement parameters must keep the same tensor geometry and global output conversion;
grouping coupled bytes into one region is what prevents a partial update from leaving stale
bias correction ([runtime/sequence_format.md](../runtime/sequence_format.md), "Version 4"
section). The whole point of a *named region* is that the runtime addresses it and not a
hand-computed offset.

## The experimental async API

These are documented as experimental in the header and carry the measured
`docs/plans/pipelining-plan.md` caveats ([runtime/open_rknpu.h](../runtime/open_rknpu.h),
[research/barrier_probe/README.md](../research/barrier_probe/README.md)):

* `ornpu_submit_flags(model, ORNPU_JOB_NONBLOCK, NULL)` submits and **returns before the
  job completes**; flag `0x2`.
* `ORNPU_JOB_FENCE_IN` `0x8` / `ORNPU_JOB_FENCE_OUT` `0x10` are accepted by the API but
  return `-EINVAL` on the attached board: its kernel is built without
  `CONFIG_ROCKCHIP_RKNPU_FENCE`, so there is no pollable completion fd.
* Because jobs run **in order per core**, a queued job can be drained by the next blocking
  submission. For lag 0, submit the model non-blocking and run a small blocking **barrier**
  container, then `ornpu_sync_outputs(model)` before reading:

```c
ornpu_submit_flags(model, ORNPU_JOB_NONBLOCK, NULL);   /* queue the inference */
ornpu_run(barrier, barrier_in, barrier_in_bytes,        /* blocking: runs after it */
          barrier_out, barrier_out_bytes);
ornpu_sync_outputs(model);                              /* outputs are now final */
```

* `ornpu_set_input(model, tensor_index, data, size)` packs the next input while the NPU
  works. For a v5 model `tensor_index` is the table index; for a legacy model it must be 0
  and `size` must equal `input_bytes`.
* `ornpu_wait_fence(fd, timeout_ms)` polls a fence fd; it returns `-ETIMEDOUT` when the
  timeout expires.
* `ornpu_set_submit_core` exists so a board probe can show that `core_mask` and
  `subcore_task[]` are inert on RV1106: the driver forces `core_mask = CORE0` when the
  config has one IRQ and reads `subcore_task[]` only when `num_irqs > 1`
  ([runtime/open_rknpu.h](../runtime/open_rknpu.h),
  [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S8).

## Error handling

Every function except `ornpu_close` returns `0` on success and a **negative errno** on
failure; `ornpu_close` is `void` and accepts `NULL`. `runtime/main.c`'s `report()` is the
idiomatic formatter: `strerror(-rc)` prints the reason
([runtime/main.c](../runtime/main.c)).

| Value | Meaning | Typical source |
| --- | --- | --- |
| `-EINVAL` | malformed argument, malformed container, or a flag/tensor the loader rejects | most loader checks, size mismatches, `ornpu_submit_flags` with an undocumented bit |
| `-ENOMEM` | a `malloc` in the loader failed | `ornpu_open` / `ornpu_inspect` |
| `-ENOENT`, `-EACCES`, `-ENODEV`, … | `fopen`/`open`/`ioctl` failed; the runtime returns `-errno` | missing file, missing `/dev/rknpu` |
| `-ETIMEDOUT` | `ornpu_wait_fence` timed out | async path |

What the loader rejects, before any device work
([runtime/open_rknpu.c](../runtime/open_rknpu.c), `load_program`/`load_v5`; the same checks
are mirrored by the Python decoder in `src/open_rknpu/sequence.py`):

* wrong magic, unsupported version/header size, trailing bytes after the payload;
* legacy geometry bounds and the fixed 8192-byte payload / 12288-byte-plus-planes arena;
* stride not equal to the layout's expected stride; payload not a multiple of 64 or over
  1 MiB; arena not a multiple of 4096 or over 4 MiB; IO offsets not 64-byte aligned, below
  the payload, or overlapping;
* a task descriptor whose `(enable, mask, register count)` is not one of the verified
  families, whose command offset is not 8-byte aligned, or whose program plus its four tail
  words runs past the payload;
* a v5 tensor whose name is empty/duplicated/unterminated, whose role or layout is
  unknown, whose declared `size` disagrees with its layout geometry, or whose arena range
  is out of bounds or overlaps an external tensor;
* non-contiguous external indices, a missing primary (index 0) tensor, or primary-tensor
  geometry that disagrees with the header;
* non-finite or non-positive scales, zero points out of range;
* **a checksum mismatch** (see [container-example.md](container-example.md)).

The loader is a format validator, not a sandbox: register programs are trusted compiler
output, and validation does not prove their memory accesses safe
([runtime/sequence_format.md](../runtime/sequence_format.md)).

## A complete minimal program

`run_one.c` inspects, opens, runs one inference from a file, writes the INT8 output, and
cleans up. It works for a single-input/single-output model (legacy or v5). Save it as
`run_one.c` next to the compile command below.

```c
/* SPDX-License-Identifier: MIT */
#include "open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int report(int rc, const char *what) {
    if (rc) fprintf(stderr, "%s: %s\n", what, strerror(-rc));
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.bin input.u8 output.i8\n", argv[0]);
        return 2;
    }
    ornpu_info info;
    if (report(ornpu_inspect(argv[1], &info), "inspect")) return 1;
    printf("input UINT8 NHWC [%u,%u,%u,%u] = %u B, scale=%.9g zp=%u\n",
           info.batch, info.height, info.width, info.input_channels,
           info.input_bytes, info.input_scale, info.input_zero_point);
    printf("output INT8 NHWC [%u,%u,%u,%u] = %u B, scale=%.9g zp=%d\n",
           info.batch, info.output_height, info.output_width, info.output_channels,
           info.output_bytes, info.output_scale, info.output_zero_point);

    uint8_t *input = malloc(info.input_bytes);
    int8_t *output = malloc(info.output_bytes);
    if (!input || !output) { fprintf(stderr, "allocation failed\n"); return 1; }

    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(input, 1, info.input_bytes, f) != info.input_bytes || fgetc(f) != EOF) {
        fprintf(stderr, "input must contain exactly %u bytes\n", info.input_bytes);
        return 1;
    }
    fclose(f);

    ornpu_model *model = NULL;
    if (report(ornpu_open(argv[1], &model), "open")) return 1;
    int rc = ornpu_run(model, input, info.input_bytes, output, info.output_bytes);
    ornpu_close(model);                       /* safe even if run failed */
    if (report(rc, "inference")) return 1;

    f = fopen(argv[3], "wb");
    if (!f || fwrite(output, 1, info.output_bytes, f) != info.output_bytes ||
        fclose(f) != 0) {
        fprintf(stderr, "output write failed\n");
        return 1;
    }
    free(input);
    free(output);
    return 0;
}
```

Build and run it on the board — the cross-compile and staging commands are the ones in
[docs/board.md](board.md) and [docs/getting-started.md](getting-started.md):

```sh
# 1. cross-compile (vendor toolchain; any ARM uClibc compiler with the board sysroot works)
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime run_one.c runtime/open_rknpu.c \
  -o /tmp/run_one

# 2. stage it (never write to /tmp on the board, never stop rkipc)
adb shell mkdir -p /userdata/open-npu-research/run-one
adb push /tmp/run_one /userdata/open-npu-research/run-one/run_one
adb push model.bin input000.u8 /userdata/open-npu-research/run-one/

# 3. run and check the output against the reference bytes from the host
adb shell 'cd /userdata/open-npu-research/run-one && chmod +x run_one && \
  ./run_one model.bin input000.u8 output.i8'
adb pull /userdata/open-npu-research/run-one/output.i8 /tmp/
cmp /tmp/output.i8 expected000.i8 && echo "byte-exact"
```

A quick host-side sanity check before touching hardware is `open-rknpu inspect model.bin`,
which prints the same header/tasks/tensor table as JSON
([docs/getting-started.md](getting-started.md)).

## Cross-compiling the reference runner

The reference runner is the same three calls with file I/O around them:

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime runtime/main.c \
  runtime/open_rknpu.c -o /tmp/open-rknpu-run

adb push /tmp/open-rknpu-run /userdata/open-npu-research/open-rknpu-run
adb shell /userdata/open-npu-research/open-rknpu-run --inspect model.bin
adb shell /userdata/open-npu-research/open-rknpu-run model.bin input000.u8 output.i8
```

`runtime/Makefile` can also build it natively for a host sanity check (`make -C runtime`);
that build validates compilation and the loader, not NPU execution
([runtime/Makefile](../runtime/Makefile), [Makefile](../Makefile) target `runtime`).

For a whole suite, `tests/board_io.c` (v5 named tensors) or `tests/board_api.c` (legacy
single-IO) is the reference harness, and
`PYTHONPATH=src python research/run_v5_suite.py <suite> --binary /tmp/board_io` stages and
runs it ([docs/board.md](board.md), [tests/README.md](../tests/README.md)).
