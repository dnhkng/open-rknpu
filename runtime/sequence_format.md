# Explicit task-table executable

`ORNPUSEQ`, format versions 3, 4 and 5, targets RV1103. The existing C API and file
runner accept it alongside legacy `ORNPUBIN` versions 1/2. The compiler-side
encoder/validator is `open_rknpu.sequence`; `open-rknpu inspect` recognizes it.
The sequence compiler lowers its supported ONNX profiles to this format.
Version 5 adds a named tensor table for bounded DAG graphs (fan-out, several
external inputs and outputs); versions 3/4 are unchanged.

The file contains a 96-byte little-endian header, `task_count` 16-byte task
descriptors, optional v4 constant descriptors, then the command/constant payload.
The header is `<8s22I`:

| Byte offset | Field |
| --- | --- |
| 0 | Eight-byte magic `ORNPUSEQ` |
| 8, 12 | Version 3 or 4, header size 96 |
| 16, 20, 24 | Input height, width, channels |
| 28, 32, 36 | Output height, width, channels |
| 40 | Input row stride: aligned to 16 for packed mode, width for native16 mode |
| 44, 48 | Payload bytes, DMA arena bytes |
| 52, 56 | Input and output offsets within arena |
| 60 | Number of task descriptors |
| 64, 68 | Float32 input scale bits, UINT8 input zero point |
| 72, 76 | Float32 output scale bits, signed INT8 output zero point in int32 |
| 80 | FNV-1a checksum of complete file with this field zeroed |
| 84 | Low bits: submission mode and input count; bits 8+ are v4 constant count |
| 88 | Input layout: 0 packed, 1 native16 |
| 92 | Reserved (v3/v4: must be zero; v5: `batch − 1`, accepted 0..15) |

Each descriptor is four uint32 values: command offset, register count,
enable mask, interrupt mask. The current combinations are convolution
`29/768` (1–256 register words), pooling `96/3072` (1–256 words), elementwise
`24/768` with exactly 78 words, and the long activation lookup-table setup
`24/768` with exactly 1106 words. Older runtimes reject the elementwise and LUT
descriptors; rebuild the runtime when using the Add/Mul or Sigmoid/Tanh compiler
profiles. Command offsets refer to the start of the
payload, not the file. The payload is copied to arena offset zero; descriptors
become kernel task records with DMA addresses resolved after allocation.
The compiler remains responsible for command terminal links and intermediates.
## Submission modes

**Serial** (header flag bit 0 set, the default): one descriptor per ioctl, each waited
for before the next; every task program ends terminal. Correct for any graph, any task
count, any engine mix.

**Batched / engine runs** (flag bit 0 clear): the container links every task, and the
runtime submits **one ioctl per maximal linked run**. A task whose tail link (register
`0x10`, payload-relative offset) is zero ends its run; a linked task continues it. The
tail's control word (register `0x14`) is the **successor program's fetch amount** - the
same value the driver programs into `PC_DATA_AMOUNT` for a task:

```
control = (successor regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT + scale - 1) / scale - 1
        = (words + 4 + 2 - 1) / 2 - 1          # RV1106: extra amount 4, scale 2
```

so `0x40` (64) links to a 126-word Conv program, `0x14` (20) to a 37-word pool program,
`0x28` (40) to a 78-word elementwise program and `0x22A` (554) to the 1106-word LUT setup
(`rknpu_job.c`, `open_rknpu.compose.amount_control`; the vendor captures write exactly
those values). A terminal tail keeps the `0x28` sentinel and a zero link. With the right
amount every transition links, whatever engines it crosses, so a whole DAG is one job;
a container that links nothing is one run per task (correct, if not fast). Depth is
bounded only by the 64-task table and the 16-bit count in `PC_TASK_CONTROL`. Cost: one
job of N tasks is roughly 30-90 us plus 3-7 us/task, against about 14 us/task plus a
per-ioctl round trip serially, so serial wins below roughly 4-8 tasks in a run and one
job wins above it (a 12-layer chain measured 1.9 ms serial against 0.18 ms in one job).
`open-rknpu compile --sequence --submission batched` emits the linked form and refuses a
profile whose emitter ignores the request (`docs/plans/pipelining-plan.md` S1/S2/S8/S10).

The driver's `core_mask` and `subcore_task[]` submission fields are inert on RV1106:
`rknpu_job_alloc` forces `core_mask = CORE0` when the config has one IRQ (RV1106 uses the
single-entry `rknpu_irqs`) and `subcore_task[]` is read only when `num_irqs > 1`
(`tests/board_core.c`, `research/job_field_probe/core_fields.txt`). The runtime programs
`PC_DMA_BASE_ADDR` with the payload DMA base, which is what makes the link
payload-relative.

**Non-blocking queue-then-drain** (`JOB_NONBLOCK`): the ioctl returns without waiting;
a following blocking submission drains the queue, because the driver runs one job at a
time per core in order. Measured 1.5-2.25x over synchronous submission on small graphs
(`research/async_probe/`). `JOB_FENCE_IN`/`JOB_FENCE_OUT` return -EINVAL on the attached
board: its kernel was built without `CONFIG_ROCKCHIP_RKNPU_FENCE`, so there is no
pollable completion fd. Completion can still be observed with **lag 0** by draining with
a small *barrier* job - any blocking submission after the queued one completes only after
it - which is faster than the synchronous path for a serial container and costs one small
job for an already-batched one (`research/barrier_probe/`, `tests/board_barrier.c`).

Version 4 follows the task table with up to64 constant descriptors, each
`<24s4I>`: a NUL-terminated UTF-8 name, payload offset, byte size, kind and a
zero reserved word. Described regions may not overlap command programs. The
opt-in native Conv profile currently emits `conv.parameters` kind1, covering
its complete packed weight and bias/per-channel conversion region. C callers
enumerate it with `ornpu_get_constant()` and replace the whole region with
`ornpu_set_constant()`. Replacement parameters must use the same tensor
geometry and global output conversion; grouping coupled bytes in one region
prevents partial updates from leaving stale bias correction. Version3 files
have zero descriptors and preserve their existing bytes and behavior.
The opt-in constant-Mul profile emits `mul.factor` kind3. Its packed INT8 codes
may change at runtime while retaining the compiled factor scale and output
conversion; scalar/per-channel and compressed per-batch vectors are supported.

Limits: 64 tasks, 1 MiB payload, 4 MiB arena; dimensions 1–1024, one or three
external input channels in packed mode, 1–64 in native16 mode, 1–128 output channels. These are loader bounds, not
claims of compiler or hardware coverage. Input and output must be disjoint
from each other and the payload and remain within the arena. Payload size
is a multiple of 64, arena size a multiple of 4096, IO offsets multiples of 64.
Native output uses 16-channel planes, each ceil(H*W/4)*64 bytes; the API
returns logical channels in NHWC order. The complete output allocation multiplies
that plane size by ceil(output_channels/16).
Input/output API buffers remain packed NHWC without row/pixel padding.
Native16 mode converts UINT8 input values to signed bytes by subtracting 128,
packs channels into 16-lane planes, and fills unused lanes with input_zero_point−128.
Its input allocation is ceil(H*W/4)*64*ceil(input_channels/16) bytes. Rebuild older runtimes before
using this layout; they reject the new flag. Internal runtime profile 9 denotes
this sequence packing mode and is not a legacy executable profile.

Like legacy executables, payloads are trusted compiler output. Validation
checks layout, descriptors, quantization, lengths, and checksum, but does not
interpret register programs or prove their memory accesses safe.

The independently emitted native14 graph is verified through this format:
`research/emit_native14.py` creates `research/sequence_suite/model000.bin`.
The public C API ran 36 cases with 112896 exact output bytes. The same runtime
also passed 896 existing five-task graph runs after legacy profile conversion
was moved onto the common task-table submission path.

Version 3/4 describe one output and at most two matching inputs. General DAG
graphs (fan-out, multiple consumers, unequal runtime inputs, multiple outputs)
use the version 5 named-tensor table below; v3/v4 decoding is unchanged. A
lifetime allocator that reuses arena bytes for non-overlapping internal tensors
is still future work (plan P1).

## Version 5 named tensors

Version 5 replaces the fixed input/output contract with a tensor table. The file
is the 96-byte header, a 16-byte extension, `task_count` 16-byte task
descriptors, the 64-byte tensor descriptors, then the payload. There are no v4
constant descriptors in v5 (`r0` bits 8+ are zero).

The extension is four uint32 values: `tensor_count`, `tensor_size` (64),
`input_count`, `output_count`.

Each tensor descriptor is `<24s10I>` (64 bytes): a NUL-terminated UTF-8 name,
then `role` (0 external input, 1 external output, 2 internal), `layout`
(0 packed UINT8 with 16-aligned row stride, 1 native16 signed INT8 planes,
2 packed INT8), `index` (ordering within the role, contiguous from zero),
`batch`, `height`, `width`, `channels`, `byte_offset` in the arena, `size`
(arena storage bytes) and a zero reserved word. Arena sizes are
`batch*H*ceil(W/16)*16*C` for packed layouts and
`batch*ceil(H*W/4)*64*ceil(C/16)` for native16. External tensors must be
disjoint from each other and from the payload and internal tensors; internal
tensors may share arena bytes (a future lifetime plan), but they may not
overlap an external tensor. The header's primary shape/offset fields must match
the external input and output with index 0.

The C API enumerates the table with `ornpu_get_tensor()` and binds one packed
NHWC buffer per external tensor with `ornpu_run_io()`. `ornpu_run()` keeps its
legacy behaviour for v1–v4 and accepts a v5 file only when it has exactly one
input and one output. All external tensors must be supplied exactly once.

Verified on RV1103: the two-head fan-out profile (`Conv->Relu->{Conv,Conv}`,
hidden C3..16, dense 1x1/3x3 heads) compiles to v5 and passed 14 models,
448 inferences and 172,032 exact output bytes through `ornpu_run_io`
(`research/two_head_suite/`, `tests/board_io.c`).
