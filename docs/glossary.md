# Glossary

The vocabulary a newcomer needs to read this project's docs and error messages.
Each entry is two or three sentences and points at the document that explains it in
full. Register numbers are hexadecimal; signed grid values are INT8 with a zero
point unless stated otherwise ([docs/README.md](README.md#conventions-used-throughout)).

---

**accumulator**: The INT32 arithmetic result inside the convolution or elementwise engine,
before requantization. A Conv accumulates `Σ (code − zero_point) ×
(weight − weight_zero_point) + bias`; on this hardware the verified range is about
±2,076,148,792, and the compiler rejects a weight/bias set whose bound would overflow it
before emitting any command. See [quantization.md](quantization.md) and the dense Conv row
in [support-matrix.md](support-matrix.md).

**arena**: The single contiguous device buffer, after the command payload, that holds every
tensor a container uses (input surface, intermediate grids, output). Its offset and size are
recorded in the header and in each tensor descriptor, and the loader bounds it at 4 MiB.
See [container-format.md](container-format.md).

**band**: The affine map between integer codes and real values for one tensor,
`real = (code − zero_point) × scale`. Every intermediate grid in a container carries exactly
one band, and choosing those bands is what separates a working model from one that scores at
chance. See [quantization.md](quantization.md).

**baseline**: The checked-in map `research/container_baseline.json` from every published suite
model to the sha256 of the container the compiler produces for it (or `ERR:<Exception>` for
models a profile deliberately rejects). It is the "no emitted container may change" contract:
a refactor that alters an accepted container, or starts accepting a rejected one, fails
`research/verify_suites.py`. See [verification.md](verification.md).

**binding**: A declaration that a stage reads or writes a named tensor through a specific
register, for example `Binding(0x1070, "input0", "read")`. Every emitter publishes its
bindings, and `compose.check_declared_bindings` re-derives the same read/write sets from the
finished container's address registers to prove the two agree. See
[architecture.md](architecture.md#the-composer).

**board ledger**: `research/COVERAGE_EXPANSION_RESULTS.md`, the evidence index: one row per
suite with its model count, inference count and exact output bytes, plus the per-suite
directories that hold the containers, inputs and recorded `board_results_*.json`. A claim is
only "board-verified" if it appears here. See [verification.md](verification.md#3-the-board-ledger).

**calibration (minmax/percentile/kl)**: `open_rknpu.calibration.measure` runs ONNX's host
reference over a directory of `.npy` batches and returns per-tensor `scale`/`zero_point`
ranges for the compiler. `minmax` takes the observed extremes, `percentile` takes the lower
bound plus an upper-tail quantile so outliers cannot stretch the scale, and `kl` runs a
TensorRT-style saturation search. Calibration is required for trained networks; analytic bands
are wrong for them. See [quantization.md](quantization.md#calibration).

**channel multiplier**: The per-output-channel factor
`round(weight_scale / max_weight_scale × 16384)` that scales each output channel relative to
the widest one before the shared multiplier/shift stage. It is stored per channel in the bias
block (24 bytes per four depthwise channels, 32 bytes per four dense channels). See
[quantization.md](quantization.md).

**checksum**: A FNV-1a 32-bit hash over the container bytes (with the checksum field zeroed
during the computation), stored in the header. The Python decoder and the C loader both
verify it before any ioctl, so a truncated or edited container fails closed. See
[container-format.md](container-format.md).

**CNA**: The convolution engine of the NPU — the fixed-function block that reads the input
surface, multiplies against packed weights and writes the output grid. A dense Conv, depthwise
Conv and ConvTranspose all become CNA tasks. See
[plans/pipelining-plan.md](plans/pipelining-plan.md).

**composer**: `open_rknpu.compose`, the pass that turns declared `Stage` objects into one v5
container: it allocates the arena (optionally reusing dead tensors' bytes via `liveness`),
substitutes final addresses into each stage's register fields, packs constants and links the
task tails. Emitters describe *what* to compute; the composer decides *where*. See
[architecture.md](architecture.md#the-composer).

**constant**: An immutable packed region a task reads at run time — weights, bias, a broadcast
Mul factor. v4 containers may expose selected constants as named descriptors the runtime can
overwrite between inferences (`--mutable-weights`, `--mutable-constants`); otherwise a constant
is fixed for the life of the container. See [container-format.md](container-format.md).

**container (v1–v5)**: The `.bin` the compiler writes and the runtime maps. `ORNPUBIN` v1/v2 is
the legacy single-Conv model; `ORNPUSEQ` v3/v4/v5 is a task table, where v4 adds overwritable
constant descriptors and v5 adds a named tensor table for fan-out and multiple inputs/outputs.
See [container-format.md](container-format.md).

**DPU**: The NPU's second fixed-function engine — pooling, elementwise arithmetic and
activation. A `MaxPool`/`AveragePool`, an Add/Sub/Max/Mul join, a Clip clamp task and the LUT
activation all become DPU tasks. See [plans/pipelining-plan.md](plans/pipelining-plan.md).

**emitter**: One module per profile (`native.py`, `depthwise.py`, `transposed.py`, …) that
validates a graph class, builds the register fields, weights and reference, and returns
`(container_bytes, meta)`. `meta["profile"]` records which emitter ran. See
[architecture.md](architecture.md) and [support-matrix.md](support-matrix.md).

**engine run**: One ioctl submission carrying one or more linked tasks. Serial submission
issues one run per task; `--submission batched` links a same-engine task list into a single
run, and `meta["engine_runs"]` reports the run structure. See
[plans/pipelining-plan.md](plans/pipelining-plan.md).

**ERDMA**: The elementwise engine's read path for a second (auxiliary) operand; the
configuration register is `ERDMA_CFG` (`0x5034`). It reads its secondary operand strictly
linearly, one 16-byte atom per output pixel, which is why a spatial Mul constant must be
materialized and there is no compact spatial broadcast. See [primitives.md](primitives.md) and
[plans/completion-plan.md](plans/completion-plan.md) P4.

**fence-free completion**: This kernel has no `CONFIG_ROCKCHIP_RKNPU_FENCE`, so `FENCE_IN`/
`FENCE_OUT` return `-EINVAL` and no pollable completion fd exists. Completion is instead
signalled by queueing the inference non-blocking and then running a small blocking barrier job,
which finishes only after the queued work. See [board.md](board.md) and
[plans/pipelining-plan.md](plans/pipelining-plan.md) S4.

**job / submission**: A *job* is the driver-level unit the runtime hands to `/dev/rknpu`;
a *submission* is how the container's tasks are grouped into jobs (serial, one linked batched
job, or a non-blocking queue-then-drain pipeline). Because each task tail is the successor's
fetch amount, a whole mixed-engine DAG can be one job. See
[container-format.md](container-format.md) and [board.md](board.md).

**lifetime / reuse**: The interval, in task order, during which an intermediate tensor is
live. `open_rknpu.liveness` computes those intervals and a first-fit arena allocator reuses
bytes only where two tensors' intervals are disjoint; in-place execution is never assumed.
`--reuse-intermediates` applies this to chains (layer L writes the buffer layer L−1 last read).
See [architecture.md](architecture.md) and [primitives.md](primitives.md).

**LSB**: Least significant bit — one integer step of a quantized tensor, i.e. one `scale` in
real units. Error metrics are reported in LSB (for example the mel-CNN residual is ≤4 LSB on
one output cell); it is a quantization-quality number, never the correctness criterion, which
is byte equality. See [quantization.md](quantization.md#measuring-accuracy-not-vibes).

**multiplier/shift**: The integer requantization pair that converts an accumulator to an output
code: roughly `(acc × multiplier) >> shift`, with `multiplier` normalized to ≤32767 and the
hardware adding `8191 + ((product >> 14) & 1)` so half ties round away from zero at the right
step. The Python reference reproduces the same formula. See [quantization.md](quantization.md).

**native16 layout**: The internal grid layout where 16 channels (lanes) are stored per pixel
and each plane occupies `ceil(H·W/4)·64` bytes; an image input is staged as a native16 surface
storing `byte − 128`. It is the layout the CNA reads, and packed descriptors are larger than
their flat API buffers because the 16-aligned row stride is kept. See
[container-format.md](container-format.md).

**oracle**: A reference used to check the compiler that is independent of it — the Python
integer reference next to each emitter, the retained board run, or (during register recovery
only) a vendor-toolkit capture. Vendor oracle builders are named `*_oracle.py` and never feed
the public emitters. See [plans/completion-plan.md](plans/completion-plan.md) §7 and
[verification.md](verification.md).

**packed UINT8 / NHWC**: The public API boundary: inputs are UINT8 in packed NHWC with
`real = (byte − input_zero_point) × input_scale`, and outputs are INT8 in packed NHWC. "Packed"
means rows are padded to a 16-byte stride rather than the native16 plane layout. See
[quantization.md](quantization.md) and [container-format.md](container-format.md).

**profile**: A bounded graph class with its own emitter, validation bounds, reference and
`meta["profile"]` name (for example `native16-input`, `chain-walk`, `diamond-tail`).
`scheduler.py` tries profiles in a fixed order and the first one that accepts the graph wins;
there is no silent fallback. See [architecture.md](architecture.md#the-schedulers-profile-order).

**register profile**: `open_rknpu.register_profile.REGISTERS`, the list of register offsets,
reset values and command tags an emitted Conv task writes (126 words), recovered by
experiment and recorded with the byte-level format in `runtime/sequence_format.md`. Many
fields are still opaque. See [architecture.md](architecture.md) and
[container-format.md](container-format.md).

**requantization**: Converting an accumulator (or a product) from the internal integer domain
back to the tensor's output band using the channel multiplier, multiplier/shift, output zero
point and rounding. It is the step where ties-even versus ties-away and the order of the
offset matter. See [quantization.md](quantization.md).

**role / layout**: The two per-tensor descriptor fields in a v5 tensor table: `role` is
`input`, `output` or `internal`, and `layout` is `packed` (UINT8 rows padded to 16),
`native16` or `packed int8`. `ornpu_run_io` binds external tensors by role and index. See
[container-format.md](container-format.md).

**scale**: The real value of one integer step, `real = (code − zero_point) × scale`. A scale
is always paired with a zero point; the pair is the tensor's band. See
[quantization.md](quantization.md).

**serial vs batched**: Serial submission issues one ioctl per task; batched links a same-engine
task list into one job. Batched is much faster for deep runs (up to 11× on a 12-layer chain)
and slower for short ones (crossover near eight tasks), so serial is the default. See
[plans/pipelining-plan.md](plans/pipelining-plan.md).

**stage**: A declared unit of composition — a name, a task family, the tensors it reads and
writes, a `fields(addresses, constants)` function and its constants and bindings. Emitters
that use the composer build a list of stages and let `compose()` place them. See
[architecture.md](architecture.md#the-composer).

**tail control / PC_DATA_AMOUNT**: The four words appended to each task program. The control
word is the **successor program's fetch amount**
(`PC_DATA_AMOUNT = (register_config_words + 4 + 2 − 1)/2 − 1`), not an engine code, with `0x28`
as the terminal sentinel; a pool successor gives `0x14` and an elementwise successor `0x28`.
Because it is a size, every transition links and a DAG is one job. See
[architecture.md](architecture.md#the-composer) and [container-format.md](container-format.md).

**tensor table**: The v5 structure listing every external and internal tensor by name, role,
layout, shape, arena offset and byte size. It is what lets the runtime expose intermediates
and bind several inputs/outputs. See [container-format.md](container-format.md).

**undecoded field**: A register bit whose semantics are not yet established by experiment, left
at its reset/profile value by the emitter. The LUT table interpolation and domain-calibration
fields beyond the two banks are the canonical open example. See
[investigation-log.md](investigation-log.md) and
[plans/completion-plan.md](plans/completion-plan.md) P6.

**vendor toolkit**: Rockchip's RKNN Toolkit2, used only as a development oracle (for example
to establish that it rejects `do_quantization=False` for RV1103, and to produce marker
captures used as hypotheses). No vendor code, RKNN model or library is in the emitted path.
See [plans/primitive-roadmap.md](plans/primitive-roadmap.md) and
[research/hardware_refs/README.md](../research/hardware_refs/README.md).

**walk**: `open_rknpu.walk`, the op-level lowering path that validates a linear graph, walks it
in node order and emits one stage per node from the same verified per-op builders as the branch
profiles. It is the only path where a pool's position in a chain is free, and it also lowers a
fan-in join class; a large image input is staged as a native16 surface. See
[architecture.md](architecture.md#the-op-level-walk).

**zero point**: The integer code that represents real 0.0 in a tensor's band. The input image's
zero point is part of the API band; a Conv reads its input band's zero point from register
`0x1184`, and a Conv feeding a DPU pool is re-quantized onto a zero-point-0 grid because the
verified pool program assumes it. See [quantization.md](quantization.md).

---

## See also

* [support-matrix.md](support-matrix.md) — the bounds and evidence for every accepted op.
* [troubleshooting.md](troubleshooting.md) — symptom-first diagnosis.
* [architecture.md](architecture.md) — the pipeline these terms describe.
