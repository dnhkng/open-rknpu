# Container migration and compatibility

This page is for **third-party producers and consumers**: anyone writing a `.bin` that this
runtime will load, or reading one this compiler emitted. It states what each container
version adds, which loader reads it, the rules a writer must satisfy, and — plainly — how
much stability to expect. The byte-level specification is
[`runtime/sequence_format.md`](../runtime/sequence_format.md); the orientation is
[container-format.md](container-format.md); a real v5 file is walked field by field in
[container-example.md](container-example.md).

## Version matrix

| Magic | Versions | Family | What the version adds | Encoder |
| --- | --- | --- | --- | --- |
| `ORNPUBIN` | 1, 2 | legacy single-model executable | fixed geometry, packed weights, one to five tasks per profile 1–8; v2 assigns header bytes 88/92 to a float32 input scale and a UINT8 input zero point, v1 keeps them reserved and fixed at scale 1 / zero point 0 | `open_rknpu.model.encode` |
| `ORNPUSEQ` | 3 | task-table sequence | 96-byte header plus `task_count` 16-byte descriptors (command offset, register count, enable mask, interrupt mask) and a payload; one output, at most two matching inputs | `open_rknpu.sequence.encode_sequence` |
| `ORNPUSEQ` | 4 | task table + constants | up to 64 **constant descriptors** after the task table: named, runtime-replaceable payload regions | `encode_sequence` (emits v4 iff constants were given) |
| `ORNPUSEQ` | 5 | task table + **named tensor table** | a 16-byte extension and 64-byte tensor descriptors for fan-out, several external inputs/outputs and exposed intermediates; **no** constant descriptors | `encode_sequence_v5` |

Sources: [`open_rknpu/model.py`](../src/open_rknpu/model.py),
[`open_rknpu/sequence.py`](../src/open_rknpu/sequence.py),
[`runtime/sequence_format.md`](../runtime/sequence_format.md),
[docs/container-format.md](container-format.md). The default `open-rknpu compile` path
emits the legacy `ORNPUBIN` form; `--sequence` selects the task-table family
([getting-started.md](getting-started.md), [cli.py](../src/open_rknpu/cli.py)).

### v1/v2 — legacy `ORNPUBIN`

One fixed-geometry model per file. The header is the 96-byte `<8s17Iif3I>` structure with
magic `ORNPUBIN`; the decoder accepts only versions 1 and 2, header size 96, target 1103
and profiles 1–8, and enforces the fixed memory layout (payload 8192, 126 register words,
input/output offsets 8192/12288, a derived arena size)
([model.py](../src/open_rknpu/model.py), [runtime/open_rknpu.c](../runtime/open_rknpu.c)
`read_model`). The task count follows the profile: 1 (profile 1), 2 (pools / two-layer),
4 (multi-stage pools) or 5 (pool + second Conv)
([model.py](../src/open_rknpu/model.py), [runtime/open_rknpu.c](../runtime/open_rknpu.c)).

Version selection is automatic: the encoder writes **version 1 when the input band is
`(scale 1.0, zero point 0)` and version 2 otherwise**, and version 1 requires the reserved
fields to be zero — a v1 file never carries an explicit input band
([model.py](../src/open_rknpu/model.py)). Version 2 is restricted to profile 1 and
validates the input quantization ([model.py](../src/open_rknpu/model.py)).

### v3 — task table

`ORNPUSEQ` replaces the fixed memory layout with a task list. The file is the 96-byte
little-endian header, `task_count` 16-byte task descriptors, then the command/constant
payload. Each descriptor is four `uint32` values — command offset, register count, enable
mask, interrupt mask — and the payload is copied to arena offset zero; command offsets are
relative to the start of the **payload**, not the file
([runtime/sequence_format.md](../runtime/sequence_format.md),
[sequence.py](../src/open_rknpu/sequence.py)).

Header flags: bit 0 selects **serial** submission (one descriptor per ioctl); bit 1 encodes
the input-tensor count (`input_count − 1`); bits 8+ are the v4 constant count
([sequence.py](../src/open_rknpu/sequence.py), [runtime/sequence_format.md](../runtime/sequence_format.md)).
A v3 file has **zero** constant descriptors, and both Python and C enforce the equivalence
`version == 3` ⟺ `constant_count == 0`
([sequence.py](../src/open_rknpu/sequence.py), [runtime/open_rknpu.c](../runtime/open_rknpu.c)).
Accepted task kinds are the bounded set in the spec — Conv `29/768`, pooling `96/3072`,
elementwise `24/768` with exactly 78 words, and LUT setup `24/768` with exactly 1106 words
([runtime/sequence_format.md](../runtime/sequence_format.md)).

### v4 — named constant descriptors

Version 4 follows the task table with up to **64** constant descriptors, each `<24s4I>`
(40 bytes): a NUL-terminated UTF-8 **name** (≤23 bytes, unique), a **payload offset**, a
**byte size**, a **kind** (1–3), and a **zero reserved word**. A described region must sit
inside the payload and **may not overlap a command program**
([runtime/sequence_format.md](../runtime/sequence_format.md),
[sequence.py](../src/open_rknpu/sequence.py)). The C loader applies the same checks and
additionally rejects duplicate names and a nonzero reserved word
([runtime/open_rknpu.c](../runtime/open_rknpu.c), `load_program`).

The opt-in native Conv profile emits one region, `conv.parameters` (kind 1), covering its
complete packed weight and per-channel conversion region; the constant-Mul profile emits
`mul.factor` (kind 3), whose packed INT8 codes may change at runtime while retaining the
compiled factor scale and output conversion
([runtime/sequence_format.md](../runtime/sequence_format.md)). Consumers enumerate with
`ornpu_get_constant()` and replace the whole region with `ornpu_set_constant()`, whose size
must match exactly; grouping coupled bytes in one region is what prevents a partial update
from leaving stale bias correction ([c-api.md](c-api.md),
[runtime/sequence_format.md](../runtime/sequence_format.md)).

Encoding is what picks the version: `encode_sequence` writes v4 when any descriptor is
present and v3 otherwise, and packs the constant count into flags bits 8+
([sequence.py](../src/open_rknpu/sequence.py)).

### v5 — named tensor table

Version 5 keeps the 96-byte header and adds a **16-byte extension**, then `task_count`
16-byte task descriptors, then the 64-byte tensor descriptors, then the payload. The
extension is four `uint32` values: `tensor_count`, `tensor_size` (fixed 64), `input_count`,
`output_count` ([runtime/sequence_format.md](../runtime/sequence_format.md),
[sequence.py](../src/open_rknpu/sequence.py)).

Each tensor descriptor is `<24s10I>` (64 bytes): name, `role` (0 external input, 1 external
output, 2 internal), `layout` (0 packed UINT8, 1 native16, 2 packed INT8), `index`
(ordering within the role, contiguous from zero), `batch`, `H`, `W`, `C`, `byte_offset` in
the arena, `size`, and a zero reserved word. Arena sizes are
`batch·H·ceil(W/16)·16·C` for packed layouts and `batch·ceil(H·W/4)·64·ceil(C/16)` for
native16 ([runtime/sequence_format.md](../runtime/sequence_format.md),
[sequence.py](../src/open_rknpu/sequence.py)). External tensors must be disjoint from each
other, from the payload and from internal tensors, and the header's primary shape/offset
fields must match the external input and output with index 0
([runtime/sequence_format.md](../runtime/sequence_format.md)).

The C API enumerates the table with `ornpu_get_tensor()` and binds one packed NHWC buffer
per external tensor with `ornpu_run_io()`; `ornpu_run()` keeps its legacy behaviour for
v1–v4 and accepts a v5 file only when it has exactly one input and one output
([runtime/sequence_format.md](../runtime/sequence_format.md), [c-api.md](c-api.md)).

### The v5 `batch − 1` header byte

Header word 21 (byte **92**) carries `batch − 1` in v5, **validated as 0..15** (`batch`
1..16); `_decode_v5` also cross-checks that the primary input descriptor's batch equals
`r2 + 1` ([sequence.py](../src/open_rknpu/sequence.py),
[runtime/sequence_format.md](../runtime/sequence_format.md)). The C loader does the same
(`load_v5`, [runtime/open_rknpu.c](../runtime/open_rknpu.c)). `container-example.md` flags
the spec table's wording explicitly: byte 92 is listed as "Reserved, must be zero" in the
v3/v4 part of `sequence_format.md`, but for **v5** it is `batch − 1`, so "a batch-16 v5
container would fail a literal reading of the old table"
([container-example.md](container-example.md)).

For completeness: the shared slot is used for batch on **v3/v4** as well — `encode_sequence`
writes `batch − 1` and `decode_sequence` derives `batch = r2 + 1`, rejecting `r2 > 15` and
requiring native16 for a batch other than 1
([sequence.py](../src/open_rknpu/sequence.py)). The `batch − 1` field is therefore not
v5-only in the implementation, even though the v5 documentation is where it is called out.

## Which loader reads which version

| Loader | v1/v2 `ORNPUBIN` | v3 | v4 | v5 |
| --- | --- | --- | --- | --- |
| C runtime `ornpu_inspect` / `ornpu_open` | yes | yes | yes | yes |
| Python `open_rknpu.model.decode` | yes | yes | yes | yes |
| Python `open_rknpu.sequence.decode_sequence` | **no** (`invalid sequence magic`) | yes | yes | yes |
| `open-rknpu inspect` (calls `model.decode`) | yes | yes | yes | yes |
| `ornpu_run` | yes | yes | yes | only with exactly one input and one output |
| `ornpu_run_io` | **no** (`-EINVAL`) | **no** (`-EINVAL`) | **no** (`-EINVAL`) | yes, multi-tensor |

Sources: [`runtime/sequence_format.md`](../runtime/sequence_format.md) ("The existing C API
and file runner accept it alongside legacy `ORNPUBIN` versions 1/2"),
[`runtime/open_rknpu.c`](../runtime/open_rknpu.c) (`load_program` dispatches
`ORNPUBIN` → `read_model`, v5 → `load_v5`, otherwise v3/v4),
[`model.py`](../src/open_rknpu/model.py) (`decode` dispatches on the magic),
[`sequence.py`](../src/open_rknpu/sequence.py) (`decode_sequence` raises on a non-sequence
magic), [`cli.py`](../src/open_rknpu/cli.py) (`inspect` prints `decode(...)`),
[c-api.md](c-api.md) (`ornpu_run_io` on a legacy container returns `-EINVAL`).

Note the asymmetry: the **runtime** is the universal reader, while `decode_sequence` is a
sequence-family parser by design and will reject a legacy file. If you need one Python
entry point for both, use `open_rknpu.model.decode`.

## Why v5 rejects a nonzero constant count

A v5 container has **no** constant descriptor table. The encoder makes that explicit
rather than silently dropping the request — `encode_sequence_v5(..., constants=...)` raises
`v5 containers have no constant descriptor table; runtime replaceable parameters require a
v4 container` ([sequence.py](../src/open_rknpu/sequence.py)). Both loaders enforce the same
rule from the other side: the v5 decoder rejects `(r0 >> 8) != 0` as
`invalid v5 sequence header`, and the C `load_v5` rejects `(v[19] >> 8) != 0`
([sequence.py](../src/open_rknpu/sequence.py), [runtime/open_rknpu.c](../runtime/open_rknpu.c)).
`container-example.md` states the reason in one line: "byte 84's bits 8+ are the v4
constant count; a v5 container must have them zero, which is why this file has no constant
descriptors" ([container-example.md](container-example.md)).

If you need runtime-replaceable parameters (mutable Conv weights, a constant-Mul factor),
emit a **v4** container with `encode_sequence(..., constants=...)`
([sequence.py](../src/open_rknpu/sequence.py), [c-api.md](c-api.md)).

## Writing your own producer

The writer is deliberately small (~100 lines) and the two encoders validate themselves
before returning: they call the matching decoder and raise if the bytes they just wrote do
not parse ([container-format.md](container-format.md),
[sequence.py](../src/open_rknpu/sequence.py)). To locate the payload from outside, use the
same rule the decoder uses — `payload_base(info)` is `112 + 16·task_count + 64·tensor_count`
for v5 and `96 + 16·task_count + 40·constant_count` for v3/v4
([sequence.py](../src/open_rknpu/sequence.py)).

Cross-check anything you write before loading it on hardware:

```sh
open-rknpu inspect model.bin          # Python decoder: header, tasks, constants, tensors as JSON
```

```python
from open_rknpu.model import decode                 # both families
from open_rknpu.sequence import decode_sequence     # ORNPUSEQ only (v3/v4/v5)
```

The C `tests/board_io.c` prints a readable failure for every malformed field it can detect
(wrong magic, bad checksum, short payload, tensor index out of range, input/output count
mismatch) ([container-format.md](container-format.md),
[tests/board_io.c](../tests/board_io.c)). The full list of rejection strings is in
[troubleshooting.md](troubleshooting.md#the-container-will-not-open).

## Compatibility statement

**The project is 0.x and gives no format-stability guarantee yet.** The current release is
`0.1.0` ([pyproject.toml](../pyproject.toml), [CHANGELOG.md](../CHANGELOG.md)), the
changelog has an active `[Unreleased]` section, and format churn is a recorded risk:
"Mitigate by versioning and by never mutating v3/v4"
([plans/completion-plan.md](plans/completion-plan.md) P1). Treat the byte layout as
documented and implemented, but not frozen.

What a consumer can reasonably assume today:

* **The runtime accepts v1 through v5.** All five are loaded by the same C entry points,
  and `open-rknpu inspect` decodes all of them
  ([runtime/sequence_format.md](../runtime/sequence_format.md),
  [runtime/open_rknpu.c](../runtime/open_rknpu.c), [cli.py](../src/open_rknpu/cli.py)).
* **v3/v4 decoding does not change when a new version is added.** v5 leaves v3/v4 decoding
  untouched, and v3 files "have zero descriptors and preserve their existing bytes and
  behavior" ([runtime/sequence_format.md](../runtime/sequence_format.md),
  [plans/completion-plan.md](plans/completion-plan.md) P1).
* **The loaders fail closed.** A container is validated (magic, version, geometry,
  descriptors, quantization, lengths, checksum) before `/dev/rknpu` is opened; the checksum
  is FNV-1a over the header with the checksum field zeroed, plus the v5 extension, task
  table, tensor table and payload ([container-format.md](container-format.md),
  [c-api.md](c-api.md)). Payload register programs are trusted compiler output — validation
  is not a sandbox ([runtime/sequence_format.md](../runtime/sequence_format.md)).
* **Published suites are pinned.** The container bytes of the evidence suites are checked
  against `research/container_baseline.json` by `research/verify_suites.py`, so a change to
  an accepted container is meant to be a visible, deliberate event
  ([verification.md](verification.md)).

What may change before 1.0 (`[Unreleased]` already contains format-documentation
corrections, including the v5 `batch − 1` header byte, [CHANGELOG.md](../CHANGELOG.md)):

* new **versions** may be added for new capabilities (arena lifetimes/aliasing and more
  general DAG scheduling are listed as future work), and the intent recorded in the plan is
  to add a version rather than mutate an existing one
  ([runtime/sequence_format.md](../runtime/sequence_format.md),
  [plans/completion-plan.md](plans/completion-plan.md) P1);
* documentation of a field can be corrected without a byte change (the v5 byte-92 wording
  is the worked example, [container-example.md](container-example.md));
* ABI additions are intended to be backward-compatible, so a newer runtime should keep
  reading older files ([plans/completion-plan.md](plans/completion-plan.md) P1 step 3);
* **do not** assume a hard "v5 never changes" promise. There is none yet.

### Migrating an older container to v5

**Recompile from the source model; do not patch container bytes.** The payload is
compiler-generated register programs and packed weights, so there is no meaningful
byte-level upgrade path between versions — the compiler regenerates the payload, the task
table and (for v5) the tensor table together, and the encoder validates the result before
writing it ([container-format.md](container-format.md),
[sequence.py](../src/open_rknpu/sequence.py)). This is also the general fix for a container
that will not open: "Recompile the model (`open-rknpu compile model.onnx -o model.bin
--sequence`) rather than repairing bytes"
([troubleshooting.md](troubleshooting.md#the-container-will-not-open)).

Practical recipe:

```sh
# legacy v1/v2 or v3/v4 -> the modern task-table family
open-rknpu compile model.onnx -o model.bin --sequence

# mutable parameters need the v4 constant table (not v5)
open-rknpu compile model.onnx -o model.bin --sequence --mutable-weights

# fan-out / several external inputs or outputs need v5
open-rknpu compile model.onnx -o model.bin --sequence --expose-intermediates

open-rknpu inspect model.bin          # confirm the version and tables you expected
```

`--mutable-weights` emits a v4 packed Conv parameter descriptor and `--mutable-constants` a
v4 packed Mul factor descriptor; both require `--sequence`
([cli.py](../src/open_rknpu/cli.py), [c-api.md](c-api.md),
[container-format.md](container-format.md)). A v5 container is emitted by the profiles that
need the named tensor table (for example `--expose-intermediates` on a chain), so the
"migrate to v5" step is the same recompile — only the profile choice differs
([runtime/sequence_format.md](../runtime/sequence_format.md),
[container-example.md](container-example.md)). Note that some published suite containers
predate later serializer fixes and cannot be reproduced byte-for-byte by a fresh compile;
that is an artifact drift, not a format-version issue
([troubleshooting.md](troubleshooting.md#the-container-will-not-open),
[verification.md](verification.md#the-campaign-sweep)).

## See also

* [runtime/sequence_format.md](../runtime/sequence_format.md) — the byte-level specification for v3/v4/v5.
* [container-format.md](container-format.md) — the two families, the v5 anatomy, and the runtime's lifecycle.
* [container-example.md](container-example.md) — one real v5 file, field by field.
* [c-api.md](c-api.md) — `ornpu_inspect`/`ornpu_open`, `ornpu_get_tensor`/`ornpu_run_io`, and the v4 constants API.
* [troubleshooting.md](troubleshooting.md#the-container-will-not-open) — every rejection string and what it means.
