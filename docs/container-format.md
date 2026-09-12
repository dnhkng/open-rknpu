# Container format

The byte-level specification is [`runtime/sequence_format.md`](../runtime/sequence_format.md);
this page is the orientation you need before reading it.

## Two container families

| Family | Magic | Versions | Where it comes from |
| --- | --- | --- | --- |
| Legacy single-Conv model | `ORNPUBIN` | 1, 2 | the first milestone: fixed geometry and packed weights for one of the eight encoded profiles - profile 1 is a single Conv, 2 a two-layer Conv, 3/4 Max/AveragePool, 5/6 a triple pool, 7/8 Conv plus triple pool - so one to five tasks (`open_rknpu.model`) |
| Task-table sequence | `ORNPUSEQ` | 3, 4, 5 | every modern profile: a list of tasks with register words and a payload (`open_rknpu.sequence`) |

Version 5 adds a **named tensor table**, which is what makes fan-out, several external
inputs/outputs and exposed intermediates possible. The runtime transparently supports all
of them; `open-rknpu inspect file.bin` prints the header, tasks, constants and tensor table
as JSON.

## Anatomy of a v5 container

```text
┌───────────────────────────────┐
│ 96-byte header                │  magic, version, geometry, checksum byte 80,
│                               │  task count byte 60, flags byte 84 (bit 0 = serial)
├───────────────────────────────┤
│ 16-byte v5 extension          │
├───────────────────────────────┤
│ task_count × 16-byte tasks    │  each task: payload offset, register count, enable mask
├───────────────────────────────┤
│ tensor table (v5)             │  name, role, layout, geometry, arena offset/size, API offset/size
├───────────────────────────────┤
│ payload                       │  per task: 126 register words (+4 tail words) then
│                               │  packed weights/biases; then the arena surfaces
└───────────────────────────────┘
```

* **Register words** are `tag << 48 | value << 16 | register`, 8 bytes each; the register
  profile (`open_rknpu.register_profile`) lists the 126 fields an emitted Conv task writes.
* **Tail control** is the successor's fetch amount: `PC_DATA_AMOUNT =
  (register_config_words + 4 + 2 − 1) / 2 − 1`, with `0x28` as the terminal sentinel. Because
  the tail is a fetch size rather than an engine code, every transition links and a whole
  mixed-engine DAG can be one submitted job.
* **Arena** holds the tensors. Layouts: `packed` (UINT8 rows padded to 16, C interleaved)
  for a legacy image input, `native16` (16 lanes per pixel, `((H·W+3)/4)·64` per plane) for
  internal grids and the native16 image stage, and `packed int8` for a few profiles.
* **Checksum** is FNV-1a over the header (with the checksum field itself zeroed), the v5
  extension, the task table, the tensor table and the payload; the runtime verifies it in
  `ornpu_inspect`/`ornpu_open` (both read and validate the whole file) before opening
  `/dev/rknpu`.
* **Flags bit 0** distinguishes serial (one descriptor per task) from a container whose
  tasks are already linked into engine runs.

## What the runtime does with it

`ornpu_open` reads the container with `stdio`, validates it, allocates two device buffers
through the driver (a 4 KiB task buffer and the payload/arena buffer) and `mmap`s *those*
buffers — the container file itself is never mapped. It then copies the task descriptors and
relocates each task's register addresses to the device's DMA base. `ornpu_run` then:

1. packs the caller's NHWC UINT8 buffer into the input tensor's arena layout (subtracting
   128 for native16 surfaces, filling padding with the input zero point);
2. zeroes the output tensors, syncs both buffers to the device;
3. submits the container's runs (one ioctl per run, or batched `--submission batched`);
4. syncs back and unpacks the output tensors to NHWC INT8.

`ornpu_run_io` is the multi-tensor variant used by v5 containers: every external input and
output must be bound exactly once by tensor index.

## Writing your own producer

The format is intentionally simple, and the writer is ~100 lines
(`open_rknpu/sequence.py`). If you want to emit a task program from another toolchain,
`encode_sequence_v5` takes the payload, task descriptors, tensor table and quantization
metadata, and validates the checksum and geometry for you. Cross-check what you wrote with
`open-rknpu inspect` and the C loader (`tests/board_io.c` prints a readable failure for
every malformed field it can detect: wrong magic, bad checksum, short payload, tensor index
out of range, input/output count mismatch).
