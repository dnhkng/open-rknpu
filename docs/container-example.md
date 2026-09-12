# Container walkthrough: one real v5 file, field by field

This page opens one published container and reads every region of it against the byte
specification in [runtime/sequence_format.md](../runtime/sequence_format.md). The example is

```
research/walk_chain_suite/model000.bin        3,872 bytes
```

It is the smallest interesting v5 container in the tree: 3 tasks, 4 named tensors (one
external input, two internals, one external output), no constants, serial submission. It
is the op-level walk's `Conv3x3+Relu -> MaxPool -> Conv3x3` graph at 8×8 with 3 input
channels, and it is board-verified: model 0 of `walk_chain_suite` passed 16 inputs with
byte-exact output, as part of the suite's **12 models / 192 inferences / 6,368 exact
output bytes** ([research/walk_chain_suite/README.md](../research/walk_chain_suite/README.md),
[research/walk_chain_suite/board_results_0.json](../research/walk_chain_suite/board_results_0.json),
[research/walk_chain_suite/board_summary.txt](../research/walk_chain_suite/board_summary.txt)).

## Region map

| Range (offset–end) | Size | Region | Contents |
| --- | ---: | --- | --- |
| `0x0000`–`0x005F` | 96 | header | `<8s22I>` little-endian: magic, geometry, sizes, quantization, checksum, flags |
| `0x0060`–`0x006F` | 16 | v5 extension | `tensor_count`, `tensor_size`, `input_count`, `output_count` |
| `0x0070`–`0x009F` | 48 | task table | 3 × 16-byte descriptors |
| `0x00A0`–`0x019F` | 256 | tensor table | 4 × 64-byte tensor descriptors |
| `0x01A0`–`0x0D7F` | 3,456 | payload | register programs, tails, and packed weights/biases |
| — | — | (no constants) | v5 has no v4 constant descriptors |

Arithmetic check: `112 + 3×16 + 4×64 + 3456 = 3872`, the file size, and the Python decoder
asserts exactly that ([src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py),
`_decode_v5`).

## Dump it yourself

The decoder is the compiler's own parser, so anything below can be reproduced exactly:

```sh
PYTHONPATH=src python -c "from open_rknpu.sequence import decode_sequence; import json; \
  print(json.dumps(decode_sequence(open('research/walk_chain_suite/model000.bin','rb').read()), indent=2))"
```

Real output (unchanged, except that the decoder's `input_tensors`/`output_tensors` arrays,
which re-list the same descriptors filtered by role, are abbreviated here):

```json
{
  "target": "rv1103",
  "format_version": 5,
  "task_count": 3,
  "tasks": [
    { "command_offset": 0,    "register_count": 126, "enable": 29, "mask": 768  },
    { "command_offset": 1088, "register_count": 37,  "enable": 96, "mask": 3072 },
    { "command_offset": 1472, "register_count": 126, "enable": 29, "mask": 768  }
  ],
  "constants": [],
  "constant_count": 0,
  "serial": true,
  "input_tensor_count": 1,
  "output_tensor_count": 1,
  "tensors": [
    { "name": "input0",   "role": 0, "role_name": "input",    "layout": 0, "layout_name": "packed",
      "index": 0, "batch": 1, "height": 8, "width": 8, "channels": 3,
      "byte_offset": 4096, "bytes": 384 },
    { "name": "conv0",    "role": 2, "role_name": "internal", "layout": 1, "layout_name": "native16",
      "index": 0, "batch": 1, "height": 8, "width": 8, "channels": 8,
      "byte_offset": 4480, "bytes": 1024 },
    { "name": "maxpool1", "role": 2, "role_name": "internal", "layout": 1, "layout_name": "native16",
      "index": 0, "batch": 1, "height": 4, "width": 4, "channels": 3,
      "byte_offset": 5504, "bytes": 256 },
    { "name": "output",   "role": 1, "role_name": "output",   "layout": 1, "layout_name": "native16",
      "index": 0, "batch": 1, "height": 4, "width": 4, "channels": 3,
      "byte_offset": 5760, "bytes": 256 }
  ],
  "input_tensors":  [ { "name": "input0", "index": 0, "layout_name": "packed",   "byte_offset": 4096, "bytes": 384 } ],
  "output_tensors": [ { "name": "output", "index": 0, "layout_name": "native16", "byte_offset": 5760, "bytes": 256 } ],
  "tensor_count": 4,
  "shape_nhwc": [1, 8, 8, 3],
  "output_shape_nhwc": [1, 4, 4, 3],
  "batch": 1,
  "input_layout": "packed",
  "input_dtype": "uint8",
  "output_dtype": "int8",
  "input_scale": 1.0,
  "input_zero_point": 0,
  "output_scale": 538.692138671875,
  "output_zero_point": 0,
  "input_bytes": 192,
  "output_bytes": 48,
  "input_stride": 16,
  "payload_bytes": 3456,
  "arena_bytes": 8192,
  "input_offset": 4096,
  "output_offset": 5760,
  "task_bytes": 4096
}
```

The same header and tables come out of the C loader through `ornpu_inspect` /
`ornpu_get_info` / `ornpu_get_tensor` ([docs/c-api.md](c-api.md)); `open-rknpu inspect`
prints them from the CLI.

## The 96-byte header

The header is `<8s22I>`: an 8-byte magic then 22 little-endian `uint32` values, which the C
loader copies into `v[0..21]` ([runtime/open_rknpu.c](../runtime/open_rknpu.c),
`load_program`). The field names in [runtime/sequence_format.md](../runtime/sequence_format.md)
map to offsets as follows, with this file's bytes:

| Offset | `v[]` | Field | Bytes here | Value | Reading |
| ---: | ---: | --- | --- | ---: | --- |
| 0 | — | magic | `4f 52 4e 50 55 53 45 51` | `ORNPUSEQ` | task-table family |
| 8 | 0 | version | `05 00 00 00` | 5 | v5 named-tensor table |
| 12 | 1 | header size | `60 00 00 00` | 96 | must be 96 |
| 16 | 2 | input height | `08 00 00 00` | 8 | primary input (role index 0) |
| 20 | 3 | input width | `08 00 00 00` | 8 | |
| 24 | 4 | input channels | `03 00 00 00` | 3 | packed UINT8 RGB |
| 28 | 5 | output height | `04 00 00 00` | 4 | primary output |
| 32 | 6 | output width | `04 00 00 00` | 4 | |
| 36 | 7 | output channels | `03 00 00 00` | 3 | |
| 40 | 8 | input row stride | `10 00 00 00` | 16 | packed: `ceil(8/16)×16` |
| 44 | 9 | payload bytes | `80 0d 00 00` | 3456 | multiple of 64 |
| 48 | 10 | arena bytes | `00 20 00 00` | 8192 | multiple of 4096 |
| 52 | 11 | input offset in arena | `00 10 00 00` | 4096 | 64-aligned, ≥ payload |
| 56 | 12 | output offset in arena | `80 16 00 00` | 5760 | |
| 60 | 13 | task count | `03 00 00 00` | 3 | 1..64 |
| 64 | 14 | input scale bits | `00 00 80 3f` | `0x3f800000` = 1.0f | `real = (byte − zp) × scale` |
| 68 | 15 | input zero point | `00 00 00 00` | 0 | UINT8, 0..255 |
| 72 | 16 | output scale bits | `4c ac 06 44` | `0x4406ac4c` ≈ 538.692 | |
| 76 | 17 | output zero point | `00 00 00 00` | 0 | signed INT8, −128..127 |
| 80 | 18 | checksum | `6a 0a 81 5b` | `0x5b810a6a` | whole-file FNV-1a, field zeroed |
| 84 | 19 | flags / constant count | `01 00 00 00` | bit 0 = 1 | serial submission; bits 8+ = 0 constants |
| 88 | 20 | input layout | `00 00 00 00` | 0 | packed UINT8 (1 would be native16) |
| 92 | 21 | batch − 1 | `00 00 00 00` | 0 | v5: batch = 1 |

Two notes on the spec table's wording:

* byte 92 is listed as "Reserved, must be zero" in the v3/v4 part of
  [runtime/sequence_format.md](../runtime/sequence_format.md), but for **v5** it is
  `batch − 1` (validated `≤ 15`, [runtime/open_rknpu.c](../runtime/open_rknpu.c), `load_v5`);
  a batch-16 v5 container would fail a literal reading of the old table.
* byte 84's bits 8+ are the v4 constant count; a v5 container must have them zero, which is
  why this file has no constant descriptors.

## The 16-byte v5 extension

Immediately after the header, four `uint32` values
([runtime/sequence_format.md](../runtime/sequence_format.md), "Version 5 named tensors"):

| Offset | Field | Bytes here | Value |
| ---: | --- | --- | ---: |
| 96 | `tensor_count` | `04 00 00 00` | 4 |
| 100 | `tensor_size` | `40 00 00 00` | 64 |
| 104 | `input_count` | `01 00 00 00` | 1 |
| 108 | `output_count` | `01 00 00 00` | 1 |

`tensor_size` is fixed at 64 and checked by both the C loader and the Python decoder; the
extension is what tells a reader how to find the tensor table after the task descriptors.

## One task descriptor

The task table starts at 112 (`0x70`). Each descriptor is `<4I>`: command offset (relative
to the **start of the payload**, not the file), register-word count, enable mask, interrupt
mask ([runtime/sequence_format.md](../runtime/sequence_format.md)):

| Task | Offset | `command_offset` | `register_count` | `enable` | `mask` | Family |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `0x70` | 0 | 126 | 29 | 768 (`0x300`) | dense Conv, CNA |
| 1 | `0x80` | 1088 (`0x440`) | 37 | 96 | 3072 (`0xC00`) | pool, DPU |
| 2 | `0x90` | 1472 (`0x5C0`) | 126 | 29 | 768 | dense Conv, CNA |

Task 0 in raw bytes: `00 00 00 00 | 7e 00 00 00 | 1d 00 00 00 | 00 03 00 00`. The
`(register_count, enable)` pairs `(126, 29)` and `(37, 96)` are the composer's `NATIVE` and
`POOL_TASK` families ([src/open_rknpu/compose.py](../src/open_rknpu/compose.py)), and the
same signatures the C loader accepts ([runtime/open_rknpu.c](../runtime/open_rknpu.c),
`valid_kind`). The 126-word Conv slot is `(126 + 4) × 8 = 1040` bytes, but the next task
starts at 1088: program slots are aligned to 64 and carry four terminal words
([src/open_rknpu/compose.py](../src/open_rknpu/compose.py), placement policy).

Because this is a **serial** container (header byte 84 bit 0 = 1), every task is submitted
in its own ioctl. A linked container would clear bit 0 and write a next-command offset into
each task's tail word; the rule and the batched form are in
[runtime/sequence_format.md](../runtime/sequence_format.md) and
[docs/plans/pipelining-plan.md](plans/pipelining-plan.md).

## The first register words

Each register word is 8 bytes: `tag << 48 | value << 16 | register`. The payload begins at
`0x01A0` (416), so task 0's program is the first 126 words of the payload. The first words
are the profile defaults from
[src/open_rknpu/register_profile.py](../src/open_rknpu/register_profile.py); later words
carry the emitter's geometry and addresses:

| Word | `tag` | register | value | Note |
| ---: | ---: | ---: | ---: | --- |
| 0 | `0x0201` | `0x1004` | `0x0e` | profile default `(0x1004, 0x0000000e, 0x0201)` |
| 1 | `0x0801` | `0x3004` | `0x0e` | profile default `(0x3004, 0x0000000e, 0x0801)` |
| 2 | `0x0201` | `0x1040` | `0x08` | profile default `(0x1040, 0x00000008, 0x0201)` |
| 3 | `0x1001` | `0x4004` | `0x0e` | profile default `(0x4004, 0x0000000e, 0x1001)` |
| 4 | `0x2001` | `0x5004` | `0x0e` | profile default `(0x5004, 0x0000000e, 0x2001)` |
| 5 | `0x0201` | `0x100c` | `0x2000a000` | profile default |
| 6 | `0x0201` | `0x1010` | `0x000003ff` | profile default; the walk changes this for later Convs (`0x108`/`0x104` for K3/K1, [research/walk_chain_suite/README.md](../research/walk_chain_suite/README.md)) |
| 7 | `0x0201` | `0x1014` | `0x00000009` | profile default |
| 8 | `0x0201` | `0x101c` | `0xffe00000` | profile default |
| 9 | `0x0201` | `0x1020` | `0x00080008` | profile default |
| 10 | `0x0201` | `0x1024` | `0x00020010` | profile default |
| 11 | `0x0201` | `0x1028` | `0x00000008` | profile default |
| 29 | `0x0201` | `0x1070` | `0x00001000` = **4096** | the **input** address: profile default `0x2000`, overridden to the `input0` arena offset. `NATIVE.reads = (0x1070,)` |
| 50 | `0x1001` | `0x400c` | `0x000001e4` | profile default |
| 55 | `0x1001` | `0x4020` | `0x00001180` = **4480** | the **output** address: profile default `0x3000`, overridden to the `conv0` arena offset. `NATIVE.writes = (0x4020,)` |
| 56 | `0x1001` | `0x4024` | `0x00000400` = **1024** | `conv0` arena bytes: native16 `ceil(64/4)×64×ceil(8/16)` |
| 71 | `0x1001` | `0x4060` | `0x00000012` | activation **on** (profile default `0x13`): the Relu after this Conv |
| 74 | `0x1001` | `0x406c` | `0x00000000` | activation clamp low |
| 87 | `0x1001` | `0x40c0` | `0x00000400` = 1024 | bias block size (profile default) |
| 91 | `0x1001` | `0x40e0` | `0x00000000` | activation clamp high |

Task 1 is the pool; its address registers follow `POOL_TASK`'s declared bindings
`reads = (0x701c,)`, `writes = (0x6070,)`
([src/open_rknpu/compose.py](../src/open_rknpu/compose.py)):

| Word | register | value | Reading |
| ---: | ---: | ---: | --- |
| 24 | `0x6070` | `0x00001580` = **5504** | write `maxpool1` |
| 31 | `0x701c` | `0x00001180` = **4480** | read `conv0` |

Task 2 shows the geometry-dependent registers changing as the walk predicts: `0x1010` is
`0x108` (a different kernel/geometry path), `0x101c` is 0, `0x1020`/`0x1024`/`0x1028` are
`0x00040004`/`0x00070010`/`0x00000004`, `0x1070` is 5504 (read `maxpool1`), and `0x4020`
is 5760 (write `output`, 256 arena bytes). Its activation registers are **off**
(`0x4060 = 0x13`, `0x406c = 0x40e0 = 0x80000000`), as the graph's final Conv should be
([src/open_rknpu/register_profile.py](../src/open_rknpu/register_profile.py);
the activation encoding is explained in
[docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S9).

Each program's last four words are the tail. In this serial container the terminal tail is
`reg 0x10 = 0` (no next program) and `reg 0x14 = 0x28` (the terminal sentinel), followed by
two setup words:

```
tail+0  0x0101000000000010   reg 0x0010 = 0
tail+8  0x0101000000280014   reg 0x0014 = 0x28   (terminal)
tail+16 0x0041000000000000
tail+24 0x00810000001d0008
```

`0x28` as the terminal control word and the successor's fetch amount for a link are
documented in [runtime/sequence_format.md](../runtime/sequence_format.md) and
[src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py) (`TERMINAL_CONTROL`,
`amount_control`).

## The tensor table

The tensor table starts at `112 + 3×16 = 160` (`0xA0`). Each entry is `<24s10I>` (64
bytes): name, role, layout, index, batch, height, width, channels, arena offset, arena
size, reserved ([runtime/sequence_format.md](../runtime/sequence_format.md)):

| Name | role | layout | index | batch | H | W | C | arena offset | arena bytes | API bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `input0` | 0 input | 0 packed | 0 | 1 | 8 | 8 | 3 | 4096 | 384 | **192** |
| `conv0` | 2 internal | 1 native16 | 0 | 1 | 8 | 8 | 8 | 4480 | 1024 | — |
| `maxpool1` | 2 internal | 1 native16 | 0 | 1 | 4 | 4 | 3 | 5504 | 256 | — |
| `output` | 1 output | 1 native16 | 0 | 1 | 4 | 4 | 3 | 5760 | 256 | **48** |

The arena sizes follow the spec's formulas
([runtime/sequence_format.md](../runtime/sequence_format.md)):

* packed: `1 × 8 × ceil(8/16)×16 × 3 = 384` — but the API buffer is the flat
  `1 × 8 × 8 × 3 = 192` bytes, because the packed row stride of 16 is an arena detail;
* native16: `ceil(H·W/4) × 64 × ceil(C/16)`, giving `16×64×1 = 1024` for the 8×8/C8
  `conv0` and `4×64×1 = 256` for the 4×4/C3 surfaces.

The arena holding them is 8,192 bytes: the payload occupies `0..3455`, the external input
starts at the 4096-aligned payload end, `conv0` follows it at 4480, the internals are
placed by liveness (`maxpool1` at 5504), the external output follows at 5760, and the whole
arena is rounded up to the next 4096. That is the placement policy in
[src/open_rknpu/compose.py](../src/open_rknpu/compose.py), and it is why
`arena_offset ≠ api_offset`, `arena_bytes ≠ api_bytes`.

## The checksum

Byte 80 holds `0x5b810a6a` (1535183466). The checksum is FNV-1a computed over the file with
**field 80 itself zeroed**; the C loader reconstructs exactly this prefix — header,
extension, task table, tensor table, payload — and compares before it opens the device
([runtime/open_rknpu.c](../runtime/open_rknpu.c), `load_v5`). The Python encoder computes
the same thing with `open_rknpu.model.checksum` ([src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py)).

Recompute it in two lines:

```python
from open_rknpu.model import checksum
data = bytearray(open('research/walk_chain_suite/model000.bin','rb').read())
data[80:84] = b'\0\0\0\0'
print(hex(checksum(bytes(data))))     # 0x5b810a6a
```

The coverage is broader than "the payload": an FNV over the 3,456 payload bytes **only**
would be `0x82e48bb2`, not `0x5b810a6a`. The checksum protects every table from silent
edits, which is why the runtime can reject a hand-edited container before submitting it.

## A real hexdump

```text
00000000: 4f524e50 55534551 05000000 60000000  ORNPUSEQ....`...
00000010: 08000000 08000000 03000000 04000000  ................
00000020: 04000000 03000000 10000000 800d0000  ................
00000030: 00200000 00100000 80160000 03000000  . ..............
00000040: 0000803f 00000000 4cac0644 00000000  ...?....L..D....
00000050: 6a0a815b 01000000 00000000 00000000  j..[............
00000060: 04000000 40000000 01000000 01000000  ....@...........
00000070: 00000000 7e000000 1d000000 00030000  ....~...........
00000080: 40040000 25000000 60000000 000c0000  @...%...`.......
00000090: c0050000 7e000000 1d000000 00030000  ....~...........
000000a0: 696e7075 74300000 00000000 00000000  input0..........
000000b0: 00000000 00000000 00000000 00000000  ................
000000c0: 00000000 01000000 08000000 08000000  ................
000000d0: 03000000 00100000 80010000 00000000  ................
000000e0: 636f6e76 30000000 00000000 00000000  conv0...........
000000f0: 00000000 00000000 02000000 01000000  ................
00000100: 00000000 01000000 08000000 08000000  ................
00000110: 08000000 80110000 00040000 00000000  ................
00000120: 6d617870 6f6f6c31 00000000 00000000  maxpool1........
00000130: 00000000 00000000 02000000 01000000  ................
00000140: 00000000 01000000 04000000 04000000  ................
00000150: 03000000 80150000 00010000 00000000  ................
00000160: 6f757470 75740000 00000000 00000000  output..........
00000170: 00000000 00000000 01000000 01000000  ................
00000180: 00000000 01000000 04000000 04000000  ................
00000190: 03000000 80160000 00010000 00000000  ................
000001a0: 04100e00 00000102 04300e00 00000108  .........0......
000001b0: 40100800 00000102 04400e00 00000110  @.......@.......
000001c0: 04500e00 00000120 0c1000a0 00200102  .P..... ..... ..
000001d0: 1010ff03 00000102 14100900 00000102  ................
000001e0: 1c100000 e0ff0102 20100800 08000102  ........ .......
000001f0: 24101000 02000102 28100800 00000102  $.......(.......
```

Read it left to right: `0x00`–`0x5F` is the header (`05000000` is version 5, `60000000`
is header size 96, `0000803f` is the input scale 1.0f, `6a0a815b` is the checksum);
`0x60`–`0x6F` is the extension; `0x70`–`0x9F` is the task table (`7e000000` = 126,
`1d000000` = 29, `00030000` = 768); `0xA0`–`0x19F` is the tensor table (`696e7075 7430` =
`inpu`/`t0`); and `0x1A0` begins task 0's program (`04100e00 00000102` is the little-endian
word `0x02010000000e1004`, i.e. the `0x1004` register). The `xxd -g 4` grouping puts four
little-endian `uint32`s per 16-byte line.

## How to write your own producer

Do not hand-assemble the header. The encoder validates shapes, tensor sizes, arena ranges,
task signatures, quantization and the checksum, and it re-decodes its own output as a
self-check ([src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py)):

```python
from open_rknpu.sequence import encode_sequence_v5

data = encode_sequence_v5(
    payload,                       # register programs + tails + packed parameters
    tensors=[                     # one dict per tensor descriptor
        {"name": "input0", "role": 0, "layout": 0, "index": 0,
         "shape": (1, 8, 8, 3), "offset": 4096, "size": 384},
        {"name": "output", "role": 1, "layout": 1, "index": 0,
         "shape": (1, 4, 4, 3), "offset": 4608, "size": 256},
    ],
    tasks=[(0, 126, 29, 768), (1088, 37, 96, 3072)],   # (offset, words, enable, mask)
    arena_bytes=8192,
    input_scale=1.0, input_zero_point=0,
    output_scale=sc, output_zero_point=zp,
    serial=True,                   # see the tail note below for a linked container
    constants=(),                  # v5 has no constant descriptors
)
open("model.bin", "wb").write(data)
```

`encode_sequence_v5` packs the header, extension, task table, tensor table and checksum;
it does **not** rewrite the four tail words inside `payload`. A serial container is already
terminal there, so `serial=True` with terminal tails is complete. To emit a linked
(batched) container, either write each task's next-command link and successor fetch amount
yourself, or emit the serial form and pass it through
`open_rknpu.sequence.relink_for_batched(data)`, which rewrites every tail, clears header
flag bit 0 and re-checksums the file without touching the programs
([src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py), `relink_for_batched`).

`encode_sequence_v5` is the v5 entry point; v3/v4 use `encode_sequence`
([src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py)). After writing, always:

1. `decode_sequence(data)` (or `open-rknpu inspect model.bin`) and check the header,
   tensor table and task list round-trip;
2. run it through the C loader — `tests/board_io.c` / `tests/board_api.c` print a
   readable failure for every malformed field they can detect (wrong magic, bad checksum,
   short payload, tensor index out of range, input/output count mismatch);
3. compare every output byte against an independent integer reference for the same
   quantization parameters ([docs/verification.md](verification.md)).

For the container family as a whole — the v1/v2 legacy executable, v3/v4 task tables, and
the v5 named-tensor table — see [container-format.md](container-format.md), and for the
runtime contract that consumes the file see [docs/c-api.md](c-api.md).
