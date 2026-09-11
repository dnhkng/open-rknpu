"""Probe native spatial broadcast using the decoded ERDMA notch registers.

The RK3588 TRM (the same NPU IP) decodes the registers the earlier sweep only guessed
at:

* `0x5034` `erdma_cfg`: `data_mode` 31:30 (0 per channel, 1 per pixel, 2 per channel
  by pixel, 3 reserved), `surf_mode` 29 (1 or 2 surface series), `data_size` 3:2
  (1 = 8-bit), `erdma_disable` 0;
* `0x5040` `ew_surf_stride`: the EW operand surface stride, in 16-byte units (the TRM
  requires 1 for per-channel mode, which is what the verified profile stores as 0x10);
* `0x5010` bits 28:16 `ew_line_notch_addr`: "line notch of EW";
* `0x506c` `ew_surf_notch`: "how many pixels from the end of this process feature map
  to the end of the shape feature map".

The verified per-row constant materializes a full `[H,W,16]` operand plane
(`ew_surf_stride` = plane atoms, `ew_surf_notch` = 0). The earlier compact sweep left
both notches at zero, so the ERDMA kept walking past the compact table - which is why
every variant hung instead of broadcasting.

This probe keeps the compact `H`-row table (one 16-byte atom per row) but sets the
pixel notch to the operand row's shortfall, so the address should wrap at the end of
each operand row and repeat it across the output row. Development experiment only.
"""
from pathlib import Path
import json
import struct

import numpy as np

from open_rknpu.model import checksum
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "mul_broadcast_suite"
OUT = ROOT / "mul_broadcast_notch_suite"
OUT.mkdir(exist_ok=True)
SOURCE = SUITE / "model002.bin"          # verified per-row constant: 7 rows x 5 cols x 3 ch
HEIGHT, WIDTH, CHANNELS = 7, 5, 3
OPERAND = 0x1000                          # payload-relative EW operand base (0x5038)
PLANE_ATOMS = ((HEIGHT * WIDTH + 3) // 4) * 4   # 36 atoms = the materialized plane


def ew_words(binary):
    info = decode_sequence(binary)
    header = 96 + 16 * info["task_count"]
    task = info["tasks"][-1]
    assert (task["enable"], task["mask"]) == (24, 768), task
    words = {}
    for index in range(task["register_count"]):
        offset = header + task["command_offset"] + index * 8
        word = struct.unpack_from("<Q", binary, offset)[0]
        words[word & 0xFFFF] = (index, offset, word >> 48, (word >> 16) & 0xFFFFFFFF)
    return info, task, words


def variant(name, table, fields):
    binary = bytearray(SOURCE.read_bytes())
    info, task, words = ew_words(bytes(binary))
    header = 96 + 16 * info["task_count"]
    body = bytearray(binary[header:])
    if table is not None:
        body[OPERAND:OPERAND + PLANE_ATOMS * 16] = bytes(PLANE_ATOMS * 16)
        body[OPERAND:OPERAND + len(table)] = table
    # Patch the register words *inside the body*: writing them into `binary` and then
    # restoring the body from a pre-patch copy silently dropped every change once.
    for register, value in fields.items():
        index, offset, tag, _ = words[register]
        struct.pack_into("<Q", body, offset - header, tag << 48 | value << 16 | register)
    binary[header:] = body
    binary[80:84] = b"\0" * 4
    struct.pack_into("<I", binary, 80, checksum(bytes(binary)))
    index = len(manifest)
    (OUT / f"model{index:03}.bin").write_bytes(bytes(binary))
    (OUT / f"input{index:03}.u8").write_bytes((SUITE / "input002.u8").read_bytes())
    (OUT / f"expected{index:03}.i8").write_bytes((SUITE / "expected002.i8").read_bytes())
    manifest.append(dict(index=index, variant=name, **{hex(k): hex(v) for k, v in fields.items()}))


# The compact per-row table: one 16-byte atom per row holds that row's three channels,
# packed exactly as the materialized plane packs pixel (row, 0).
import onnx
from onnx import numpy_helper as nh

manifest_all = json.loads((SUITE / "manifest.json").read_text())
constant_scale = manifest_all[2]["constant_scale"]
factor = None
for tensor in onnx.load(str(SUITE / "model002.onnx")).graph.initializer:
    if tensor.name == "factor":
        factor = nh.to_array(tensor)
constant = np.clip(np.rint(np.broadcast_to(factor, (1, CHANNELS, HEIGHT, WIDTH)) / constant_scale),
                   -128, 127).astype(np.int8)[0].transpose(1, 2, 0)
table = np.zeros((HEIGHT, 16), np.int8)
for row in range(HEIGHT):
    table[row, :CHANNELS] = constant[row, 0, :]

manifest = []
# Control: the verified materialized per-row operand.
variant("baseline_materialized", None, {0x5040: 0x240, 0x506C: 0x0})
# Compact table, per-pixel mode: the pixel notch ends the operand row after one pixel.
variant("compact_stride_plane_notch_w-1", table.tobytes(),
        {0x5040: PLANE_ATOMS << 4, 0x506C: (WIDTH - 1) << 4})
variant("compact_stride_1_notch_w-1", table.tobytes(),
        {0x5040: 1 << 4, 0x506C: (WIDTH - 1) << 4})
variant("compact_stride_plane_notch_w", table.tobytes(),
        {0x5040: PLANE_ATOMS << 4, 0x506C: WIDTH << 4})
# Mode 2 is documented "per channel by pixel" - the likely broadcast mode.
variant("compact_mode2_notch_w-1", table.tobytes(),
        {0x5034: 0x80000004, 0x5040: PLANE_ATOMS << 4, 0x506C: (WIDTH - 1) << 4})
# Mode 1 with the second surface series, in case the operand rows are a second series.
variant("compact_mode1_surf1_notch_w-1", table.tobytes(),
        {0x5034: 0x60000004, 0x5040: PLANE_ATOMS << 4, 0x506C: (WIDTH - 1) << 4})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"built {len(manifest)} notch variants in {OUT}")
for entry in manifest:
    print(" ", entry)
