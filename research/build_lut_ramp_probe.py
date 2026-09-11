"""Ramp-table probe: read the LUT index directly.

Replaces the Sigmoid/Sigmoid tables with `value(i) = clip((i-512)*4)` so the
output code is an affine, saturating readout of the table index actually used by
the hardware. Combined with the declared-scale variants from
`check_lut_domain.py`, this separates "the table argument is the dequantized
value" from "the argument is the wide requantized code".

Development experiment only; the public emitter never reads these files.
"""
from pathlib import Path
import json
import struct
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.lut import compile_lut
from open_rknpu.model import checksum
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lut_domain_probe"
FACTOR = 1 / 32
VARIANTS = [("ramp_scale2048", 1 / 2048, 0), ("ramp_scale1024", 1 / 1024, 0),
            ("ramp_scale4096", 1 / 4096, 0), ("ramp_scale512", 1 / 512, 0)]


def sigmoid_model():
    w = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1) * FACTOR
    b = np.zeros(3, np.float32)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node("Sigmoid", ["conv"], ["output"])], "ramp",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(w, "w"), nh.from_array(b, "b")]), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def patch_table(binary):
    info = decode_sequence(binary)
    start = 96 + 16 * info["task_count"]
    data = bytearray(binary[start:])
    setup = info["tasks"][0]
    bank = None
    entry = 0
    for i in range(setup["register_count"]):
        offset = setup["command_offset"] + i * 8
        word = struct.unpack_from("<Q", data, offset)[0]
        reg = word & 0xffff
        value = word >> 16 & 0xffffffff
        if reg == 0x4100:
            bank = {0x20000: 0, 0x30000: 512}.get(value)
            entry = 0
            continue
        if reg == 0x4104 and bank is not None:
            index = bank + entry
            ramp = int(np.clip((index - 512) * 4, -32768, 32767)) & 0xffff
            struct.pack_into("<Q", data, offset, (word >> 48) << 48 | ramp << 16 | reg)
            entry += 1
    result = bytearray(binary[:start]) + data
    result[80:84] = b"\0" * 4
    struct.pack_into("<I", result, 80, checksum(result))
    return bytes(result)


manifest = []
for index, (name, scale, zero) in enumerate(VARIANTS):
    binary, meta = compile_lut(sigmoid_model(), 1, 128, declared_scale=scale, declared_zero_point=zero)
    binary = patch_table(binary)
    (OUT / f"model{100 + index:03}.bin").write_bytes(binary)
    pixels = np.zeros((2, 8, 8, 3), np.uint8)
    pixels[0, :, :, :] = (np.arange(64) * 4).reshape(8, 8, 1)
    pixels[1, :, :, :] = (255 - np.arange(64) * 4).reshape(8, 8, 1)
    for j in range(2):
        (OUT / f"run{100 + index:03}_{j}.u8").write_bytes(pixels[j].tobytes())
    manifest.append({"index": 100 + index, "variant": name, "declared_scale": scale, "declared_zero_point": zero})
    print(index, name, "scale", scale, flush=True)
(OUT / "ramp_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("built", len(manifest), "ramp variants")
