"""Probe ERDMA data modes for native spatial broadcast of an immutable constant.

Register 0x5034 is RDMA_ERDMA_CFG (Mesa): ERDMA_DATA_MODE bits 30..31,
ERDMA_SURF_MODE bit 29, ERDMA_DATA_SIZE bits 2..3. 0x5040 is EW_SURF_STRIDE.
The verified per-pixel constant stores a full [H,W,16] tensor with DATA_MODE=1;
the verified per-channel constant stores one 16-byte vector with DATA_MODE=0.

Hypothesis: a compact per-row table (H rows of 16 bytes) plus a small
EW_SURF_STRIDE reproduces the materialized per-row constant for some
DATA_MODE/SURF_MODE combination. All models share the suite's expected bytes, so
any variant that broadcasts correctly must match exactly.

Development experiment only; the public emitter never reads these variants.
"""
from pathlib import Path
import struct
import numpy as np
from open_rknpu.sequence import decode_sequence
from open_rknpu.model import checksum
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization, reference
from open_rknpu.elementwise import mul_reference

root = Path(__file__).resolve().parent
suite = root / "mul_broadcast_suite"
out = root / "mul_broadcast_mode_suite"
out.mkdir(exist_ok=True)

# Index 2 is kind='row' (per-row constant): height 7, width 5, input scale .5 zp 255.
model_path = suite / "model002.onnx"
import json
manifest_all = json.loads((suite / "manifest.json").read_text())
meta = manifest_all[2]
binary = (suite / "model002.bin").read_bytes()
info = decode_sequence(binary)
payload_start = 96 + 16 * info["task_count"]
payload = bytearray(binary[payload_start:])
constant_offset = 0x1000
height, width = 7, 5

# Expected bytes already exist from the verified suite.
inputs = np.fromfile(suite / "input002.u8", np.uint8).reshape(-1, height, width, 3)
expected = (suite / "expected002.i8").read_bytes()
(out / "input000.u8").write_bytes((suite / "input002.u8").read_bytes())

# Materialized constant bytes (int8 per pixel) as the suite computed them.
q = Quantization(**{k: np.array(v) if isinstance(v, list) else v for k, v in meta["branches"][0]["quantization"].items()})
factor = None
import onnx
from onnx import numpy_helper as nh
for tensor in onnx.load(str(model_path)).graph.initializer:
    if tensor.name == "factor":
        factor = nh.to_array(tensor)
b = np.clip(np.rint(np.broadcast_to(factor, (1, 3, height, width)) / meta["constant_scale"]), -128, 127).astype(np.int8)[0].transpose(1, 2, 0)

ew_task = info["tasks"][-1]
assert (ew_task["enable"], ew_task["mask"]) == (24, 768)
registers = {}
for i in range(ew_task["register_count"]):
    word = struct.unpack_from("<Q", payload, ew_task["command_offset"] + i * 8)[0]
    registers[word & 0xffff] = (i, (word >> 16) & 0xffffffff, word >> 48)
assert 0x5034 in registers and 0x5040 in registers and 0x5038 in registers

def variant(name, data_mode, surf_mode, stride, compact):
    data = bytearray(binary)
    body = bytearray(binary[payload_start:])
    if compact:
        table = np.zeros((height, 16), np.int8)
        for y in range(height):
            table[y, :3] = b[y, 0, :]
        body[constant_offset:constant_offset + table.nbytes] = table.tobytes()
    for reg, value in ((0x5034, (data_mode << 30) | (surf_mode << 29) | 4),
                       (0x5040, stride)):
        index, old, tag = registers[reg]
        offset = payload_start + ew_task["command_offset"] + index * 8
        struct.pack_into("<Q", data, offset, tag << 48 | value << 16 | reg)
    data[payload_start:] = bytes(body)
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(data))
    (out / f"model{len(manifest):03}.bin").write_bytes(bytes(data))
    (out / f"expected{len(manifest):03}.i8").write_bytes(expected)
    manifest.append(dict(index=len(manifest), variant=name, data_mode=data_mode, surf_mode=surf_mode,
                         stride=stride, compact=compact))

manifest = []
# Baseline: materialized per-pixel constant, unchanged registers.
variant("baseline_materialized", 1, 0, registers[0x5040][1], False)
# Compact row table with each candidate mode and the verified surface stride.
for data_mode in (0, 1, 2, 3):
    for stride in (16, ((height * width + 3) // 4) * 64):
        variant(f"compact_dm{data_mode}_stride{stride}", data_mode, 0, stride, True)
# Compact row table with SURF_MODE set.
for data_mode in (1, 2, 3):
    variant(f"compact_dm{data_mode}_surf1", data_mode, 1, 16, True)

import json
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} variants in {out}")
for entry in manifest:
    print(entry)
