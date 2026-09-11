"""Check the candidate 1x1 output-channel packing against synchronized oracles.

Development evidence only: this reads vendor captures, not production artifacts.
The single-output compiler specialization is intentionally not covered here.
"""
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parent


def verify(channels):
    name = f"out{channels}"
    capture = ROOT / f"capture_{name}"
    data = (capture / "run0_before_mem1.bin").read_bytes()
    registers = {word & 65535: (word >> 16) & 0xffffffff
                 for word in struct.unpack_from("<126Q", data)}
    aligned = (channels + 3) // 4 * 4
    assert registers[0x1030] == aligned * 4
    assert registers[0x1038] == 0x01010000 | aligned
    assert registers[0x403c] == ((aligned - 1) << 16) | 15
    weights = np.array(struct.unpack_from(f"<{channels * 4}b", data,
                                         registers[0x1110]), dtype=np.int64)
    weights = weights.reshape(channels, 4)[:, :3]
    bias, offsets, multipliers = [], [], []
    for c in range(channels):
        block = registers[0x5020] + (c // 4) * 32
        lane = c % 4
        bias.append(struct.unpack_from("<i", data, block + lane * 4)[0])
        offsets.append(struct.unpack_from("<h", data, block + 16 + lane * 2)[0])
        multipliers.append(struct.unpack_from("<H", data, block + 24 + lane * 2)[0])
    centered = weights + np.array(offsets)[:, None]
    shift, multiplier = registers[0x4088], registers[0x4084]
    zp = registers[0x4080]
    if zp >= 1 << 31:
        zp -= 1 << 32
    outputs = [bytes.fromhex(line.split()[2]) for line in
               (ROOT / f"{name}_capture.log").read_text().splitlines()
               if line.startswith("OUTPUT ")]
    assert len(outputs) == 4
    total = 0
    for index, raw in enumerate(outputs):
        inputs = np.frombuffer((capture / f"run{index}_before_mem2.bin").read_bytes(),
                               dtype=np.uint8)[:384].reshape(8, 16, 3)[:, :8].astype(np.int64)
        accumulator = np.einsum("hwc,oc->hwo", inputs - 128, centered) + bias
        product = accumulator * multipliers
        scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
        predicted = np.clip(((scaled * multiplier + (1 << (shift - 1))) >> shift)
                            + zp, -128, 127).astype(np.int8)
        observed = np.frombuffer(raw, dtype=np.int8).reshape(8, 8, 16)[:, :, :channels]
        np.testing.assert_array_equal(predicted, observed)
        total += observed.size
    print(f"{name}: {total}/{total} bytes match; output tile={aligned}, table groups={(channels+3)//4}")


if __name__ == "__main__":
    for channels in (2, 4, 5, 8, 16):
        verify(channels)
