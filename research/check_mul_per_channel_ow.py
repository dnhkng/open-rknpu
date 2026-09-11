"""Probe per-channel Mul output conversion via BS_OW_CFG.OW_SRC and the BS table.

Register 0x4050 is `BS_OW_CFG` (Mesa): `OW_SRC@0`, `OD_BYPASS@1`. Every verified
Conv/depthwise emitter runs with `OW_SRC=1`/`OD_BYPASS=0` and reads its per-channel
bias/scale block through `0x5020`, whose decoded format is a 32-byte block per four
output channels: INT32 bias at +0..15, `-weight_zero_point` INT16 at +16..23 and the
Q14 channel multiplier UINT16 at +24..31 (`open_rknpu/depthwise.py`,
`open_rknpu/transposed.py`). The verified elementwise Mul runs `OW_SRC=0` with the
scalar `0x4084`/`0x4088` conversion.

The first experiment (2026-09-10, `model001.bin`) set `OW_SRC=1` with a 32-byte table
written at **file** offset 0x1800 while `0x5020` names a **payload** offset, i.e. 0x80
bytes away, and used a four-UINT16 multiplier layout. It hung the NPU and is retained
for history.

This script rebuilds the baseline plus *corrected* variants: the table is written at the
address `0x5020` actually names, in the decoded Conv block format, so the datapath has a
well-formed table to read. Development experiment only; the compiler never reads this
directory.

    PYTHONPATH=src python research/check_mul_per_channel_ow.py
"""
from pathlib import Path
import struct

from open_rknpu.model import checksum
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "mul_broadcast_suite"
OUT = ROOT / "mul_per_channel_ow_suite"
OUT.mkdir(exist_ok=True)
SOURCE = "model007"
TABLE = 0x1800  # payload offset named by 0x5020; the file offset is header + TABLE


def block(multipliers, biases=(0, 0, 0, 0)):
    """One decoded BS block: four INT32 biases, four INT16 weight zps, four Q14 scales."""
    data = bytearray(32)
    for lane, bias in enumerate(biases):
        struct.pack_into("<i", data, lane * 4, int(bias))
    for lane, multiplier in enumerate(multipliers):
        struct.pack_into("<H", data, 24 + lane * 2, int(multiplier))
    return data


def reseal(data):
    """Constant Mul containers are ORNPUSEQ v3: checksum at offset 80."""
    data = bytearray(data)
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(bytes(data)))
    return bytes(data)


def patch_ew(data, fields, table=None):
    """Patch the elementwise task's registers (and optionally its BS table)."""
    data = bytearray(data)
    info = decode_sequence(bytes(data))
    start = 96 + 16 * info["task_count"]
    task = info["tasks"][-1]
    assert task["enable"] == 24, task
    for register, value in fields.items():
        found = False
        for i in range(task["register_count"]):
            offset = start + task["command_offset"] + i * 8
            word = struct.unpack_from("<Q", data, offset)[0]
            if (word & 0xFFFF) == register:
                struct.pack_into("<Q", data, offset,
                                 (word >> 48) << 48 | (value & 0xFFFFFFFF) << 16 | register)
                found = True
                break
        if not found:
            raise KeyError(hex(register))
    if table is not None:
        at = start + TABLE
        data[at:at + 32] = table
    return bytes(data)


def _legacy(binary):
    """The superseded 2026-09-10 variant: file-offset table, 4xUINT16 at +24."""
    data = bytearray(binary)
    data = bytearray(patch_ew(data, {0x4050: 0x30000001, 0x5020: TABLE}))
    table = bytearray(32)
    struct.pack_into("<4H", table, 24, 16384, 0, 0, 0)
    data[TABLE:TABLE + 32] = table
    return bytes(data)


def main():
    binary = bytearray((SUITE / f"{SOURCE}.bin").read_bytes())
    input_bytes = (SUITE / f"input{SOURCE[5:]}.u8").read_bytes()
    identity = block([16384] * 4)
    per_channel = block([16384, 0, 0, 0])

    variants = {
        # unchanged baseline (scalar conversion, OW_SRC=0)
        "model000": bytes(binary),
        # superseded 2026-09-10 attempt: table written at the *file* offset 0x1800
        # (0x80 bytes past the payload address 0x5020 names) with a 4xUINT16 layout.
        # Reproduced byte-for-byte so the retained hang stays regenerable.
        "model001": reseal(_legacy(binary)),
        # corrected table address and decoded block format, OW_SRC=1 / OD_BYPASS=0
        "model002": patch_ew(binary, {0x4050: 0x30000001, 0x5020: TABLE}, identity),
        # same, but only channel 0 keeps a unit multiplier
        "model003": patch_ew(binary, {0x4050: 0x30000001, 0x5020: TABLE}, per_channel),
        # OW_SRC=1 with OD_BYPASS kept set, to isolate the two bits
        "model004": patch_ew(binary, {0x4050: 0x30000003, 0x5020: TABLE}, per_channel),
        # table present but the verified OW_SRC=0 config, as a table-presence control
        "model005": patch_ew(binary, {0x4050: 0x30000002, 0x5020: TABLE}, per_channel),
    }
    for name, data in variants.items():
        (OUT / f"{name}.bin").write_bytes(reseal(data))
        (OUT / f"input{name[5:]}.u8").write_bytes(input_bytes)
    print("built", " ".join(sorted(variants)), "in", OUT)


if __name__ == "__main__":
    main()
