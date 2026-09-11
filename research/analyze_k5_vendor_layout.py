"""Retained analysis: vendor K5 depthwise ConvTranspose register set and weights.

The public emitter rejects K5, so this script reconstructs the *vendor* ground
truth from `research/capture_transpose_k5/` and diffs it against the retained
`research/transpose_k5_suite/model000.bin` that the old experiment shipped:

* the vendor task's 126-word register set, including the fields the stale binary
  left at K3 depthwise defaults;
* the 25-tap weight table layout (32 bytes per tap: 16 lanes of
  `(value, -weight_zero_point)` pairs);
* the stale binary's own weight table, which populates only 17 of 25 taps.

Development-only: the public emitter never reads these artifacts.

    PYTHONPATH=src python research/analyze_k5_vendor_layout.py
"""
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT / "capture_transpose_k5" / "run0_before_mem1.bin"
STALE = ROOT / "transpose_k5_suite" / "model000.bin"
TAGS = (0x1001, 0x201, 0x101, 0x81, 0x41, 0x801, 0x2001, 0x401, 0x1)
INTERESTING = (0x100c, 0x1010, 0x1014, 0x1018, 0x101c, 0x1020, 0x1024, 0x1028, 0x102c,
               0x1030, 0x1034, 0x1038, 0x103c, 0x1040, 0x1044, 0x1048, 0x104c, 0x1050,
               0x1054, 0x1058, 0x105c, 0x1060, 0x1064, 0x1068, 0x106c, 0x1070, 0x1074,
               0x1078, 0x107c, 0x1080, 0x1084, 0x1088, 0x1094, 0x1100, 0x1104, 0x1110,
               0x1184, 0x1188, 0x118c, 0x3010, 0x3014, 0x3018, 0x301c, 0x400c, 0x4010,
               0x4014, 0x4018, 0x401c, 0x4020, 0x4024, 0x4028, 0x402c, 0x4030, 0x4034,
               0x4038, 0x403c, 0x4040, 0x4044, 0x4048, 0x4050, 0x4054, 0x4058, 0x405c,
               0x4060, 0x4068, 0x406c, 0x4070, 0x4074, 0x4078, 0x4080, 0x4084, 0x4088,
               0x40c0, 0x500c, 0x5010, 0x5014, 0x5018, 0x501c, 0x5020, 0x5044)


def vendor_task():
    words = struct.unpack_from("<%dQ" % (CAPTURE.stat().st_size // 8), CAPTURE.read_bytes())
    tail = [index for index, word in enumerate(words[:400])
            if (word & 0xffff) == 0x10 and (word >> 48) == 0x101]
    start = tail[1] - 126 if len(tail) > 1 else 0
    registers = {}
    for index in range(start, start + 126):
        word = words[index]
        if (word >> 48) & 0xffff in TAGS:
            registers[word & 0xffff] = (word >> 16) & 0xffffffff
    return registers, words


def stale_task():
    binary = STALE.read_bytes()
    from open_rknpu.sequence import decode_sequence
    info = decode_sequence(binary)
    start = 96 + 16 * info["task_count"]
    payload = binary[start:]
    task = info["tasks"][1]
    registers = {}
    for index in range(task["register_count"]):
        word = struct.unpack_from("<Q", payload, task["command_offset"] + index * 8)[0]
        registers[word & 0xffff] = (word >> 16) & 0xffffffff
    return registers, payload


def table(payload, offset, rows=25):
    raw = np.frombuffer(payload[offset:offset + rows * 32], np.int8).reshape(rows, 32)
    values = raw[:, 0:16:2].astype(int)
    zero_points = raw[:, 1:16:2].astype(int)
    return values, zero_points


def main():
    vendor, words = vendor_task()
    stale, payload = stale_task()
    print("K5 depthwise ConvTranspose: vendor vs retained experiment\n")
    print("%-8s %-12s %-12s %s" % ("reg", "vendor", "retained", "note"))
    missing = []
    for reg in INTERESTING:
        left = vendor.get(reg)
        right = stale.get(reg)
        if left is None and right is None:
            continue
        note = "match" if left == right else ""
        if left is not None and right is None:
            note = "retained uses its default"
            missing.append(reg)
        elif left is not None and right is not None and left != right:
            note = "differs"
        print("%-8s %-12s %-12s %s" % (hex(reg), hex(left) if left is not None else "-",
                                       hex(right) if right is not None else "-", note))
    print("\nfields the retained binary never wrote (%d): %s"
          % (len(missing), " ".join(hex(reg) for reg in missing)))
    print("retained output conversion 0x4080/0x4084/0x4088:",
          [hex(stale.get(reg, 0)) for reg in (0x4080, 0x4084, 0x4088)],
          "vendor:", [hex(vendor[reg]) for reg in (0x4080, 0x4084, 0x4088)])

    vendor_values, vendor_zp = table(np.frombuffer(CAPTURE.read_bytes(), np.int8), 0xf00)
    stale_values, stale_zp = table(payload, 0xb00)
    print("\nvendor weight table: %d taps x 32 bytes = 16 lanes of (value, -wzp)"
          % len(vendor_values))
    print("  vendor lane values (first 3 lanes, 25 taps):\n", vendor_values[:, :3].T)
    print("  vendor per-lane -wzp (first 3 lanes):\n", vendor_zp[:, :3].T)
    print("  stale lane values (first 3 lanes, 25 taps):\n", stale_values[:, :3].T)
    print("  stale populated taps:", int(np.count_nonzero(stale_values.any(axis=1))), "of 25")
    print("  stale -wzp any nonzero:", bool(stale_zp.any()))
    # The stale table's populated rows are the reversed kernel of the ONNX model.
    import onnx
    from onnx import numpy_helper as nh
    model = onnx.load(ROOT / "transpose_k5_suite" / "model000.onnx")
    constants = {t.name: nh.to_array(t) for t in model.graph.initializer}
    weights = constants[model.graph.node[1].input[1]].reshape(3, 25)
    print("  ONNX kernel (channel 0, 25 taps):", np.round(weights[0], 4).tolist())


if __name__ == "__main__":
    main()
