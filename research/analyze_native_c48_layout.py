"""Retained analysis: the native C33..128 input weight layout.

Prints the recovered 32-lane-part layout and validates it against the retained
vendor marker captures. Run from the repo root:

    PYTHONPATH=src python research/analyze_native_c48_layout.py

The open emitter (open_rknpu/native.py) never reads these captures; they are
evidence for the formula reproduced in tests/test_native_c48.py.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def weight_byte(o, t, c, oc, ic, k):
    """Byte offset of one quantized weight for the 32-lane-part layout."""
    lanes = (ic + 15) // 16 * 16
    block, lane = divmod(o, 16)
    OCb = min(16, oc - block * 16)
    part, within = divmod(c, 32)
    part_size = min(32, lanes - part * 32)
    base = block * 16 * k * k * lanes
    for previous in range(part):
        base += k * k * min(32, lanes - previous * 32) * OCb
    return base + t * part_size * OCb + lane * part_size + within


# folder, weight-table offset in the dump, window size, marker predicate.
# Zero weights store the 0x80 weight zero point; the C65 oc=17 capture uses the
# 127 code (the quantized value of the marker weight 1) because its table also
# holds a bias/scale region inside the window.
CAPTURES = {
    "c48_taps": ("capture_native_c48_taps", 0xAC0, 432, lambda v: v != 0x80),
    "c48_channels": ("capture_native_c48_channels", 0xAC0, 432, lambda v: v != 0x80),
    "c48_oci": ("capture_native_c48_oci", 0xBC0, 6912, lambda v: v != 0x80),
    "c64_channels": ("capture_native_c64_channels", 0xAC0, 576, lambda v: v != 0x80),
    "c65_channels": ("capture_native_c65_channels", 0xAC0, 720, lambda v: v != 0x80),
    "c65_oci": ("capture_native_c65_oci", 0xCC0, 12288, lambda v: v == 127),
    "c128_channels": ("capture_native_c128_channels", 0xAC0, 1152, lambda v: v != 0x80),
}


def markers(folder, offset, size, is_marker):
    memory = (ROOT / folder / "run0_before_mem1.bin").read_bytes()
    weights = np.frombuffer(memory[offset:offset + size], np.uint8)
    return sorted(i for i, v in enumerate(weights) if is_marker(int(v)))


def main():
    checks = {
        "c48_taps": sorted(weight_byte(0, t, 0, 1, 48, 3) for t in range(9)),
        "c48_channels": sorted(weight_byte(0, 0, c, 1, 48, 3) for c in range(48)),
        "c48_oci": sorted(weight_byte(o, 0, o, 16, 48, 3) for o in range(16)),
        "c64_channels": sorted(weight_byte(0, 0, c, 1, 64, 3) for c in range(64)),
        "c65_channels": sorted(weight_byte(0, 0, c, 1, 65, 3) for c in range(65)),
        "c65_oci": sorted(weight_byte(o, 0, c, 17, 65, 3) for o in range(17) for c in range(65)),
        "c128_channels": sorted(weight_byte(0, 0, c, 1, 128, 3) for c in range(128)),
    }
    ok = True
    for name, predicted in checks.items():
        folder, offset, size, is_marker = CAPTURES[name]
        actual = markers(folder, offset, size, is_marker)
        match = predicted == actual
        ok = ok and match
        print(f"{name:14s} {'MATCH' if match else 'MISMATCH'}  ({len(actual)} markers)")
        if not match:
            print("  predicted", predicted[:12], "...")
            print("  observed ", actual[:12], "...")
    print("\nRecovered layout: 16-lane input planes grouped in 32-lane parts, all taps")
    print("of a part first, inside each 16-output block; a trailing part holds the")
    print("last 16 or 32 lanes.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
