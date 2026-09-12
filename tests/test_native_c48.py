"""Native input C33..128 weight layout, validated against retained vendor markers."""
from pathlib import Path
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]


def weight_byte(o, t, c, oc, ic, k):
    """Mirror of the emitter's 32-lane-part layout (see open_rknpu/native.py)."""
    lanes = (ic + 15) // 16 * 16
    block, lane = divmod(o, 16)
    OCb = min(16, oc - block * 16)
    part, within = divmod(c, 32)
    part_size = min(32, lanes - part * 32)
    base = block * 16 * k * k * lanes
    for previous in range(part):
        base += k * k * min(32, lanes - previous * 32) * OCb
    return base + t * part_size * OCb + lane * part_size + within


# folder, weight-table offset inside the dump, window size, marker predicate.
CAPTURES = {
    "taps": ("capture_native_c48_taps", 0xAC0, 432, lambda v: v != 0x80),
    "channels": ("capture_native_c48_channels", 0xAC0, 432, lambda v: v != 0x80),
    "oci": ("capture_native_c48_oci", 0xBC0, 6912, lambda v: v != 0x80),
    "c64channels": ("capture_native_c64_channels", 0xAC0, 576, lambda v: v != 0x80),
    "c65channels": ("capture_native_c65_channels", 0xAC0, 720, lambda v: v != 0x80),
    "c65oci": ("capture_native_c65_oci", 0xCC0, 12288, lambda v: v == 127),
    "c128channels": ("capture_native_c128_channels", 0xAC0, 1152, lambda v: v != 0x80),
}


def marker_positions(capture):
    """Marker weight byte offsets from a retained vendor capture."""
    folder, offset, size, is_marker = CAPTURES[capture]
    memory = (ROOT / "research" / folder / "run0_before_mem1.bin").read_bytes()
    weights = np.frombuffer(memory[offset:offset + size], np.uint8)
    return sorted(i for i, v in enumerate(weights) if is_marker(int(v)))


def c48_model(ic=48, oc=8, kernel=3, height=6, width=5, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-.7, .8, (oc, ic, kernel, kernel)).astype(np.float32)
    b = rng.uniform(-2, 2, (oc,)).astype(np.float32)
    graph = h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)],
        "c48", [h.make_tensor_value_info("input", 1, [1, ic, height, width])],
        [h.make_tensor_value_info("output", 1, [1, oc, height, width])],
        [nh.from_array(w, "w"), nh.from_array(b, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class NativeC48Tests(unittest.TestCase):
    def test_layout_reproduces_tap_order_markers(self):
        # One input channel, nine taps valued 1..9: tap t at 32*t for oc=1.
        predicted = sorted(weight_byte(0, t, 0, 1, 48, 3) for t in range(9))
        self.assertEqual(predicted, marker_positions("taps"))

    def test_layout_reproduces_input_channel_markers(self):
        predicted = sorted(weight_byte(0, 0, c, 1, 48, 3) for c in range(48))
        self.assertEqual(predicted, marker_positions("channels"))
        # planes 0-1 stay contiguous, plane 2 is the trailing 16-lane region
        self.assertEqual([weight_byte(0, 0, c, 1, 48, 3) for c in range(32)], list(range(32)))
        self.assertEqual(weight_byte(0, 0, 32, 1, 48, 3), 288)

    def test_layout_reproduces_output_channel_markers(self):
        # Identity markers w[o,o,0,0]=1 for oc=16: output o at 33*o.
        predicted = sorted(weight_byte(o, 0, o, 16, 48, 3) for o in range(16))
        self.assertEqual(predicted, marker_positions("oci"))

    def test_c48_compiles_to_native16_container(self):
        for ic in (33, 40, 48):
            with self.subTest(input_channels=ic):
                model = c48_model(ic=ic)
                path = ROOT / "research" / f"_native_c48_{ic}.onnx"
                onnx.save(model, path)
                try:
                    binary, meta = compile_sequence(path)
                finally:
                    path.unlink()
                info = decode_sequence(binary)
                self.assertEqual(info["input_layout"], "native16")
                self.assertEqual(info["shape_nhwc"][3], ic)
                self.assertEqual(meta["shape_nhwc"][3], ic)

    def test_layout_reproduces_c64_four_plane_markers(self):
        predicted = sorted(weight_byte(0, 0, c, 1, 64, 3) for c in range(64))
        self.assertEqual(predicted, marker_positions("c64channels"))
        self.assertEqual(weight_byte(0, 0, 32, 1, 64, 3), 288)
        self.assertEqual(weight_byte(0, 0, 63, 1, 64, 3), 319)

    def test_c64_compiles_to_native16_container(self):
        model = c48_model(ic=64)
        path = ROOT / "research" / "_native_c64.onnx"
        onnx.save(model, path)
        try:
            binary, meta = compile_sequence(path)
        finally:
            path.unlink()
        self.assertEqual(decode_sequence(binary)["shape_nhwc"][3], 64)
        self.assertEqual(meta["shape_nhwc"][3], 64)

    def test_layout_reproduces_c65_five_plane_markers(self):
        # 32 + 32 + 16 lanes: parts start at 0, 288 and 576.
        predicted = sorted(weight_byte(0, 0, c, 1, 65, 3) for c in range(65))
        self.assertEqual(predicted, marker_positions("c65channels"))
        self.assertEqual([weight_byte(0, 0, c, 1, 65, 3) for c in (0, 31, 32, 63, 64)], [0, 31, 288, 319, 576])

    def test_layout_reproduces_c128_eight_plane_markers(self):
        # Four full 32-lane parts: 0, 288, 576, 864.
        predicted = sorted(weight_byte(0, 0, c, 1, 128, 3) for c in range(128))
        self.assertEqual(predicted, marker_positions("c128channels"))
        self.assertEqual([weight_byte(0, 0, c, 1, 128, 3) for c in (0, 32, 64, 96, 127)],
                         [0, 288, 576, 864, 895])

    def test_layout_reproduces_c65_seventeen_output_markers(self):
        # Tap 0 nonzero for every (o, c): 17 output channels exercise the second
        # 16-output block (block base 16*k*k*lanes = 11520).
        predicted = sorted(weight_byte(o, 0, c, 17, 65, 3) for o in range(17) for c in range(65))
        self.assertEqual(predicted, marker_positions("c65oci"))
        self.assertEqual(len(predicted), 17 * 65)
        self.assertEqual(weight_byte(16, 0, 0, 17, 65, 3), 11520)

    def test_c65_to_c128_compile_to_native16_container(self):
        for ic in (65, 80, 96, 128):
            with self.subTest(input_channels=ic):
                model = c48_model(ic=ic, oc=8)
                path = ROOT / "research" / f"_native_c{ic}.onnx"
                onnx.save(model, path)
                try:
                    binary, meta = compile_sequence(path)
                finally:
                    path.unlink()
                self.assertEqual(decode_sequence(binary)["input_layout"], "native16")
                self.assertEqual(decode_sequence(binary)["shape_nhwc"][3], ic)
                self.assertEqual(meta["shape_nhwc"][3], ic)

    def test_above_128_is_now_lowered_and_the_wide_wall_is_enforced(self):
        # F10 (2026-09-12): the old C1..128 front-end cap was a vendor-fixture artifact, not a
        # hardware wall. C144 lowers; the real wall is the 511-part weight table, so C16352 is
        # the largest accepted input and C16368 (512 parts) is refused (it hangs the CNA job).
        model = c48_model(ic=144)
        path = ROOT / "research" / "_native_c144.onnx"
        onnx.save(model, path)
        try:
            binary, meta = compile_sequence(path)
        finally:
            path.unlink()
        self.assertEqual(decode_sequence(binary)["input_layout"], "native16")
        self.assertEqual(meta["shape_nhwc"][3], 144)
        with self.assertRaises(ValueError) as caught:
            compile_sequence(c48_model(ic=16368))
        self.assertIn("input C1..16352", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
