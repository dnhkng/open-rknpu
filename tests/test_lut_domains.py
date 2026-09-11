"""Generalized LUT stems: sign-split table domain, family bounds and rejections.

The board suite that pins exact bytes is `research/lut_domain_suite/`; this module
checks the emitted table content, the accepted gain family and the retained
mixed-weight blocker (`research/lut_domain_failed/`).
"""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.lut import BASE_WEIGHT_SCALE, lut_reference, stem_range
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]


def lut_model(weights, bias, kind="Sigmoid"):
    weights = np.asarray(weights, np.float32).reshape(3, 3)
    bias = np.asarray(bias, np.float32)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node(kind, ["conv"], ["output"])], "lut",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(weights.reshape(3, 3, 1, 1), "w"), nh.from_array(bias, "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class LutDomainTests(unittest.TestCase):
    def compile(self, model, scale=1, zero=128):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path, scale, zero)

    def table(self, binary):
        info = decode_sequence(binary)
        start = 96 + 16 * info["task_count"]
        payload = binary[start:]
        setup = info["tasks"][0]
        banks = {0x20000: [], 0x30000: []}
        bank = None
        for i in range(setup["register_count"]):
            word = struct.unpack_from("<Q", payload, setup["command_offset"] + i * 8)[0]
            reg, value = word & 0xffff, word >> 16 & 0xffffffff
            if reg == 0x4100 and value in banks:
                bank = value
            elif reg == 0x4104 and bank is not None:
                banks[bank].append(value - 65536 if value > 32767 else value)
        return banks

    def test_table_samples_the_corrected_argument(self):
        for kind in ("Sigmoid", "Tanh"):
            for gain, correction in ((1 / 32, 1.0), (1 / 16, 2.0), (1 / 64, 0.5)):
                with self.subTest(kind=kind, gain=gain):
                    binary, meta = self.compile(lut_model(np.eye(3) * gain, np.zeros(3), kind))
                    self.assertAlmostEqual(meta["negative_gain"], correction, places=6)
                    banks = self.table(binary)
                    self.assertEqual(len(banks[0x20000]), 513)
                    self.assertEqual(len(banks[0x30000]), 513)
                    fn = (lambda x: 1 / (1 + np.exp(-x))) if kind == "Sigmoid" else np.tanh
                    negative = np.arange(513)
                    expected = np.clip(np.rint(fn((negative - 512) / 64 * correction) * 32768), -32768, 32767)
                    np.testing.assert_array_equal(banks[0x20000], expected)
                    positive = np.arange(513)
                    expected = np.clip(np.rint(fn(positive / 64) * 32768), -32768, 32767)
                    np.testing.assert_array_equal(banks[0x30000], expected)

    def test_reference_reproduces_the_verified_identity_suite(self):
        for index, kind in enumerate(("Sigmoid", "Tanh")):
            folder = ROOT / "research" / "lut_public_suite"
            cases = np.frombuffer((folder / f"input{index:03}.u8").read_bytes(), np.uint8)
            expected = np.frombuffer((folder / f"expected{index:03}.i8").read_bytes(), np.int8)
            shape = (-1, 8, 8, 3)
            weights = (np.eye(3, dtype=np.float32) / 32).reshape(3, 3)
            got = np.stack([lut_reference(case, weights, np.zeros(3, np.float32), kind)
                            for case in cases.reshape(shape)])
            self.assertTrue(np.array_equal(got.reshape(-1), expected))

    def test_reference_tracks_the_float_function(self):
        rng = np.random.default_rng(4)
        cases = rng.integers(0, 256, (8, 8, 8, 3), dtype=np.uint8)
        for kind in ("Sigmoid", "Tanh"):
            for gain, bias in ((1 / 32, [0.0, 0.0, 0.0]), (1 / 64, [1.0, -1.0, 0.0]),
                               (-1 / 32, [0.5, 0.5, -0.5])):
                with self.subTest(kind=kind, gain=gain):
                    weights = np.diag(np.array([gain, gain, gain], np.float32))
                    got = np.stack([lut_reference(case, weights, np.array(bias, np.float32), kind)
                                    for case in cases])
                    stem = np.einsum("nhwc,oc->nhwo", cases.astype(np.float64) - 128, weights) + np.array(bias)
                    fn = (lambda x: 1 / (1 + np.exp(-x))) if kind == "Sigmoid" else np.tanh
                    scale, zero = (255, -128) if kind == "Sigmoid" else (127, 0)
                    exact = np.rint(np.clip(np.rint(fn(stem) * 32768), -32768, 32767) * scale / 32768) + zero
                    # The table index grid (1/64 on the positive half, scaled on the
                    # negative half) allows at most a one-step difference.
                    self.assertLessEqual(int(np.abs(got.astype(int) - exact.astype(int)).max()), 1)

    def test_accepts_sign_flips_and_biases(self):
        for kind in ("Sigmoid", "Tanh"):
            for weights, bias in ((np.diag([-1 / 32, 1 / 32, -1 / 32]), [0.5, 0.5, -0.5]),
                                  (np.diag([1 / 64, -1 / 64, 1 / 64]), [-0.5, 0.0, 0.5])):
                with self.subTest(kind=kind):
                    binary, meta = self.compile(lut_model(np.asarray(weights, np.float32),
                                                          np.asarray(bias, np.float32), kind))
                    self.assertEqual(decode_sequence(binary)["task_count"], 2)
                    self.assertEqual(meta["kind"], kind)

    def test_rejects_profiles_outside_the_family(self):
        mixed = np.array([[0.02, 0.005, 0.0], [0.005, 0.02, 0.0], [0.0, 0.005, 0.02]], np.float32)
        cases = [
            ("mixed weights", mixed, np.zeros(3, np.float32)),
            ("off diagonal", np.array([[0.01, 0.01, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], np.float32), np.zeros(3, np.float32)),
            ("unequal gains", np.diag([1 / 32, 1 / 16, 1 / 32]), np.zeros(3, np.float32)),
            ("non power of two", np.eye(3) / 24, np.zeros(3, np.float32)),
            ("range too wide", np.eye(3) / 8, np.zeros(3, np.float32)),
            ("negative range too wide", np.eye(3) / 128, np.array([-3.0, 0.0, 0.0], np.float32)),
            ("zero gain", np.zeros((3, 3), np.float32), np.zeros(3, np.float32)),
        ]
        for name, weights, bias in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.compile(lut_model(weights, bias))

    def test_stem_range_matches_the_analytic_interval(self):
        weights = np.diag([1 / 32, -1 / 32, 1 / 16]).astype(np.float32)
        bias = np.array([0.5, -0.5, 0.0], np.float32)
        low, high = stem_range(weights.reshape(3, 3, 1, 1), bias)
        self.assertAlmostEqual(low, -8.0, places=5)
        self.assertAlmostEqual(high, 127 / 16, places=5)
        self.assertAlmostEqual(BASE_WEIGHT_SCALE, (1 / 32) / 255, places=12)


if __name__ == "__main__":
    unittest.main()
