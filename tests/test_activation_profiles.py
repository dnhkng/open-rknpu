# SPDX-License-Identifier: MIT
"""Pins the fused activation profiles: LeakyRelu, PRelu and Conv+Clip[0,6].

`activation.py` lowers one dense Conv followed by a scalar LeakyRelu or a
scalar/per-channel PRelu.  The device converts the accumulator to the output band
with a channel multiplier (Q14) followed by a global multiplier, and for the
negative branch it scales the Q14 value by ``round(alpha * 16384)`` before the
same global conversion.  The modules here pin:

* the accepted scalar/per-channel slope domains (0..1) and their compiled bands;
* that the module integer reference tracks an independent float64 computation
  that dequantizes the compiled conv output with the container band, applies the
  activation, and requantizes with the output band (<= 1 LSB; exact for slope 0
  and 1 when every channel shares one weight scale, so the Q14 conversion is
  transparent);
* the conservative INT32 positive-path overflow guard and the documented
  connection/spatial/stride/range rejections;
* the native Conv+Clip[0,6]/ReLU6 fusion: the implicit band ``6/255``/``-128``,
  the equal CNA/DPU upper clamp at 0x4028/0x40e4 and the constant-range check.

No board is used: the integer references are the emitters' own, and the
independent values are recomputed here from the compiled quantization metadata.
"""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.activation import leaky_reference, prelu_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence, payload_base


def balanced_negative_weights(channels, input_channels, kernel, seed=0):
    """One repeated negative channel: every channel shares a weight scale.

    Repeating one all-negative kernel keeps ``(max - min)`` equal across channels
    (so the Q14 channel multiplier is exactly 16384) and keeps the positive
    accumulator bound at the bias, away from the INT32 guard.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.4, -0.15, (1, input_channels, kernel, kernel)).astype(np.float32)
    return np.repeat(base, channels, axis=0)

def random_weights(channels, input_channels, kernel, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, (channels, input_channels, kernel, kernel)).astype(np.float32)

def conv_activation_model(weights, bias, activation, shape, out_channels):
    input_channels = weights.shape[1]
    height, width = shape
    kernel = weights.shape[2]
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    nodes.append(activation)
    graph = h.make_graph(
        nodes, "activation",
        [h.make_tensor_value_info("input", 1, [1, input_channels, height, width])],
        [h.make_tensor_value_info("output", 1, [1, out_channels, height, width])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model

def leaky_model(alpha, channels=3, input_channels=3, kernel=1, height=8, width=8, seed=0, weights=None):
    weights = balanced_negative_weights(channels, input_channels, kernel, seed) if weights is None else weights
    bias = np.random.default_rng(seed + 1).uniform(-1, 1, channels).astype(np.float32)
    return conv_activation_model(weights, bias, h.make_node("LeakyRelu", ["conv"], ["output"], alpha=alpha),
                                 (height, width), channels)

def prelu_model(slopes, channels=3, input_channels=3, kernel=1, height=8, width=8, seed=0, weights=None):
    weights = balanced_negative_weights(channels, input_channels, kernel, seed) if weights is None else weights
    bias = np.random.default_rng(seed + 1).uniform(-1, 1, channels).astype(np.float32)
    slope_array = np.asarray(slopes)
    kernel = weights.shape[2]
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4),
             h.make_node("PRelu", ["conv", "slopes"], ["output"])]
    graph = h.make_graph(
        nodes, "prelu",
        [h.make_tensor_value_info("input", 1, [1, input_channels, height, width])],
        [h.make_tensor_value_info("output", 1, [1, channels, height, width])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b"), nh.from_array(slope_array, "slopes")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model

def clip_model(lower=0.0, upper=6.0, channels=3, input_channels=3, kernel=1, height=8, width=8, seed=0):
    weights = random_weights(channels, input_channels, kernel, seed)
    bias = np.random.default_rng(seed + 1).uniform(-1, 1, channels).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4),
             h.make_node("Clip", ["conv", "lower", "upper"], ["output"])]
    graph = h.make_graph(
        nodes, "clip",
        [h.make_tensor_value_info("input", 1, [1, input_channels, height, width])],
        [h.make_tensor_value_info("output", 1, [1, channels, height, width])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b"),
         nh.from_array(np.float32(lower), "lower"), nh.from_array(np.float32(upper), "upper")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model

def compile_graph(model, *args, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path, *args, **kwargs)

def quantization_of(meta):
    return Quantization(**{key: np.array(value) if isinstance(value, list) else value
                           for key, value in meta["quantization"].items()})

def independent_activation(inputs, meta, slope):
    """Dequantize the compiled conv output, activate in float64, requantize."""
    codes = reference(inputs, quantization_of(meta)).astype(np.int64)
    scale = meta["output_scale"]
    real = (codes - meta["output_zero_point"]) * scale
    if isinstance(slope, np.ndarray):
        activated = np.where(real < 0, real * slope[None, None, :], real)
    else:
        activated = np.where(real < 0, real * slope, real)
    return np.clip(np.rint(activated / scale) + meta["output_zero_point"], -128, 127).astype(np.int8)

def read_registers(binary, info, task_index=0):
    base = payload_base(info)
    task = info["tasks"][task_index]
    registers = {}
    for index in range(task["register_count"]):
        word = struct.unpack_from("<Q", binary, base + task["command_offset"] + index * 8)[0]
        registers[word & 0xFFFF] = (word >> 16) & 0xFFFFFFFF
    return registers

class LeakyReluProfileTests(unittest.TestCase):
    def test_scalar_slopes_track_independent_float_activation(self):
        rng = np.random.default_rng(42)
        for alpha in (0.0, 0.01, 0.1, 0.25, 0.5, 1.0):
            for channels, input_channels, kernel in ((1, 3, 1), (3, 3, 1), (3, 3, 3), (2, 1, 5)):
                with self.subTest(alpha=alpha, channels=channels, kernel=kernel):
                    binary, meta = compile_graph(leaky_model(alpha, channels, input_channels, kernel))
                    info = decode_sequence(binary)
                    self.assertEqual(info["task_count"], 1)
                    self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, channels])
                    self.assertAlmostEqual(meta["leaky_alpha"], alpha, places=6)
                    self.assertEqual(meta["leaky_alpha_encoded"], round(alpha * 16384) / 16384)
                    q = quantization_of(meta)
                    # Repeating one kernel gives every channel the same weight scale.
                    np.testing.assert_array_equal(q.channel_multipliers, 16384)
                    samples = rng.integers(0, 256, (24, 8, 8, input_channels), dtype=np.uint8)
                    got = np.stack([leaky_reference(x, q, meta["leaky_alpha"]) for x in samples])
                    want = np.stack([independent_activation(x, meta, meta["leaky_alpha"]) for x in samples])
                    difference = np.abs(got.astype(np.int64) - want.astype(np.int64))
                    self.assertLessEqual(int(difference.max()), 1)
                    if alpha in (0.0, 1.0):
                        # Equal channel scales make the Q14 conversion transparent, so the
                        # identity/relu slopes must match the float reference byte for byte.
                        np.testing.assert_array_equal(got, want)

    def test_unequal_channel_scales_stay_within_one_lsb(self):
        # Unequal channel scales add a second rounding to the negative branch, so only
        # the one-LSB bound is claimed for those slopes.
        samples = np.random.default_rng(21).integers(0, 256, (16, 8, 8, 1), dtype=np.uint8)
        for alpha in (0.0, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                weights = random_weights(3, 1, 1, seed=5)
                _binary, meta = compile_graph(leaky_model(alpha, channels=3, input_channels=1, kernel=1,
                                                          weights=weights))
                q = quantization_of(meta)
                self.assertGreater(len(set(q.channel_multipliers.tolist())), 1)
                got = np.stack([leaky_reference(x, q, meta["leaky_alpha"]) for x in samples])
                want = np.stack([independent_activation(x, meta, meta["leaky_alpha"]) for x in samples])
                self.assertLessEqual(int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()), 1)

    def test_affine_input_band(self):
        model = leaky_model(0.5, channels=3, input_channels=3, kernel=1)
        binary, meta = compile_graph(model, 0.5, 128)
        info = decode_sequence(binary)
        q = quantization_of(meta)
        self.assertEqual((info["input_scale"], info["input_zero_point"]), (0.5, 128))
        self.assertEqual((q.input_scale, q.input_zero_point), (0.5, 128))
        samples = np.random.default_rng(7).integers(0, 256, (16, 8, 8, 3), dtype=np.uint8)
        samples[0] = 128
        got = np.stack([leaky_reference(x, q, meta["leaky_alpha"]) for x in samples])
        want = np.stack([independent_activation(x, meta, meta["leaky_alpha"]) for x in samples])
        self.assertLessEqual(int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()), 1)

    def test_rejects_out_of_range_alpha(self):
        for alpha in (-0.1, 1.5, float("nan"), float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    compile_graph(leaky_model(alpha))

    def test_rejects_overflowing_positive_path(self):
        rng = np.random.default_rng(3)
        weights = rng.uniform(0.9, 1.0, (3, 3, 1, 1)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "positive conversion may overflow INT32"):
            compile_graph(leaky_model(0.5, weights=weights))

    def test_rejects_unsupported_spatial_and_stride(self):
        rng = np.random.default_rng(5)
        weights = rng.uniform(-0.4, 0.4, (3, 3, 1, 1)).astype(np.float32)
        bias = np.zeros(3, np.float32)
        nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1],
                             strides=[2, 2]),
                 h.make_node("LeakyRelu", ["conv"], ["output"], alpha=0.5)]
        graph = h.make_graph(nodes, "stride", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                             [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        with self.assertRaisesRegex(ValueError, "unsupported Conv attribute"):
            compile_graph(model)

    def test_rejects_wrong_follower_connection(self):
        weights = balanced_negative_weights(3, 3, 1)
        bias = np.zeros(3, np.float32)
        # The activation reads the graph input, not the Conv output.
        wrong_input = conv_activation_model(
            weights, bias, h.make_node("LeakyRelu", ["input"], ["output"], alpha=0.5), (8, 8), 3)
        with self.assertRaisesRegex(ValueError, "unsupported LeakyRelu parameters/connections"):
            compile_graph(wrong_input)
        # The activation result is not the graph output.
        nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                 h.make_node("LeakyRelu", ["conv"], ["lr"], alpha=0.5)]
        graph = h.make_graph(nodes, "dangling", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("conv", 1, [1, 3, 8, 8])],
                             [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        with self.assertRaisesRegex(ValueError, "unsupported LeakyRelu parameters/connections"):
            compile_graph(model)
        # More than the single Conv[/activation] pair is not the LeakyRelu profile.
        weights = balanced_negative_weights(3, 3, 1)
        nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                 h.make_node("Relu", ["conv"], ["relu"]),
                 h.make_node("LeakyRelu", ["relu"], ["output"], alpha=0.5)]
        graph = h.make_graph(nodes, "chained", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                             [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        with self.assertRaisesRegex(ValueError, "requires one Conv followed by LeakyRelu"):
            compile_graph(model)

class PReluProfileTests(unittest.TestCase):
    def test_scalar_and_per_channel_slopes(self):
        samples = np.random.default_rng(11).integers(0, 256, (20, 8, 8, 3), dtype=np.uint8)
        cases = [
            (np.float32(0.25), [0.25, 0.25, 0.25]),
            (np.array([0.5], np.float32), [0.5, 0.5, 0.5]),
            (np.array([0.0, 0.5, 1.0], np.float32).reshape(3, 1, 1), [0.0, 0.5, 1.0]),
            (np.array([1.0, 0.0, 0.25], np.float32).reshape(3, 1, 1), [1.0, 0.0, 0.25]),
        ]
        for slopes, expected in cases:
            with self.subTest(slopes=expected):
                binary, meta = compile_graph(prelu_model(slopes))
                info = decode_sequence(binary)
                self.assertEqual(info["task_count"], 1)
                np.testing.assert_allclose(meta["prelu_slopes"], expected, rtol=0, atol=1e-6)
                self.assertEqual(meta["prelu_slopes_q14"], [int(round(value * 16384)) for value in expected])
                q = quantization_of(meta)
                np.testing.assert_array_equal(q.channel_multipliers, 16384)
                slope = np.asarray(expected)
                got = np.stack([prelu_reference(x, q, np.array(meta["prelu_slopes_q14"])) for x in samples])
                want = np.stack([independent_activation(x, meta, slope) for x in samples])
                difference = np.abs(got.astype(np.int64) - want.astype(np.int64))
                self.assertLessEqual(int(difference.max()), 1)
                if set(expected) <= {0.0, 1.0}:
                    np.testing.assert_array_equal(got, want)
                # The emitted table holds the Q14 slopes.
                registers = read_registers(binary, info)
                table_offset = registers[0x502C]
                table_size = registers[0x5028]
                self.assertGreaterEqual(table_size, 2 * 3)
                entry = struct.unpack_from("<3H", binary, payload_base(info) + table_offset)
                self.assertEqual([int(v) for v in entry], meta["prelu_slopes_q14"])

    def test_rejects_bad_slopes(self):
        cases = {
            "above_one": np.float32(1.25),
            "negative": np.float32(-0.25),
            "non_float32": np.float64(0.5),
            "wrong_shape": np.array([0.1, 0.2, 0.3], np.float32),
        }
        for name, slopes in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    compile_graph(prelu_model(slopes))

    def test_rejects_overflowing_positive_path(self):
        rng = np.random.default_rng(2)
        positive = rng.uniform(0.9, 1.0, (3, 3, 1, 1)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "positive conversion may overflow INT32"):
            compile_graph(prelu_model(np.float32(0.5), weights=positive))

class ClipRelu6ProfileTests(unittest.TestCase):
    def test_default_band_and_upper_clamp(self):
        for channels, kernel in ((3, 1), (5, 1), (3, 3)):
            with self.subTest(channels=channels, kernel=kernel):
                binary, meta = compile_graph(clip_model(channels=channels, kernel=kernel))
                info = decode_sequence(binary)
                self.assertEqual(meta["fused_activation"], "Clip[0,6]")
                self.assertEqual(meta["profile"], "native16-input")
                self.assertAlmostEqual(info["output_scale"], 6 / 255, places=8)
                self.assertEqual(info["output_scale"], float(np.float32(6 / 255)))
                self.assertEqual(info["output_zero_point"], -128)
                # The emitter keeps the exact Python band in metadata and the float32
                # encoding in the container header.
                self.assertEqual(meta["output_scale"], 6 / 255)
                self.assertEqual(meta["output_zero_point"], -128)
                weight_scales = np.asarray(meta["quantization"]["weight_scales"])
                expected = round(6 / (float(np.max(weight_scales)) * meta["input_scale"]))
                registers = read_registers(binary, info)
                self.assertEqual(registers[0x4028], expected)
                self.assertEqual(registers[0x40E4], expected)
                self.assertNotEqual(registers[0x4028], 0x7FFFFFFF)

    def test_affine_input_scale_scales_the_clamp(self):
        binary, meta = compile_graph(clip_model(channels=3), 0.5, 128)
        info = decode_sequence(binary)
        weight_scales = np.asarray(meta["quantization"]["weight_scales"])
        expected = round(6 / (float(np.max(weight_scales)) * 0.5))
        registers = read_registers(binary, info)
        self.assertEqual((registers[0x4028], registers[0x40E4]), (expected, expected))

    def test_rejects_non_relu6_range(self):
        for lower, upper in ((1.0, 6.0), (0.0, 7.0), (0.0, 6.5), (-1.0, 6.0)):
            with self.subTest(range=(lower, upper)):
                with self.assertRaisesRegex(ValueError, r"constant scalar range \[0,6\]"):
                    compile_graph(clip_model(lower, upper))

    def test_rejects_non_scalar_range(self):
        weights = random_weights(3, 3, 1)
        bias = np.zeros(3, np.float32)
        nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                 h.make_node("Clip", ["conv", "lower", "upper"], ["output"])]
        graph = h.make_graph(
            nodes, "clip", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
            [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
            [nh.from_array(weights, "w"), nh.from_array(bias, "b"),
             nh.from_array(np.array([0.0, 0.0], np.float32), "lower"),
             nh.from_array(np.float32(6.0), "upper")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        with self.assertRaisesRegex(ValueError, r"constant scalar range \[0,6\]"):
            compile_graph(model)

if __name__ == "__main__":
    unittest.main()
