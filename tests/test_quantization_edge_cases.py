"""SPDX-License-Identifier: MIT

Exact-number and boundary regression tests for `open_rknpu.quantization`.

`quantize` chooses one affine INT8 band per output channel: a weight scale from the
observed band, a weight zero point at a rail, a common accumulator scale and an INT8
bias, a `channel_multipliers` entry on the 16384 grid, and a multiplier/shift pair
that folds the channel step and the output conversion. This module pins those numbers
rather than smoke-testing them:

* the weight scale/zero point/quantized weights, the analytic output scale and zero
  point, the INT32 biases and the `round(scale / max_scale * 16384)` invariant;
* the three weight-zero-point boundaries (-128, 0, 127) and the `_adjusted_scale`
  widening a pool-fed Conv applies (covering and minimal on the pool's symmetric grid);
* the two hardware rounding steps at exact `.5` accumulators (round half to even) and
  INT8 rail saturation, because a sign or parity mistake there is invisible until a
  board run;
* `reference` versus `native_input_reference`: the two profiles share one accumulator
  and agree whenever they read the same band, but they place the output zero point on
  opposite sides of the final rounding, which is observable at an odd-zero-point half
  tie and is pinned here as the documented intentional difference;
* the `Quantization.metadata()` -> `load_quantizations` round trip after a real graph
  compiles, so the declared container parameters rebuild into identical arrays.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu.chain import native_quantize
from open_rknpu.compiler import compile_model
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization, quantize, reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.walk import _adjusted_scale, load_quantizations

ARRAY_KEYS = ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers")


def rebuild(metadata):
    """A `Quantization` rebuilt from `Quantization.metadata()` output."""
    values = dict(metadata)
    for key in ARRAY_KEYS:
        values[key] = np.array(values[key])
    return Quantization(**values)


def synthetic_quantization(*, channel_multipliers=16384, multiplier=1, shift=0, accumulator=0,
                           output_zero_point=0, weight_zero_point=0, weight=1,
                           output_scale=1.0, relu=False, input_zero_point=128):
    """A one-channel 1x1 grid whose accumulator is `accumulator` for a zero input.

    `reference` centers a UINT8 input by subtracting 128, so with a centered weight of
    one and `biases = 128 + accumulator` the integer accumulator is exactly the value
    requested - which is how the `.5` rounding cases below are constructed.
    """
    return Quantization(
        weights=np.array([[weight]], dtype=np.int64),
        weight_zero_points=np.array([weight_zero_point], dtype=np.int64),
        weight_scales=np.array([1.0], dtype=np.float32),
        biases=np.array([128 + accumulator], dtype=np.int64),
        channel_multipliers=np.array([channel_multipliers], dtype=np.int64),
        multiplier=multiplier, shift=shift, output_scale=output_scale,
        output_zero_point=output_zero_point, kernel_size=1, relu=relu,
        input_scale=1.0, input_zero_point=input_zero_point)


def single_reference(quantization, value=0):
    return int(reference(np.array([[[value]]], dtype=np.uint8), quantization).ravel()[0])


def one_conv_graph(seed=5, channels=4, kernel=1):
    """One Conv (no Relu) that fits the single-Conv compiler profile."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.6, .7, (channels, 3, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-2, 2, (channels,)).astype(np.float32)
    node = h.make_node("Conv", ["input", "weights", "bias"], ["output"], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4)
    graph = h.make_graph([node], "single", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, channels, 8, 8])],
                         [nh.from_array(weights, "weights"), nh.from_array(bias, "bias")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def walk_chain_graph(seed=3, hidden=5):
    """Conv -> Relu -> MaxPool -> Conv -> Relu: the op-level chain walk."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c1"], ["r1"]),
             h.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
             h.make_node("Conv", ["p1", "w2", "b2"], ["c2"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c2"], ["output"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (hidden, hidden, 1, 1)).astype(np.float32), "w2"),
                    nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b2")]
    graph = h.make_graph(nodes, "walk_chain", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, hidden, 4, 4])], initializers)
    for name, shape in (("c1", [1, hidden, 8, 8]), ("r1", [1, hidden, 8, 8]),
                        ("p1", [1, hidden, 4, 4]), ("c2", [1, hidden, 4, 4])):
        graph.value_info.append(h.make_tensor_value_info(name, 1, shape))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def save(model):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "model.onnx"
    onnx.save(model, path)
    return path


class BandSelectionTests(unittest.TestCase):
    """The exact band, channel multipliers and biases `quantize` chooses."""

    def test_weight_band_scale_zero_point_and_quantized_weights(self):
        weights = np.array([[-.5, .25, .75], [.3, -.3, .1]], dtype=np.float32)
        bias = np.array([.25, -.75], dtype=np.float32)
        q = quantize(weights, bias, output_scale=0.5, output_zero_point=-10,
                     input_scale=1 / 255, input_zero_point=0)
        # scale = (max(w, 0) - min(w, 0)) / 255 per channel, exactly as float32.
        expected_scales = np.array([(0.75 - -.5), (.3 - -.3)], dtype=np.float32) / np.float32(255)
        np.testing.assert_array_equal(q.weight_scales, expected_scales)
        np.testing.assert_array_equal(q.weight_zero_points, np.array([-26, 0], dtype=np.int64))
        np.testing.assert_array_equal(q.weights, np.array([[-128, 25, 127], [127, -128, 42]],
                                                          dtype=np.int64))
        np.testing.assert_array_equal(q.biases, np.array([26061, -76033], dtype=np.int64))
        self.assertEqual((q.multiplier, q.shift), (20641, 29))
        self.assertEqual((q.output_scale, q.output_zero_point), (0.5, -10))
        self.assertEqual(q.kernel_size, 1)
        self.assertFalse(q.relu)

    def test_quantized_weights_saturate_the_int8_rails_and_round_ties_to_even(self):
        # Range 510 makes scale exactly 2.0, so w/scale lands on exact halves and the
        # +127.5 weight rounds to 128 before the clip to the 127 rail.
        weights = np.array([-255.0, -1.0, 1.0, 0.0, 3.0, -3.0, 127.5, -127.5, 255.0],
                           dtype=np.float32).reshape(1, 1, 3, 3)
        q = quantize(weights, [0.0], output_scale=1.0, output_zero_point=0)
        self.assertEqual(float(q.weight_scales[0]), 2.0)
        self.assertEqual(int(q.weight_zero_points[0]), 0)
        # -127.5 -> -128 and 127.5 -> 128 -> clip 127; the .5 values round to even (0).
        np.testing.assert_array_equal(q.weights, np.array([[-128, 0, 0, 0, 2, -2, 64, -64, 127]],
                                                          dtype=np.int64))
        self.assertEqual(int(q.weights.min()), -128)
        self.assertEqual(int(q.weights.max()), 127)

    def test_channel_multipliers_are_round_scale_over_max_scale_on_the_16384_grid(self):
        rng = np.random.default_rng(8)
        weights = rng.uniform(-.7, .8, (7, 3, 1, 1)).astype(np.float32)
        q = quantize(weights, np.zeros(7, np.float32), output_scale=1.0, output_zero_point=0)
        expected = np.rint(q.weight_scales / np.float32(q.weight_scales.max()) * 16384).astype(np.int64)
        np.testing.assert_array_equal(q.channel_multipliers, expected)
        self.assertEqual(int(q.channel_multipliers.max()), 16384)
        self.assertTrue(np.all(q.channel_multipliers >= 1))
        self.assertTrue(np.all(q.channel_multipliers <= 16384))

    def test_biases_are_int32_and_follow_the_documented_accumulator_formula(self):
        weights = np.array([[-.5, .25, .75], [.3, -.3, .1]], dtype=np.float32)
        bias = np.array([.25, -.75], dtype=np.float32)
        for input_scale, input_zero_point in ((1 / 255, 0), (1 / 255, 128), (2.0, 255)):
            with self.subTest(input_zero_point=input_zero_point):
                q = quantize(weights, bias, output_scale=0.5, output_zero_point=-10,
                             input_scale=input_scale, input_zero_point=input_zero_point)
                centered = q.weights - q.weight_zero_points[:, None]
                expected = (np.rint(bias.astype(np.float32) / (q.weight_scales * np.float32(input_scale)))
                            .astype(np.int64)
                            + (128 - input_zero_point) * centered.sum(axis=1))
                np.testing.assert_array_equal(q.biases, expected)
                self.assertTrue(np.all(np.abs(q.biases) < (1 << 31)))

    def test_analytic_output_scale_covers_the_interval_and_picks_the_zero_point(self):
        weights = np.array([[-.25, .5, .75], [.5, -.25, 0], [0, .125, -.25]], dtype=np.float32)
        bias = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        for relu in (False, True):
            for input_scale, input_zero_point in ((1.0, 0), (1 / 255, 0), (1 / 255, 128)):
                with self.subTest(relu=relu, input_zero_point=input_zero_point):
                    q = quantize(weights, bias, relu=relu, input_scale=input_scale,
                                 input_zero_point=input_zero_point)
                    lo = -input_zero_point * input_scale
                    hi = (255 - input_zero_point) * input_scale
                    lower = min(0.0, float((np.minimum(weights * lo, weights * hi).sum(axis=1) + bias).min()))
                    upper = max(0.0, float((np.maximum(weights * lo, weights * hi).sum(axis=1) + bias).max()))
                    if relu:
                        lower = 0.0
                    expected_scale = float(np.float32((upper - lower) / 255)) or 1.0
                    self.assertEqual(q.output_scale, expected_scale)
                    self.assertEqual(q.output_zero_point,
                                     int(np.clip(np.rint(-128 - lower / expected_scale), -128, 127)))
                    # The scale really covers the analytic band (float32 rounding aside).
                    self.assertGreaterEqual(q.output_scale * 255, (upper - lower) * (1 - 1e-6))

    def test_weight_zero_point_boundaries(self):
        positive = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 255.0],
                            dtype=np.float32).reshape(1, 1, 3, 3)
        q_low = quantize(positive, [0.0], output_scale=1.0, output_zero_point=0)
        self.assertEqual(int(q_low.weight_zero_points[0]), -128)
        np.testing.assert_array_equal(q_low.weights, np.array([[-128, -127, -126, -125, -124, -123,
                                                               -122, -121, 127]], dtype=np.int64))
        q_zero = quantize(np.array([-255.0, -1.0, 1.0, 0.0, 3.0, -3.0, 63.75, -63.75, 255.0],
                                   dtype=np.float32).reshape(1, 1, 3, 3), [0.0],
                          output_scale=1.0, output_zero_point=0)
        self.assertEqual(float(q_zero.weight_scales[0]), 2.0)
        self.assertEqual(int(q_zero.weight_zero_points[0]), 0)
        negative = np.array([-255.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0],
                            dtype=np.float32).reshape(1, 1, 3, 3)
        q_high = quantize(negative, [0.0], output_scale=1.0, output_zero_point=0)
        self.assertEqual(int(q_high.weight_zero_points[0]), 127)
        np.testing.assert_array_equal(q_high.weights, np.array([[-128, 120, 121, 122, 123, 124, 125,
                                                                 126, 127]], dtype=np.int64))

    def test_zero_point_and_scale_arguments_are_validated_at_the_boundaries(self):
        weights = np.array([[-.5, .25, .75]], dtype=np.float32)
        for output_zero_point in (-128, 127):
            self.assertEqual(quantize(weights, [0.0], 1.0, output_zero_point).output_zero_point,
                             output_zero_point)
        for output_zero_point in (-129, 128):
            with self.assertRaisesRegex(ValueError, "invalid output quantization"):
                quantize(weights, [0.0], 1.0, output_zero_point)
        for input_zero_point in (0, 255):
            self.assertEqual(quantize(weights, [0.0], output_scale=1.0, output_zero_point=0,
                                      input_zero_point=input_zero_point).input_zero_point,
                             input_zero_point)
        for input_zero_point in (-1, 256):
            with self.assertRaisesRegex(ValueError, "invalid UINT8 input quantization"):
                quantize(weights, [0.0], output_scale=1.0, output_zero_point=0,
                         input_zero_point=input_zero_point)
        with self.assertRaisesRegex(ValueError, "specify output scale and zero point together"):
            quantize(weights, [0.0], output_scale=1.0)


class AdjustedScaleTests(unittest.TestCase):
    """The `_adjusted_scale` widening a pool-fed Conv applies to its band."""

    def test_widening_covers_the_band_and_is_minimal_on_the_symmetric_grid(self):
        for scale in (0.013, 1.0, 0.0004):
            for zero_point in (-128, -64, -1, 0, 1, 63, 127):
                with self.subTest(scale=scale, zero_point=zero_point):
                    adjusted = _adjusted_scale(scale, zero_point)
                    self.assertEqual(adjusted,
                                     float(np.float32(scale * max(128 + zero_point, 127 - zero_point) / 127)))
                    # The destination grid is the pool profile's symmetric [-127, 127] at
                    # zero point 0, so the widened band must cover the original one.
                    self.assertGreaterEqual(127 * adjusted, (128 + zero_point) * scale - 1e-6)
                    self.assertGreaterEqual(127 * adjusted, (127 - zero_point) * scale - 1e-6)
                    # ... and it is the smallest such multiple: one step down no longer covers.
                    steps = max(128 + zero_point, 127 - zero_point)
                    smaller = adjusted * (steps - 1) / steps
                    self.assertLess(127 * smaller, steps * scale)
                    self.assertGreaterEqual(adjusted, scale)


class RoundingAndSaturationTests(unittest.TestCase):
    """The two fixed-point rounding steps and the INT8 rails, on exact `.5` inputs."""

    def test_channel_step_rounds_exact_halves_to_even(self):
        # channel_multipliers=8192 puts an odd accumulator exactly on k + 0.5.
        accumulators = (1, 3, 5, 7, -1, -3, -5, -7)
        observed = [single_reference(synthetic_quantization(channel_multipliers=8192, accumulator=a))
                    for a in accumulators]
        self.assertEqual(observed, [0, 2, 2, 4, 0, -2, -2, -4])
        for accumulator, value in zip(accumulators, observed):
            self.assertEqual(value, min((accumulator // 2, accumulator // 2 + 1), key=lambda v: (v % 2, v)))
        # Round-half-away-from-zero would disagree at the 0.5 and 2.5 ties.
        half_up = [(a + 1) // 2 if a >= 0 else -((-a + 1) // 2) for a in accumulators]
        self.assertNotEqual(observed, half_up)
        self.assertEqual([half_up[i] - observed[i] for i in (0, 2)], [1, 1])
        self.assertEqual([half_up[i] - observed[i] for i in (4, 6)], [-1, -1])

    def test_output_step_rounds_exact_halves_to_even(self):
        # channel_multipliers=16384 leaves the channel step exact, so shift=1 with
        # multiplier=1 exposes the output step's own tie handling.
        accumulators = (1, 3, 5, 7, -1, -3, -5, -7)
        observed = [single_reference(synthetic_quantization(multiplier=1, shift=1, accumulator=a))
                    for a in accumulators]
        self.assertEqual(observed, [0, 2, 2, 4, 0, -2, -2, -4])
        # Shifting the zero point by an even value moves the tie to the same even result.
        for offset in (0, 2, -4):
            shifted = [single_reference(synthetic_quantization(multiplier=1, shift=1, accumulator=a,
                                                               output_zero_point=offset))
                       for a in accumulators]
            self.assertEqual(shifted, [value + offset for value in observed])

    def test_int8_rail_saturation_and_zero_point_placement(self):
        big = synthetic_quantization(accumulator=100000, output_zero_point=10)
        self.assertEqual(single_reference(big), 127)
        low = synthetic_quantization(accumulator=-100000, output_zero_point=10)
        self.assertEqual(single_reference(low), -128)
        # The zero point is added before the clip, so a moderate value shifts without
        # saturating - clamping the converted byte would use the wrong origin.
        mid = synthetic_quantization(accumulator=-127, output_zero_point=10)
        self.assertEqual(single_reference(mid), -117)
        high = synthetic_quantization(accumulator=120, output_zero_point=127)
        self.assertEqual(single_reference(high), 127)
        self.assertEqual(single_reference(synthetic_quantization(accumulator=-1, output_zero_point=127)), 126)

    def test_relu_clamps_the_accumulator_before_the_output_conversion(self):
        # A negative accumulator with Relu must clamp to the +zero_point origin, not to
        # a converted negative byte.
        quantization = synthetic_quantization(accumulator=-50, output_zero_point=10, relu=True)
        self.assertEqual(single_reference(quantization), 10)
        self.assertEqual(single_reference(synthetic_quantization(accumulator=-50, output_zero_point=10)), -40)


class ReferenceConsistencyTests(unittest.TestCase):
    """`reference` and `native_input_reference` must agree on a shared band."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.weights = rng.uniform(-.7, .8, (3, 3, 3, 3)).astype(np.float32)
        self.bias = rng.uniform(-2, 2, (3,)).astype(np.float32)
        self.inputs = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)

    def test_both_references_agree_on_the_same_input_band(self):
        # They share the accumulator and the padding value, so byte equality is the
        # documented contract whenever the signed zero point matches `q.input_zero_point`.
        for input_scale, input_zero_point in ((1.0, 0), (1 / 255, 0), (1 / 255, 128), (2.0, 255)):
            with self.subTest(input_zero_point=input_zero_point):
                q = quantize(self.weights, self.bias, input_scale=input_scale,
                             input_zero_point=input_zero_point)
                left = reference(self.inputs, q)
                right = native_input_reference(self.inputs, q, q.input_zero_point)
                np.testing.assert_array_equal(left, right)

    def test_a_native_grid_agrees_with_the_legacy_reference(self):
        q = native_quantize(self.weights, self.bias, 1.0, 0)
        q.input_scale = 1.0
        q.input_zero_point = 0
        np.testing.assert_array_equal(reference(self.inputs, q),
                                      native_input_reference(self.inputs, q, 0))

    def test_odd_zero_point_half_tie_is_the_documented_intentional_difference(self):
        # `reference` adds the output zero point after the final rounding;
        # `native_input_reference` folds it in before. The two are identical except at an
        # exact half with an odd zero point, where the tie-break parity flips by one LSB.
        pairs = [(0, 0, 0), (2, 2, 2), (-2, -2, -2), (1, 1, 2), (-1, -1, 0), (3, 3, 4)]
        for output_zero_point, legacy, native in pairs:
            with self.subTest(output_zero_point=output_zero_point):
                q = synthetic_quantization(multiplier=1, shift=1, accumulator=1,
                                           output_zero_point=output_zero_point)
                self.assertEqual(single_reference(q), legacy)
                self.assertEqual(int(native_input_reference(np.array([[[0]]], np.uint8), q, 128).ravel()[0]),
                                 native)


class MetadataRoundTripTests(unittest.TestCase):
    """`Quantization.metadata()` survives a rebuild into identical arrays."""

    def assert_same_arrays(self, left, right):
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(getattr(left, key), getattr(right, key))

    def test_native_quantize_metadata_round_trip(self):
        rng = np.random.default_rng(9)
        weights = rng.uniform(-.7, .8, (4, 3, 3, 3)).astype(np.float32)
        bias = rng.uniform(-2, 2, (4,)).astype(np.float32)
        q = native_quantize(weights, bias, 0.013, -37)
        metadata = q.metadata()
        rebuilt = rebuild(metadata)
        self.assert_same_arrays(q, rebuilt)
        self.assertEqual(rebuilt.metadata(), metadata)

    def test_quantize_metadata_round_trip(self):
        weights = np.array([[-.5, .25, .75], [.3, -.3, .1]], dtype=np.float32)
        q = quantize(weights, np.array([.25, -.75], np.float32), 0.5, -10)
        metadata = q.metadata()
        rebuilt = rebuild(metadata)
        self.assert_same_arrays(q, rebuilt)
        self.assertEqual(rebuilt.metadata(), metadata)
        self.assertEqual(rebuilt.relu, q.relu)

    def test_compiled_walk_chain_quantizations_rebuild_identically(self):
        binary, meta = compile_sequence(save(walk_chain_graph()))
        self.assertEqual(meta["profile"], "chain-walk")
        loaded = load_quantizations(meta)
        self.assertEqual(len(loaded), len(meta["quantizations"]))
        # The pool position carries no Conv band and stays None.
        self.assertIsNone(loaded[1])
        for original, rebuilt in zip(meta["quantizations"], loaded):
            if original is None:
                self.assertIsNone(rebuilt)
                continue
            self.assert_same_arrays(rebuild(original), rebuilt)
            self.assertEqual(rebuilt.metadata(), original)

    def test_compile_model_quantization_metadata_round_trip(self):
        _, meta = compile_model(save(one_conv_graph()))
        metadata = meta["quantization"]
        rebuilt = rebuild(metadata)
        self.assertEqual(rebuilt.metadata(), metadata)
        self.assertEqual(rebuilt.output_scale, meta["output_scale"])
        self.assertEqual(rebuilt.output_zero_point, meta["output_zero_point"])
        self.assertGreaterEqual(int(rebuilt.weights.min()), -128)
        self.assertLessEqual(int(rebuilt.weights.max()), 127)


if __name__ == "__main__":
    unittest.main()
