"""SPDX-License-Identifier: MIT

Front-end rejection paths pinned exactly once, by message.

The front end is where malformed user input is turned away, so a guard that
silently starts accepting bad input is a correctness bug that tests of the
emission paths cannot see. Each test below builds the minimal input that reaches
one guard and asserts the exact text (or exact observable side effect), so a
reworded or deleted guard fails loudly:

* `accuracy`: the shape contracts of the two public metric helpers;
* `calibration.measure`: a single static graph input is required;
* `chain_n._validate`: every native-chain structural rejection;
* `cli.main`: the post-parse target/quantization guard. `argparse` normally
  rejects unknown values through `choices` before the guard runs, so the test
  narrows the declared set and relies on argparse not validating string defaults,
  which is the only way to actually execute the guard;
* `normalize`: an existing Conv bias keeps the Add unfolded, and a folded bias
  whose name is already taken gets a `_` suffix;
* `padding.pad_input`: only HWC/NHWC arrays are accepted;
* `quantization`: weight geometry, INT32 accumulator overflow, the multiplier/
  shift band, and unknown rounding candidates.
"""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu import cli
from open_rknpu.accuracy import classification_accuracy, error_metrics
from open_rknpu.calibration import measure
from open_rknpu.chain_n import compile_chain_n
from open_rknpu.normalize import normalize_model
from open_rknpu.padding import pad_input
from open_rknpu.quantization import quantize, reference


def _weights(shape, seed):
    """Deterministic float32 weights plus a same-length float32 bias."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.7, .8, shape).astype(np.float32)
    bias = rng.uniform(-2, 2, (shape[0],)).astype(np.float32)
    return weights, bias


def _conv(source, output, weights, bias, kernel=1):
    return h.make_node("Conv", [source, weights, bias], [output], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4)


def _relu(source, output):
    return h.make_node("Relu", [source], [output])


def _native_model(nodes, initializers, inputs=None, outputs=None):
    if inputs is None:
        inputs = [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])]
    if outputs is None:
        outputs = [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])]
    graph = h.make_graph(nodes, "native_chain_rejection", inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def run_cli(*argv):
    """Drive cli.main() with argv, returning (exit status, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    previous = sys.argv
    sys.argv = ["open-rknpu", *[str(value) for value in argv]]
    status = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main()
    except SystemExit as exit_status:
        status = int(exit_status.code or 0)
    finally:
        sys.argv = previous
    return status, stdout.getvalue(), stderr.getvalue()


class AccuracyShapeTests(unittest.TestCase):
    def test_error_metrics_requires_equal_shapes(self):
        with self.assertRaises(ValueError) as raised:
            error_metrics(np.zeros((2, 2), np.int8), np.zeros((2, 3), np.float64), 1.0, 0)
        self.assertEqual(str(raised.exception), "integer and reference outputs must share a shape")

    def test_classification_accuracy_requires_logits_and_labels(self):
        for logits, labels in ((np.zeros((3, 2, 2)), np.zeros(3, np.int64)),
                               (np.zeros((3, 2)), np.zeros(2, np.int64))):
            with self.subTest(logits=logits.shape, labels=labels.shape):
                with self.assertRaises(ValueError) as raised:
                    classification_accuracy(logits, labels)
                self.assertEqual(str(raised.exception), "expected [N,C] logits and [N] labels")


class CalibrationInputTests(unittest.TestCase):
    def test_measure_requires_exactly_one_graph_input(self):
        weights, bias = _weights((3, 3, 3, 3), 11)
        graph = h.make_graph(
            [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
            "two_inputs",
            [h.make_tensor_value_info("input", 1, [1, 3, 8, 8]),
             h.make_tensor_value_info("extra", 1, [1, 3, 8, 8])],
            [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
            [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            with self.assertRaises(ValueError) as raised:
                measure(path, folder)
            self.assertEqual(str(raised.exception), "calibration requires exactly one graph input")


class NativeChainRejectionTests(unittest.TestCase):
    def _rejects(self, model):
        """Run the native-chain validator through its public entry point."""
        with self.assertRaises(ValueError) as raised:
            compile_chain_n(model)
        return raised.exception

    def test_first_node_must_be_conv(self):
        w0, b0 = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model([_relu("input", "r"), _conv("r", "c0", "w0", "b0"),
                               _conv("c0", "out", "w1", "b1")],
                              [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
                               nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
        self.assertEqual(str(self._rejects(model)), "native chain requires alternating Conv and Relu")

    def test_nodes_must_be_connected_in_order(self):
        w0, b0 = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model([_conv("input", "c0", "w0", "b0"), _relu("input", "r1"),
                               _conv("r1", "out", "w1", "b1")],
                              [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
                               nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
        self.assertEqual(str(self._rejects(model)), "native chain nodes must be connected in order")

    def test_exactly_one_input_and_one_output(self):
        w0, b0 = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model(
            [_conv("input", "c0", "w0", "b0"), _relu("c0", "r0"), _conv("r0", "out", "w1", "b1")],
            [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
             nh.from_array(w1, "w1"), nh.from_array(b1, "b1")],
            inputs=[h.make_tensor_value_info("input", 1, [1, 3, 8, 8]),
                    h.make_tensor_value_info("extra", 1, [1, 3, 8, 8])])
        self.assertEqual(str(self._rejects(model)), "native chain requires one input and one output")

    def test_external_tensors_must_be_float32_1x3x8x8(self):
        w0, b0 = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model(
            [_conv("input", "c0", "w0", "b0"), _relu("c0", "r0"), _conv("r0", "out", "w1", "b1")],
            [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
             nh.from_array(w1, "w1"), nh.from_array(b1, "b1")],
            inputs=[h.make_tensor_value_info("input", 1, [1, 3, 4, 4])],
            outputs=[h.make_tensor_value_info("out", 1, [1, 3, 4, 4])])
        self.assertEqual(str(self._rejects(model)),
                         "native chain external tensors must be float32 [1,3,8,8]")

    def test_convolutions_require_constant_weights_and_bias(self):
        w0, _ = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model([h.make_node("Conv", ["input", "w0"], ["c0"], kernel_shape=[1, 1]),
                               _relu("c0", "r0"), _conv("r0", "out", "w1", "b1")],
                              [nh.from_array(w0, "w0"), nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
        self.assertEqual(str(self._rejects(model)),
                         "native chain convolutions require constant weights and bias")

    def test_weights_and_bias_must_be_float32_rank_four(self):
        w0, b0 = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model([_conv("input", "c0", "w0", "b0"), _relu("c0", "r0"),
                               _conv("r0", "out", "w1", "b1")],
                              [nh.from_array(w0.astype(np.float64), "w0"), nh.from_array(b0, "b0"),
                               nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
        self.assertEqual(str(self._rejects(model)),
                         "native chain weights and bias must be float32 rank-four")

    def test_three_external_input_and_output_channels(self):
        w0, b0 = _weights((3, 5, 1, 1), 1)
        w1, b1 = _weights((3, 3, 1, 1), 2)
        model = _native_model([_conv("input", "c0", "w0", "b0"), _relu("c0", "r0"),
                               _conv("r0", "out", "w1", "b1")],
                              [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
                               nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
        self.assertEqual(str(self._rejects(model)),
                         "native chain requires three external input and output channels")

    def test_hidden_channels_must_match_between_layers(self):
        w0, b0 = _weights((3, 3, 1, 1), 1)
        w1, b1 = _weights((3, 5, 1, 1), 2)
        model = _native_model([_conv("input", "c0", "w0", "b0"), _relu("c0", "r0"),
                               _conv("r0", "out", "w1", "b1")],
                              [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
                               nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
        self.assertEqual(str(self._rejects(model)),
                         "native chain hidden channel counts must match between layers")


class CliGuardTests(unittest.TestCase):
    def test_unsupported_target_guard(self):
        # Narrowing the declared set makes argparse's default pass `choices` but
        # fail the post-parse membership guard at cli.py:61.
        with mock.patch.object(cli, "TARGETS", ("rv1106",)):
            status, _, stderr = run_cli("compile", "unused.onnx", "-o", "unused.bin")
        self.assertEqual(status, 1)
        self.assertEqual(stderr, "open-rknpu: unsupported target rv1103 (supported: rv1106)\n")

    def test_unsupported_quantization_guard(self):
        with mock.patch.object(cli, "QUANTIZATIONS", ("int16",)):
            status, _, stderr = run_cli("compile", "unused.onnx", "-o", "unused.bin")
        self.assertEqual(status, 1)
        self.assertEqual(stderr, "open-rknpu: unsupported quantization int8 (supported: int16)\n")

    def test_calibration_with_input_quantization_override_guard(self):
        # The override check runs before measure(), so the calibration path need not exist.
        status, _, stderr = run_cli("compile", "unused.onnx", "-o", "unused.bin",
                                    "--calibration", "calibration-dir", "--input-scale", "0.5")
        self.assertEqual(status, 1)
        self.assertEqual(stderr,
                         "open-rknpu: calibration with input quantization overrides is not supported yet\n")


class NormalizeFoldGuardTests(unittest.TestCase):
    def _model(self, nodes, initializers, output_name="sum"):
        graph = h.make_graph(nodes, "normalize_fold",
                             [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info(output_name, 1, [1, 3, 8, 8])],
                             initializers)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        return model

    def test_existing_bias_plus_add_is_left_unfolded(self):
        weights = np.full((3, 3, 1, 1), 0.1, np.float32)
        bias = np.zeros(3, np.float32)
        add_bias = np.full((3, 1, 1), 0.5, np.float32)
        model = self._model([_conv("input", "c0", "w", "b"), h.make_node("Add", ["c0", "add_bias"], ["sum"])],
                            [nh.from_array(weights, "w"), nh.from_array(bias, "b"),
                             nh.from_array(add_bias, "add_bias")])
        normalized = normalize_model(model)
        # normalize.py:136 continues instead of folding an Add into an already biased Conv.
        self.assertEqual([node.op_type for node in normalized.graph.node], ["Conv", "Add"])
        self.assertEqual(len(normalized.graph.node[0].input), 3)

    def test_folded_bias_name_collision_gets_a_suffix(self):
        weights = np.full((3, 3, 1, 1), 0.1, np.float32)
        add_bias = np.full((3, 1, 1), 0.5, np.float32)
        collision = np.zeros(3, np.float32)
        model = self._model([h.make_node("Conv", ["input", "w"], ["c0"], kernel_shape=[1, 1]),
                             h.make_node("Add", ["c0", "add_bias"], ["sum"])],
                            [nh.from_array(weights, "w"), nh.from_array(add_bias, "add_bias"),
                             nh.from_array(collision, "sum_conv_bias")])
        normalized = normalize_model(model)
        # normalize.py:139 appends '_' because the Add's preferred bias name is taken.
        self.assertEqual([node.op_type for node in normalized.graph.node], ["Conv"])
        self.assertEqual(normalized.graph.node[0].input[2], "sum_conv_bias_")
        self.assertIn("sum_conv_bias_", {tensor.name for tensor in normalized.graph.initializer})


class PaddingShapeTests(unittest.TestCase):
    def test_pad_input_rejects_non_hwc_rank(self):
        for array in (np.zeros((2, 2), np.uint8), np.zeros((1, 1, 2, 2, 3), np.uint8)):
            with self.subTest(shape=array.shape):
                with self.assertRaises(ValueError) as raised:
                    pad_input(array, (1, 1, 1, 1))
                self.assertEqual(str(raised.exception), "pad_input expects HWC or NHWC data")


class QuantizationRejectionTests(unittest.TestCase):
    def test_weight_geometry_must_match_the_affine_profile(self):
        with self.assertRaises(ValueError) as raised:
            quantize(np.zeros((3, 2, 3, 3), np.float32), np.zeros(3, np.float32))
        self.assertEqual(str(raised.exception),
                         "expected [O,I,K,K], I=1 or 3, 1<=O<=16, K=1, 3 or 5 weights")

    def test_int32_accumulator_overflow(self):
        # All-(-255) weights make every centered weight -255, so one output channel
        # accumulates 75 * 32640 = 2_448_000. The bias is chosen so the INT32 bias
        # guard (qbias = 2**31 - 1_948_032) still passes while acc_max crosses 2**31.
        weights = np.full((1, 3, 5, 5), -255.0, np.float32)
        bias = np.array([2 ** 31 + 500000], np.float32)
        with self.assertRaises(ValueError) as raised:
            quantize(weights, bias)
        self.assertEqual(str(raised.exception), "INT32 accumulator overflow")

    def test_output_scale_must_stay_in_the_multiplier_shift_band(self):
        weights = np.array([[-.5, .25, .75]], np.float32)
        for output_scale in (1e-30, 1e30):
            with self.subTest(output_scale=output_scale):
                with self.assertRaises(ValueError) as raised:
                    quantize(weights, [0.0], output_scale=output_scale, output_zero_point=0)
                self.assertEqual(str(raised.exception),
                                 "output scale outside validated multiplier/shift range")

    def test_reference_rejects_unknown_rounding_candidate(self):
        quantization = quantize(np.array([[-.5, .25, .75]], np.float32), [0.0],
                                output_scale=1.0, output_zero_point=0)
        with self.assertRaises(ValueError) as raised:
            reference(np.zeros((8, 8, 3), np.uint8), quantization, rounding="nearest")
        self.assertEqual(str(raised.exception), "unknown rounding candidate")


if __name__ == "__main__":
    unittest.main()
