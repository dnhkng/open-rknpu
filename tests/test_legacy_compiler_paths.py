"""SPDX-License-Identifier: MIT

Why this file exists
--------------------

`compiler.py` is the oldest emitter in the tree and the only one with a real CLI
entry point. The published suites drive its *happy* single-Conv path (through
`compile_chain`, `compile_diamond`, ... which all compile a one-Conv stem with
`compile_model`), but none of the rejection branches of the legacy single-Conv
floor were pinned, and `main()` had never been executed by the suite at all
(0% of lines 152-165).

What it pins
------------

* Every rejection branch of the legacy single-Conv emitter with the exact
  message it raises, each paired with a valid neighbour that compiles, so the
  boundary is proven rather than just the failure: dispatch guards (input
  overrides, six-node network, non-Conv graphs), the Relu connection rule, the
  I/O arity rules, weight shape, explicit padding, static NCHW I/O, float32
  weights and bias shape.
* `compile_model`'s accepted metadata: register count, kernel, Relu flag and the
  UINT8 output band for the Conv and Conv+Relu shapes.
* `main()` in process with `sys.argv` patched: the success path writes the
  binary and the sibling `.json`, `--output-scale/--output-zero-point` are
  threaded through, argparse errors exit 2, and a compile rejection surfaces as
  an exception.
* The `python -m open_rknpu.compiler` entry point (`runpy`), which is the only
  way the `__main__` guard line runs.

Everything is offline and deterministic; models are written to a temporary
directory, never to the checkout.
"""
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu import compiler

SPATIAL = [1, 3, 8, 8]


def _value(name, shape, elem_type=1):
    return h.make_tensor_value_info(name, elem_type, list(shape))


def _initializer(name, array):
    return nh.from_array(np.asarray(array), name)


def _model(nodes, outputs, initializers, inputs=None, value_info=(), opset=13, opset_imports=None):
    """A graph plus model with a default float32 [1,3,8,8] input unless overridden."""
    if inputs is None:
        inputs = [_value("input", SPATIAL)]
    graph = h.make_graph(nodes, "legacy", inputs, outputs, initializers, value_info=list(value_info))
    imports = list(opset_imports) if opset_imports is not None else [h.make_opsetid("", opset)]
    model = h.make_model(graph, opset_imports=imports)
    model.ir_version = 8
    return model


def conv_weights(out_channels=3, in_channels=3, kernel=1):
    rng = np.random.default_rng(out_channels * 31 + in_channels * 7 + kernel)
    return rng.uniform(-0.6, 0.7, (out_channels, in_channels, kernel, kernel)).astype(np.float32)


def single_conv_model(relu=False, in_channels=3, out_channels=3, kernel=1, pads=None, seed=5,
                      weights=None, bias=None, input_shape=None, output_shape=None):
    """A one/two node Conv[/Relu] graph, the profile `compile_model` is built for."""
    rng = np.random.default_rng(seed)
    weight = conv_weights(out_channels, in_channels, kernel) if weights is None else np.asarray(weights)
    bias = rng.uniform(-2, 2, (out_channels,)).astype(np.float32) if bias is None else np.asarray(bias)
    shape = list(SPATIAL) if input_shape is None else list(input_shape)
    attributes = {"kernel_shape": [kernel, kernel]}
    if pads is not None:
        attributes["pads"] = list(pads)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], **attributes)]
    output = "conv"
    if relu:
        nodes.append(h.make_node("Relu", ["conv"], ["output"]))
        output = "output"
    out_shape = [1, out_channels, shape[2], shape[3]] if output_shape is None else list(output_shape)
    return _model(nodes, [_value(output, out_shape)],
                  [_initializer("w", weight), _initializer("b", bias)],
                  inputs=[_value("input", shape)])


def write_model(directory, model, name="model.onnx"):
    path = Path(directory) / name
    onnx.save(model, path)
    return path


class SingleConvAcceptanceTests(unittest.TestCase):
    """The accepted neighbours: prove the floor still compiles before rejecting at it."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_plain_conv_compiles_with_expected_metadata(self):
        for kernel in (1, 3, 5):
            with self.subTest(kernel=kernel):
                model = single_conv_model(kernel=kernel, pads=[kernel // 2] * 4
                                          if kernel > 1 else [0, 0, 0, 0])
                data, meta = compiler.compile_model(write_model(self.folder.name, model))
                self.assertEqual(len(data), 8192)
                self.assertEqual(meta["register_count"], 126)
                self.assertEqual(meta["kernel_size"], kernel)
                self.assertFalse(meta["relu"])
                self.assertEqual(meta["shape_nhwc"], [1, 8, 8, 3])
                self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 3])
                self.assertGreater(meta["output_scale"], 0)
                self.assertEqual(meta["quantization"]["kernel_size"], kernel)

    def test_conv_relu_compiles_with_activation_metadata(self):
        model = single_conv_model(relu=True)
        _, meta = compiler.compile_model(write_model(self.folder.name, model))
        self.assertTrue(meta["relu"])
        self.assertEqual(meta["quantization"]["relu"], True)

    def test_input_and_output_overrides_are_threaded_into_the_band(self):
        model = single_conv_model()
        path = write_model(self.folder.name, model)
        _, meta = compiler.compile_model(path, output_scale=0.25, output_zero_point=-7)
        self.assertAlmostEqual(meta["output_scale"], 0.25, places=6)
        self.assertEqual(meta["output_zero_point"], -7)
        _, overridden = compiler.compile_model(path, input_scale=2.0, input_zero_point=3)
        self.assertEqual(overridden["input_scale"], 2.0)
        self.assertEqual(overridden["input_zero_point"], 3)


class LegacyRejectionTests(unittest.TestCase):
    """One subtest per rejection branch, each asserting the exact message."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def _reject(self, model, fragment, name="reject.onnx", **kwargs):
        path = write_model(self.folder.name, model, name)
        with self.assertRaisesRegex(ValueError, fragment):
            compiler.compile_model(path, **kwargs)

    def test_input_override_requires_single_conv(self):
        # Three nodes is already a chain, so a UINT8 override has no legal home.
        rng = np.random.default_rng(11)
        nodes = [
            h.make_node("Conv", ["input", "w1", "b1"], ["half"], kernel_shape=[1, 1]),
            h.make_node("Relu", ["half"], ["mid"]),
            h.make_node("Conv", ["mid", "w2", "b2"], ["output"], kernel_shape=[1, 1]),
        ]
        model = _model(nodes, [_value("output", SPATIAL)],
                       [_initializer("w1", conv_weights(4, 3, 1)),
                        _initializer("b1", rng.uniform(-1, 1, (4,)).astype(np.float32)),
                        _initializer("w2", conv_weights(3, 4, 1)),
                        _initializer("b2", rng.uniform(-1, 1, (3,)).astype(np.float32))])
        self._reject(model, "input quantization overrides currently require a single Conv",
                     input_scale=2.0)
        # The valid neighbour: the same three-node graph without the override compiles
        # through the chain dispatch instead.
        _, meta = compiler.compile_model(write_model(self.folder.name, model, "ok.onnx"))
        self.assertEqual(meta["shape_nhwc"], [1, 8, 8, 3])

    def test_six_node_network_requires_calibration(self):
        model = six_node_network_model()
        self._reject(model, "network output overrides require calibration instead",
                     output_scale=0.5, output_zero_point=0)

    def test_non_conv_graph_is_rejected(self):
        # Four Conv/Relu nodes miss every specialised profile and reach the floor.
        rng = np.random.default_rng(13)
        nodes = [
            h.make_node("Conv", ["input", "w1", "b1"], ["a"], kernel_shape=[1, 1]),
            h.make_node("Relu", ["a"], ["b"]),
            h.make_node("Conv", ["b", "w2", "b2"], ["c"], kernel_shape=[1, 1]),
            h.make_node("Relu", ["c"], ["output"]),
        ]
        model = _model(nodes, [_value("output", SPATIAL)],
                       [_initializer("w1", conv_weights(4, 3, 1)),
                        _initializer("b1", rng.uniform(-1, 1, (4,)).astype(np.float32)),
                        _initializer("w2", conv_weights(3, 4, 1)),
                        _initializer("b2", rng.uniform(-1, 1, (3,)).astype(np.float32))])
        self._reject(model, "one Conv with optional following Relu is supported")
        # A custom-domain Conv passes the ONNX checker once its opset is imported
        # but is not the default-domain op the emitter supports.
        custom = _model([h.make_node("Conv", ["input", "w", "b"], ["output"],
                                     kernel_shape=[1, 1], domain="com.example")],
                        [_value("output", SPATIAL)],
                        [_initializer("w", conv_weights()), _initializer("b", np.zeros(3, np.float32))],
                        opset_imports=[h.make_opsetid("", 13), h.make_opsetid("com.example", 1)])
        self._reject(custom, "one Conv with optional following Relu is supported", name="domain.onnx")

    def test_relu_must_consume_the_conv_output(self):
        model = _model(
            [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["input"], ["output"])],
            [_value("output", SPATIAL)],
            [_initializer("w", conv_weights()), _initializer("b", np.zeros(3, np.float32))])
        self._reject(model, "only a directly connected standard Relu may follow Conv")

    def test_one_input_output_and_optional_bias(self):
        # Two graph inputs: the weight is an external tensor, not an initializer.
        model = _model([h.make_node("Conv", ["input", "w"], ["output"], kernel_shape=[1, 1])],
                       [_value("output", SPATIAL)],
                       [],
                       inputs=[_value("input", SPATIAL), _value("w", [3, 3, 1, 1])])
        self._reject(model, "one input/output and optional constant bias required")

    def test_unexpected_graph_connections_are_rejected(self):
        # The Conv reads a constant while the declared graph input stays unused.
        model = _model([h.make_node("Conv", ["const", "w", "b"], ["output"], kernel_shape=[1, 1])],
                       [_value("output", SPATIAL)],
                       [_initializer("const", np.zeros((1, 3, 8, 8), np.float32)),
                        _initializer("w", conv_weights()), _initializer("b", np.zeros(3, np.float32))])
        self._reject(model, "unexpected graph connections")

    def test_weight_shape_is_bounded(self):
        model = single_conv_model(out_channels=17)
        self._reject(model, r"weights must be \[O,I,K,K\]")
        model = single_conv_model(in_channels=2)
        self._reject(model, r"weights must be \[O,I,K,K\]", name="channels.onnx")

    def test_spatial_conv_requires_explicit_symmetric_pads(self):
        # Omitting `pads` reaches the explicit check; the same 3x3 Conv with the
        # symmetric pads spelled out is the valid neighbour that compiles.
        model = single_conv_model(kernel=3, pads=None)
        self._reject(model, "spatial Conv requires explicit symmetric padding K//2")
        valid = single_conv_model(kernel=3, pads=[1, 1, 1, 1])
        _, meta = compiler.compile_model(write_model(self.folder.name, valid, "padded.onnx"))
        self.assertEqual(meta["kernel_size"], 3)

    def test_static_nchw_io_is_required(self):
        model = single_conv_model(output_shape=[1, 3, 4, 4])
        self._reject(model, "input/output must be float32 tensors with matching spatial shapes")

    def test_float32_weights_are_required(self):
        model = single_conv_model(weights=conv_weights().astype(np.float16))
        self._reject(model, "float32 weights required")

    def test_bias_must_match_output_channels(self):
        model = single_conv_model(bias=np.zeros(5, np.float32))
        self._reject(model, r"bias must be float32 \[output_channels\]")


def six_node_network_model():
    """Three Conv+Relu pairs: the six-node network profile's shape."""
    rng = np.random.default_rng(17)
    nodes = []
    source = "input"
    initializers = []
    channels = [3, 4, 5, 3]
    for index in range(3):
        conv_out = f"hidden{index}"
        relu_out = "output" if index == 2 else f"relu{index}"
        nodes.append(h.make_node("Conv", [source, f"w{index}", f"b{index}"], [conv_out],
                                 kernel_shape=[1, 1]))
        nodes.append(h.make_node("Relu", [conv_out], [relu_out]))
        initializers.append(_initializer(f"w{index}", conv_weights(channels[index + 1], channels[index], 1)))
        initializers.append(_initializer(f"b{index}", rng.uniform(-1, 1, (channels[index + 1],)).astype(np.float32)))
        source = relu_out
    return _model(nodes, [_value("output", SPATIAL)], initializers)


class CompilerMainTests(unittest.TestCase):
    """`main()` and the module entry point, driven in process and as a subprocess."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model = write_model(self.folder.name, single_conv_model(relu=True))

    def _run(self, argv):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["open-rknpu", *argv]):
            with contextlib.redirect_stdout(output):
                compiler.main()
        return output.getvalue()

    def test_main_writes_binary_and_metadata(self):
        target = Path(self.folder.name) / "nested" / "out.bin"
        printed = self._run([str(self.model), "-o", str(target)])
        self.assertTrue(target.is_file())
        self.assertEqual(target.stat().st_size, 8192)
        self.assertIn(f"Emitted 8192 bytes: {target}", printed)
        metadata = json.loads(target.with_suffix(".json").read_text())
        self.assertEqual(metadata["kernel_size"], 1)
        self.assertTrue(metadata["relu"])
        self.assertEqual(metadata["shape_nhwc"], [1, 8, 8, 3])

    def test_main_threads_output_quantization_overrides(self):
        target = Path(self.folder.name) / "quantized.bin"
        self._run([str(self.model), "-o", str(target), "--output-scale", "0.125",
                   "--output-zero-point", "-3"])
        metadata = json.loads(target.with_suffix(".json").read_text())
        self.assertAlmostEqual(metadata["output_scale"], 0.125, places=6)
        self.assertEqual(metadata["output_zero_point"], -3)

    def test_main_argparse_errors_exit_two(self):
        for argv in ([str(self.model)], ["--unknown", "1"], []):
            with self.subTest(argv=argv):
                with mock.patch.object(sys, "argv", ["open-rknpu", *argv]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            compiler.main()
                self.assertEqual(caught.exception.code, 2)

    def test_main_surfaces_compile_rejections(self):
        missing = Path(self.folder.name) / "missing.onnx"
        target = Path(self.folder.name) / "out.bin"
        with mock.patch.object(sys, "argv", ["open-rknpu", str(missing), "-o", str(target)]):
            with self.assertRaises(FileNotFoundError):
                compiler.main()
        self.assertFalse(target.exists())

    def test_python_m_entry_point_runs_main(self):
        # The only way to execute the `if __name__ == "__main__"` guard line.
        target = Path(self.folder.name) / "module.bin"
        with mock.patch.object(sys, "argv",
                               ["open_rknpu.compiler", str(self.model), "-o", str(target)]):
            with contextlib.redirect_stdout(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    runpy.run_module("open_rknpu.compiler", run_name="__main__")
        self.assertTrue(target.is_file())
        self.assertTrue(target.with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()
