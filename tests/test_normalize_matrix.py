"""SPDX-License-Identifier: MIT

Front-end semantics matrix for `open_rknpu.normalize` and `open_rknpu.network`.

The normalizer is the only place where an ONNX graph is rewritten before lowering, so
its rewrites must be *semantics preserving*, not merely crash-free. This module pins
each rewrite by running the same inputs through ONNX's `ReferenceEvaluator` on the
original and the normalized graph and comparing the outputs: bias folding, constant
`Conv -> Mul` folding (scalar and per-channel), even/rectangular to odd-square kernel
rewrite, and grouped-Conv lowering to a zero-filled dense kernel. It also pins the
`auto_pad` materialization, the leading-`Pad` modes (with the negative-pad and unknown-
mode rejections), the native `Relu`/`Clip[0,6]` fusion boundary, the quantized import
routes, the loud rejection of unsupported operators (1-D Conv, MatMul, Softmax, Concat,
dynamic shapes, a second graph input) and the three-pool `network` profile.

The one exception is `auto_pad=VALID`: ONNX's `ReferenceEvaluator` computes SAME
padding for it (its `_conv_implementation` lumps VALID with SAME_UPPER), so the VALID
case is checked against the equivalent explicit-pads graph rather than the reference.
"""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.native import compile_native_input
from open_rknpu.network import compile_network
from open_rknpu.normalize import normalize_model
from open_rknpu.scheduler import compile_sequence


def conv_model(ic=3, oc=3, size=8, kernel=3, pads=None, auto_pad=None, strides=(1, 1),
               bias=True, seed=11):
    """A single Conv with either explicit pads or an auto_pad attribute."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(oc, ic, kernel, kernel)).astype(np.float32)
    attrs = dict(kernel_shape=[kernel, kernel], strides=list(strides))
    if auto_pad:
        attrs["auto_pad"] = auto_pad
        out = ((size - kernel) // strides[0] + 1 if auto_pad == "VALID"
               else (size + strides[0] - 1) // strides[0])
    else:
        use_pads = [kernel // 2] * 4 if pads is None else list(pads)
        attrs["pads"] = use_pads
        out = (size + use_pads[0] + use_pads[2] - kernel) // strides[0] + 1
    nodes = [h.make_node("Conv", ["x", "w"] + (["b"] if bias else []), ["y"], **attrs)]
    initializers = [nh.from_array(weights, "w")]
    if bias:
        initializers.append(nh.from_array(rng.normal(size=oc).astype(np.float32), "b"))
    graph = h.make_graph(nodes, "conv", [h.make_tensor_value_info("x", 1, [1, ic, size, size])],
                         [h.make_tensor_value_info("y", 1, [1, oc, out, out])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def conv_mul_model(factor_shape=(1,), bias=True, seed=12):
    """`Conv -> Mul` with a constant factor; the normalizer should fold it."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.5, .5, (4, 3, 3, 3)).astype(np.float32)
    factor = (rng.uniform(.5, 1.5, int(np.prod(factor_shape))).astype(np.float32)
              .reshape(factor_shape))
    nodes = [h.make_node("Conv", ["x", "w"] + (["b"] if bias else []), ["conv"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             h.make_node("Mul", ["conv", "factor"], ["y"])]
    initializers = [nh.from_array(weights, "w"), nh.from_array(factor, "factor")]
    if bias:
        initializers.append(nh.from_array(rng.uniform(-1, 1, 4).astype(np.float32), "b"))
    graph = h.make_graph(nodes, "conv_mul", [h.make_tensor_value_info("x", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def conv_kernel_model(kh, kw, pads, seed=13):
    """A Conv whose declared kernel is kh x kw."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.5, .5, (3, 3, kh, kw)).astype(np.float32)
    out_h = 8 + pads[0] + pads[2] - kh + 1
    out_w = 8 + pads[1] + pads[3] - kw + 1
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[kh, kw], pads=list(pads))]
    graph = h.make_graph(nodes, "kernel", [h.make_tensor_value_info("x", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("y", 1, [1, 3, out_h, out_w])],
                         [nh.from_array(weights, "w"), nh.from_array(np.zeros(3, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def grouped_model(group=2, ic=4, oc=4, kernel=3, seed=14):
    """A depthwise-style grouped Conv over a non-RGB input channel count."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.5, .5, (oc, ic // group, kernel, kernel)).astype(np.float32)
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["y"], group=group,
                         kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)]
    graph = h.make_graph(nodes, "grouped", [h.make_tensor_value_info("x", 1, [1, ic, 8, 8])],
                         [h.make_tensor_value_info("y", 1, [1, oc, 8, 8])],
                         [nh.from_array(weights, "w"),
                          nh.from_array(rng.uniform(-1, 1, oc).astype(np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def padded_conv(mode="constant", pads=(1, 1, 1, 1), with_axes=False, values=None):
    """A leading Pad feeding a 3x3 Conv, mirroring the padding suite."""
    weights = np.arange(81, dtype=np.float32).reshape(3, 3, 3, 3) / 81
    pad_values = np.array([0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]], np.int64)
    pad_inputs = ["input", "pads", "value"]
    initializers = [nh.from_array(weights, "w"), nh.from_array(np.zeros(3, np.float32), "b"),
                    nh.from_array(pad_values, "pads"), nh.from_array(np.array(0, np.float32), "value")]
    if with_axes:
        pad_inputs.append("axes")
        initializers.append(nh.from_array(np.array([2, 3], np.int64), "axes"))
    nodes = [h.make_node("Pad", pad_inputs, ["padded"], mode=mode),
             h.make_node("Conv", ["padded", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "padding", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1,
                                                   [1, 3, 8 + pads[0] + pads[2], 8 + pads[1] + pads[3]])],
                         initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def conv_activation_model(kind="Relu", low=0.0, high=6.0, seed=15):
    """A Conv followed by one activation, for the native fusion boundary."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.5, .5, (4, 3, 3, 3)).astype(np.float32)
    bias = rng.uniform(-1, 1, 4).astype(np.float32)
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    initializers = [nh.from_array(weights, "w"), nh.from_array(bias, "b")]
    if kind == "Clip":
        nodes.append(h.make_node("Clip", ["conv", "low", "high"], ["y"]))
        initializers += [nh.from_array(np.array(low, np.float32), "low"),
                         nh.from_array(np.array(high, np.float32), "high")]
    else:
        nodes.append(h.make_node(kind, ["conv"], ["y"]))
    graph = h.make_graph(nodes, "activation", [h.make_tensor_value_info("x", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def unsupported_model(op_type, input_shape=(1, 3, 8, 8), operand_shape=None, output_shape=None,
                      attrs=None):
    """A single unsupported op with one external input and a constant operand."""
    inputs = ["x"]
    initializers = []
    if operand_shape is not None:
        inputs.append("operand")
        initializers.append(nh.from_array(np.zeros(operand_shape, np.float32), "operand"))
    nodes = [h.make_node(op_type, inputs, ["y"], **(attrs or {}))]
    output = list(input_shape) if output_shape is None else list(output_shape)
    graph = h.make_graph(nodes, op_type, [h.make_tensor_value_info("x", 1, list(input_shape))],
                         [h.make_tensor_value_info("y", 1, output)], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def one_dimensional_conv():
    rng = np.random.default_rng(16)
    weights = rng.uniform(-.5, .5, (4, 3, 3)).astype(np.float32)
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3], pads=[1, 1])]
    graph = h.make_graph(nodes, "conv1d", [h.make_tensor_value_info("x", 1, [1, 3, 8])],
                         [h.make_tensor_value_info("y", 1, [1, 4, 8])],
                         [nh.from_array(weights, "w"), nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def dynamic_conv():
    rng = np.random.default_rng(17)
    weights = rng.uniform(-.5, .5, (4, 3, 3, 3)).astype(np.float32)
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "dynamic", [h.make_tensor_value_info("x", 1, [1, 3, "H", 8])],
                         [h.make_tensor_value_info("y", 1, [1, 4, "H", 8])],
                         [nh.from_array(weights, "w"), nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def two_input_conv():
    """A Conv whose weights are a second external input, not an initializer."""
    graph = h.make_graph(
        [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        "two_input",
        [h.make_tensor_value_info("x", 1, [1, 3, 8, 8]),
         h.make_tensor_value_info("w", 1, [4, 3, 3, 3])],
        [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])],
        [nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def network_model(kind="MaxPool", out_shape=(1, 3, 1, 1), seed=21):
    """Conv-Relu-Conv followed by three 2x2 stride-2 pools ending at [1,3,1,1]."""
    rng = np.random.default_rng(seed)
    w0 = rng.uniform(-.5, .5, (8, 3, 1, 1)).astype(np.float32)
    b0 = rng.uniform(-1, 1, 8).astype(np.float32)
    w1 = rng.uniform(-.5, .5, (3, 8, 1, 1)).astype(np.float32)
    b1 = rng.uniform(-1, 1, 3).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["c0"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c0"], ["r0"]),
             h.make_node("Conv", ["r0", "w1", "b1"], ["c1"], kernel_shape=[1, 1])]
    source = "c1"
    for index in range(3):
        nodes.append(h.make_node(kind, [source], [f"p{index}"], kernel_shape=[2, 2], strides=[2, 2]))
        source = f"p{index}"
    graph = h.make_graph(nodes, "network", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info(source, 1, list(out_shape))],
                         [nh.from_array(w0, "w0"), nh.from_array(b0, "b0"),
                          nh.from_array(w1, "w1"), nh.from_array(b1, "b1")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def qlinear_model(per_channel=False, axis=1):
    rng = np.random.default_rng(18)
    in_channels, out_channels, kernel = 3, 5, 3
    weights = rng.integers(-128, 128, (out_channels, in_channels, kernel, kernel)).astype(np.int8)
    scales = (rng.uniform(.001, .02, out_channels).astype(np.float32) if per_channel
              else np.array(.01, np.float32))
    zero_points = (rng.integers(-8, 9, out_channels).astype(np.int16).astype(np.int8) if per_channel
                   else np.array(-3, np.int8))
    names = ["input", "xs", "xz", "qw", "ws", "wz", "ys", "yz"]
    initializers = [nh.from_array(np.array(.25, np.float32), "xs"),
                    nh.from_array(np.array(0, np.uint8), "xz"),
                    nh.from_array(weights, "qw"), nh.from_array(scales, "ws"),
                    nh.from_array(zero_points, "wz"),
                    nh.from_array(np.array(2.0, np.float32), "ys"),
                    nh.from_array(np.array(-17, np.int8), "yz")]
    node = h.make_node("QLinearConv", names, ["output"], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4)
    model = h.make_model(h.make_graph(
        [node], "qlinearconv",
        [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, out_channels, 8, 8])],
        initializers), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def qdq_model(per_channel=True, axis=0):
    rng = np.random.default_rng(19)
    in_channels, out_channels, kernel = 3, 5, 3
    weights = rng.integers(-128, 128, (out_channels, in_channels, kernel, kernel)).astype(np.int8)
    scales = (rng.uniform(.001, .02, out_channels).astype(np.float32) if per_channel
              else np.array(.01, np.float32))
    zero_points = (rng.integers(-8, 9, out_channels).astype(np.int16).astype(np.int8) if per_channel
                   else np.array(-2, np.int8))
    initializers = [nh.from_array(np.array(.25, np.float32), "xs"),
                    nh.from_array(np.array(0, np.uint8), "xz"),
                    nh.from_array(weights, "qw"), nh.from_array(scales, "ws"),
                    nh.from_array(zero_points, "wz"),
                    nh.from_array(np.array(2.0, np.float32), "ys"),
                    nh.from_array(np.array(-17, np.int8), "yz")]
    weight_dq = (h.make_node("DequantizeLinear", ["qw", "ws", "wz"], ["wf"], axis=axis)
                 if per_channel else h.make_node("DequantizeLinear", ["qw", "ws", "wz"], ["wf"]))
    nodes = [h.make_node("DequantizeLinear", ["input", "xs", "xz"], ["xf"]), weight_dq,
             h.make_node("Conv", ["xf", "wf"], ["yf"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4),
             h.make_node("QuantizeLinear", ["yf", "ys", "yz"], ["output"])]
    model = h.make_model(h.make_graph(
        nodes, "qdq",
        [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, out_channels, 8, 8])],
        initializers), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def reference_equal(model, feeds):
    """Run the same inputs through the original and normalized graphs and compare."""
    normalized = normalize_model(model)
    expected = ReferenceEvaluator(model).run(None, feeds)
    actual = ReferenceEvaluator(normalized).run(None, feeds)
    for before, after in zip(expected, actual):
        np.testing.assert_allclose(before, after, atol=1e-5, rtol=1e-5)
    return normalized


def init_map(model):
    return {tensor.name: nh.to_array(tensor) for tensor in model.graph.initializer}


class _PathMixin:
    """Save a graph to a temporary path and run `compile_sequence` on it."""

    def compile_saved(self, model, *args, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path, *args, **kwargs)

    def reject_saved(self, message, model, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            self.compile_saved(model, *args, **kwargs)

    def compile_network_saved(self, model):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "network.onnx"
            onnx.save(model, path)
            return compile_network(path)


class NormalizeConvTests(unittest.TestCase):
    """Bias-less Conv, explicit pads and `auto_pad` materialization."""

    def test_biasless_conv_is_semantics_preserving(self):
        feeds = {"x": np.random.default_rng(1).normal(size=(1, 3, 8, 8)).astype(np.float32)}
        normalized = reference_equal(conv_model(bias=False), feeds)
        self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv"])
        self.assertEqual(len(normalized.graph.node[0].input), 2)
        self.assertEqual(len(init_map(normalized)), 1)

    def test_explicit_auto_pad_is_materialized(self):
        feeds = {"x": np.random.default_rng(2).normal(size=(1, 3, 8, 8)).astype(np.float32)}
        # Stride 1: the ONNX reference and the normalizer agree on the SAME pads, so
        # the output equality check is meaningful here.
        reference_equal(conv_model(auto_pad="SAME_UPPER"), feeds)
        # Stride 2: the ONNX reference reads the batch/channel extents when it computes
        # SAME pads, so the materialization is pinned against the equivalent explicit
        # pads graph instead.
        for auto, pads in (("SAME_UPPER", [0, 0, 1, 1]), ("SAME_LOWER", [1, 1, 0, 0])):
            with self.subTest(auto_pad=auto):
                normalized = normalize_model(conv_model(auto_pad=auto, strides=(2, 2)))
                attrs = {a.name: h.get_attribute_value(a) for a in normalized.graph.node[0].attribute}
                self.assertNotIn("auto_pad", attrs)
                self.assertEqual(attrs["pads"], pads)
                expected = ReferenceEvaluator(conv_model(pads=pads, strides=(2, 2))).run(None, feeds)[0]
                actual = ReferenceEvaluator(normalized).run(None, feeds)[0]
                np.testing.assert_allclose(expected, actual, atol=1e-5, rtol=1e-5)
                self.assertEqual([d.dim_value for d in normalized.graph.output[0].type.tensor_type.shape.dim],
                                 [1, 3, 4, 4])

    def test_valid_auto_pad_becomes_zero_padding(self):
        # ONNX's ReferenceEvaluator materializes SAME padding for VALID as well, so the
        # VALID case is compared against the equivalent explicit zero-padding graph.
        normalized = normalize_model(conv_model(auto_pad="VALID"))
        attrs = {a.name: h.get_attribute_value(a) for a in normalized.graph.node[0].attribute}
        self.assertEqual(attrs["pads"], [0, 0, 0, 0])
        self.assertEqual([d.dim_value for d in normalized.graph.output[0].type.tensor_type.shape.dim],
                         [1, 3, 6, 6])
        feeds = {"x": np.random.default_rng(3).normal(size=(1, 3, 8, 8)).astype(np.float32)}
        explicit = conv_model(pads=[0, 0, 0, 0])
        expected = ReferenceEvaluator(explicit).run(None, feeds)
        actual = ReferenceEvaluator(normalized).run(None, feeds)
        np.testing.assert_allclose(expected[0], actual[0], atol=1e-5, rtol=1e-5)

    def test_explicit_pads_are_preserved(self):
        feeds = {"x": np.random.default_rng(4).normal(size=(1, 3, 8, 8)).astype(np.float32)}
        normalized = reference_equal(conv_model(pads=[1, 2, 1, 2]), feeds)
        attrs = {a.name: h.get_attribute_value(a) for a in normalized.graph.node[0].attribute}
        self.assertEqual(attrs["pads"], [1, 2, 1, 2])


class NormalizeFoldTests(unittest.TestCase):
    """Constant `Conv -> Mul` folding, checked against the ONNX reference."""

    def feeds(self):
        return {"x": np.random.default_rng(5).normal(size=(1, 3, 8, 8)).astype(np.float32)}

    def assert_folded(self, model):
        normalized = reference_equal(model, self.feeds())
        self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv"])
        self.assertTrue(normalized.graph.node[0].input[1].endswith("_mul_weights"))
        before = init_map(model)["w"]
        after = init_map(normalized)[normalized.graph.node[0].input[1]]
        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(after.shape, before.shape)
        return normalized

    def test_scalar_factor_folds_into_the_conv(self):
        self.assert_folded(conv_mul_model(factor_shape=(1,)))

    def test_per_channel_factors_fold_into_the_conv(self):
        for shape in ((4, 1, 1), (1, 4, 1, 1)):
            with self.subTest(shape=shape):
                normalized = reference_equal(conv_mul_model(factor_shape=shape), self.feeds())
                self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv"])

    def test_biasless_conv_mul_folds_and_materializes_a_bias(self):
        normalized = reference_equal(conv_mul_model(bias=False), self.feeds())
        self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv"])
        self.assertEqual(len(normalized.graph.node[0].input), 3)

    def test_non_scalar_factor_shape_is_left_alone(self):
        # `(1,1,1,1)` broadcasts but is not on the folding whitelist, so the
        # unverified broadcast arithmetic is preserved rather than absorbed.
        normalized = reference_equal(conv_mul_model(factor_shape=(1, 1, 1, 1)), self.feeds())
        self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv", "Mul"])


class NormalizeKernelRewriteTests(unittest.TestCase):
    """Even/rectangular kernel rewrite and grouped-Conv lowering."""

    def feeds(self, channels=3):
        return {"x": np.random.default_rng(6).normal(size=(1, channels, 8, 8)).astype(np.float32)}

    def test_even_and_rectangular_kernels_become_odd_square(self):
        cases = ((2, 2, (0, 0, 0, 0)), (1, 3, (0, 1, 0, 1)), (2, 4, (0, 1, 1, 1)))
        for kh, kw, pads in cases:
            with self.subTest(kernel=(kh, kw)):
                model = conv_kernel_model(kh, kw, pads)
                normalized = reference_equal(model, self.feeds())
                attrs = {a.name: h.get_attribute_value(a) for a in normalized.graph.node[0].attribute}
                square = 3 if max(kh, kw) <= 3 else 5
                self.assertEqual(attrs["kernel_shape"], [square, square])
                after = init_map(normalized)[normalized.graph.node[0].input[1]]
                self.assertEqual(after.shape[2:], (square, square))
                self.assertEqual(attrs["pads"][2], pads[2] + square - kh)
                self.assertEqual(attrs["pads"][3], pads[3] + square - kw)

    def test_square_k5_kernel_is_not_rewritten(self):
        model = conv_kernel_model(5, 5, (2, 2, 2, 2))
        normalized = reference_equal(model, self.feeds())
        attrs = {a.name: h.get_attribute_value(a) for a in normalized.graph.node[0].attribute}
        self.assertEqual(attrs["kernel_shape"], [5, 5])

    def test_grouped_conv_lowers_to_a_zero_filled_dense_kernel(self):
        for group in (2, 4):
            with self.subTest(group=group):
                model = grouped_model(group=group)
                normalized = reference_equal(model, self.feeds(channels=4))
                node = normalized.graph.node[0]
                attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
                self.assertEqual(attrs.get("group"), 1)
                self.assertTrue(node.input[1].endswith("_dense"))
                dense = init_map(normalized)[node.input[1]]
                self.assertEqual(dense.shape, (4, 4, 3, 3))
                per_group = 4 // group
                for index in range(group):
                    rows = slice(index * per_group, (index + 1) * per_group)
                    self.assertGreater(np.count_nonzero(dense[rows, rows]), 0)
                    for other in range(group):
                        if other == index:
                            continue
                        columns = slice(other * per_group, (other + 1) * per_group)
                        self.assertEqual(np.count_nonzero(dense[rows, columns]), 0)


class NormalizePadTests(_PathMixin, unittest.TestCase):
    """Leading-Pad folding: accepted modes and the documented rejections."""

    def test_supported_modes_are_folded_and_recorded(self):
        for mode in ("constant", "reflect", "edge"):
            with self.subTest(mode=mode):
                _, meta = self.compile_saved(padded_conv(mode=mode, pads=(1, 1, 1, 1)))
                padding = meta["input_padding"]
                self.assertEqual(padding["mode"], mode)
                self.assertEqual(padding["pads"], [1, 1, 1, 1])
                self.assertEqual(padding["padded_shape"], [1, 3, 10, 10])
                self.assertTrue(padding["requires_host_preprocessing"])

    def test_negative_pad_is_rejected(self):
        self.reject_saved("negative Pad amounts are not supported", _negative_pad_model())

    def test_unknown_mode_is_rejected(self):
        self.reject_saved("unsupported Pad mode: bogus", padded_conv(mode="bogus"))

    def test_explicit_axes_are_rejected(self):
        self.reject_saved("leading Pad with explicit axes is not supported",
                          padded_conv(with_axes=True))


def _negative_pad_model():
    model = padded_conv(mode="edge", pads=(1, 1, 1, 1))
    pads = np.array([0, 0, 1, 1, 0, 0, -1, 1], np.int64)
    for tensor in model.graph.initializer:
        if tensor.name == "pads":
            tensor.CopyFrom(nh.from_array(pads, "pads"))
    return model


class ActivationFusionTests(_PathMixin, unittest.TestCase):
    """Native `Relu`/`Clip[0,6]` fusion and the rejection of other activations."""

    def test_relu_and_clip_are_fused(self):
        _, relu = compile_native_input(conv_activation_model("Relu"))
        self.assertEqual(relu["fused_activation"], "Relu")
        _, clip = compile_native_input(conv_activation_model("Clip", 0.0, 6.0))
        self.assertEqual(clip["fused_activation"], "Clip[0,6]")

    def test_other_clip_ranges_are_rejected(self):
        message = "native Clip fusion requires constant scalar range [0,6]"
        for low, high in ((0.0, 7.0), (1.0, 6.0)):
            with self.subTest(low=low, high=high):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    compile_native_input(conv_activation_model("Clip", low, high))

    def test_other_activations_are_rejected(self):
        self.reject_saved("sequence lowering currently supports Conv[/Relu] followed by 2x2 pooling",
                          conv_activation_model("HardSigmoid"))
        with self.assertRaisesRegex(ValueError, re.escape("native input profile requires one Conv")):
            compile_native_input(conv_activation_model("HardSigmoid"))


class QuantizedImportFrontEndTests(_PathMixin, unittest.TestCase):
    """Quantized import routes are accepted and carry their own parameters."""

    def test_qlinearconv_import_is_accepted(self):
        _, meta = self.compile_saved(qlinear_model())
        self.assertEqual(meta["quantized_import"], "QLinearConv weights preserved bit-for-bit")
        self.reject_saved("QLinearConv carries its own quantization parameters", qlinear_model(), 0.5, 0)

    def test_qdq_import_is_accepted_and_per_channel_requires_axis0(self):
        _, meta = self.compile_saved(qdq_model(per_channel=True, axis=0))
        self.assertIn("Q/DQ Conv INT8 weights preserved bit-for-bit", meta["quantized_import"])
        self.reject_saved("per-channel weight Q/DQ requires axis0", qdq_model(per_channel=True, axis=1))


class UnsupportedInputTests(_PathMixin, unittest.TestCase):
    """Unsupported ONNX shapes and operators are rejected loudly."""

    def test_one_dimensional_convolution_is_promoted_to_height_one(self):
        """F2: a rank-3 [N,C,L] Conv compiles as the rank-4 [N,C,1,L] form.

        The bytes are the same, so the promotion is algebraic; the container reports the
        promoted geometry and the decoded shape must match the rank-4 equivalent exactly.
        """
        from open_rknpu.sequence import decode_sequence
        binary, _meta = self.compile_saved(one_dimensional_conv())
        info = decode_sequence(binary)
        # decode_sequence reports NHWC: [N, H, W, C].
        self.assertEqual(info["shape_nhwc"], [1, 1, 8, 3])
        self.assertEqual(info["output_shape_nhwc"], [1, 1, 8, 4])
        self.assertEqual(info["task_count"], 1)

        # A rank-3 graph with a node the promotion cannot rewrite stays rejected, with the
        # node named, rather than being half-promoted.
        model = one_dimensional_conv()
        node = h.make_node("Flatten", ["y"], ["z"], axis=1)
        model.graph.node.append(node)
        model.graph.output[0].name = "z"
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 32
        del model.graph.output[0].type.tensor_type.shape.dim[2:]
        with self.assertRaisesRegex(ValueError, re.escape("1-D rank promotion requires rank-3 outputs")):
            self.compile_saved(model)

    def test_matmul_softmax_and_concat(self):
        message = "sequence lowering requires one input, one output, and an initial Conv"
        models = (unsupported_model("MatMul", operand_shape=(8, 8)),
                  unsupported_model("Softmax"),
                  unsupported_model("Concat", operand_shape=(1, 1, 8, 8),
                                    output_shape=(1, 4, 8, 8), attrs=dict(axis=1)))
        for model in models:
            with self.subTest(op=model.graph.node[0].op_type):
                self.reject_saved(message, model)

    def test_dynamic_shapes(self):
        self.reject_saved("native Conv requires static batch1..16, H/W1..128, input C1..128",
                          dynamic_conv())

    def test_second_graph_input(self):
        self.reject_saved("sequence lowering requires one input, one output, and an initial Conv",
                          two_input_conv())


class NetworkProfileTests(_PathMixin, unittest.TestCase):
    """The three-pool `network` profile bounds."""

    def test_valid_network_compiles(self):
        for kind, profile in (("MaxPool", 7), ("AveragePool", 8)):
            with self.subTest(kind=kind):
                _, meta = self.compile_network_saved(network_model(kind=kind))
                self.assertEqual(meta["profile"], profile)
                self.assertEqual(meta["output_shape_nhwc"], [1, 1, 1, 3])
                self.assertEqual(meta["pool_levels"], 3)

    def test_node_count_and_prefix_are_checked(self):
        model = network_model()
        del model.graph.node[5]
        model.graph.output[0].name = model.graph.node[-1].output[0]
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 2
        model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 2
        with self.assertRaisesRegex(ValueError, re.escape("expected Conv-Relu-Conv followed by three pools")):
            self.compile_network_saved(model)

    def test_mixed_and_malformed_pools_are_rejected(self):
        message = "three matching 2x2 stride-2 pools required"
        mixed = network_model()
        mixed.graph.node[4].op_type = "AveragePool"
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            self.compile_network_saved(mixed)
        malformed = network_model()
        pool = malformed.graph.node[3]
        del pool.attribute[:]
        pool.attribute.extend([h.make_attribute("kernel_shape", [3, 3]),
                               h.make_attribute("strides", [2, 2])])
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            self.compile_network_saved(malformed)

    def test_output_shape_is_checked(self):
        model = network_model()
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 2
        with self.assertRaisesRegex(ValueError, re.escape("network output must be float32 [1,3,1,1]")):
            self.compile_network_saved(model)


if __name__ == "__main__":
    unittest.main()
