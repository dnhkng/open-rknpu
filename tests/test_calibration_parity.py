# SPDX-License-Identifier: MIT
"""Calibration parity: every scheduler profile classified in the doc and exercised.

The cookbook table ("Which profiles accept calibration",
`docs/calibration-cookbook.md`) is the contract this module checks. Each profile in
`open_rknpu.scheduler.DISPATCH_PROFILES` has one row; a profile that carries measured
bands is compiled here and its per-stage bands are compared with the requested ranges,
and a profile that refuses calibration must raise exactly the documented message.

No board, no network, no sleeps: every graph is built in memory and compiled through a
temporary file.
"""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.calibration import measured_tensor_names
from open_rknpu.scheduler import DISPATCH_PROFILES, compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "calibration-cookbook.md"


# --------------------------------------------------------------------------- #
# Graph builders (one per profile, deliberately tiny and deterministic)
# --------------------------------------------------------------------------- #

def _graph(nodes, inputs, outputs, initializers, value_info=()):
    graph = h.make_graph(nodes, "parity", inputs, outputs, list(initializers),
                         value_info=list(value_info))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def _conv(source, out, inits, channels, input_channels, kernel, pads=None, strides=(1, 1),
          group=1, seed=0, activation=None):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-.7, .8, (channels, input_channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-2, 2, (channels,)).astype(np.float32)
    name = f"w_{out}"
    inits += [nh.from_array(weight, name), nh.from_array(bias, f"b_{out}")]
    attrs = dict(kernel_shape=[kernel, kernel], strides=list(strides), dilations=[1, 1], group=group)
    attrs["pads"] = [kernel // 2] * 4 if pads is None else pads
    nodes = [h.make_node("Conv", [source, name, f"b_{out}"], [out], **attrs)]
    if activation:
        nodes.append(h.make_node(activation, [out], [out + "_act"]))
    return nodes


def _block(source, out, inits, channels, input_channels, kernel, seed, activation=None, **kwargs):
    return _conv(source, out, inits, channels, input_channels, kernel, seed=seed,
                 activation=activation, **kwargs)


def _rgb(name="input", height=8, width=8):
    return h.make_tensor_value_info(name, 1, [1, 3, height, width])


def _vi(name, channels, height=8, width=8):
    return h.make_tensor_value_info(name, 1, [1, channels, height, width])


def _qlinear():
    """A minimal QLinearConv graph: the importer owns every band."""
    rng = np.random.default_rng(3)
    qw = rng.integers(-6, 7, (3, 3, 1, 1)).astype(np.int8)
    constants = [nh.from_array(np.array(1 / 255, np.float32), "xscale"),
                 nh.from_array(np.array(128, np.uint8), "xzp"),
                 nh.from_array(qw, "w"),
                 nh.from_array(np.full(3, 0.02, np.float32), "wscale"),
                 nh.from_array(np.zeros(3, np.int8), "wzp"),
                 nh.from_array(np.array(0.03, np.float32), "yscale"),
                 nh.from_array(np.array(0, np.int8), "yzp")]
    node = h.make_node("QLinearConv", ["input", "xscale", "xzp", "w", "wscale", "wzp",
                                       "yscale", "yzp"], ["output"],
                       kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1])
    graph = h.make_graph([node], "qlinearconv",
                         [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, 3, 8, 8])],
                         constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _qdq():
    """A minimal DequantizeLinear/Conv/QuantizeLinear graph: it owns every band."""
    rng = np.random.default_rng(5)
    qw = rng.integers(-6, 7, (3, 3, 1, 1)).astype(np.int8)
    constants = [nh.from_array(np.array(1 / 255, np.float32), "xscale"),
                 nh.from_array(np.array(128, np.uint8), "xzp"),
                 nh.from_array(qw, "w"),
                 nh.from_array(np.full(3, 0.02, np.float32), "ws"),
                 nh.from_array(np.zeros(3, np.int8), "wz"),
                 nh.from_array(np.array(0.03, np.float32), "yscale"),
                 nh.from_array(np.array(0, np.int8), "yzp"),
                 nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bias")]
    nodes = [h.make_node("DequantizeLinear", ["input", "xscale", "xzp"], ["xdq"]),
             h.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wdq"], axis=0),
             h.make_node("Conv", ["xdq", "wdq", "bias"], ["conv"], kernel_shape=[1, 1],
                         pads=[0, 0, 0, 0], strides=[1, 1]),
             h.make_node("QuantizeLinear", ["conv", "yscale", "yzp"], ["output"])]
    graph = h.make_graph(nodes, "qdq",
                         [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, 3, 8, 8])],
                         constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _join_chain(tail=None):
    inits = []
    nodes = _block("input", "stem", inits, 3, 3, 1, seed=1)
    for index, (channels, kernel) in enumerate(((3, 1), (3, 3), (3, 1))):
        nodes += _block("stem", f"h{index}", inits, channels, 3, kernel, seed=10 + index)
    nodes += [h.make_node("Add", ["h0", "h1"], ["j0"]),
              h.make_node("Mul", ["j0", "h2"], ["output"])]
    value_info = [_vi(f"h{i}", 3) for i in range(3)]
    if tail:
        inits += []
        nodes[-1].output[0] = "joined"
        nodes += _block("joined", "output", inits, 3, 3, tail, seed=20)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, value_info)


def _join_dag():
    inits = []
    nodes = _block("image", "stem", inits, 3, 3, 1, seed=31)
    for index, kernel in enumerate((1, 3, 1)):
        nodes += _block("stem", f"h{index}", inits, 3, 3, kernel, seed=32 + index)
    nodes += [h.make_node("Add", ["h0", "h1"], ["j0"]),
              h.make_node("Mul", ["j0", "h0"], ["output"])]
    value_info = [_vi(f"h{i}", 3) for i in range(3)]
    return _graph(nodes, [_rgb("image")], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, value_info)


def _depthwise_join():
    inits = []
    nodes = _block("input", "stem", inits, 3, 3, 1, seed=41)
    nodes += _block("stem", "dense0", inits, 3, 3, 1, seed=42)
    nodes += _block("stem", "depthwise", inits, 3, 1, 3, pads=[1, 1, 1, 1], group=3, seed=43)
    nodes.append(h.make_node("Add", ["dense0", "depthwise"], ["output"]))
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("dense0", 3), _vi("depthwise", 3), _vi("stem", 3)])


def _pool_join():
    inits = []
    nodes = _block("input", "stem0", inits, 8, 3, 1, seed=51) + \
        [h.make_node("Relu", ["stem0"], ["stem"])]
    for name, kernel, seed in (("a", 1, 52), ("b", 3, 53)):
        nodes += _block("stem", f"head_{name}", inits, 3, 8, kernel, seed=seed)
        nodes.append(h.make_node("MaxPool", [f"head_{name}"], [f"pool_{name}"],
                                 kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(h.make_node("Add", ["pool_a", "pool_b"], ["output"]))
    value_info = [_vi("stem", 8), _vi("head_a", 3), _vi("head_b", 3),
                  _vi("pool_a", 3, 4, 4), _vi("pool_b", 3, 4, 4)]
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                  inits, value_info)


def _pooled_branches():
    inits = []
    nodes = _block("image", "stem", inits, 3, 3, 1, seed=61)
    for branch in range(2):
        source = "stem"
        for layer in range(2):
            name = f"b{branch}_{layer}"
            nodes += _block(source, name, inits, 3, 3, 1, seed=62 + branch * 2 + layer)
            source = name
        nodes.append(h.make_node("MaxPool", [source], [f"pool{branch}"],
                                 kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(h.make_node("Add", ["pool0", "pool1"], ["output"]))
    value_info = [_vi(f"b{b}_{l}", 3) for b in range(2) for l in range(2)]
    value_info += [_vi("pool0", 3, 4, 4), _vi("pool1", 3, 4, 4)]
    return _graph(nodes, [_rgb("image")], [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                  inits, value_info)


def _lut():
    inits = []
    weight = np.zeros((3, 3, 1, 1), np.float32)
    for channel in range(3):
        weight[channel, channel, 0, 0] = np.float32(1 / 32)
    inits.append(nh.from_array(weight, "w"))
    inits.append(nh.from_array(np.zeros(3, np.float32), "b"))
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Sigmoid", ["stem"], ["output"])]
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("stem", 3)])


def _mul_relu():
    nodes = [h.make_node("Mul", ["a", "b"], ["product"]),
             h.make_node("Relu", ["product"], ["output"])]
    return _graph(nodes, [_rgb("a"), _rgb("b")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], [])


def _mul_clip():
    inits = [nh.from_array(np.float32(0.0), "lo"), nh.from_array(np.float32(6.0), "hi")]
    nodes = [h.make_node("Mul", ["a", "b"], ["product"]),
             h.make_node("Clip", ["product", "lo", "hi"], ["output"])]
    return _graph(nodes, [_rgb("a"), _rgb("b")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], inits)


def _mul_add():
    inits = [nh.from_array(np.float32(1.0), "offset")]
    nodes = [h.make_node("Mul", ["a", "b"], ["product"]),
             h.make_node("Add", ["product", "offset"], ["output"])]
    return _graph(nodes, [_rgb("a"), _rgb("b")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], inits)


def _activation(kind, alpha=0.1):
    inits = []
    nodes = _block("input", "stem", inits, 3, 3, 1, seed=71)
    if kind == "LeakyRelu":
        nodes.append(h.make_node("LeakyRelu", ["stem"], ["output"], alpha=alpha))
    else:
        inits.append(nh.from_array(np.array([.1, .2, .3], np.float32), "slopes"))
        nodes.append(h.make_node("PRelu", ["stem", "slopes"], ["output"]))
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("stem", 3)])


def _transposed():
    inits = []
    nodes = _block("input", "stem", inits, 8, 3, 1, seed=81)
    weight = np.zeros((8, 3, 3, 3), np.float32)
    rng = np.random.default_rng(82)
    weight[:, :, 1, 1] = rng.uniform(-.7, .8, (8, 3)).astype(np.float32)
    inits.append(nh.from_array(weight, "tw"))
    inits.append(nh.from_array(np.zeros(3, np.float32), "tb"))
    nodes.append(h.make_node("ConvTranspose", ["stem", "tw", "tb"], ["output"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("stem", 8)])


def _depthwise_pointwise():
    inits = []
    nodes = _block("input", "d0", inits, 3, 3, 3, seed=91)
    nodes += _block("d0", "dw", inits, 3, 1, 3, pads=[1, 1, 1, 1], group=3, seed=92)
    nodes += _block("dw", "output", inits, 3, 3, 1, seed=93)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("d0", 3), _vi("dw", 3)])


def _reshape():
    inits = []
    nodes = _block("input", "conv", inits, 3, 3, 1, seed=101)
    inits.append(nh.from_array(np.array([1, 3, 8, 8], np.int64), "shape"))
    nodes.append(h.make_node("Reshape", ["conv", "shape"], ["output"]))
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("conv", 3)])


def _standalone_mul():
    return _graph([h.make_node("Mul", ["a", "b"], ["output"])], [_rgb("a"), _rgb("b")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], [])


def _constant_mul(spatial=False):
    inits = []
    if spatial:
        values = np.array([1.0, 2.0, 3.0], np.float32).reshape(1, 3, 1, 1)
    else:
        values = np.array(2.0, np.float32)
    inits.append(nh.from_array(values, "factor"))
    return _graph([h.make_node("Mul", ["input", "factor"], ["output"])], [_rgb()],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], inits)


def _runtime_scale_mul():
    return _graph([h.make_node("Mul", ["image", "scale"], ["output"])],
                  [_rgb("image"), h.make_tensor_value_info("scale", 1, [1, 3, 1, 1])],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], [])


def _two_head():
    inits = []
    nodes = _block("input", "stem", inits, 8, 3, 1, seed=111) + \
        [h.make_node("Relu", ["stem"], ["relu"])]
    nodes += _block("relu", "head_a", inits, 3, 8, 1, seed=112)
    nodes += _block("relu", "head_b", inits, 3, 8, 1, seed=113)
    return _graph(nodes, [_rgb()],
                  [h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]),
                   h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8])],
                  inits, [_vi("relu", 8)])


def _join_walk():
    inits = []
    nodes = _block("input", "stem", inits, 8, 3, 1, seed=121) + \
        [h.make_node("Relu", ["stem"], ["relu1"])]
    for name, seed in (("a", 122), ("b", 123)):
        nodes += _block("relu1", f"head_{name}", inits, 3, 8, 1, seed=seed)
        nodes.append(h.make_node("MaxPool", [f"head_{name}"], [f"pool_{name}"],
                                 kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(h.make_node("Add", ["pool_a", "pool_b"], ["output"]))
    value_info = [_vi("relu1", 8), _vi("head_a", 3), _vi("head_b", 3),
                  _vi("pool_a", 3, 4, 4), _vi("pool_b", 3, 4, 4)]
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                  inits, value_info)


def _diamond():
    inits = []
    nodes = _block("input", "stem", inits, 8, 3, 1, seed=131) + \
        [h.make_node("Relu", ["stem"], ["relu1"])]
    nodes += _block("relu1", "head_a", inits, 3, 8, 1, seed=132)
    nodes += _block("relu1", "head_b", inits, 3, 8, 3, seed=133)
    nodes.append(h.make_node("Add", ["head_a", "head_b"], ["output"]))
    return _graph(nodes, [_rgb()],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("relu1", 8), _vi("head_a", 3), _vi("head_b", 3)])


def _native_chain():
    inits = []
    nodes = _block("input", "c0", inits, 5, 3, 1, seed=141, activation="Relu")
    nodes += _block("c0_act", "c1", inits, 5, 5, 3, seed=142, activation="Relu")
    nodes += _block("c1_act", "output", inits, 3, 5, 1, seed=143)
    return _graph(nodes, [_rgb()],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("c0_act", 5), _vi("c1_act", 5)])


def _legacy_chain():
    inits = []
    nodes = _block("input", "c0", inits, 5, 3, 1, seed=151) + \
        [h.make_node("Relu", ["c0"], ["r0"])]
    nodes += _block("r0", "output", inits, 3, 5, 1, seed=152)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("r0", 5)])


def _multi_input_dag():
    inits = []
    nodes = _block("a", "A", inits, 3, 3, 1, seed=161)
    nodes += _block("b", "B", inits, 3, 3, 1, seed=162)
    nodes.append(h.make_node("Add", ["A", "B"], ["C"]))
    nodes += _block("c", "D", inits, 3, 3, 1, seed=163)
    nodes.append(h.make_node("Mul", ["C", "D"], ["output"]))
    return _graph(nodes, [_rgb("a"), _rgb("b"), _rgb("c")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], inits,
                  [_vi("A", 3), _vi("B", 3), _vi("C", 3), _vi("D", 3)])


def _elementwise_dag():
    inits = []
    nodes = _block("a", "A", inits, 3, 3, 1, seed=171)
    nodes += _block("b", "B", inits, 3, 3, 1, seed=172)
    nodes.append(h.make_node("Add", ["A", "B"], ["C"]))
    nodes.append(h.make_node("Mul", ["C", "A"], ["output"]))
    return _graph(nodes, [_rgb("a"), _rgb("b")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], inits,
                  [_vi("A", 3), _vi("B", 3), _vi("C", 3)])


def _elementwise_join(kind="Mul"):
    inits = []
    nodes = _block("a", "A", inits, 3, 3, 1, seed=181)
    nodes += _block("b", "B", inits, 3, 3, 1, seed=182)
    nodes.append(h.make_node(kind, ["A", "B"], ["output"]))
    return _graph(nodes, [_rgb("a"), _rgb("b")],
                  [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], inits,
                  [_vi("A", 3), _vi("B", 3)])


def _chain_walk():
    inits = []
    nodes = _block("input", "c0", inits, 8, 3, 3, seed=191) + \
        [h.make_node("Relu", ["c0"], ["r0"]),
         h.make_node("MaxPool", ["r0"], ["p0"], kernel_shape=[2, 2], strides=[2, 2])]
    nodes += _block("p0", "output", inits, 3, 8, 3, seed=192)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("r0", 8), _vi("p0", 8, 4, 4)])


def _native_input():
    inits = []
    nodes = _block("input", "output", inits, 32, 3, 3, seed=201)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 32, 8, 8])], inits)


def _strided():
    inits = []
    nodes = _block("input", "output", inits, 3, 3, 3, pads=[0, 0, 0, 0], strides=(2, 2), seed=211)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 3, 3])], inits)


def _depthwise():
    inits = []
    nodes = _block("input", "c0", inits, 3, 3, 3, seed=221) + \
        [h.make_node("Relu", ["c0"], ["r0"])]
    nodes += _block("r0", "output", inits, 3, 1, 3, pads=[1, 1, 1, 1], group=3, seed=222)
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                  inits, [_vi("r0", 3)])


def _pooling_sequence():
    inits = []
    nodes = _block("input", "c0", inits, 3, 3, 3, seed=231) + \
        [h.make_node("Relu", ["c0"], ["r0"]),
         h.make_node("MaxPool", ["r0"], ["output"], kernel_shape=[2, 2], strides=[2, 2])]
    return _graph(nodes, [_rgb()], [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                  inits, [_vi("r0", 3)])


# --------------------------------------------------------------------------- #
# The profile table
# --------------------------------------------------------------------------- #

def _profile(key, support, fixture, kwargs=None, message=None, output_override=None,
             tensors="convs+output", compare_analytic=True, output_hand_supplied=False):
    return dict(key=key, support=support, fixture=fixture, kwargs=kwargs or {},
                message=message, output_override=output_override, tensors=tensors,
                compare_analytic=compare_analytic, output_hand_supplied=output_hand_supplied)


PROFILES = (
    _profile("qlinearconv", "rejected", _qlinear,
             message="QLinearConv carries its own quantization parameters", tensors="none"),
    _profile("qdq-conv", "rejected", _qdq,
             message="Q/DQ Conv carries its own quantization parameters", tensors="none"),
    _profile("join-chain", "supported", _join_chain),
    _profile("join-dag", "supported", _join_dag),
    _profile("depthwise-join", "supported", _depthwise_join),
    _profile("pool-join", "supported", _pool_join),
    _profile("pooled-branches", "rejected", _pooled_branches,
             message="calibration is unsupported for pooled branches", tensors="none"),
    _profile("lut", "rejected", _lut,
             message="output override unsupported for LUT profile", tensors="none"),
    _profile("mul-relu", "supported", _mul_relu, tensors="output", output_hand_supplied=True),
    _profile("mul-clip", "supported", _mul_clip, tensors="output",
             output_override={"scale": 6 / 255, "zero_point": -128}, compare_analytic=False,
             output_hand_supplied=True),
    _profile("mul-add", "supported", _mul_add, tensors="output",
             output_override={"scale": 0.01, "zero_point": 0}, compare_analytic=False,
             output_hand_supplied=True),
    _profile("leaky-relu", "rejected", lambda: _activation("LeakyRelu"),
             message="output override unsupported for LeakyRelu profile", tensors="none"),
    _profile("prelu", "rejected", lambda: _activation("PRelu"),
             message="output override unsupported for PRelu profile", tensors="none"),
    _profile("transposed-conv", "supported", _transposed, tensors="output", output_hand_supplied=True),
    _profile("depthwise-pointwise", "supported", _depthwise_pointwise, tensors="output"),
    _profile("reshape", "rejected", _reshape,
             message="output override unsupported for Reshape profile", tensors="none"),
    _profile("standalone-mul", "supported", _standalone_mul, tensors="output", output_hand_supplied=True),
    _profile("constant-mul", "supported", _constant_mul, tensors="output", output_hand_supplied=True),
    _profile("per-channel-constant-mul", "supported", lambda: _constant_mul(spatial=True),
             kwargs={"per_channel_mul": True}, tensors="output", output_hand_supplied=True),
    _profile("runtime-scale-mul", "supported", _runtime_scale_mul, tensors="output", output_hand_supplied=True),
    _profile("two-head", "rejected", _two_head,
             message="output override unsupported for the two-head profile", tensors="none"),
    _profile("join-walk", "supported", _join_walk),
    _profile("diamond", "supported", _diamond),
    _profile("native-chain", "supported", _native_chain, tensors="convs"),
    _profile("legacy-conv-chain", "supported", _legacy_chain),
    _profile("multi-input-elementwise-dag", "rejected", _multi_input_dag,
             message="output override unsupported for the multi-input elementwise DAG",
             tensors="none"),
    _profile("elementwise-dag", "rejected", _elementwise_dag,
             message="output override unsupported for the elementwise DAG", tensors="none"),
    _profile("elementwise-join", "supported", _elementwise_join, tensors="output", output_hand_supplied=True),
    _profile("chain-walk", "supported", _chain_walk),
    _profile("native-input", "supported", _native_input, tensors="output"),
    _profile("strided", "rejected", _strided,
             message="output override unsupported for this strided profile", tensors="none"),
    _profile("depthwise", "supported", _depthwise, tensors="output"),
    _profile("pooling-sequence", "supported", _pooling_sequence, tensors="convs"),
)

JOIN_FAMILY = {"join-chain", "join-dag", "depthwise-join", "pool-join", "join-walk", "diamond"}


def _compile(model, calibration_ranges, kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path, calibration_ranges=calibration_ranges, **kwargs)


def _requested_ranges(model, output_override=None):
    names = list(measured_tensor_names(model))
    output = model.graph.output[0].name
    if output not in names:
        names.append(output)
    ranges = {name: {"scale": 0.001 * (index + 1), "zero_point": -100 + index}
              for index, name in enumerate(names)}
    if output_override is not None:
        ranges[output] = dict(output_override)
    return ranges, names


def _bands(key, meta):
    if key == "native-chain":
        return tuple((round(q["output_scale"], 6), q["output_zero_point"]) for q in meta["quantizations"])
    if key in JOIN_FAMILY:
        stem = meta["stem_quantization"]
        heads = [(round(q["output_scale"], 6), q["output_zero_point"])
                 for q in meta.get("head_quantization", [])]
        tails = [(round(q["output_scale"], 6), q["output_zero_point"])
                 for q in meta.get("tail_quantization", [])]
        return ((round(stem["output_scale"], 6), stem["output_zero_point"]), tuple(heads), tuple(tails))
    if key == "legacy-conv-chain":
        first = meta["first"]
        return ((round(first["output_scale"], 6), first["output_zero_point"]),
                (round(meta["output_scale"], 6), meta["output_zero_point"]))
    return (round(meta["output_scale"], 6), meta["output_zero_point"])


def _assert_stage_bands(test, key, model, ranges, binary, meta):
    output = model.graph.output[0].name
    requested_out = (round(ranges[output]["scale"], 6), ranges[output]["zero_point"])
    # Every container reports the same output band it decodes to.
    info = decode_sequence(binary)
    test.assertEqual(round(float(info["output_scale"]), 6), round(float(meta["output_scale"]), 6), key)
    test.assertEqual(info["output_zero_point"], meta["output_zero_point"], key)
    if key in JOIN_FAMILY:
        stem = meta["stem_quantization"]
        stem_tensor = measured_tensor_names(model)[0]
        test.assertEqual((round(stem["output_scale"], 6), stem["output_zero_point"]),
                         (round(ranges[stem_tensor]["scale"], 6), ranges[stem_tensor]["zero_point"]), key)
        for quantization in meta.get("head_quantization", []):
            test.assertEqual(quantization["output_zero_point"], 0, key)
            test.assertGreater(quantization["output_scale"], 0, key)
        if key == "join-chain":
            tail_ranges, _ = _requested_ranges(_join_chain(tail=3))
            _, tail_meta = _compile(_join_chain(tail=3), tail_ranges, {})
            last = tail_meta["tail_quantization"][-1]
            test.assertEqual((round(last["output_scale"], 6), last["output_zero_point"]),
                             (round(tail_ranges["output"]["scale"], 6), tail_ranges["output"]["zero_point"]), key)
        return
    if key == "native-chain":
        expected = [(round(ranges[name]["scale"], 6), ranges[name]["zero_point"])
                    for name in measured_tensor_names(model)]
        test.assertEqual([(round(q["output_scale"], 6), q["output_zero_point"])
                          for q in meta["quantizations"]], expected, key)
        return
    if key == "legacy-conv-chain":
        first = meta["first"]
        test.assertEqual((round(first["output_scale"], 6), first["output_zero_point"]),
                         (round(ranges["r0"]["scale"], 6), ranges["r0"]["zero_point"]), key)
        test.assertEqual((round(meta["output_scale"], 6), meta["output_zero_point"]), requested_out, key)
        return
    if key == "chain-walk":
        layers = [q for q in meta["quantizations"] if q]
        test.assertEqual((round(layers[-1]["output_scale"], 6), layers[-1]["output_zero_point"]),
                         requested_out, key)
        test.assertEqual(layers[0]["output_zero_point"], 0, key)
        return
    if key == "pooling-sequence":
        quantization = meta["quantization"]
        first = measured_tensor_names(model)[0]
        test.assertEqual((round(quantization["output_scale"], 6), quantization["output_zero_point"]),
                         (round(ranges[first]["scale"], 6), ranges[first]["zero_point"]), key)
        return
    test.assertEqual((round(meta["output_scale"], 6), meta["output_zero_point"]), requested_out, key)


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #

def doc_profile_keys():
    section = DOC.read_text().split("## Which profiles accept calibration", 1)[1]
    section = section.split("\n## ", 1)[0]
    keys = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("| "):
            continue
        first = line[2:].split("|", 1)[0].strip()
        match = re.fullmatch(r"`([a-z0-9-]+)`", first)
        if match:
            keys.append(match.group(1))
    return keys


class CalibrationParityTests(unittest.TestCase):
    def test_doc_table_matches_scheduler_dispatch(self):
        self.assertEqual(doc_profile_keys(), list(DISPATCH_PROFILES))

    def test_every_dispatch_profile_has_a_row_in_the_test_table(self):
        self.assertEqual([profile["key"] for profile in PROFILES], list(DISPATCH_PROFILES))

    def test_supported_profiles_reproduce_the_measured_bands(self):
        for profile in PROFILES:
            key = profile["key"]
            if profile["support"] != "supported":
                continue
            with self.subTest(profile=key):
                model = profile["fixture"]()
                ranges, _ = _requested_ranges(model, profile["output_override"])
                binary, meta = _compile(model, ranges, profile["kwargs"])
                self.assertIsInstance(binary, bytes)
                _assert_stage_bands(self, key, model, ranges, binary, meta)
                if profile["compare_analytic"]:
                    _, analytic = _compile(model, None, profile["kwargs"])
                    self.assertNotEqual(_bands(key, analytic), _bands(key, meta),
                                        "measured ranges must change the emitted bands")

    def test_rejected_profiles_raise_the_documented_message(self):
        for profile in PROFILES:
            key = profile["key"]
            if profile["support"] != "rejected":
                continue
            with self.subTest(profile=key):
                model = profile["fixture"]()
                ranges, _ = _requested_ranges(model)
                with self.assertRaisesRegex(ValueError, re.escape(profile["message"])):
                    _compile(model, ranges, profile["kwargs"])

    def test_hand_supplied_output_keys_are_not_measured_tensors(self):
        """The cookbook says these profiles need an output key `measure` cannot produce."""
        for profile in PROFILES:
            if not profile["output_hand_supplied"]:
                continue
            with self.subTest(profile=profile["key"]):
                model = profile["fixture"]()
                names = measured_tensor_names(model)
                self.assertNotIn(model.graph.output[0].name, names)

    def test_missing_tensor_message_per_supporting_profile(self):
        for profile in PROFILES:
            key = profile["key"]
            if profile["support"] != "supported":
                continue
            with self.subTest(profile=key):
                model = profile["fixture"]()
                with self.assertRaisesRegex(ValueError, r"calibration ranges lack tensor"):
                    _compile(model, {}, profile["kwargs"])


if __name__ == "__main__":
    unittest.main()
