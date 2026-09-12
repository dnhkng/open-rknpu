"""SPDX-License-Identifier: MIT

Profile bounds and rejection matrix for the scheduled families in `open_rknpu.scheduler`.

Every profile in the scheduler is a *bounded* lowering: some minimally valid graph
compiles, and the first out-of-bounds neighbour is rejected with a message that names
the violated bound. A bound test that only asserts "it raises" cannot tell a widened
bound from a reworded error, and it cannot tell a tightened bound from a broken valid
path. This module therefore pairs, for each family, the valid neighbour that compiles
with the invalid neighbour's exact message.

Covered families: the dense/image native Conv (`native.compile_native_input`), the
N-chain family (`chain_n`/`tiled_chain`), the op-level walk (`walk.parse_chain` /
`compile_chain_walk`), depthwise, pooling/reduction, the elementwise joins, the LUT,
ConvTranspose, the QLinearConv/Q/DQ import paths, and the scheduler's own argument
validation. Everything is deterministic (fixed seeds), uses temporary graph paths and
touches no board or vendor artifact.
"""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.chain_n import compile_chain_n
from open_rknpu.depthwise import compile_depthwise
from open_rknpu.lut import compile_lut
from open_rknpu.native import compile_native_input
from open_rknpu.pooling import compile_pool
from open_rknpu.quantized_import import compile_qdq_conv, compile_qlinearconv
from open_rknpu.reduction import compile_reduction
from open_rknpu.scheduler import compile_sequence
from open_rknpu.tiled_chain import compile_tiled_chain
from open_rknpu.transposed import compile_transposed
from open_rknpu.walk import compile_chain_walk, parse_chain

NATIVE_SHAPE = ("native Conv requires static batch1..16, H/W1..128, input C1..16352, "
                "constant weights/bias")
NATIVE_ATTRS = ("native Conv supports odd K1..31, explicit padding, stride 1..4, "
                "input C1..16352/output C1..8192")
NATIVE_GEOMETRY = "native padding/stride/dilation unsupported"
NATIVE_OUTPUT = "invalid native Conv output geometry"


# ---------------------------------------------------------------------------
# Small deterministic graph builders (the house style: helper.make_node).
# ---------------------------------------------------------------------------

def native_model(ic=3, oc=4, ih=8, iw=8, k=3, pads=None, strides=None, dilations=None,
                 group=None, weight_dtype=np.float32, bias=True, input_dtype=1, rank=4,
                 dynamic=False, output_shape=None):
    """A single Conv graph for `native.compile_native_input` boundary probes."""
    pads = [k // 2] * 4 if pads is None else list(pads)
    strides = [1, 1] if strides is None else list(strides)
    dilations = [1, 1] if dilations is None else list(dilations)
    attrs = dict(kernel_shape=[k, k], pads=pads, strides=strides, dilations=dilations)
    if group is not None:
        attrs["group"] = group
    weights = np.zeros((oc, ic, k, k), weight_dtype)
    inputs = ["x", "w"] + (["b"] if bias else [])
    nodes = [h.make_node("Conv", inputs, ["y"], **attrs)]

    def spatial(size, begin, end, stride, dilation):
        try:
            return (size + begin + end - ((k - 1) * dilation + 1)) // stride + 1
        except ZeroDivisionError:
            return 1

    oh = spatial(ih, pads[0], pads[2], strides[0], dilations[0])
    ow = spatial(iw, pads[1], pads[3], strides[1], dilations[1])
    if dynamic:
        xinfo = h.make_tensor_value_info("x", input_dtype, [1, ic, "H", iw])
    elif rank != 4:
        xinfo = h.make_tensor_value_info("x", input_dtype, [1, ic, ih])
    else:
        xinfo = h.make_tensor_value_info("x", input_dtype, [1, ic, ih, iw])
    initializers = [nh.from_array(weights, "w")]
    if bias:
        initializers.append(nh.from_array(np.zeros(oc, np.float32), "b"))
    graph = h.make_graph(nodes, "native", [xinfo],
                         [h.make_tensor_value_info("y", 1, output_shape or [1, oc, oh, ow])],
                         initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def native_non_square():
    """A Conv whose declared kernel is the rectangular 3x5."""
    weights = np.zeros((4, 3, 3, 5), np.float32)
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 5], pads=[1, 2, 1, 2])]
    graph = h.make_graph(nodes, "rect", [h.make_tensor_value_info("x", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])],
                         [nh.from_array(weights, "w"), nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def native_weights_as_input():
    """Weights supplied as a second external input: the one-input profile rejects it."""
    graph = h.make_graph(
        [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        "external_weights",
        [h.make_tensor_value_info("x", 1, [1, 3, 8, 8]),
         h.make_tensor_value_info("w", 1, [4, 3, 3, 3])],
        [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])],
        [nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def native_undeclared_weights():
    """The weight name has no initializer, so the Conv is not constant-foldable."""
    graph = h.make_graph(
        [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        "undeclared",
        [h.make_tensor_value_info("x", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])],
        [nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def native_rank_three_weights():
    """A rank-three weight tensor reaches the explicit rank check."""
    graph = h.make_graph(
        [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        "rank_three",
        [h.make_tensor_value_info("x", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("y", 1, [1, 4, 8, 8])],
        [nh.from_array(np.zeros((4, 3, 3), np.float32), "w"),
         nh.from_array(np.zeros(4, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def chain_model(hidden=4, kernels=(1, 1, 1), layers=3, seed=1, tail_relu=False):
    """`[Conv, Relu]*(N-1) + [Conv]` at 8x8 C3, the `chain_n` family."""
    rng = np.random.default_rng(seed)
    nodes, initializers = [], []
    source, channels = "input", 3
    for index in range(layers):
        kernel = kernels[index]
        out_channels = hidden if index < layers - 1 else 3
        weights = rng.uniform(-.7, .8, (out_channels, channels, kernel, kernel)).astype(np.float32)
        bias = rng.uniform(-2, 2, out_channels).astype(np.float32)
        nodes.append(h.make_node("Conv", [source, f"w{index}", f"b{index}"], [f"c{index}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        initializers += [nh.from_array(weights, f"w{index}"), nh.from_array(bias, f"b{index}")]
        source, channels = f"c{index}", out_channels
        if index < layers - 1:
            nodes.append(h.make_node("Relu", [source], [f"r{index}"]))
            source = f"r{index}"
    if tail_relu:
        nodes.append(h.make_node("Relu", [source], ["tail"]))
        source = "tail"
    graph = h.make_graph(nodes, "chain", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info(source, 1, [1, channels, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def even_conv_relu_model():
    """Four nodes `Conv, Relu, Conv, Relu`: an even-length pattern, not a chain."""
    rng = np.random.default_rng(2)
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["c0"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c0"], ["r0"]),
             h.make_node("Conv", ["r0", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c1"], ["output"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (4, 3, 1, 1)).astype(np.float32), "w0"),
                    nh.from_array(rng.uniform(-1, 1, 4).astype(np.float32), "b0"),
                    nh.from_array(rng.uniform(-.7, .8, (3, 4, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-1, 1, 3).astype(np.float32), "b1")]
    graph = h.make_graph(nodes, "even", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def walk_model(ops, height=8, width=8, channels=3, seed=1):
    """A linear walk graph: `("conv", C, K, act)` / `("MaxPool",)` / `("Sigmoid",)`."""
    rng = np.random.default_rng(seed)
    nodes, initializers = [], []
    source, source_channels, out_h, out_w = "input", channels, height, width
    for index, spec in enumerate(ops):
        if spec[0] == "conv":
            _, out_channels, kernel, activation = spec
            nodes.append(h.make_node("Conv", [source, f"w{index}", f"b{index}"], [f"c{index}"],
                                     kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
            initializers += [
                nh.from_array(rng.uniform(-.8, .8, (out_channels, source_channels, kernel, kernel)).astype(np.float32),
                              f"w{index}"),
                nh.from_array(rng.uniform(-1, 1, out_channels).astype(np.float32), f"b{index}")]
            source, source_channels = f"c{index}", out_channels
            if activation == "Relu":
                nodes.append(h.make_node("Relu", [source], [f"r{index}"]))
                source = f"r{index}"
        elif spec[0] in ("MaxPool", "AveragePool"):
            attrs = spec[1] if len(spec) > 1 else dict(kernel_shape=[2, 2], strides=[2, 2])
            nodes.append(h.make_node(spec[0], [source], [f"p{index}"], **attrs))
            source = f"p{index}"
            out_h, out_w = out_h // 2, out_w // 2
        else:
            nodes.append(h.make_node(spec[0], [source], [f"x{index}"]))
            source = f"x{index}"
    graph = h.make_graph(nodes, "walk", [h.make_tensor_value_info("input", 1, [1, channels, height, width])],
                         [h.make_tensor_value_info(source, 1, [1, source_channels, out_h, out_w])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def depthwise_model(channels=3, stem=3, kernel=3, strides=(1, 1), pads=None, size=8,
                    weight_shape=None, group=None):
    """`Conv[/Relu] -> depthwise Conv`, the fixed depthwise family."""
    rng = np.random.default_rng(4)
    group = channels if group is None else group
    pads = (kernel // 2,) * 4 if pads is None else pads
    stem_w = rng.uniform(-.5, .5, (channels, 3, stem, stem)).astype(np.float32)
    stem_b = rng.uniform(-1, 1, channels).astype(np.float32)
    depth_w = (rng.uniform(-.5, .5, (channels, 1, kernel, kernel)).astype(np.float32)
               if weight_shape is None else np.zeros(weight_shape, np.float32))
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["stem"],
                         kernel_shape=[stem, stem], pads=[stem // 2] * 4),
             h.make_node("Conv", ["stem", "w1", "b1"], ["output"], group=group,
                         kernel_shape=[kernel, kernel], strides=list(strides), pads=list(pads))]
    out_h = (size + pads[0] + pads[2] - kernel) // strides[0] + 1
    out_w = (size + pads[1] + pads[3] - kernel) // strides[1] + 1
    graph = h.make_graph(nodes, "depthwise",
                         [h.make_tensor_value_info("input", 1, [1, 3, size, size])],
                         [h.make_tensor_value_info("output", 1, [1, channels, out_h, out_w])],
                         [nh.from_array(stem_w, "w0"), nh.from_array(stem_b, "b0"),
                          nh.from_array(depth_w, "w1"), nh.from_array(np.zeros(channels, np.float32), "b1")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, channels, size, size]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def pool_model(kind="MaxPool", pool_attrs=None, out_shape=(1, 3, 4, 4)):
    attrs = dict(kernel_shape=[2, 2], strides=[2, 2]) if pool_attrs is None else pool_attrs
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[1, 1]),
             h.make_node(kind, ["c"], ["output"], **attrs)]
    graph = h.make_graph(nodes, "pool", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, list(out_shape))],
                         [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                          nh.from_array(np.zeros(3, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def reduction_model(levels=3, kind="MaxPool", out_shape=(1, 3, 1, 1)):
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[1, 1])]
    source = "c"
    for index in range(levels):
        nodes.append(h.make_node(kind, [source], [f"p{index}"], kernel_shape=[2, 2], strides=[2, 2]))
        source = f"p{index}"
    graph = h.make_graph(nodes, "reduction", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info(source, 1, list(out_shape))],
                         [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                          nh.from_array(np.zeros(3, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def two_branch_model(kind="Add", out_channels_b=3, scale_a=0.7, scale_b=0.7):
    """Two independent Conv branches feeding one join, the elementwise profile."""
    rng = np.random.default_rng(7)
    weights_a = rng.uniform(-scale_a, scale_a, (3, 3, 1, 1)).astype(np.float32)
    weights_b = rng.uniform(-scale_b, scale_b, (out_channels_b, 3, 1, 1)).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["input", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
             h.make_node(kind, ["A", "B"], ["output"])]
    initializers = [nh.from_array(weights_a, "wa"), nh.from_array(np.zeros(3, np.float32), "ba"),
                    nh.from_array(weights_b, "wb"), nh.from_array(np.zeros(out_channels_b, np.float32), "bb")]
    graph = h.make_graph(nodes, "elementwise", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def lut_model(weights, bias, kind="Sigmoid", channels=3):
    weights = np.asarray(weights, np.float32).reshape(channels, channels)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node(kind, ["conv"], ["output"])], "lut",
        [h.make_tensor_value_info("input", 1, [1, channels, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, channels, 8, 8])],
        [nh.from_array(weights.reshape(channels, channels, 1, 1), "w"),
         nh.from_array(np.asarray(bias, np.float32).reshape(channels), "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def transpose_model(channels=3, kernel=3, strides=(1, 1), pads=None, output_padding=(0, 0),
                    stem_kernel=1, out_shape=None):
    """`Conv -> depthwise ConvTranspose`, the 8x8/C3 transposed family."""
    rng = np.random.default_rng(8)
    pads = (kernel // 2,) * 4 if pads is None else pads
    stem_w = rng.uniform(-.3, .3, (channels, 3, stem_kernel, stem_kernel)).astype(np.float32)
    stem_b = rng.uniform(-1, 1, channels).astype(np.float32)
    weights = rng.uniform(-.25, .25, (channels, 1, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-.5, .5, channels).astype(np.float32)
    out_h = 7 * strides[0] + kernel - pads[0] - pads[2] + output_padding[0]
    out_w = 7 * strides[1] + kernel - pads[1] - pads[3] + output_padding[1]
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["stem"],
                         kernel_shape=[stem_kernel, stem_kernel], pads=[stem_kernel // 2] * 4),
             h.make_node("ConvTranspose", ["stem", "w1", "b1"], ["output"], group=channels,
                         kernel_shape=[kernel, kernel], strides=list(strides), pads=list(pads),
                         output_padding=list(output_padding))]
    graph = h.make_graph(nodes, "transpose", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, out_shape or [1, channels, out_h, out_w])],
                         [nh.from_array(stem_w, "w0"), nh.from_array(stem_b, "b0"),
                          nh.from_array(weights, "w1"), nh.from_array(bias, "b1")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, channels, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def dense_transpose_model(kernel=3, out_channels=3, strides=(1, 1)):
    """`Conv -> dense ConvTranspose`, the dense transposed family."""
    rng = np.random.default_rng(9)
    in_channels = 3
    stem_w = rng.uniform(-.3, .3, (in_channels, 3, 1, 1)).astype(np.float32)
    stem_b = rng.uniform(-1, 1, in_channels).astype(np.float32)
    weights = rng.uniform(-.25, .25, (in_channels, out_channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-.5, .5, out_channels).astype(np.float32)
    out_h = 7 * strides[0] + kernel - 2 * (kernel // 2)
    out_w = 7 * strides[1] + kernel - 2 * (kernel // 2)
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("ConvTranspose", ["stem", "w1", "b1"], ["output"],
                         kernel_shape=[kernel, kernel], strides=list(strides), pads=[kernel // 2] * 4)]
    graph = h.make_graph(nodes, "dense_transpose",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, out_channels, out_h, out_w])],
                         [nh.from_array(stem_w, "w0"), nh.from_array(stem_b, "b0"),
                          nh.from_array(weights, "w1"), nh.from_array(bias, "b1")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, in_channels, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def qlinear_model(per_channel=False, bias_on=True, weight_scale_shape=None, weight_zp_shape=None):
    rng = np.random.default_rng(1)
    in_channels, out_channels, kernel = 3, 5, 3
    weights = rng.integers(-128, 128, (out_channels, in_channels, kernel, kernel)).astype(np.int8)
    if per_channel:
        scale_shape = tuple(weight_scale_shape or (out_channels,))
        zp_shape = tuple(weight_zp_shape or (out_channels,))
        scales = rng.uniform(.001, .02, int(np.prod(scale_shape))).astype(np.float32).reshape(scale_shape)
        zero_points = (rng.integers(-8, 9, int(np.prod(zp_shape))).astype(np.int16).astype(np.int8)
                       .reshape(zp_shape))
    else:
        scales = np.array(.01, np.float32)
        zero_points = np.array(-3, np.int8)
    names = ["input", "xs", "xz", "qw", "ws", "wz", "ys", "yz"]
    initializers = [nh.from_array(np.array(.25, np.float32), "xs"),
                    nh.from_array(np.array(0, np.uint8), "xz"),
                    nh.from_array(weights, "qw"),
                    nh.from_array(np.asarray(scales, np.float32), "ws"),
                    nh.from_array(np.asarray(zero_points, np.int8), "wz"),
                    nh.from_array(np.array(2.0, np.float32), "ys"),
                    nh.from_array(np.array(-17, np.int8), "yz")]
    if bias_on:
        names.append("qb")
        initializers.append(nh.from_array(rng.integers(-2000, 2001, out_channels).astype(np.int32), "qb"))
    node = h.make_node("QLinearConv", names, ["output"], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4)
    model = h.make_model(h.make_graph(
        [node], "qlinearconv",
        [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, out_channels, 8, 8])],
        initializers), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def qdq_model(per_channel=False, axis=0, weight_dtype=np.int8):
    rng = np.random.default_rng(2)
    in_channels, out_channels, kernel = 3, 5, 3
    weights = rng.integers(-128, 128, (out_channels, in_channels, kernel, kernel)).astype(weight_dtype)
    scales = (rng.uniform(.001, .02, out_channels).astype(np.float32) if per_channel
              else np.array(.01, np.float32))
    zero_points = (rng.integers(-8, 9, out_channels).astype(np.int16).astype(np.int8) if per_channel
                   else np.array(-2, np.int8))
    initializers = [nh.from_array(np.array(.25, np.float32), "xs"),
                    nh.from_array(np.array(0, np.uint8), "xz"),
                    nh.from_array(weights, "qw"),
                    nh.from_array(scales, "ws"), nh.from_array(zero_points, "wz"),
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


def sequence_model(hidden=4):
    """The 3-node `Conv, Relu, Conv` legacy chain, used for scheduler arguments."""
    rng = np.random.default_rng(3)
    w1 = rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32)
    b1 = rng.uniform(-2, 2, hidden).astype(np.float32)
    w2 = rng.uniform(-.7, .8, (3, hidden, 1, 1)).astype(np.float32)
    b2 = rng.uniform(-2, 2, 3).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c1"], ["r1"]),
             h.make_node("Conv", ["r1", "w2", "b2"], ["output"], kernel_shape=[1, 1])]
    graph = h.make_graph(nodes, "sequence", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                         [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"),
                          nh.from_array(w2, "w2"), nh.from_array(b2, "b2")])
    graph.value_info.append(h.make_tensor_value_info("r1", 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def constant_mul_model():
    graph = h.make_graph(
        [h.make_node("Mul", ["input", "c"], ["output"])], "constant_mul",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(np.array([1.5, .5, 2.], np.float32).reshape(3, 1, 1), "c")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def leaky_model(alpha=None):
    rng = np.random.default_rng(31)
    attrs = {} if alpha is None else dict(alpha=alpha)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[1, 1]),
             h.make_node("LeakyRelu", ["c"], ["output"], **attrs)]
    graph = h.make_graph(nodes, "leaky", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 4, 8, 8])],
                         [nh.from_array(rng.uniform(-.5, .5, (4, 3, 1, 1)).astype(np.float32), "w"),
                          nh.from_array(rng.uniform(-1, 1, 4).astype(np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def prelu_model(slope):
    rng = np.random.default_rng(32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[1, 1]),
             h.make_node("PRelu", ["c", "slope"], ["output"])]
    graph = h.make_graph(nodes, "prelu", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 4, 8, 8])],
                         [nh.from_array(rng.uniform(-.5, .5, (4, 3, 1, 1)).astype(np.float32), "w"),
                          nh.from_array(rng.uniform(-1, 1, 4).astype(np.float32), "b"),
                          nh.from_array(np.asarray(slope, np.float32), "slope")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def reshape_model(shape, immutable=True):
    rng = np.random.default_rng(33)
    inputs = [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])]
    initializers = [nh.from_array(rng.uniform(-.5, .5, (4, 3, 1, 1)).astype(np.float32), "w"),
                    nh.from_array(rng.uniform(-1, 1, 4).astype(np.float32), "b")]
    if immutable:
        initializers.append(nh.from_array(np.array(shape, np.int64), "shape"))
    else:
        inputs.append(h.make_tensor_value_info("shape", onnx.TensorProto.INT64, [4]))
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[1, 1]),
             h.make_node("Reshape", ["c", "shape"], ["output"])]
    graph = h.make_graph(nodes, "reshape", inputs,
                         [h.make_tensor_value_info("output", 1, list(shape))], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class _PathMixin:
    """Save a graph to a temporary path (the scheduler takes a path) and compile it."""

    def compile_saved(self, model, *args, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path, *args, **kwargs)

    def reject_saved(self, message, model, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            self.compile_saved(model, *args, **kwargs)


class NativeConvBoundsTests(unittest.TestCase):
    """Dense/image Conv bounds in `native.compile_native_input`."""

    def test_input_channel_boundary(self):
        # 16352 is the largest 16-lane count inside the 511-part weight-table budget; the
        # board hangs at 16368 (512 parts) and the driver soft-resets the core, so the
        # emitter refuses it. See docs/investigation-log.md ("channel-plane wall").
        for channels, size in ((1, 8), (128, 8), (129, 8), (1024, 8), (16352, 2)):
            with self.subTest(channels=channels):
                _, meta = compile_native_input(native_model(ic=channels, k=1, pads=[0, 0, 0, 0],
                                                            ih=size, iw=size))
                self.assertEqual(meta["shape_nhwc"][3], channels)
        for channels in (0, 16353, 16368):
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_SHAPE)):
                    compile_native_input(native_model(ic=channels, k=1, pads=[0, 0, 0, 0]))

    def test_output_channel_boundary(self):
        # 8192 output channels (512 16-channel surface blocks) is the largest exact
        # geometry; C16384 writes the first 8192 channels and then wrong bytes.
        for channels in (1, 128, 129, 1024, 8192):
            with self.subTest(channels=channels):
                _, meta = compile_native_input(native_model(oc=channels, k=1, pads=[0, 0, 0, 0]))
                self.assertEqual(meta["output_shape_nhwc"][3], channels)
        for channels in (0, 8193, 16384):
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
                    compile_native_input(native_model(oc=channels, k=1, pads=[0, 0, 0, 0]))

    def test_spatial_boundary(self):
        for size in (1, 128):
            with self.subTest(size=size):
                _, meta = compile_native_input(native_model(ih=size, iw=size, k=1, pads=[0, 0, 0, 0]))
                self.assertEqual(meta["shape_nhwc"][1:3], [size, size])
        for size in (0, 129):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_SHAPE)):
                    compile_native_input(native_model(ih=size, iw=size))

    def test_kernel_boundary(self):
        for kernel in (3, 31):
            with self.subTest(kernel=kernel):
                _, meta = compile_native_input(native_model(k=kernel))
                self.assertEqual(meta["output_shape_nhwc"][1:3], [8, 8])
        for kernel in (2, 33):
            with self.subTest(kernel=kernel):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
                    compile_native_input(native_model(k=kernel))

    def test_stride_boundary(self):
        for stride in (1, 4):
            with self.subTest(stride=stride):
                _, meta = compile_native_input(native_model(strides=[stride, stride], pads=[0, 0, 0, 0]))
                self.assertEqual(meta["output_shape_nhwc"][1:3], [(8 - 3) // stride + 1] * 2)
        for stride in (0, 5):
            with self.subTest(stride=stride):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_GEOMETRY)):
                    compile_native_input(native_model(strides=[stride, stride]))

    def test_dilation_boundary(self):
        # Dilation 17 is the largest supported; it is only geometrically usable on a
        # surface larger than its 33-tap receptive field.
        _, meta = compile_native_input(native_model(dilations=[1, 1]))
        self.assertEqual(meta["output_shape_nhwc"][1:3], [8, 8])
        _, wide = compile_native_input(native_model(ih=128, iw=128, pads=[0, 0, 0, 0],
                                                    dilations=[17, 17]))
        self.assertEqual(wide["output_shape_nhwc"][1:3], [94, 94])
        for dilation in (0, 18):
            with self.subTest(dilation=dilation):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_GEOMETRY)):
                    compile_native_input(native_model(ih=128, iw=128, pads=[0, 0, 0, 0],
                                                      dilations=[dilation, dilation]))

    def test_padding_boundary(self):
        for pads in ([0, 0, 0, 0], [1, 1, 1, 1]):
            with self.subTest(pads=pads):
                compile_native_input(native_model(pads=pads))
        # Negative and above-255 amounts fail the range check ...
        for pads in ([-1, 0, 0, 0], [256, 0, 0, 0]):
            with self.subTest(pads=pads):
                with self.assertRaisesRegex(ValueError, re.escape(NATIVE_GEOMETRY)):
                    compile_native_input(native_model(pads=pads))
        # ... while an in-range amount that swallows the whole image fails geometry.
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_OUTPUT)):
            compile_native_input(native_model(pads=[255, 0, 0, 0]))

    def test_kernel_shape_and_group_are_rejected(self):
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
            compile_native_input(native_non_square())
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
            compile_native_input(native_model(ic=4, oc=4, group=2))

    def test_weight_rank_and_constant_requirement(self):
        for model in (native_rank_three_weights(), native_undeclared_weights()):
            with self.subTest(model=model.graph.name):
                expected = ("native Conv weights must have rank four"
                            if model.graph.name == "rank_three" else NATIVE_SHAPE)
                with self.assertRaisesRegex(ValueError, re.escape(expected)):
                    compile_native_input(model)
        with self.assertRaisesRegex(ValueError, re.escape("native input profile requires one Conv")):
            compile_native_input(native_weights_as_input())

    def test_missing_bias_is_zero(self):
        # A bias-less native Conv is the zero-bias Conv: same container, same bytes.
        without, meta = compile_native_input(native_model(bias=False))
        with_bias, _ = compile_native_input(native_model(bias=True))
        self.assertEqual(len(native_model(bias=False).graph.node[0].input), 2)
        self.assertEqual(without, with_bias)
        self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 4])
        self.assertTrue(without.startswith(b"ORNPUSEQ"))

    def test_dtype_rank_and_dynamic_shape_boundaries(self):
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
            compile_native_input(native_model(weight_dtype=np.float64))
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
            compile_native_input(native_model(input_dtype=2))
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_SHAPE)):
            compile_native_input(native_model(rank=3))
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_SHAPE)):
            compile_native_input(native_model(dynamic=True))

    def test_declared_output_shape_must_match(self):
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_ATTRS)):
            compile_native_input(native_model(output_shape=[1, 4, 7, 8]))
        _, meta = compile_native_input(native_model(output_shape=[1, 4, 8, 8]))
        self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 4])


class ChainFamilyBoundsTests(_PathMixin, unittest.TestCase):
    """`chain_n` and the height-strip `tiled_chain` family."""

    def test_hidden_channel_boundary(self):
        for hidden in (3, 16):
            with self.subTest(hidden=hidden):
                _, meta = self.compile_saved(chain_model(hidden=hidden))
                self.assertEqual(meta["profile"], "native-chain")
        for hidden in (2, 17):
            with self.subTest(hidden=hidden):
                self.reject_saved("native chain hidden channels must be 3..16 with matching bias",
                                  chain_model(hidden=hidden))

    def test_kernel_boundary(self):
        for kernels in ((1, 1, 1), (3, 3, 3)):
            with self.subTest(kernels=kernels):
                self.compile_saved(chain_model(kernels=kernels))
        self.reject_saved("native chain supports 1x1 or padded 3x3, stride1, group1",
                          chain_model(kernels=(1, 5, 1)))

    def test_even_length_pattern_is_rejected(self):
        # The scheduler has no chain profile for the even pattern; it hands the graph
        # to the depthwise profile, which names the shape mismatch.
        self.reject_saved("depthwise sequence requires Conv[/Relu] -> depthwise Conv",
                          even_conv_relu_model())
        with self.assertRaisesRegex(ValueError, re.escape("native chain requires [Conv, Relu]*(N-1) + [Conv]")):
            compile_chain_n(even_conv_relu_model())

    def test_tiles_boundary(self):
        for tiles in (0, 1, 2.5):
            with self.subTest(tiles=tiles):
                self.reject_saved("tiles must be an integer of at least 2",
                                  chain_model(), tiles=tiles)
        _, meta = self.compile_saved(chain_model(), tiles=2)
        self.assertEqual(meta["profile"], "tiled-native-chain")
        self.assertEqual(meta["tiles"], 2)
        # 3 passes the scheduler's ">= 2" check and then fails tiled_chain's own
        # "divides the 8-row height" bound.
        self.reject_saved("tiles must divide the 8-row chain height (1, 2, 4 or 8)",
                          chain_model(), tiles=3)
        with self.assertRaisesRegex(ValueError, re.escape("tiles must divide the 8-row chain height")):
            compile_tiled_chain(chain_model(), tiles=3)


class WalkBoundsTests(unittest.TestCase):
    """The op-level Conv/pool walk (`walk.parse_chain` / `compile_chain_walk`)."""

    VALID = [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)]

    def test_valid_chain_compiles(self):
        spec = parse_chain(walk_model(self.VALID).graph)
        self.assertEqual([op["kind"] for op in spec["ops"]], ["conv", "MaxPool", "conv"])
        binary, meta = compile_chain_walk(walk_model(self.VALID))
        self.assertEqual(meta["profile"], "chain-walk")
        self.assertTrue(binary.startswith(b"ORNPUSEQ"))

    def test_odd_spatial_before_pool(self):
        model = walk_model(self.VALID, height=7)
        self.assertIsNone(parse_chain(model.graph))
        with self.assertRaisesRegex(ValueError, re.escape("unsupported chain walk graph")):
            compile_chain_walk(model)

    def test_input_channels_must_be_three(self):
        model = walk_model(self.VALID, channels=4)
        self.assertIsNone(parse_chain(model.graph))
        with self.assertRaisesRegex(ValueError, re.escape("unsupported chain walk graph")):
            compile_chain_walk(model)
        self.assertIsNotNone(parse_chain(walk_model(self.VALID, channels=3).graph))

    def test_pool_kernel_and_stride_boundary(self):
        message = "walk pool supports 2x2 stride-2 MaxPool/AveragePool only"
        for pool in (("MaxPool", dict(kernel_shape=[3, 3], strides=[2, 2])),
                     ("MaxPool", dict(kernel_shape=[2, 2], strides=[1, 1])),
                     ("AveragePool", dict(kernel_shape=[3, 3], strides=[2, 2]))):
            model = walk_model([("conv", 8, 3, "Relu"), pool, ("conv", 3, 3, None)])
            with self.subTest(pool=pool):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    parse_chain(model.graph)
        self.assertIsNotNone(parse_chain(walk_model(self.VALID).graph))

    def test_unsupported_op_in_chain(self):
        model = walk_model([("conv", 8, 3, "Relu"), ("Sigmoid",)])
        self.assertIsNone(parse_chain(model.graph))
        with self.assertRaisesRegex(ValueError, re.escape("unsupported chain walk graph")):
            compile_chain_walk(model)

    def test_large_image_requires_height_tiling(self):
        # 80*80 atoms on one native16 surface exceeds the observed 6144-atom task limit.
        model = walk_model(self.VALID, height=80, width=80)
        with self.assertRaisesRegex(ValueError, re.escape("walk native input geometry requires height tiling")):
            compile_chain_walk(model)
        _, meta = compile_chain_walk(walk_model(self.VALID, height=32, width=32))
        self.assertEqual(meta["profile"], "chain-walk")


class DepthwisePoolingBoundsTests(_PathMixin, unittest.TestCase):
    """Depthwise, pooling and staged-reduction families."""

    def test_depthwise_channel_and_kernel_boundary(self):
        _, meta = compile_depthwise(depthwise_model())
        self.assertIn("depthwise_profile", meta)
        for channels in (0, 17):
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, re.escape("depthwise supports 1..16 channels")):
                    compile_depthwise(depthwise_model(channels=channels))
        for kernel in (2, 7):
            with self.subTest(kernel=kernel):
                with self.assertRaisesRegex(ValueError, re.escape("depthwise kernels 1/3/5 only")):
                    compile_depthwise(depthwise_model(kernel=kernel))

    def test_depthwise_stride_and_spatial_boundary(self):
        with self.assertRaisesRegex(ValueError, re.escape("depthwise supports stride 1 or 2")):
            compile_depthwise(depthwise_model(strides=(3, 3)))
        with self.assertRaisesRegex(ValueError, re.escape("depthwise input must be RGB, H/W 5..8")):
            compile_depthwise(depthwise_model(size=9))
        compile_depthwise(depthwise_model(strides=(2, 2)))

    def test_depthwise_padding_weights_and_stem_boundary(self):
        padded = "depthwise profile requires supported group/kernel/padding/stride and constant weights/bias"
        with self.assertRaisesRegex(ValueError, re.escape(padded)):
            compile_depthwise(depthwise_model(pads=(0, 0, 0, 0)))
        with self.assertRaisesRegex(ValueError, re.escape("depthwise profile requires matching float32 weights")):
            compile_depthwise(depthwise_model(weight_shape=(3, 2, 3, 3)))
        with self.assertRaisesRegex(ValueError, re.escape("C5..16 requires a 1x1 stem")):
            compile_depthwise(depthwise_model(channels=5, stem=3))
        compile_depthwise(depthwise_model(channels=5, stem=1))

    def test_pooling_boundary(self):
        for kind, profile in (("MaxPool", 3), ("AveragePool", 4)):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.onnx"
                    onnx.save(pool_model(kind), path)
                    _, meta = compile_pool(path)
                self.assertEqual(meta["profile"], profile)
        for attrs, out_shape in ((dict(kernel_shape=[3, 3], strides=[2, 2]), (1, 3, 4, 4)),
                                 (dict(kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1), (1, 3, 4, 4)),
                                 (dict(kernel_shape=[2, 2], strides=[2, 2]), (1, 3, 5, 4))):
            with self.subTest(attrs=attrs, out_shape=out_shape):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.onnx"
                    onnx.save(pool_model(pool_attrs=attrs, out_shape=out_shape), path)
                    with self.assertRaisesRegex(ValueError, re.escape("unsupported pooling graph")):
                        compile_pool(path)

    def test_reduction_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(reduction_model(), path)
            _, meta = compile_reduction(path)
        self.assertEqual(meta["profile"], 5)
        for levels, out_shape in ((2, (1, 3, 2, 2)), (3, (1, 3, 2, 2))):
            with self.subTest(levels=levels, out_shape=out_shape):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.onnx"
                    onnx.save(reduction_model(levels=levels, out_shape=out_shape), path)
                    with self.assertRaisesRegex(ValueError, re.escape("unsupported staged pooling graph")):
                        compile_reduction(path)


class ElementwiseJoinBoundsTests(_PathMixin, unittest.TestCase):
    """The two-branch elementwise joins and the elementwise DAG."""

    def test_valid_join_compiles(self):
        _, meta = self.compile_saved(two_branch_model())
        self.assertEqual(meta["elementwise_profile"], "add-8x8-c3-equal-scale")

    def test_missing_join_operand(self):
        model = two_branch_model()
        model.graph.node[-1].input[1] = "constant"
        model.graph.initializer.append(nh.from_array(np.zeros(3, np.float32), "constant"))
        self.reject_saved("elementwise operands must come from Conv[/Relu] branches", model)

    def test_mismatched_branch_shapes(self):
        # A one-channel second branch broadcasts against the three-channel first;
        # the profile refuses silent broadcasting.
        self.reject_saved("elementwise branch shapes must agree; broadcasting unsupported",
                          two_branch_model(out_channels_b=1))
        self.compile_saved(two_branch_model(out_channels_b=3))

    def test_add_requires_equal_branch_scales(self):
        # The Add/Sub/Max profile quantizes both branches onto one shared scale, so a
        # large weight-magnitude imbalance cannot produce unequal operand scales.
        _, equalized = self.compile_saved(two_branch_model(kind="Add", scale_a=0.02, scale_b=0.8))
        branch_a, branch_b = equalized["branches"]
        self.assertEqual(branch_a["output_scale"], branch_b["output_scale"])
        self.assertAlmostEqual(equalized["output_scale"], 2 * branch_a["output_scale"])
        self.assertTrue(equalized["elementwise_profile"].endswith("equal-scale"))
        # Mul is the one join that declares independent operand scales.
        _, independent = self.compile_saved(two_branch_model(kind="Mul", scale_a=0.02, scale_b=0.8))
        self.assertTrue(independent["elementwise_profile"].endswith("independent-scales"))

    def test_dag_external_shapes_must_be_8x8(self):
        rng = np.random.default_rng(17)
        nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
                 h.make_node("Add", ["A", "B"], ["C"]),
                 h.make_node("Mul", ["C", "A"], ["out"])]
        initializers = [nh.from_array(rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32), "wa"),
                        nh.from_array(np.zeros(3, np.float32), "ba"),
                        nh.from_array(rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32), "wb"),
                        nh.from_array(np.zeros(3, np.float32), "bb")]
        graph = h.make_graph(nodes, "dag",
                             [h.make_tensor_value_info("a", 1, [1, 3, 8, 9]),
                              h.make_tensor_value_info("b", 1, [1, 3, 8, 9])],
                             [h.make_tensor_value_info("out", 1, [1, 3, 8, 9])], initializers)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        self.reject_saved("elementwise DAG external tensors must be float32 [1,3,8,8]", model)


class LutBoundsTests(_PathMixin, unittest.TestCase):
    """The bounded Sigmoid/Tanh LUT profile."""

    def test_valid_stem_compiles(self):
        for kind in ("Sigmoid", "Tanh"):
            with self.subTest(kind=kind):
                _, meta = compile_lut(lut_model(np.eye(3) / 32, np.zeros(3), kind), 1, 128)
                self.assertEqual(meta["lut_profile"], kind.lower() + "-8x8-c3")

    def test_stem_shape_and_quantization_boundary(self):
        message = "bounded LUT profile requires a C3 1x1 Conv stem, 8x8, input scale1 zero point128"
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            compile_lut(lut_model(np.eye(4) / 32, np.zeros(4), channels=4), 1, 128)
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            compile_lut(lut_model(np.eye(3) / 32, np.zeros(3)), 0.5, 128)
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            compile_lut(lut_model(np.eye(3) / 32, np.zeros(3)), 1, 0)
        # The scheduler forwards the boundary and reaches the same check.
        self.reject_saved(message, lut_model(np.eye(3) / 32, np.zeros(3)), 0.5, 128)
        self.reject_saved("output override unsupported for LUT profile",
                          lut_model(np.eye(3) / 32, np.zeros(3)), 1, 128,
                          output_range=dict(scale=1.0, zero_point=0))

    def test_stem_gain_and_structure_boundary(self):
        with self.assertRaisesRegex(
                ValueError, re.escape("LUT stem weight scale must make BASE_WEIGHT_SCALE/scale a power of two")):
            compile_lut(lut_model(np.eye(3) * 0.02, np.zeros(3)), 1, 128)
        with self.assertRaisesRegex(ValueError, re.escape("LUT stem must be a diagonal 1x1 Conv")):
            compile_lut(lut_model(np.array([[1, .1, 0], [0, 1, 0], [0, 0, 1]], np.float32) / 32, np.zeros(3)), 1, 128)
        with self.assertRaisesRegex(ValueError, re.escape("LUT stem weights and bias must be finite")):
            compile_lut(lut_model(np.full((3, 3), np.nan, np.float32), np.zeros(3)), 1, 128)

    def test_analytic_range_must_fit_the_table_domain(self):
        with self.assertRaisesRegex(ValueError, re.escape("leaves the table domain")):
            compile_lut(lut_model(np.eye(3) / 32, np.full(3, 9.0, np.float32)), 1, 128)
        _, meta = compile_lut(lut_model(np.eye(3) / 32, np.zeros(3)), 1, 128)
        self.assertLessEqual(meta["stem_range"][1], 8.0)


class ConvTransposeBoundsTests(unittest.TestCase):
    """Depthwise and dense ConvTranspose bounds."""

    def test_valid_geometry_compiles(self):
        for strides in ((1, 1), (2, 2)):
            with self.subTest(strides=strides):
                _, meta = compile_transposed(transpose_model(strides=strides))
                self.assertIn("transposed_profile", meta)
        _, dense = compile_transposed(dense_transpose_model(kernel=3))
        self.assertIn("transposed_profile", dense)

    def test_depthwise_shape_boundary(self):
        message = "ConvTranspose K2/K3/K5 stride1/2 requires padding below K and output_padding below its axis stride"
        for model in (transpose_model(kernel=4, pads=(2, 2, 2, 2)),
                      transpose_model(channels=17),
                      transpose_model(strides=(3, 3), pads=(0, 0, 0, 0)),
                      transpose_model(pads=(3, 3, 3, 3))):
            with self.subTest(channels=model.graph.output[0].type.tensor_type.shape.dim[1].dim_value):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    compile_transposed(model)

    def test_output_geometry_boundary(self):
        with self.assertRaisesRegex(
                ValueError,
                re.escape("ConvTranspose profile requires depthwise C1..16, K2/K3/K5, "
                          "per-axis stride1/2 and matching output geometry")):
            compile_transposed(transpose_model(out_shape=(1, 3, 7, 7)))

    def test_dilation_and_dense_kernel_boundary(self):
        with self.assertRaisesRegex(
                ValueError,
                re.escape("ConvTranspose dilation is supported only for square K2 or depthwise K3 with dilation2")):
            model = transpose_model()
            for node in model.graph.node:
                if node.op_type == "ConvTranspose":
                    node.attribute.append(h.make_attribute("dilations", [4, 4]))
            compile_transposed(model)
        with self.assertRaisesRegex(ValueError, re.escape("dense ConvTranspose supports square K3 or K5")):
            compile_transposed(dense_transpose_model(kernel=4))


class QuantizedImportBoundsTests(_PathMixin, unittest.TestCase):
    """The QLinearConv and Q/DQ import paths."""

    def test_qlinearconv_is_accepted_and_carries_its_own_quantization(self):
        _, meta = self.compile_saved(qlinear_model())
        self.assertEqual(meta["quantized_import"], "QLinearConv weights preserved bit-for-bit")
        _, per_channel = self.compile_saved(qlinear_model(per_channel=True))
        self.assertEqual(len(per_channel["source_weight_scales"]), 5)
        self.reject_saved("QLinearConv carries its own quantization parameters", qlinear_model(), 0.5, 0)
        self.reject_saved("QLinearConv carries its own quantization parameters", qlinear_model(), 1.0, 0,
                          output_range=dict(scale=1.0, zero_point=0))

    def test_qlinearconv_constant_boundary(self):
        scalar = "QLinearConv requires scalar activation quantization constants"
        with self.assertRaisesRegex(ValueError, re.escape(scalar)):
            compile_qlinearconv(_qlinear_vector_scale())
        message = "weight quantization must be scalar or per output channel"
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            compile_qlinearconv(qlinear_model(per_channel=True, weight_scale_shape=(3,),
                                              weight_zp_shape=(5,)))
        with self.assertRaisesRegex(ValueError, re.escape(message)):
            compile_qlinearconv(qlinear_model(per_channel=True, weight_scale_shape=(5,),
                                              weight_zp_shape=(3,)))

    def test_qdq_import_is_accepted_and_carries_its_own_quantization(self):
        _, meta = self.compile_saved(qdq_model())
        self.assertIn("Q/DQ Conv INT8 weights preserved bit-for-bit", meta["quantized_import"])
        self.compile_saved(qdq_model(per_channel=True, axis=0))
        self.reject_saved("Q/DQ Conv carries its own quantization parameters", qdq_model(), 0.5, 0)
        self.reject_saved("Q/DQ Conv carries its own quantization parameters", qdq_model(), 1.0, 0,
                          output_range=dict(scale=1.0, zero_point=0))

    def test_qdq_weight_boundary(self):
        with self.assertRaisesRegex(ValueError, re.escape("per-channel weight Q/DQ requires axis0")):
            compile_qdq_conv(qdq_model(per_channel=True, axis=1))
        with self.assertRaisesRegex(ValueError, re.escape("Q/DQ weights must be constant INT8/float32/INT8")):
            compile_qdq_conv(qdq_model(weight_dtype=np.uint8))


def _qlinear_vector_scale():
    """QLinearConv whose input scale is a length-two vector instead of a scalar."""
    rng = np.random.default_rng(1)
    in_channels, out_channels, kernel = 3, 5, 3
    weights = rng.integers(-128, 128, (out_channels, in_channels, kernel, kernel)).astype(np.int8)
    names = ["input", "xs", "xz", "qw", "ws", "wz", "ys", "yz"]
    initializers = [nh.from_array(np.array([.25, .5], np.float32), "xs"),
                    nh.from_array(np.array(0, np.uint8), "xz"),
                    nh.from_array(weights, "qw"),
                    nh.from_array(np.array(.01, np.float32), "ws"),
                    nh.from_array(np.array(-3, np.int8), "wz"),
                    nh.from_array(np.array(2.0, np.float32), "ys"),
                    nh.from_array(np.array(-17, np.int8), "yz")]
    node = h.make_node("QLinearConv", names, ["output"], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4)
    model = h.make_model(h.make_graph(
        [node], "qlinearconv_vector",
        [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, out_channels, 8, 8])],
        initializers), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class SchedulerArgumentBoundsTests(_PathMixin, unittest.TestCase):
    """Scheduler-level argument validation, independent of the graph family."""

    def test_calibration_and_output_range_are_exclusive(self):
        # The chain lowers its first layer through `compile_model`, so calibration
        # needs the first Relu tensor as well as the graph output.
        ranges = {"r1": dict(scale=.25, zero_point=0), "output": dict(scale=.5, zero_point=0)}
        self.reject_saved("calibration and output quantization overrides cannot be combined",
                          sequence_model(), 1.0, 0,
                          output_range=dict(scale=1.0, zero_point=0), calibration_ranges=ranges)
        _, meta = self.compile_saved(sequence_model(), 1.0, 0, calibration_ranges=ranges)
        self.assertAlmostEqual(meta["output_scale"], .5)

    def test_mul_operand_zero_points_require_a_mul(self):
        self.reject_saved("Mul operand zero points require a Mul profile",
                          sequence_model(), 1.0, 0, mul_operand_zero_points=(1, 2))
        _, meta = self.compile_saved(two_branch_model(kind="Mul"), 1.0, 0,
                                     mul_operand_zero_points=(1, 2))
        self.assertEqual(meta["operand_zero_points"], [1, 2])

    def test_mutable_weights_precondition(self):
        self.reject_saved("mutable weights currently require one native Conv[/Relu]",
                          sequence_model(), 1.0, 0, mutable_weights=True)
        _, meta = self.compile_saved(native_model(ic=4), 1.0, 0, mutable_weights=True)
        self.assertEqual(meta["mutable_constants"], ["conv.parameters"])

    def test_mutable_constants_precondition(self):
        self.reject_saved("mutable constants currently require one constant Mul",
                          sequence_model(), 1.0, 0, mutable_constants=True)
        _, meta = self.compile_saved(constant_mul_model(), 1.0, 0, mutable_constants=True)
        self.assertIn("constant_data_mode", meta)

    def test_submission_and_tiles(self):
        self.reject_saved("submission must be None, serial or batched",
                          sequence_model(), 1.0, 0, submission="turbo")
        self.compile_saved(sequence_model(), 1.0, 0, submission="serial")
        for tiles in (0, 1, -3, 2.5):
            with self.subTest(tiles=tiles):
                self.reject_saved("tiles must be an integer of at least 2",
                                  chain_model(), 1.0, 0, tiles=tiles)


class ActivationAndLayoutBoundsTests(_PathMixin, unittest.TestCase):
    """The LeakyRelu, PRelu and spatial-Reshape terminal profiles."""

    def test_leaky_relu_profile_bounds(self):
        for alpha in (None, 0.5):
            with self.subTest(alpha=alpha):
                _, meta = self.compile_saved(leaky_model(alpha))
                self.assertIn("leaky_alpha_encoded", meta)
        for alpha in (-0.1, 2.0):
            with self.subTest(alpha=alpha):
                self.reject_saved("unsupported LeakyRelu parameters/connections", leaky_model(alpha))

    def test_prelu_profile_bounds(self):
        for slope in (np.array(.1, np.float32), np.full((4, 1, 1), .1, np.float32)):
            with self.subTest(shape=np.shape(slope)):
                _, meta = self.compile_saved(prelu_model(slope))
                self.assertIn("prelu_slopes", meta)
        self.reject_saved("PRelu requires constant float32 slopes in [0,1]",
                          prelu_model(np.full((4, 1, 1), 2.0, np.float32)))
        self.reject_saved("PRelu slope must be scalar or [C,1,1]",
                          prelu_model(np.full((4,), .1, np.float32)))

    def test_spatial_reshape_bounds(self):
        _, meta = self.compile_saved(reshape_model([1, 4, 16, 4]))
        self.assertTrue(meta["spatial_reshape"])
        self.reject_saved("Reshape must preserve batch, channels and spatial pixel count",
                          reshape_model([1, 4, 8, 7]))
        self.reject_saved("terminal Reshape requires an immutable shape and one output",
                          reshape_model([1, 4, 16, 4], immutable=False))


if __name__ == "__main__":
    unittest.main()
