"""SPDX-License-Identifier: MIT

Generated-graph property sweep across the scheduler and the emitter families.

Where the retained suites pin *chosen* models against a baseline and the
per-family tests pin the accepted bounds with the first out-of-bounds neighbour,
this module pins the same contracts over *generated* graphs: seeded random
shapes and weights inside the documented bounds, one family at a time.  For every
generated graph it asserts

* the profile dispatch is stable for the same graph (`profile_marker`, because
  the legacy families identify themselves in a family-specific meta key such as
  `depthwise_profile` rather than in `meta['profile']`);
* the container decodes, and its declared geometry and quantization band are the
  ones the graph and the metadata claim (a decoded container is the only thing
  the loader sees, so this is the loader contract);
* that family's own integer reference runs on a deterministic sample and returns
  the declared output shape, dtype and band;
* compiling the same graph twice is byte-identical (determinism), which is what
  the retained 2,244-model baseline relies on.

Every accepted graph is paired with one *documented* out-of-bounds perturbation
(one bound per family, named in the subtest) and the scheduler must reject it
with `ValueError`.  Seeds are fixed, no network and no board is used.

The module also closes reachable defensive branches in `walk.py`,
`elementwise.py` and `join_dag.py` that the generated sweep does not enter; the
unreachable ones are reported in the accompanying coverage report, not exercised
here.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu.depthwise import depthwise_reference
from open_rknpu.elementwise import (add_reference, compile_constant_mul, compile_elementwise,
                                    compile_mul_add, compile_mul_clip, compile_mul_relu,
                                    compile_per_channel_constant_mul, compile_standalone_mul,
                                    max_reference, mul_output_conversion, mul_reference,
                                    mul_requant_reference, sub_reference)
from open_rknpu.join_dag import compile_join_dag, parse_join_dag
from open_rknpu.lut import lut_reference
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.walk import (chain_walk_reference, compile_chain_walk, compile_join_walk,
                             join_walk_reference, load_quantizations, parse_chain, parse_join_walk,
                             _as_conv, _peel_operand, _value_shape)

PROFILE_KEYS = ("profile", "elementwise_profile", "depthwise_profile",
                "transposed_profile", "lut_profile")


def profile_marker(meta):
    """The family identity a compiled `meta` carries, independent of key choice."""
    marker = tuple((key, meta[key]) for key in PROFILE_KEYS if key in meta)
    if not marker:
        raise AssertionError("compiled metadata has no profile identity: %r" % sorted(meta))
    return marker


def nchw(shape):
    height, width, channels = shape
    return [1, channels, height, width]


def build_model(nodes, input_shape, output_shape, initializers, value_info=(), inputs=()):
    """NHWC shapes are declared NCHW; `inputs` adds extra external tensors."""
    declared = [h.make_tensor_value_info("input", 1, nchw(input_shape))]
    declared += [h.make_tensor_value_info(name, 1, nchw(shape)) for name, shape in inputs]
    graph = h.make_graph(nodes, "fuzz", declared,
                         [h.make_tensor_value_info("output", 1, nchw(output_shape))],
                         list(initializers),
                         value_info=[h.make_tensor_value_info(name, 1, nchw(shape))
                                     for name, shape in value_info])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def conv_node(name, source, out, weight, bias, kernel, pads, strides=(1, 1), group=1, dilations=(1, 1)):
    return h.make_node("Conv", [source, name + "_w", name + "_b"], [out], kernel_shape=[kernel, kernel],
                       pads=list(pads), strides=list(strides), group=group, dilations=list(dilations))


def build_batch_model(batch, height, width, channels, factor):
    """A batched constant-Mul graph (NCHW declared, factor broadcastable)."""
    graph = h.make_graph([h.make_node("Mul", ["input", "k"], ["output"])], "batched-mul",
                         [h.make_tensor_value_info("input", 1, [batch, channels, height, width])],
                         [h.make_tensor_value_info("output", 1, [batch, channels, height, width])],
                         [nh.from_array(np.asarray(factor, np.float32), "k")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def q_of(entry):
    """Live `Quantization` from emitter metadata (dict or Quantization)."""
    if isinstance(entry, Quantization):
        return entry
    if "quantization" in entry:
        entry = entry["quantization"]
    return Quantization(**{key: (np.array(value) if isinstance(value, list) else value)
                           for key, value in entry.items()})


def sample_for(shape, seed):
    """Deterministic UINT8 sample that exercises the full code range."""
    values = (np.arange(np.prod(shape), dtype=np.int64) * 37 + seed * 11) % 256
    return values.astype(np.uint8).reshape(shape)


def walk_join_graph(head_pools=(None, None), tail=(), stem_k=1, head_k=1, stem_relu=False,
                    hidden=3, input_shape=(8, 8, 3), join_kind="Add"):
    """A stem -> two heads -> join [-> tail] graph for the join-walk parser.

    Returns `(graph, final_tensor)`; the shapes are declared for the unpooled
    case, which is the one the compiler accepts.  Parser-only tests mutate the
    result and never run shape inference.
    """
    rng = np.random.default_rng(61)
    height, width, channels = input_shape
    nodes = []
    initializers = []
    stem = rng.uniform(-.3, .3, (hidden, 3, stem_k, stem_k)).astype(np.float32)
    stem_bias = rng.uniform(-.3, .3, hidden).astype(np.float32)
    nodes.append(h.make_node("Conv", ["input", "sw", "sb"], ["stem"], kernel_shape=[stem_k, stem_k],
                             pads=[stem_k // 2] * 4))
    initializers += [nh.from_array(stem, "sw"), nh.from_array(stem_bias, "sb")]
    source = "stem"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem"], ["stem_r"]))
        source = "stem_r"
    heads = []
    for index in range(2):
        weight = rng.uniform(-.3, .3, (3, hidden, head_k, head_k)).astype(np.float32)
        bias = rng.uniform(-.3, .3, 3).astype(np.float32)
        nodes.append(h.make_node("Conv", [source, "hw%d" % index, "hb%d" % index], ["h%d" % index],
                                 kernel_shape=[head_k, head_k], pads=[head_k // 2] * 4))
        initializers += [nh.from_array(weight, "hw%d" % index), nh.from_array(bias, "hb%d" % index)]
        last = "h%d" % index
        if head_pools[index]:
            nodes.append(h.make_node(head_pools[index], [last], ["hp%d" % index],
                                     kernel_shape=[2, 2], strides=[2, 2]))
            last = "hp%d" % index
        heads.append(last)
    nodes.append(h.make_node(join_kind, heads, ["join"]))
    previous = "join"
    for index, op in enumerate(tail):
        name = "t%d" % index
        if op == "Conv":
            weight = rng.uniform(-.3, .3, (3, 3, 1, 1)).astype(np.float32)
            bias = rng.uniform(-.3, .3, 3).astype(np.float32)
            nodes.append(h.make_node("Conv", [previous, "tw%d" % index, "tb%d" % index], [name],
                                     kernel_shape=[1, 1]))
            initializers += [nh.from_array(weight, "tw%d" % index), nh.from_array(bias, "tb%d" % index)]
        else:
            nodes.append(h.make_node(op, [previous], [name]))
        previous = name
    graph = h.make_graph(nodes, "join-walk",
                         [h.make_tensor_value_info("input", 1, [1, channels, height, width])],
                         [h.make_tensor_value_info(previous, 1, [1, 3, height, width])],
                         initializers)
    return graph, previous


# ---------------------------------------------------------------------------
# 1. Dense image Conv (native16 native-input profile)
# ---------------------------------------------------------------------------


def native_case(seed, height, width, channels, out_channels, kernel, pads, strides, dilations):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-.3, .3, (out_channels, channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-.5, .5, out_channels).astype(np.float32)
    ekh = (kernel - 1) * dilations[0] + 1
    ekw = (kernel - 1) * dilations[1] + 1
    oh = (height + pads[0] + pads[2] - ekh) // strides[0] + 1
    ow = (width + pads[1] + pads[3] - ekw) // strides[1] + 1
    model = build_model([conv_node("c", "input", "output", weight, bias, kernel, pads, strides, dilations=dilations)],
                        (height, width, channels), (oh, ow, out_channels),
                        [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")])

    def run(sample, meta):
        q = q_of(meta["quantization"])
        return native_input_reference(sample, q, meta["input_zero_point"], meta["conv_pads"],
                                      meta["conv_strides"], meta["conv_dilations"])

    return dict(family="image-conv", model=model, input_hwc=(height, width, channels),
                output_hwc=(oh, ow, out_channels), seed=seed, reference=run)


def native_perturbation(seed):
    """Stride 5 is outside the documented stride 1..4 bound."""
    return native_case(seed, 10, 10, 3, 5, 3, (1, 1, 1, 1), (5, 5), (1, 1))["model"]


# ---------------------------------------------------------------------------
# 2. Conv chain with an interior pool (walk, packed small image)
# ---------------------------------------------------------------------------


def chain_case(seed, height, width, hidden, kernel, pool, large=False):
    rng = np.random.default_rng(seed)
    first = rng.uniform(-.3, .3, (hidden, 3, kernel, kernel)).astype(np.float32)
    first_bias = rng.uniform(-.3, .3, hidden).astype(np.float32)
    last = rng.uniform(-.3, .3, (3, hidden, kernel, kernel)).astype(np.float32)
    last_bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    nodes = [conv_node("c0", "input", "c0", first, first_bias, kernel, (kernel // 2,) * 4),
             h.make_node("Relu", ["c0"], ["r0"]),
             h.make_node(pool, ["r0"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
             conv_node("c1", "p0", "output", last, last_bias, kernel, (kernel // 2,) * 4)]
    model = build_model(nodes, (height, width, 3), (height // 2, width // 2, 3),
                        [nh.from_array(first, "c0_w"), nh.from_array(first_bias, "c0_b"),
                         nh.from_array(last, "c1_w"), nh.from_array(last_bias, "c1_b")],
                        value_info=[("c0", (height, width, hidden)), ("r0", (height, width, hidden)),
                                    ("p0", (height // 2, width // 2, hidden))])

    def run(sample, meta, model=model):
        spec = parse_chain(model.graph)
        return chain_walk_reference(sample, load_quantizations(meta), spec["ops"],
                                    meta.get("input_zero_point", 0))

    return dict(family="chain-walk-large" if large else "chain-walk", model=model,
                input_hwc=(height, width, 3), output_hwc=(height // 2, width // 2, 3),
                seed=seed, reference=run)


def chain_perturbation(seed):
    """A 3x3 pool is outside the documented 2x2 stride-2 walk-pool bound."""
    rng = np.random.default_rng(seed)
    first = rng.uniform(-.3, .3, (6, 3, 3, 3)).astype(np.float32)
    first_bias = rng.uniform(-.3, .3, 6).astype(np.float32)
    last = rng.uniform(-.3, .3, (3, 6, 3, 3)).astype(np.float32)
    last_bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    nodes = [conv_node("c0", "input", "c0", first, first_bias, 3, (1, 1, 1, 1)),
             h.make_node("Relu", ["c0"], ["r0"]),
             h.make_node("MaxPool", ["r0"], ["p0"], kernel_shape=[3, 3], strides=[2, 2]),
             conv_node("c1", "p0", "output", last, last_bias, 3, (1, 1, 1, 1))]
    return build_model(nodes, (8, 8, 3), (4, 4, 3),
                       [nh.from_array(first, "c0_w"), nh.from_array(first_bias, "c0_b"),
                        nh.from_array(last, "c1_w"), nh.from_array(last_bias, "c1_b")],
                       value_info=[("c0", (8, 8, 6)), ("r0", (8, 8, 6)), ("p0", (4, 4, 6))])


# ---------------------------------------------------------------------------
# 3. Depthwise
# ---------------------------------------------------------------------------


def depthwise_case(seed, height, width, channels, kernel, stride, stem_kernel):
    rng = np.random.default_rng(seed)
    stem = rng.uniform(-.3, .3, (channels, 3, stem_kernel, stem_kernel)).astype(np.float32)
    stem_bias = rng.uniform(-.3, .3, channels).astype(np.float32)
    weight = rng.uniform(-.4, .4, (channels, 1, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-.3, .3, channels).astype(np.float32)
    pad = kernel // 2
    oh = (height + stride - 1) // stride
    ow = (width + stride - 1) // stride
    nodes = [conv_node("s", "input", "stem", stem, stem_bias, stem_kernel, (stem_kernel // 2,) * 4),
             h.make_node("Conv", ["stem", "d_w", "d_b"], ["output"], kernel_shape=[kernel, kernel],
                         pads=[pad] * 4, strides=[stride, stride], group=channels)]
    model = build_model(nodes, (height, width, 3), (oh, ow, channels),
                        [nh.from_array(stem, "s_w"), nh.from_array(stem_bias, "s_b"),
                         nh.from_array(weight, "d_w"), nh.from_array(bias, "d_b")],
                        value_info=[("stem", (height, width, channels))])

    def run(sample, meta):
        stem_q = q_of(meta["first"]["quantization"])
        grid = reference(sample, stem_q)
        return depthwise_reference(grid, q_of(meta["depthwise"]), stem_q.output_zero_point)

    return dict(family="depthwise", model=model, input_hwc=(height, width, 3),
                output_hwc=(oh, ow, channels), seed=seed, reference=run)


def depthwise_perturbation(seed):
    """A per-axis stride (2, 3) is outside the documented stride 1/2 bound."""
    rng = np.random.default_rng(seed + 1)
    stem = rng.uniform(-.3, .3, (3, 3, 1, 1)).astype(np.float32)
    stem_bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    weight = rng.uniform(-.4, .4, (3, 1, 3, 3)).astype(np.float32)
    bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    nodes = [conv_node("s", "input", "stem", stem, stem_bias, 1, (0, 0, 0, 0)),
             h.make_node("Conv", ["stem", "d_w", "d_b"], ["output"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1], strides=[2, 3], group=3)]
    return build_model(nodes, (8, 8, 3), (4, 3, 3),
                       [nh.from_array(stem, "s_w"), nh.from_array(stem_bias, "s_b"),
                        nh.from_array(weight, "d_w"), nh.from_array(bias, "d_b")],
                       value_info=[("stem", (8, 8, 3))])


# ---------------------------------------------------------------------------
# 4. Elementwise joins
# ---------------------------------------------------------------------------


def elementwise_case(seed, kind, height, width, channels):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-.3, .3, (channels, 3, 1, 1)).astype(np.float32)
    a_bias = rng.uniform(-.2, .2, channels).astype(np.float32)
    b = rng.uniform(-.3, .3, (channels, 3, 1, 1)).astype(np.float32)
    b_bias = rng.uniform(-.2, .2, channels).astype(np.float32)
    nodes = [conv_node("a", "input", "ca", a, a_bias, 1, (0, 0, 0, 0)),
             conv_node("b", "input", "cb", b, b_bias, 1, (0, 0, 0, 0)),
             h.make_node(kind, ["ca", "cb"], ["output"])]
    model = build_model(nodes, (height, width, 3), (height, width, channels),
                        [nh.from_array(a, "a_w"), nh.from_array(a_bias, "a_b"),
                         nh.from_array(b, "b_w"), nh.from_array(b_bias, "b_b")],
                        value_info=[("ca", (height, width, channels)), ("cb", (height, width, channels))])
    combine = {"Add": add_reference, "Sub": sub_reference, "Max": max_reference, "Mul": mul_reference}[kind]

    def run(sample, meta):
        left = reference(sample, q_of(meta["branches"][0]["quantization"]))
        right = reference(sample, q_of(meta["branches"][0]["quantization"])
                          if len(meta["branches"]) == 1 else q_of(meta["branches"][1]["quantization"]))
        return combine(left, right)

    return dict(family="elementwise-" + kind.lower(), model=model, input_hwc=(height, width, 3),
                output_hwc=(height, width, channels), seed=seed, reference=run)


def elementwise_perturbation(seed):
    """17 output channels is outside the documented elementwise 2..16 bound."""
    rng = np.random.default_rng(seed + 2)
    weight = rng.uniform(-.3, .3, (17, 3, 1, 1)).astype(np.float32)
    bias = rng.uniform(-.2, .2, 17).astype(np.float32)
    nodes = [conv_node("a", "input", "ca", weight, bias, 1, (0, 0, 0, 0)),
             conv_node("b", "input", "cb", weight.copy(), bias.copy(), 1, (0, 0, 0, 0)),
             h.make_node("Add", ["ca", "cb"], ["output"])]
    return build_model(nodes, (8, 8, 3), (8, 8, 17),
                       [nh.from_array(weight, "a_w"), nh.from_array(bias, "a_b"),
                        nh.from_array(weight.copy(), "b_w"), nh.from_array(bias.copy(), "b_b")],
                       value_info=[("ca", (8, 8, 17)), ("cb", (8, 8, 17))])


# ---------------------------------------------------------------------------
# 5. Constant Mul
# ---------------------------------------------------------------------------


def constant_mul_case(seed, height, width, factor):
    factor = np.asarray(factor, np.float32)
    model = build_model([h.make_node("Mul", ["input", "factor"], ["output"])],
                        (height, width, 3), (height, width, 3),
                        [nh.from_array(factor, "factor")])

    def run(sample, meta):
        stem = reference(sample, q_of(meta["branches"][0]["quantization"]))
        expanded = np.broadcast_to(factor, (1, 3, height, width))
        codes = np.clip(np.rint(expanded / meta["constant_scale"]), -128, 127).astype(np.int8)[0]
        codes = codes.transpose(1, 2, 0)
        if meta["constant_data_mode"] == "per-channel":
            codes = codes[0, 0]
        return mul_reference(stem, codes)

    return dict(family="constant-mul", model=model, input_hwc=(height, width, 3),
                output_hwc=(height, width, 3), seed=seed, reference=run)


def constant_mul_perturbation(seed):
    """H/W 4 is outside the documented single-batch constant-Mul 5..8 bound."""
    return constant_mul_case(seed, 4, 4, np.float32(0.7))["model"]


# ---------------------------------------------------------------------------
# 6. ConvTranspose (symmetric depthwise K3)
# ---------------------------------------------------------------------------


def transposed_case(seed, channels, kernel, pads, strides, out_hw):
    rng = np.random.default_rng(seed)
    stem = np.zeros((channels, 3, 1, 1), np.float32)
    for channel in range(min(3, channels)):
        stem[channel, channel, 0, 0] = 1.0
    weight = rng.uniform(-.3, .3, (channels, 1, kernel, kernel)).astype(np.float32)
    # Symmetric taps make `depthwise_reference` the documented exact oracle.
    weight = 0.5 * (weight + weight[:, :, ::-1, ::-1])
    bias = np.zeros(channels, np.float32)
    nodes = [conv_node("s", "input", "stem_out", stem, np.zeros(channels, np.float32), 1, (0, 0, 0, 0)),
             h.make_node("ConvTranspose", ["stem_out", "w", "b"], ["output"], group=channels,
                         kernel_shape=[kernel, kernel], pads=list(pads), strides=list(strides))]
    model = build_model(nodes, (8, 8, 3), (out_hw[0], out_hw[1], channels),
                        [nh.from_array(stem, "s_w"), nh.from_array(np.zeros(channels, np.float32), "s_b"),
                         nh.from_array(weight, "w"), nh.from_array(bias, "b")],
                        value_info=[("stem_out", (8, 8, channels))])

    def run(sample, meta, kernel=kernel):
        stem_q = q_of(meta["first"]["quantization"])
        grid = reference(sample, stem_q)
        return depthwise_reference(grid, q_of(meta["transposed_quantization"]), stem_q.output_zero_point)

    return dict(family="transpose", model=model, input_hwc=(8, 8, 3),
                output_hwc=(out_hw[0], out_hw[1], channels), seed=seed, reference=run)


def transposed_perturbation(seed):
    """K4 is outside the documented depthwise ConvTranspose K2/K3/K5 bound."""
    rng = np.random.default_rng(seed + 3)
    stem = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
    weight = rng.uniform(-.3, .3, (3, 1, 4, 4)).astype(np.float32)
    nodes = [conv_node("s", "input", "stem_out", stem, np.zeros(3, np.float32), 1, (0, 0, 0, 0)),
             h.make_node("ConvTranspose", ["stem_out", "w", "b"], ["output"], group=3,
                         kernel_shape=[4, 4], pads=[2, 2, 2, 2], strides=[1, 1])]
    return build_model(nodes, (8, 8, 3), (9, 9, 3),
                       [nh.from_array(stem, "s_w"), nh.from_array(np.zeros(3, np.float32), "s_b"),
                        nh.from_array(weight, "w"), nh.from_array(np.zeros(3, np.float32), "b")],
                       value_info=[("stem_out", (8, 8, 3))])


# ---------------------------------------------------------------------------
# 7. LUT (Sigmoid/Tanh), only where the gain family and domain hold
# ---------------------------------------------------------------------------


def lut_case(seed, kind, gain):
    weight = (np.eye(3, dtype=np.float32) * gain).reshape(3, 3, 1, 1)
    bias = np.zeros(3, np.float32)
    model = build_model([h.make_node("Conv", ["input", "c_w", "c_b"], ["stem"], kernel_shape=[1, 1]),
                         h.make_node(kind, ["stem"], ["output"])],
                        (8, 8, 3), (8, 8, 3),
                        [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")],
                        value_info=[("stem", (8, 8, 3))])

    def run(sample, meta):
        return lut_reference(sample, np.eye(3, dtype=np.float64) * gain, np.zeros(3), kind)

    return dict(family="lut-" + kind.lower(), model=model, input_hwc=(8, 8, 3),
                output_hwc=(8, 8, 3), seed=seed, reference=run, input_zero_point=128)


def lut_perturbation(seed):
    """A gain 64x the verified family leaves the bounded table domain."""
    weight = (np.eye(3, dtype=np.float32) * (64.0 / 32.0)).reshape(3, 3, 1, 1)
    model = build_model([h.make_node("Conv", ["input", "c_w", "c_b"], ["stem"], kernel_shape=[1, 1]),
                         h.make_node("Sigmoid", ["stem"], ["output"])],
                        (8, 8, 3), (8, 8, 3),
                        [nh.from_array(weight, "c_w"), nh.from_array(np.zeros(3, np.float32), "c_b")],
                        value_info=[("stem", (8, 8, 3))])
    return model


def generated_cases():
    """The deterministic, seeded sweep: one generated graph per family variant."""
    cases = []
    for seed, height, width, channels, out_channels, kernel, pads, strides, dilations in (
            (101, 10, 10, 3, 5, 3, (1, 1, 1, 1), (1, 1), (1, 1)),
            (102, 12, 9, 4, 6, 3, (2, 1, 0, 2), (2, 2), (1, 1)),
            (103, 11, 11, 8, 4, 3, (2, 2, 2, 2), (1, 1), (2, 2)),
            (104, 16, 13, 3, 7, 1, (0, 0, 0, 0), (3, 2), (1, 1))):
        cases.append(("image-conv", native_case(seed, height, width, channels, out_channels,
                                                kernel, pads, strides, dilations),
                      native_perturbation(seed), "native padding/stride/dilation unsupported"))
    cases.append(("chain-walk", chain_case(201, 8, 8, 6, 3, "MaxPool"),
                  chain_perturbation(201), "walk pool supports 2x2 stride-2"))
    cases.append(("chain-walk", chain_case(202, 10, 10, 8, 3, "AveragePool"),
                  chain_perturbation(202), "walk pool supports 2x2 stride-2"))
    cases.append(("depthwise", depthwise_case(301, 8, 8, 3, 3, 1, 3),
                  depthwise_perturbation(301), "depthwise supports stride 1 or 2"))
    cases.append(("depthwise", depthwise_case(302, 6, 8, 4, 3, 1, 1),
                  depthwise_perturbation(302), "depthwise supports stride 1 or 2"))
    for seed, kind in ((401, "Add"), (402, "Sub"), (403, "Max"), (404, "Mul")):
        cases.append(("elementwise-" + kind.lower(), elementwise_case(seed, kind, 8, 8, 4),
                      elementwise_perturbation(seed), "2..16 output channels"))
    cases.append(("constant-mul", constant_mul_case(501, 8, 8, np.float32(0.7)),
                  constant_mul_perturbation(501), "single-batch constant Mul requires RGB H/W5..8"))
    cases.append(("constant-mul", constant_mul_case(502, 6, 7,
                                                    np.array([0.5, 1.0, 2.0], np.float32).reshape(1, 3, 1, 1)),
                  constant_mul_perturbation(502), "single-batch constant Mul requires RGB H/W5..8"))
    cases.append(("transpose", transposed_case(601, 3, 3, (1, 1, 1, 1), (1, 1), (8, 8)),
                  transposed_perturbation(601), "ConvTranspose K2/K3/K5"))
    cases.append(("transpose", transposed_case(602, 4, 3, (1, 1, 1, 1), (1, 1), (8, 8)),
                  transposed_perturbation(602), "ConvTranspose K2/K3/K5"))
    cases.append(("lut-sigmoid", lut_case(701, "Sigmoid", 1 / 32),
                  lut_perturbation(701), "leaves the table domain"))
    cases.append(("lut-tanh", lut_case(702, "Tanh", 1 / 16),
                  lut_perturbation(702), "leaves the table domain"))
    cases.append(("chain-walk-large", chain_case(801, 12, 12, 6, 3, "AveragePool", large=True),
                  chain_perturbation(801), "walk pool supports 2x2 stride-2"))
    return cases


class EmitterFuzzTests(unittest.TestCase):
    """Generated graphs: dispatch stability, geometry, reference, determinism, rejection."""

    def compile(self, model, **kwargs):
        return compile_sequence(copy.deepcopy(model), **kwargs)

    def check_case(self, case, perturbed, message):
        model = case["model"]
        compile_kwargs = {}
        if "input_zero_point" in case:
            compile_kwargs["input_zero_point"] = case["input_zero_point"]
        binary, meta = self.compile(model, **compile_kwargs)
        first_marker = profile_marker(meta)

        # (a) the same graph dispatches to the same profile (stable identity).
        _, meta_again = self.compile(model, **compile_kwargs)
        self.assertEqual(profile_marker(meta_again), first_marker, "profile dispatch changed between builds")

        # (b) the container decodes and its declared geometry is the graph's.
        info = decode_sequence(binary)
        height, width, channels = case["input_hwc"]
        out_height, out_width, out_channels = case["output_hwc"]
        self.assertEqual(info["shape_nhwc"], [1, height, width, channels])
        self.assertEqual(info["output_shape_nhwc"], [1, out_height, out_width, out_channels])
        self.assertEqual(info["input_bytes"], height * width * channels)
        self.assertEqual(info["output_bytes"], out_height * out_width * out_channels)

        # (c) the family's own integer reference runs and matches the declared band.
        self.assertGreater(meta["output_scale"], 0.0)
        self.assertTrue(-128 <= meta["output_zero_point"] <= 127)
        self.assertAlmostEqual(info["output_scale"], float(np.float32(meta["output_scale"])), places=6)
        self.assertEqual(info["output_zero_point"], meta["output_zero_point"])
        sample = sample_for(case["input_hwc"], case["seed"])
        got = case["reference"](sample, meta)
        self.assertEqual(got.dtype, np.int8)
        self.assertEqual(got.shape, (out_height, out_width, out_channels))
        self.assertTrue(int(got.min()) >= -128 and int(got.max()) <= 127)

        # (d) determinism: the same graph compiles to the same bytes.
        second, _ = self.compile(model, **compile_kwargs)
        self.assertEqual(second, binary, "recompiling the same graph is not byte-identical")

        # Paired rejection: one documented out-of-bounds perturbation.
        with self.assertRaises(ValueError) as caught:
            self.compile(perturbed, **compile_kwargs)
        self.assertIn(message, str(caught.exception))

    def test_generated_graph_properties(self):
        for family, case, perturbed, message in generated_cases():
            with self.subTest(family=family, seed=case["seed"]):
                self.check_case(case, perturbed, message)


class WalkParserRejectionTests(unittest.TestCase):
    """Reachable defensive guards in `walk.parse_chain`/`parse_join_walk`."""

    @staticmethod
    def graph(nodes, height=8, width=8, channels=3, out_height=8, out_width=8, out_channels=3, inits=()):
        graph = h.make_graph(nodes, "walk",
                             [h.make_tensor_value_info("input", 1, [1, channels, height, width])],
                             [h.make_tensor_value_info("output", 1, [1, out_channels, out_height, out_width])],
                             list(inits))
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        return onnx.shape_inference.infer_shapes(model)

    @staticmethod
    def conv(name, source, out, weight, bias, pads, group=1):
        kernel = weight.shape[2]
        return h.make_node("Conv", [source, name + "_w", name + "_b"], [out], kernel_shape=[kernel, kernel],
                           pads=list(pads), group=group)

    def test_parse_chain_rejects_wrong_input_and_empty_graph(self):
        weight = np.zeros((3, 3, 1, 1), np.float32)
        bias = np.zeros(3, np.float32)
        inits = [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")]
        # A grayscale input is not one of the two supported packed layouts.
        model = self.graph([self.conv("c", "input", "output", weight, bias, (0, 0, 0, 0))],
                           channels=1, inits=inits)
        self.assertIsNone(parse_chain(model.graph))
        # A H/W below the documented 2x2 minimum.
        model = self.graph([self.conv("c", "input", "output", weight, bias, (0, 0, 0, 0))],
                           height=1, width=1, inits=inits)
        self.assertIsNone(parse_chain(model.graph))

    def test_parse_chain_rejects_bad_conv_shapes_and_pools(self):
        bias = np.zeros(3, np.float32)
        # Weight dtype/shape outside float32 (C,3,K,K).
        int_weight = np.zeros((3, 3, 1, 1), np.int8)
        model = self.graph([h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[1, 1])],
                           inits=[nh.from_array(int_weight, "w"), nh.from_array(bias, "b")])
        self.assertIsNone(parse_chain(model.graph))
        # An unsupported op after a valid Conv.
        weight = np.zeros((3, 3, 1, 1), np.float32)
        model = self.graph([self.conv("c", "input", "c0", weight, bias, (0, 0, 0, 0)),
                            h.make_node("Sigmoid", ["c0"], ["output"])],
                           inits=[nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")])
        self.assertIsNone(parse_chain(model.graph))
        # A Relu with an attribute is not the accepted bare Relu.
        relu = h.make_node("Relu", ["c0"], ["output"])
        relu.attribute.extend([h.make_attribute("consumed_inputs", [0])])
        model = self.graph([self.conv("c", "input", "c0", weight, bias, (0, 0, 0, 0)), relu],
                           inits=[nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")])
        self.assertIsNone(parse_chain(model.graph))
        # A Clip after a Conv is accepted only where the native emitter uses it.
        clip = h.make_node("Clip", ["c0", "lo", "hi"], ["output"])
        model = self.graph([self.conv("c", "input", "c0", weight, bias, (0, 0, 0, 0)), clip],
                           inits=[nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b"),
                                  nh.from_array(np.float32(0), "lo"), nh.from_array(np.float32(6), "hi")])
        self.assertIsNone(parse_chain(model.graph))

    def test_walk_conv_and_pool_attribute_guards(self):
        from open_rknpu.walk import _conv_attributes, _pool_attributes
        shape = [3, 3, 3, 3]
        node = h.make_node("Conv", ["x"], ["y"])
        node.domain = "custom"
        with self.assertRaises(ValueError):
            _conv_attributes(node, shape)
        node = self.conv("c", "x", "y", np.zeros((3, 3, 3, 3), np.float32), np.zeros(3, np.float32), (1, 1, 1, 1))
        with self.assertRaises(ValueError):
            _conv_attributes(node, [3, 3, 4, 3])  # rectangular kernel
        with self.assertRaises(ValueError):
            _conv_attributes(node, [3, 3, 5, 5])  # K5 outside the walk's K1/K3 bound
        pool = h.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[3, 3])
        with self.assertRaises(ValueError):
            _pool_attributes(pool)
        pool = h.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2])
        pool.domain = "custom"
        with self.assertRaises(ValueError):
            _pool_attributes(pool)
        pool = h.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
        with self.assertRaises(ValueError):
            _pool_attributes(pool)

    def test_parse_chain_rejects_non_conv_start_and_bad_activation(self):
        weight = np.zeros((3, 3, 1, 1), np.float32)
        bias = np.zeros(3, np.float32)
        # A pool before any Conv cannot start the chain.
        model = self.graph([h.make_node("MaxPool", ["input"], ["output"], kernel_shape=[2, 2], strides=[2, 2])],
                           out_height=4, out_width=4,
                           inits=[nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")])
        self.assertIsNone(parse_chain(model.graph))
        # A Relu whose input does not consume the Conv output is rejected.
        model = self.graph([self.conv("c", "input", "c0", weight, bias, (0, 0, 0, 0)),
                            h.make_node("Relu", ["input"], ["output"])],
                           inits=[nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")])
        self.assertIsNone(parse_chain(model.graph))

    def test_compile_chain_walk_rejects_unsupported_graph(self):
        model = self.graph([h.make_node("Identity", ["input"], ["output"])])
        with self.assertRaises(ValueError):
            compile_chain_walk(model)
        # The native16 walk refuses more than 16 channels for a large image.
        rng = np.random.default_rng(9)
        weight = rng.uniform(-.3, .3, (20, 3, 1, 1)).astype(np.float32)
        bias = rng.uniform(-.3, .3, 20).astype(np.float32)
        model = self.graph([self.conv("c", "input", "output", weight, bias, (0, 0, 0, 0))],
                           height=12, width=12, out_channels=20, inits=[nh.from_array(weight, "c_w"),
                                                                        nh.from_array(bias, "c_b")])
        with self.assertRaises(ValueError):
            compile_chain_walk(model)

    def test_join_walk_rejects_unsupported_graph(self):
        model = self.graph([h.make_node("Identity", ["input"], ["output"])])
        with self.assertRaises(ValueError):
            compile_join_walk(model)

    def test_parse_chain_rejects_multiple_io_and_no_nodes(self):
        weight = np.zeros((3, 3, 1, 1), np.float32)
        bias = np.zeros(3, np.float32)
        conv = self.conv("c", "input", "output", weight, bias, (0, 0, 0, 0))
        init = [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")]
        # Two graph inputs do not describe a single chain.
        graph = h.make_graph([conv], "two-inputs",
                             [h.make_tensor_value_info("input", 1, [1, 3, 8, 8]),
                              h.make_tensor_value_info("second", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], init)
        self.assertIsNone(parse_chain(graph))
        # A graph with no nodes has no chain to parse.
        empty = h.make_graph([], "empty", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], [])
        self.assertIsNone(parse_chain(empty))

    def test_parse_chain_rejects_non_constant_conv_and_two_input_relu(self):
        weight = np.zeros((3, 3, 1, 1), np.float32)
        bias = np.zeros(3, np.float32)
        # A Conv whose weights are not initializers is not a walk Conv.
        graph = h.make_graph([h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[1, 1])],
                             "float-weights", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], [])
        self.assertIsNone(parse_chain(graph))
        # A follower Relu with two inputs is not the bare activation.
        nodes = [self.conv("c", "input", "c0", weight, bias, (0, 0, 0, 0)),
                 h.make_node("Relu", ["c0", "c0"], ["output"])]
        graph = h.make_graph(nodes, "two-input-relu", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                             [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")])
        self.assertIsNone(parse_chain(graph))

    def test_as_conv_guards(self):
        # A node with two inputs, a custom domain, missing constants, bad weights.
        node = h.make_node("Conv", ["stem", "w"], ["h"])
        self.assertIsNone(_as_conv(node, {}))
        node = h.make_node("Conv", ["stem", "w", "b"], ["h"])
        node.domain = "custom"
        self.assertIsNone(_as_conv(node, {}))
        node = h.make_node("Conv", ["stem", "w", "b"], ["h"])
        self.assertIsNone(_as_conv(node, {}))
        constants = {"w": np.zeros((3, 3, 1, 1), np.int8), "b": np.zeros(3, np.float32)}
        self.assertIsNone(_as_conv(node, constants))

    def test_peel_operand_guards(self):
        # No producer for the operand name.
        self.assertIsNone(_peel_operand([], {}, "missing", 0, {}))
        # A Relu with an attribute.
        relu = h.make_node("Relu", ["a"], ["b"])
        relu.attribute.extend([h.make_attribute("consumed_inputs", [0])])
        self.assertIsNone(_peel_operand([relu], {"b": 0}, "b", 1, {}))
        # A Relu whose producer is missing.
        relu = h.make_node("Relu", ["missing"], ["b"])
        self.assertIsNone(_peel_operand([relu], {"b": 0}, "b", 1, {}))
        # A pool with unsupported attributes.
        pool = h.make_node("MaxPool", ["a"], ["b"], kernel_shape=[3, 3], strides=[2, 2])
        self.assertIsNone(_peel_operand([pool], {"b": 0}, "b", 1, {}))
        # A valid pool whose producer is missing.
        pool = h.make_node("MaxPool", ["missing"], ["b"], kernel_shape=[2, 2], strides=[2, 2])
        self.assertIsNone(_peel_operand([pool], {"b": 0}, "b", 1, {}))
        # A node that is neither an activation nor a pool.
        add = h.make_node("Add", ["a", "c"], ["b"])
        self.assertIsNone(_peel_operand([add], {"b": 0}, "b", 1, {}))


class JoinWalkGuardTests(unittest.TestCase):
    """Reachable guards in the op-level join-walk parser and reference."""

    def test_value_shape_requires_an_rgb_image(self):
        graph = h.make_graph([], "gray", [h.make_tensor_value_info("input", 1, [1, 1, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 1, 8, 8])], [])
        self.assertIsNone(_value_shape(graph))

    def test_parse_join_walk_rejects_empty_and_non_rgb_graphs(self):
        empty = h.make_graph([], "empty", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], [])
        self.assertIsNone(parse_join_walk(empty))
        gray, _ = walk_join_graph(input_shape=(8, 8, 1))
        self.assertIsNone(parse_join_walk(gray))

    def test_parse_join_walk_accepts_and_rejects_each_shape(self):
        valid, _ = walk_join_graph()
        self.assertIsNotNone(parse_join_walk(valid))

        join_attr, _ = walk_join_graph()
        for node in join_attr.node:
            if node.op_type in ("Add", "Mul", "Sub", "Max"):
                node.attribute.extend([h.make_attribute("broadcast", 1)])
        self.assertIsNone(parse_join_walk(join_attr))

        ghost, _ = walk_join_graph()
        for node in ghost.node:
            if node.op_type in ("Add", "Mul", "Sub", "Max"):
                node.input[0] = "ghost"
        self.assertIsNone(parse_join_walk(ghost))

        different_stem, _ = walk_join_graph()
        for node in different_stem.node:
            if node.output[0] == "h1":
                node.input[0] = "input"
        self.assertIsNone(parse_join_walk(different_stem))

        relu_attr, _ = walk_join_graph(stem_relu=True)
        for node in relu_attr.node:
            if node.op_type == "Relu":
                node.attribute.extend([h.make_attribute("consumed_inputs", [0])])
        self.assertIsNone(parse_join_walk(relu_attr))

        stem_k3, _ = walk_join_graph(stem_k=3)
        self.assertIsNone(parse_join_walk(stem_k3))

        untraced, _ = walk_join_graph()
        extra = h.make_node("Conv", ["input", "sw", "sb"], ["extra"], kernel_shape=[1, 1])
        untraced.node.insert(len(untraced.node) - 1, extra)
        self.assertIsNone(parse_join_walk(untraced))

        wide_head, _ = walk_join_graph()
        for tensor in wide_head.initializer:
            if tensor.name == "hw1":
                tensor.CopyFrom(nh.from_array(np.zeros((3, 5, 1, 1), np.float32), "hw1"))
        self.assertIsNone(parse_join_walk(wide_head))

        odd_pool, _ = walk_join_graph(head_pools=("MaxPool", "MaxPool"), input_shape=(7, 8, 3))
        self.assertIsNone(parse_join_walk(odd_pool))

        mixed_pool, _ = walk_join_graph(head_pools=("MaxPool", None))
        self.assertIsNone(parse_join_walk(mixed_pool))

        relu_first, _ = walk_join_graph(tail=("Relu",))
        self.assertIsNone(parse_join_walk(relu_first))

        unsupported_tail, _ = walk_join_graph(tail=("Sigmoid",))
        self.assertIsNone(parse_join_walk(unsupported_tail))

        trailing_relu, _ = walk_join_graph(tail=("Conv", "Relu"))
        self.assertIsNone(parse_join_walk(trailing_relu))

    def test_join_walk_reference_folds_the_tail(self):
        graph, _ = walk_join_graph(tail=("Conv", "Relu", "Conv"))
        model = onnx.shape_inference.infer_shapes(
            h.make_model(graph, opset_imports=[h.make_opsetid("", 13)]))
        binary, meta = compile_join_walk(model)
        self.assertEqual(meta["join"], "Add")
        self.assertEqual(len(meta["tail_quantization"]), 2)
        info = decode_sequence(binary)
        self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, 3])
        plan = parse_join_walk(model.graph)
        quantizations = {key: q_of(entry) for key, entry in meta["quantizations"].items()
                         if entry is not None}
        sample = sample_for((8, 8, 3), 77)
        got = join_walk_reference(sample, quantizations, plan)
        self.assertEqual(got.shape, (8, 8, 3))
        self.assertEqual(got.dtype, np.int8)


class JoinDagRejectionTests(unittest.TestCase):
    """`join_dag.parse_join_dag`/`_validate` accept a real DAG and reject each guard."""

    @staticmethod
    def dag(heads, hidden=3, expression=(("Add", "h0_0", "h1_0"), ("Mul", "j0", "h0_0")),
            stem_attrs=(), declare_value_info=True, output=None, pool=None, stem_relu=False):
        """Three (or more) head chains off a 1x1 stem plus a join expression.

        `heads` is a list of layer lists; each layer is
        `(out_channels, kernel, group, extra_attrs)`, where `group != 1` marks a
        depthwise layer whose real group is its channel count.  Layer outputs are
        named `h{head}_{layer}`; join outputs `j{position}`.
        """
        rng = np.random.default_rng(41)
        constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                     nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b_stem")]
        nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1],
                             **dict(stem_attrs))]
        shapes = {"stem": (hidden, 8, 8)}
        stem_source = "stem"
        if stem_relu:
            nodes.append(h.make_node("Relu", ["stem"], ["stem_r"]))
            shapes["stem_r"] = (hidden, 8, 8)
            stem_source = "stem_r"
        for head_index, layers in enumerate(heads):
            source = stem_source
            for layer_index, (out_channels, kernel, group, extra) in enumerate(layers):
                name = "h%d_%d" % (head_index, layer_index)
                depthwise = group != 1
                weight = rng.uniform(.02, .06, (out_channels, 1, kernel, kernel) if depthwise
                                     else (out_channels, shapes[source][0], kernel, kernel)).astype(np.float32)
                bias = rng.uniform(-1, 1, (out_channels,)).astype(np.float32)
                constants += [nh.from_array(weight, name + "_w"), nh.from_array(bias, name + "_b")]
                attrs = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4, **dict(extra))
                if depthwise:
                    attrs["group"] = out_channels
                nodes.append(h.make_node("Conv", [source, name + "_w", name + "_b"], [name], **attrs))
                shapes[name] = (out_channels, 8, 8)
                source = name
        for position, (kind, first, second) in enumerate(expression):
            nodes.append(h.make_node(kind, [first, second], ["j%d" % position]))
            shapes["j%d" % position] = shapes.get(second, (3, 8, 8))
        if pool is not None:
            nodes.append(h.make_node("MaxPool", [pool[1]], [pool[0]], kernel_shape=[2, 2], strides=[2, 2]))
        last = pool[0] if pool is not None else "j%d" % (len(expression) - 1)
        output_name = output if output is not None else last
        out_shape = shapes.get(output_name, (3, 8, 8))
        graph = h.make_graph(nodes, "dag", [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info(output_name, 1, [1, out_shape[0], out_shape[1], out_shape[2]])],
                             constants)
        if declare_value_info:
            for name, shape in shapes.items():
                graph.value_info.append(h.make_tensor_value_info(name, 1, [1, shape[0], shape[1], shape[2]]))
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        model = onnx.shape_inference.infer_shapes(model)
        if not declare_value_info:
            del model.graph.value_info[:]
        return model

    def test_valid_dag_parses_validates_and_compiles(self):
        model = self.dag([[ (3, 1, 1, {}), ], [ (3, 3, 1, {}), ], [ (3, 1, 1, {}), ]])
        spec = parse_join_dag(list(model.graph.node))
        self.assertIsNotNone(spec)
        self.assertEqual(len(spec["branches"]), 3)
        self.assertEqual(len(spec["joins"]), 2)
        binary, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], "join-dag")
        info = decode_sequence(binary)
        self.assertEqual(info["shape_nhwc"], [1, 8, 8, 3])
        self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, 3])

    def test_dag_reference_accepts_live_quantization(self):
        from open_rknpu.join_dag import join_dag_reference
        from open_rknpu.quantization import Quantization
        model = self.dag([[(3, 1, 1, {})], [(3, 3, 1, {})], [(3, 1, 1, {})]])
        _, meta = compile_sequence(model)
        sample = (np.arange(8 * 8 * 3) % 256).astype(np.uint8).reshape(8, 8, 3)
        stem_live = q_of(meta["stem_quantization"])
        heads_live = [q_of(entry) for entry in meta["head_quantization"]]
        dict_result = join_dag_reference(sample, meta["stem_quantization"], meta["head_quantization"],
                                         meta["head_names"], meta["join_expression"])
        live_result = join_dag_reference(sample, stem_live, heads_live, meta["head_names"],
                                         meta["join_expression"])
        self.assertIsInstance(stem_live, Quantization)
        self.assertEqual(dict_result.tobytes(), live_result.tobytes())

    def test_dag_with_a_depthwise_branch_compiles(self):
        model = self.dag([[(3, 1, 1, {})], [(3, 3, 3, {})], [(3, 1, 1, {})]])
        _, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], "join-dag")

    def test_parse_join_dag_rejects_duplicate_branch_names(self):
        # Two branch Convs sharing one output name are not a valid protobuf from any
        # exporter, but the parser must reject them rather than collapse the branches.
        nodes = [h.make_node("Conv", ["image", "ws", "bs"], ["stem"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["stem", "w1", "b1"], ["h"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["stem", "w2", "b2"], ["h"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["stem", "w3", "b3"], ["h2"], kernel_shape=[1, 1]),
                 h.make_node("Add", ["h", "h2"], ["j0"]),
                 h.make_node("Mul", ["j0", "h"], ["j1"])]
        self.assertIsNone(parse_join_dag(nodes))

    def test_parse_join_dag_rejects_fewer_than_three_branches(self):
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})]])
        self.assertIsNone(parse_join_dag(list(model.graph.node)))
        with self.assertRaises(ValueError):
            compile_join_dag(model)

    def test_parse_join_dag_rejects_join_attributes(self):
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]])
        for node in model.graph.node:
            if node.op_type == "Add":
                node.attribute.extend([h.make_attribute("broadcast", 1)])
        self.assertIsNone(parse_join_dag(list(model.graph.node)))

    def test_parse_join_dag_rejects_terminal_pool_after_a_join(self):
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]],
                         pool=("pooled", "j0"))
        self.assertIsNone(parse_join_dag(list(model.graph.node)))

    def test_validate_rejects_output_that_is_not_the_final_join(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]], output="j0")
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_wrong_external_tensor_shape(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]])
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 9
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_undeclared_branch_shape(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]], declare_value_info=False)
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_hidden_channels_outside_three_to_sixteen(self):
        from open_rknpu.join_dag import _validate
        # Hidden channels 2 with three-channel heads: the stem bound fires first.
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]], hidden=2)
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_unsupported_stem_attributes(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]],
                         stem_attrs=(("auto_pad", b"NOTSET"),))
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_unsupported_dense_layer_attributes(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {"auto_pad": b"NOTSET"})], [(3, 1, 1, {})], [(3, 1, 1, {})]])
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_dense_layer_channels_out_of_range(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(17, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]],
                         expression=(("Add", "h1_0", "h2_0"), ("Mul", "j0", "h1_0")))
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_chained_depthwise_branch(self):
        from open_rknpu.join_dag import _validate
        # A second layer whose group is neither 1 nor the branch channel count.
        model = self.dag([[(4, 1, 1, {}), (3, 3, 3, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]],
                         expression=(("Add", "h0_1", "h1_0"), ("Mul", "j0", "h2_0")))
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_a_branch_that_skips_a_layer(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {}), (3, 1, 1, {}), (3, 1, 1, {})],
                          [(3, 1, 1, {})], [(3, 1, 1, {})]],
                         expression=(("Add", "h0_2", "h1_0"), ("Mul", "j0", "h2_0")))
        for node in model.graph.node:
            if node.output[0] == "h0_2":
                node.input[0] = "h0_0"
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_a_branch_layer_without_constants(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]])
        kept = [tensor for tensor in model.graph.initializer if tensor.name != "h0_0_w"]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_a_stem_relu_that_skips_the_stem(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]], stem_relu=True)
        for node in model.graph.node:
            if node.op_type == "Relu":
                node.input[0] = "image"
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_unsupported_depthwise_layer_attributes(self):
        from open_rknpu.join_dag import _validate
        model = self.dag([[(3, 1, 3, {"auto_pad": b"NOTSET"})], [(3, 1, 1, {})], [(3, 1, 1, {})]])
        with self.assertRaises(ValueError):
            _validate(model)

    def test_validate_rejects_join_over_a_non_three_channel_intermediate(self):
        from open_rknpu.join_dag import _validate
        # Both intermediates have four channels and feed the first join.
        model = self.dag([[(4, 1, 1, {}), (3, 1, 1, {})], [(4, 1, 1, {}), (3, 1, 1, {})],
                          [(3, 1, 1, {})]],
                         expression=(("Add", "h0_0", "h1_0"), ("Mul", "j0", "h2_0")))
        with self.assertRaises(ValueError):
            _validate(model)

    def test_compile_join_dag_rejects_custom_domain(self):
        model = self.dag([[(3, 1, 1, {})], [(3, 1, 1, {})], [(3, 1, 1, {})]])
        model.opset_import.append(h.make_opsetid("custom", 1))
        for node in model.graph.node:
            if node.op_type == "Conv":
                node.domain = "custom"
        with self.assertRaises(ValueError):
            compile_join_dag(model)



class ElementwiseGuardTests(unittest.TestCase):
    """Reachable rejection guards in `elementwise.py` that the sweep does not enter."""

    def test_mul_output_conversion_rejects_invalid_range(self):
        with self.assertRaises(ValueError):
            mul_output_conversion(0.5, {"scale": -1.0, "zero_point": 0})
        with self.assertRaises(ValueError):
            mul_output_conversion(0.5, {"scale": 1.0, "zero_point": 200})
        with self.assertRaises(ValueError):
            # A tiny requested output scale pushes the 14-bit conversion out of range.
            mul_output_conversion(1.0, {"scale": 1e-9, "zero_point": 0})

    def test_mul_requant_reference_shift_zero_and_nonzero(self):
        a = np.array([[10, -20, 127]], np.int8)
        b = np.array([[3, 4, -5]], np.int8)
        zero_shift = mul_requant_reference(a, b, 0x10000 | 100, 0, 7)
        self.assertEqual(zero_shift.dtype, np.int8)
        shifted = mul_requant_reference(a, b, 0x10000 | 100, 6, 0)
        self.assertEqual(shifted.dtype, np.int8)

    def test_compile_mul_relu_requires_terminal_relu(self):
        weight = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])], (8, 8, 3), (8, 8, 3),
                            [nh.from_array(weight, "k")])
        with self.assertRaises(ValueError):
            compile_mul_relu(model)

    def test_compile_mul_clip_requires_clip_0_6(self):
        weight = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        nodes = [h.make_node("Mul", ["input", "b"], ["product"]),
                 h.make_node("Clip", ["product", "lo", "hi"], ["output"])]
        inits = [nh.from_array(weight, "b"), nh.from_array(np.float32(0), "lo"),
                 nh.from_array(np.float32(7), "hi")]
        model = build_model(nodes, (8, 8, 3), (8, 8, 3), inits)
        with self.assertRaises(ValueError):
            compile_mul_clip(model)
        # A non-default output range is refused with the fixed Clip[0,6] band.
        inits[2] = nh.from_array(np.float32(6), "hi")
        model = build_model(nodes, (8, 8, 3), (8, 8, 3), inits)
        with self.assertRaises(ValueError):
            compile_mul_clip(model, output_range={"scale": 0.5, "zero_point": 0})

    def test_compile_mul_add_requires_representable_scalar(self):
        weight = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        # A terminal op that is not the scalar Add is rejected outright.
        with self.assertRaises(ValueError):
            compile_mul_add(build_model([h.make_node("Mul", ["input", "b"], ["product"]),
                                         h.make_node("Relu", ["product"], ["output"])],
                                        (8, 8, 3), (8, 8, 3), [nh.from_array(weight, "b")]))
        nodes = [h.make_node("Mul", ["input", "b"], ["product"]),
                 h.make_node("Add", ["product", "k"], ["output"])]
        inits = [nh.from_array(weight, "b"), nh.from_array(np.float32(0.25), "k")]
        model = build_model(nodes, (8, 8, 3), (8, 8, 3), inits)
        with self.assertRaises(ValueError):
            compile_mul_add(model)  # no explicit output quantization
        with self.assertRaises(ValueError):
            compile_mul_add(model, output_range={"scale": 0.1, "zero_point": 0})  # not exact
        # A missing constant operand is rejected before the arithmetic.
        nodes = [h.make_node("Mul", ["input", "b"], ["product"]),
                 h.make_node("Add", ["product", "missing"], ["output"])]
        model = build_model(nodes, (8, 8, 3), (8, 8, 3), [nh.from_array(weight, "b")])
        with self.assertRaises(ValueError):
            compile_mul_add(model, output_range={"scale": 0.1, "zero_point": 0})

    def test_compile_elementwise_rewrite_guards(self):
        weight = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        # Residual Add whose two operands are both the Conv output, not the input.
        nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                 h.make_node("Add", ["conv", "conv"], ["output"])]
        model = build_model(nodes, (8, 8, 3), (8, 8, 3),
                            [nh.from_array(weight, "w"), nh.from_array(np.zeros(3, np.float32), "b")])
        with self.assertRaises(ValueError):
            compile_elementwise(model)
        # External-plus-intermediate Mul with a non-matching RGB operand.
        nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                 h.make_node("Mul", ["conv", "second"], ["output"])]
        model = build_model(nodes, (8, 8, 3), (8, 8, 3),
                            [nh.from_array(weight, "w"), nh.from_array(np.zeros(3, np.float32), "b")],
                            inputs=[("second", (1, 1, 3))])
        with self.assertRaises(ValueError):
            compile_elementwise(model)

    def test_compile_elementwise_requires_two_conv_branches(self):
        # A bare Add with no Conv producers has no branches to lower.
        model = build_model([h.make_node("Add", ["input", "input"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [])
        with self.assertRaises(ValueError):
            compile_elementwise(model)
        # A terminal op that is not one of the four join kinds is rejected up front.
        weight = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        with self.assertRaises(ValueError):
            compile_elementwise(build_model(
                [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                 h.make_node("Relu", ["conv"], ["output"])], (8, 8, 3), (8, 8, 3),
                [nh.from_array(weight, "w"), nh.from_array(np.zeros(3, np.float32), "b")]))
        # Operand zero points must be exactly two INT8 values on a Mul join.
        rng = np.random.default_rng(4)
        a = rng.uniform(-.3, .3, (4, 3, 1, 1)).astype(np.float32)
        ab = rng.uniform(-.2, .2, 4).astype(np.float32)
        b = rng.uniform(-.3, .3, (4, 3, 1, 1)).astype(np.float32)
        bb = rng.uniform(-.2, .2, 4).astype(np.float32)
        nodes = [h.make_node("Conv", ["input", "a_w", "a_b"], ["ca"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["input", "b_w", "b_b"], ["cb"], kernel_shape=[1, 1]),
                 h.make_node("Mul", ["ca", "cb"], ["output"])]
        model = build_model(nodes, (8, 8, 3), (8, 8, 4),
                            [nh.from_array(a, "a_w"), nh.from_array(ab, "a_b"),
                             nh.from_array(b, "b_w"), nh.from_array(bb, "b_b")],
                            value_info=[("ca", (8, 8, 4)), ("cb", (8, 8, 4))])
        with self.assertRaises(ValueError):
            compile_elementwise(model, operand_zero_points=(0,))
        with self.assertRaises(ValueError):
            compile_elementwise(model, operand_zero_points=(0, 0, 0))

    def test_compile_standalone_mul_guards(self):
        model = build_model([h.make_node("Add", ["input", "input"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [])
        with self.assertRaises(ValueError):
            compile_standalone_mul(model)

    def test_compile_constant_mul_guards(self):
        # A second external input is not the immutable-constant single-input form.
        model = build_model([h.make_node("Mul", ["input", "second"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [], inputs=[("second", (8, 8, 3))])
        with self.assertRaises(ValueError):
            compile_constant_mul(model)
        # Non-constant operand.
        weight = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        model = build_model([h.make_node("Mul", ["input", "missing"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [nh.from_array(weight, "w")])
        with self.assertRaises(ValueError):
            compile_constant_mul(model)
        # Integer constant (not float32).
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [nh.from_array(np.ones((1, 3, 1, 1), np.int32), "k")])
        with self.assertRaises(ValueError):
            compile_constant_mul(model)
        # A constant whose shape cannot broadcast to the input.
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])],
                            (8, 8, 3), (8, 8, 3),
                            [nh.from_array(np.ones((1, 5, 8, 8), np.float32), "k")])
        with self.assertRaises(ValueError):
            compile_constant_mul(model)

    def test_compile_per_channel_constant_mul_guards(self):
        with self.assertRaises(ValueError):
            compile_per_channel_constant_mul(
                build_model([h.make_node("Add", ["input", "input"], ["output"])], (8, 8, 3), (8, 8, 3), []))
        # A constant operand that is absent, non-finite, or non-rank-four.
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])], (8, 8, 3), (8, 8, 3), [])
        with self.assertRaises(ValueError):
            compile_per_channel_constant_mul(model)
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])], (8, 8, 3), (8, 8, 3),
                            [nh.from_array(np.array([np.inf, 1.0, 1.0], np.float32).reshape(1, 3, 1, 1), "k")])
        with self.assertRaises(ValueError):
            compile_per_channel_constant_mul(model)
        # A constant that does not broadcast to the input shape.
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])], (8, 8, 3), (8, 8, 3),
                            [nh.from_array(np.ones((1, 4, 8, 8), np.float32), "k")])
        with self.assertRaises(ValueError):
            compile_per_channel_constant_mul(model)
        # H/W outside the per-channel 5..8 bound.
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])], (4, 4, 3), (4, 4, 3),
                            [nh.from_array(np.array([0.5, 1.0, 2.0], np.float32).reshape(1, 3, 1, 1), "k")])
        with self.assertRaises(ValueError):
            compile_per_channel_constant_mul(model)


class RuntimeScaleGuardTests(unittest.TestCase):
    """Reachable rejection guards in the runtime-scale and batched Mul paths."""

    def test_runtime_scale_mul_guards(self):
        from open_rknpu.elementwise import compile_runtime_scale_mul
        with self.assertRaises(ValueError):
            compile_runtime_scale_mul(build_model([h.make_node("Add", ["input", "input"], ["output"])],
                                                  (8, 8, 3), (8, 8, 3), []))
        # Two external inputs but neither is the [1,3,1,1] scale tensor.
        model = build_model([h.make_node("Mul", ["input", "second"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [], inputs=[("second", (8, 8, 3))])
        with self.assertRaises(ValueError):
            compile_runtime_scale_mul(model)
        # A non-positive operand scale.
        model = build_model([h.make_node("Mul", ["input", "scale"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [], inputs=[("scale", (1, 1, 3))])
        with self.assertRaises(ValueError):
            compile_runtime_scale_mul(model, operand_scale=0.0)
        # H/W outside the runtime-scale 5..8 bound.
        model = build_model([h.make_node("Mul", ["input", "scale"], ["output"])],
                            (4, 4, 3), (4, 4, 3), [], inputs=[("scale", (1, 1, 3))])
        with self.assertRaises(ValueError):
            compile_runtime_scale_mul(model)

    def test_batched_constant_mul_guards(self):
        from open_rknpu.elementwise import _compile_batched_constant_mul
        rng = np.random.default_rng(13)
        weight = rng.uniform(-.3, .3, (3, 3, 1, 1)).astype(np.float32)
        model = build_model([h.make_node("Mul", ["input", "k"], ["output"])],
                            (8, 8, 3), (8, 8, 3), [nh.from_array(weight, "k")])
        factor = np.float32(0.5)
        # Batch 1 is outside the documented N2..16 bound.
        with self.assertRaises(ValueError):
            _compile_batched_constant_mul(model, factor, np.broadcast_to(factor, (1, 3, 8, 8)),
                                          1.0, 0, None, (0, 0), False)
        # A spatially varying constant is not a scalar or [N,C,1,1] vector.
        spatial = rng.uniform(.1, .9, (2, 3, 8, 8)).astype(np.float32)
        batch_model = build_batch_model(2, 8, 8, 3, np.float32(0.5))
        with self.assertRaises(ValueError):
            _compile_batched_constant_mul(batch_model, spatial, spatial, 1.0, 0, None, (0, 0), False)

    def test_valid_batched_constant_mul_compiles_deterministically(self):
        factor = np.array([0.5, 1.0, 2.0], np.float32).reshape(1, 3, 1, 1)
        model = build_batch_model(2, 8, 8, 3, factor)
        binary, meta = compile_sequence(copy.deepcopy(model))
        info = decode_sequence(binary)
        self.assertEqual(info["shape_nhwc"], [2, 8, 8, 3])
        self.assertEqual(info["output_shape_nhwc"], [2, 8, 8, 3])
        self.assertEqual(meta["constant_data_mode"], "per-batch-channel")
        self.assertGreater(meta["output_scale"], 0.0)
        again, _ = compile_sequence(copy.deepcopy(model))
        self.assertEqual(again, binary)


class EmitterHelperTests(unittest.TestCase):
    """Small helpers used by the sweep and their documented degenerate cases."""

    def test_profile_marker_requires_an_identity(self):
        with self.assertRaises(AssertionError):
            profile_marker({"output_scale": 1.0})

    def test_q_of_accepts_dict_and_live_quantization(self):
        rng = np.random.default_rng(17)
        weight = rng.uniform(-.3, .3, (3, 3, 1, 1)).astype(np.float32)
        bias = rng.uniform(-.3, .3, 3).astype(np.float32)
        model = build_model([h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[1, 1])],
                            (6, 6, 3), (6, 6, 3),
                            [nh.from_array(weight, "w"), nh.from_array(bias, "b")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            _, meta = compile_sequence(path)
        self.assertIsInstance(q_of(meta["quantization"]), Quantization)
        self.assertIsInstance(q_of(q_of(meta["quantization"])), Quantization)


if __name__ == "__main__":
    unittest.main()
