# SPDX-License-Identifier: MIT
"""Deep coverage of the fan-out join emitters: pooled, depthwise and chained.

Four bounded RV1103 profiles share one shape - a 1x1 stem fanned out to branches
that are folded by elementwise joins:

* `pool_join`  - two Conv heads, each followed by a 2x2 stride-2 pool, one join;
* `depthwise_join` - one dense head and one group-3 depthwise head, one join;
* `pooled_branches` - two or three one-to-three layer Conv chains, each pooled
  before one or two joins;
* `chain` (the legacy two-layer `Conv -> Relu -> Conv` container).

This module pins the accepted variants (decoded container metadata: profile,
shapes, task count, schedule and band) and checks each compiled integer
reference against an *independent* float64 implementation written here: the
container bands dequantize every grid, the arithmetic runs in float64, and the
result is requantized with the output band. Agreements are asserted at 0 LSB
where the arithmetic is integer-exact (identical branches that cancel under a
Sub join, identity convolutions) and at a documented <= 1 LSB otherwise
(average pooling and the two-step fixed-point conversion round differently).
Every rejection branch is paired with a valid neighbour that compiles, and the
two defensive guards the normal flow cannot reach (`depthwise_join`'s external
overlap check and `pooled_branches`' mixed-pool check) are exercised by
substituting the internal planner/parser they defend against.

No board, vendor artifact or network is read; the expected side never calls an
emitter reference.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

import open_rknpu.depthwise_join as depthwise_join_module
import open_rknpu.pooled_branches as pooled_branches_module
from open_rknpu.chain import compile_chain, native_quantize
from open_rknpu.chain_n import chain_n_reference
from open_rknpu.depthwise_join import (compile_depthwise_join, depthwise_join_reference,
                                       parse_depthwise_join)
from open_rknpu.pool_join import compile_pool_join, pool_join_reference
from open_rknpu.pooled_branches import (compile_pooled_branches, parse_pooled_branches,
                                        pooled_branches_reference)
from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

DENSE, DEPTHWISE = "dense", "depthwise"
POOLS = ("MaxPool", "AveragePool")
JOIN_KINDS = ("Add", "Mul", "Sub", "Max")


# ---------------------------------------------------------------------------
# Independent arithmetic (the expected side).  Nothing here calls an emitter
# reference: the container only supplies its declared quantization parameters.
# ---------------------------------------------------------------------------


def quant(entry):
    """Build a live `Quantization` from emitted metadata (dict or Quantization)."""
    if isinstance(entry, Quantization):
        return entry
    if "quantization" in entry:
        entry = entry["quantization"]
    return Quantization(**{key: (np.array(value) if isinstance(value, list) else value)
                           for key, value in entry.items()})


def to_real(codes, scale, zero_point):
    return (np.asarray(codes, np.float64) - zero_point) * scale


def requantize(values, scale, zero_point):
    return np.clip(np.rint(np.asarray(values, np.float64) / scale) + zero_point,
                   -128, 127).astype(np.int8)


def dequant_weights(q):
    """`scale * (code - zero_point)` for the container's per-channel weight codes."""
    codes = np.asarray(q.weights, np.float64)
    zero = np.asarray(q.weight_zero_points, np.float64)
    scale = np.asarray(q.weight_scales, np.float64)
    kernel = int(q.kernel_size)
    outputs = codes.shape[0]
    inputs = codes.size // (outputs * kernel * kernel)
    centered = codes.reshape(outputs, inputs, kernel, kernel) - zero[:, None, None, None]
    return centered * scale[:, None, None, None]


def conv2d(x, weight, bias, pads):
    """NHWC dense cross-correlation in float64 (real zero padding)."""
    top, left, bottom, right = pads
    padded = np.pad(x, ((top, bottom), (left, right), (0, 0)))
    kh, kw = weight.shape[2:]
    oh = padded.shape[0] - kh + 1
    ow = padded.shape[1] - kw + 1
    out = np.zeros((oh, ow, weight.shape[0]))
    for oy in range(oh):
        for ox in range(ow):
            patch = padded[oy:oy + kh, ox:ox + kw, :]
            out[oy, ox] = np.tensordot(patch, weight, axes=([0, 1, 2], [2, 3, 1])) + bias
    return out


def conv2d_code_border(x, weight, bias, pad, border):
    """NHWC cross-correlation whose spatial border carries an explicit real value."""
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), constant_values=border)
    kh, kw = weight.shape[2:]
    out = np.zeros((x.shape[0], x.shape[1], weight.shape[0]))
    for oy in range(x.shape[0]):
        for ox in range(x.shape[1]):
            patch = padded[oy:oy + kh, ox:ox + kw, :]
            out[oy, ox] = np.tensordot(patch, weight, axes=([0, 1, 2], [2, 3, 1])) + bias
    return out


def depthwise2d(x, weight, bias, pads):
    """NHWC grouped (depthwise) convolution in float64."""
    top, left, bottom, right = pads
    padded = np.pad(x, ((top, bottom), (left, right), (0, 0)))
    channels, _, kh, kw = weight.shape
    oh = padded.shape[0] - kh + 1
    ow = padded.shape[1] - kw + 1
    out = np.zeros((oh, ow, channels))
    per_channel = weight[:, 0].transpose(1, 2, 0)
    for oy in range(oh):
        for ox in range(ow):
            patch = padded[oy:oy + kh, ox:ox + kw, :]
            out[oy, ox] = (patch * per_channel).sum(axis=(0, 1)) + bias
    return out


def pool_real(grid, kind):
    height, width, channels = grid.shape
    blocked = np.asarray(grid, np.float64).reshape(height // 2, 2, width // 2, 2, channels)
    return blocked.max(axis=(1, 3)) if kind == "MaxPool" else blocked.mean(axis=(1, 3))


def join_real(kind, first, second):
    return {"Add": first + second, "Sub": first - second,
            "Max": np.maximum(first, second), "Mul": first * second}[kind]


def sample_grid(seed=7):
    return np.random.default_rng(seed).integers(0, 256, (8, 8, 3), dtype=np.uint8)


class VariantCase(unittest.TestCase):
    """Shared assertion that reports the observed per-channel LSB delta."""

    def assert_codes(self, got, expected, atol, label):
        got = np.asarray(got, np.int64)
        expected = np.asarray(expected, np.int64)
        self.assertEqual(got.shape, expected.shape, label)
        delta = int(np.abs(got - expected).max()) if got.size else 0
        self.assertLessEqual(delta, atol, "%s: max delta %d LSB" % (label, delta))
        return delta

    def assert_non_degenerate(self, codes, label):
        self.assertGreater(len(np.unique(codes)), 1, "%s: output is constant" % label)


def set_initializer(graph, name, array):
    for tensor in graph.initializer:
        if tensor.name == name:
            tensor.CopyFrom(nh.from_array(np.asarray(array), name))
            return
    raise KeyError(name)


def value_info(graph, name, shape):
    graph.value_info.append(h.make_tensor_value_info(name, 1, list(shape)))


def saved_path(model):
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "model.onnx"
    onnx.save(model, path)
    return folder, path


def compile_sequence_model(model, **kwargs):
    folder, path = saved_path(model)
    try:
        return compile_sequence(path, **kwargs)
    finally:
        folder.cleanup()


def compile_chain_model(model, **kwargs):
    folder, path = saved_path(model)
    try:
        return compile_chain(path, **kwargs)
    finally:
        folder.cleanup()


# ---------------------------------------------------------------------------
# pool_join: stem, two Conv+pool branches, one join
# ---------------------------------------------------------------------------


def pool_join_model(kind="Add", pool="MaxPool", kernels=(1, 3), hidden=8, stem_relu=True,
                    tweak=None, extra_opsets=(), identical=False):
    rng = np.random.default_rng(sum(kernels) * 5 + hidden + len(pool))
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    shared = rng.uniform(.02, .06, (3, hidden, kernels[1], kernels[1])).astype(np.float32)
    shared_bias = rng.uniform(-1, 1, (3,)).astype(np.float32)
    for name, kernel in (("a", kernels[0]), ("b", kernels[1])):
        if identical:
            weight, bias = shared, shared_bias
        else:
            weight = rng.uniform(.02, .06, (3, hidden, kernel, kernel)).astype(np.float32)
            bias = rng.uniform(-1, 1, (3,)).astype(np.float32)
        constants += [nh.from_array(weight, "w" + name), nh.from_array(bias, "b" + name)]
        nodes.append(h.make_node("Conv", [source, "w" + name, "b" + name], ["head_" + name],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        nodes.append(h.make_node(pool, ["head_" + name], ["pool_" + name],
                                 kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(h.make_node(kind, ["pool_a", "pool_b"], ["output"]))
    graph = h.make_graph(nodes, "pool_join",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])], constants)
    value_info(graph, source, (1, hidden, 8, 8))
    for name in ("head_a", "head_b"):
        value_info(graph, name, (1, 3, 8, 8))
    for name in ("pool_a", "pool_b"):
        value_info(graph, name, (1, 3, 4, 4))
    if tweak is not None:
        tweak(graph)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)] + list(extra_opsets))
    model.ir_version = 8
    return model


def pool_join_expected(inputs, meta, pool):
    stem = quant(meta["stem_quantization"])
    stem_real = conv2d(to_real(inputs, meta["input_scale"], meta["input_zero_point"]),
                       dequant_weights(stem), np.asarray(meta["stem_bias"], np.float64),
                       (0, 0, 0, 0))
    stem_codes = requantize(stem_real, stem.output_scale, stem.output_zero_point)
    pooled = []
    for index, entry in enumerate(meta["head_quantization"]):
        q = quant(entry)
        real = conv2d(to_real(stem_codes, stem.output_scale, stem.output_zero_point),
                      dequant_weights(q), np.asarray(meta["bias"][index], np.float64),
                      (q.kernel_size // 2,) * 4)
        pooled.append(pool_real(real, pool))
    value = join_real(meta["join"], pooled[0], pooled[1])
    return requantize(value, meta["output_scale"], meta["output_zero_point"])


class PoolJoinSemanticsTests(VariantCase):
    """Accepted pool-join variants and the independent float64 output."""

    def test_join_pool_kernel_matrix(self):
        for kind in JOIN_KINDS:
            for pool in POOLS:
                for kernels in ((1, 1), (1, 3), (3, 3)):
                    with self.subTest(kind=kind, pool=pool, kernels=kernels):
                        model = pool_join_model(kind, pool, kernels)
                        _, meta = compile_pool_join(model)
                        self.assertEqual(meta["profile"], "pool-join")
                        self.assertEqual(meta["join"], kind)
                        self.assertEqual(meta["pool"], pool)
                        self.assertEqual(meta["hidden_channels"], 8)
                        self.assertEqual(meta["head_kernels"], list(kernels))
                        self.assertEqual(meta["output_shape_nhwc"], [1, 4, 4, 3])
                        self.assertEqual(meta["schedule"],
                                         ["stem", "head_a", "head_b", "pool_a", "pool_b", "output"])
                        inputs = sample_grid(29 + sum(kernels))
                        got = pool_join_reference(inputs, meta["stem_quantization"],
                                                  meta["head_quantization"], meta["join"], meta["pool"])
                        expected = pool_join_expected(inputs, meta, pool)
                        self.assert_codes(got, expected, 1, "%s %s %s" % (kind, pool, kernels))
                        self.assert_non_degenerate(got, "%s %s" % (kind, pool))

    def test_identical_branches_sub_is_integer_exact(self):
        model = pool_join_model("Sub", "MaxPool", (3, 3), 6, identical=True)
        _, meta = compile_pool_join(model)
        inputs = sample_grid(41)
        got = pool_join_reference(inputs, meta["stem_quantization"], meta["head_quantization"],
                                  meta["join"], meta["pool"])
        self.assert_codes(got, np.zeros((4, 4, 3), np.int64) + meta["output_zero_point"], 0,
                          "identical branches cancel")
        self.assert_codes(got, pool_join_expected(inputs, meta, "MaxPool"), 0, "independent Sub")

    def test_output_override_requires_mul_and_is_recorded(self):
        with self.assertRaisesRegex(ValueError, "output override requires a Mul join"):
            compile_pool_join(pool_join_model("Add"), output_range={"scale": .1, "zero_point": 0})
        override = {"scale": 0.25, "zero_point": 0}
        binary, meta = compile_pool_join(pool_join_model("Mul"), output_range=override)
        self.assertEqual(meta["output_scale"], np.float32(0.25))
        self.assertEqual(meta["output_zero_point"], 0)
        natural = float(np.float32(128 * meta["join_scales"][0] * meta["join_scales"][1]))
        self.assertNotAlmostEqual(meta["output_scale"], natural)
        # The override is what the container itself declares, not just the metadata.
        _, plain = compile_pool_join(pool_join_model("Mul"))
        info = decode_sequence(binary)
        self.assertEqual(info["output_scale"], np.float32(0.25))
        self.assertEqual(info["output_zero_point"], 0)
        self.assertNotEqual(info["output_scale"], np.float32(plain["output_scale"]))
        # `pool_join_reference` models the default 128*sa*sb band only, so the override is
        # pinned by the decoded container band rather than by that reference.
        inputs = sample_grid(43)
        expected = pool_join_expected(inputs, meta, meta["pool"])
        natural_meta = dict(meta, output_scale=natural, output_zero_point=0)
        self.assertFalse(np.array_equal(pool_join_expected(inputs, natural_meta, meta["pool"]),
                                        expected))

    def test_reference_accepts_live_quantization_objects(self):
        model = pool_join_model("Add", "AveragePool", (1, 3), hidden=5)
        _, meta = compile_pool_join(model)
        inputs = sample_grid(47)
        from_dicts = pool_join_reference(inputs, meta["stem_quantization"], meta["head_quantization"],
                                         meta["join"], meta["pool"])
        from_objects = pool_join_reference(
            inputs, quant(meta["stem_quantization"]),
            [quant(entry) for entry in meta["head_quantization"]], meta["join"], meta["pool"])
        np.testing.assert_array_equal(from_objects, from_dicts)


class PoolJoinRejectionTests(VariantCase):
    """Every pool-join validation branch, paired with a compiling neighbour."""

    def test_valid_neighbour_compiles(self):
        _, meta = compile_pool_join(pool_join_model())
        self.assertEqual(meta["profile"], "pool-join")

    def test_default_domain_is_required(self):
        def tweak(graph):
            graph.node[0].domain = "com.custom"
        with self.assertRaisesRegex(ValueError, "default-domain nodes"):
            compile_pool_join(pool_join_model(tweak=tweak,
                                              extra_opsets=[h.make_opsetid("com.custom", 1)]))

    def test_structure_is_checked(self):
        def extra_node(graph):
            graph.node.append(h.make_node("Relu", ["output"], ["final"]))
            graph.output[0].name = "final"
        with self.assertRaisesRegex(ValueError, "stem\\[,Relu\\], Conv, pool"):
            compile_pool_join(pool_join_model(tweak=extra_node))

        def not_conv(graph):
            graph.node[0].op_type = "Relu"
            graph.node[0].input[:] = ["input"]
            del graph.node[0].attribute[:]
        with self.assertRaisesRegex(ValueError, "stem\\[,Relu\\], Conv, pool"):
            compile_pool_join(pool_join_model(tweak=not_conv))

    def test_join_kind_is_checked(self):
        def tweak(graph):
            graph.node[-1].op_type = "Div"
        with self.assertRaisesRegex(ValueError, "two Conv branches and Add/Mul/Sub/Max"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_matching_pool_kinds_are_required(self):
        def tweak(graph):
            graph.node[5].op_type = "AveragePool"
        with self.assertRaisesRegex(ValueError, "two matching MaxPool or AveragePool"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_join_must_not_carry_attributes(self):
        def tweak(graph):
            graph.node[-1].attribute.append(h.make_attribute("axis", 0))
        model = pool_join_model(tweak=tweak)
        with mock.patch("onnx.checker.check_model"):
            with self.assertRaisesRegex(ValueError, "must not carry attributes"):
                compile_pool_join(model)

    def test_branch_wiring_is_checked(self):
        def swap(graph):
            node = graph.node[-1]
            node.input[0], node.input[1] = node.input[1], node.input[0]
        with self.assertRaisesRegex(ValueError, "both pools in order"):
            compile_pool_join(pool_join_model(tweak=swap))

        def wrong_source(graph):
            graph.node[4].input[0] = "head_a"
        with self.assertRaisesRegex(ValueError, "both pools in order"):
            compile_pool_join(pool_join_model(tweak=wrong_source))

    def test_one_input_and_one_output_are_required(self):
        def tweak(graph):
            graph.output.append(h.make_tensor_value_info("pool_a", 1, [1, 3, 4, 4]))
        with self.assertRaisesRegex(ValueError, "one input and one output"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_external_tensors_must_be_float32(self):
        def tweak(graph):
            graph.input[0].type.tensor_type.elem_type = 2
        with self.assertRaisesRegex(ValueError, "external tensors must be float32"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_convolution_constants_are_required(self):
        def tweak(graph):
            del graph.node[2].input[2]
        with self.assertRaisesRegex(ValueError, "constant float32 weights and bias"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_convolution_constants_must_be_float32(self):
        def tweak(graph):
            set_initializer(graph, "b1", np.zeros(8, np.float64))
        with self.assertRaisesRegex(ValueError, "constants must be float32"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_stem_attributes_are_checked(self):
        def tweak(graph):
            graph.node[0].attribute.append(h.make_attribute("strides", [2, 2]))
        with self.assertRaisesRegex(ValueError, "unsupported stem attributes"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_stem_relu_must_consume_the_stem(self):
        def tweak(graph):
            graph.node[1].input[0] = "input"
        with self.assertRaisesRegex(ValueError, "stem Relu must consume the stem output"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_branch_shapes_are_checked(self):
        def channels(graph):
            set_initializer(graph, "wa", np.zeros((4, 8, 1, 1), np.float32))
        with self.assertRaisesRegex(ValueError, "dense 3-output 1x1/3x3 Conv"):
            compile_pool_join(pool_join_model(tweak=channels))

        def kernel(graph):
            set_initializer(graph, "wa", np.zeros((3, 8, 2, 2), np.float32))
        with self.assertRaisesRegex(ValueError, "dense 3-output 1x1/3x3 Conv"):
            compile_pool_join(pool_join_model(tweak=kernel))

    def test_branch_attributes_are_checked(self):
        def tweak(graph):
            graph.node[2].attribute.append(h.make_attribute("strides", [2, 2]))
        with self.assertRaisesRegex(ValueError, "unsupported branch attributes"):
            compile_pool_join(pool_join_model(tweak=tweak))

    def test_3x3_branch_requires_symmetric_pad1(self):
        def tweak(graph):
            node = graph.node[4]
            node.attribute.remove(next(a for a in node.attribute if a.name == "pads"))
        with self.assertRaisesRegex(ValueError, "symmetric pad1"):
            compile_pool_join(pool_join_model(tweak=tweak))


# ---------------------------------------------------------------------------
# depthwise_join: shared stem, dense head and depthwise head, one join
# ---------------------------------------------------------------------------


def depthwise_join_model(kind="Add", dense_kernel=1, dw_kernel=3, group=3, stem_relu=True,
                         hidden=3, dense_bias=None, dw_bias=None, tweak=None, extra_opsets=()):
    rng = np.random.default_rng(dense_kernel * 7 + dw_kernel * 13 + group)
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    dense_weight = rng.uniform(.02, .06, (3, hidden, dense_kernel, dense_kernel)).astype(np.float32)
    dense_b = (np.asarray(dense_bias, np.float32) if dense_bias is not None
               else rng.uniform(-1, 1, (3,)).astype(np.float32))
    constants += [nh.from_array(dense_weight, "wd"), nh.from_array(dense_b, "bd")]
    nodes.append(h.make_node("Conv", [source, "wd", "bd"], ["dense"],
                             kernel_shape=[dense_kernel] * 2, pads=[dense_kernel // 2] * 4))
    dw_weight = rng.uniform(.02, .06, (group, 1, dw_kernel, dw_kernel)).astype(np.float32)
    dw_b = (np.asarray(dw_bias, np.float32) if dw_bias is not None
            else rng.uniform(-1, 1, (group,)).astype(np.float32))
    constants += [nh.from_array(dw_weight, "ww"), nh.from_array(dw_b, "bw")]
    nodes.append(h.make_node("Conv", [source, "ww", "bw"], ["depthwise"],
                             kernel_shape=[dw_kernel] * 2, pads=[dw_kernel // 2] * 4, group=group))
    nodes.append(h.make_node(kind, ["dense", "depthwise"], ["output"]))
    graph = h.make_graph(nodes, "depthwise_join",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    value_info(graph, source, (1, hidden, 8, 8))
    value_info(graph, "dense", (1, 3, 8, 8))
    value_info(graph, "depthwise", (1, 3, 8, 8))
    if tweak is not None:
        tweak(graph)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)] + list(extra_opsets))
    model.ir_version = 8
    return model


def depthwise_join_expected(inputs, meta, kind):
    stem = quant(meta["stem_quantization"])
    stem_real = conv2d(to_real(inputs, meta["input_scale"], meta["input_zero_point"]),
                       dequant_weights(stem), np.asarray(meta["stem_bias"], np.float64),
                       (0, 0, 0, 0))
    stem_codes = requantize(stem_real, stem.output_scale, stem.output_zero_point)
    source = to_real(stem_codes, stem.output_scale, stem.output_zero_point)
    dense_q = quant(meta["head_quantization"][0])
    dense = conv2d(source, dequant_weights(dense_q),
                   np.asarray(meta["bias"][0], np.float64), (dense_q.kernel_size // 2,) * 4)
    depth_q = quant(meta["depthwise_quantization"])
    branch = depthwise2d(source, dequant_weights(depth_q),
                         np.asarray(meta["depthwise_bias"][0], np.float64),
                         (depth_q.kernel_size // 2,) * 4)
    return requantize(join_real(kind, dense, branch), meta["output_scale"], meta["output_zero_point"])


class DepthwiseJoinSemanticsTests(VariantCase):
    """Accepted depthwise-join variants and the independent float64 output."""

    def test_join_kernel_matrix(self):
        for kind in JOIN_KINDS:
            for dense_kernel, dw_kernel in ((1, 3), (3, 5), (3, 3)):
                with self.subTest(kind=kind, dense=dense_kernel, depthwise=dw_kernel):
                    model = depthwise_join_model(kind, dense_kernel, dw_kernel)
                    _, meta = compile_depthwise_join(model)
                    self.assertEqual(meta["profile"], "depthwise-join")
                    self.assertEqual(meta["join"], kind)
                    self.assertEqual(meta["dense_kernel"], dense_kernel)
                    self.assertEqual(meta["depthwise_kernel"], dw_kernel)
                    self.assertEqual(meta["schedule"], ["stem", "dense", "depthwise", "output"])
                    self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 3])
                    inputs = sample_grid(53 + dense_kernel + dw_kernel)
                    got = depthwise_join_reference(
                        inputs, meta["stem_quantization"], meta["head_quantization"][0],
                        meta["depthwise_quantization"], meta["join"])
                    self.assert_codes(got, depthwise_join_expected(inputs, meta, kind), 1,
                                      "%s %d %d" % (kind, dense_kernel, dw_kernel))
                    self.assert_non_degenerate(got, "depthwise join")

    def test_asymmetric_depthwise_pair_is_recorded(self):
        model = depthwise_join_model("Max", 3, 3)
        _, meta = compile_depthwise_join(model, asymmetric_depthwise=True)
        self.assertTrue(meta["depthwise_asymmetric_pair"])
        inputs = sample_grid(59)
        got = depthwise_join_reference(inputs, meta["stem_quantization"],
                                       meta["head_quantization"][0], meta["depthwise_quantization"],
                                       meta["join"])
        self.assert_codes(got, depthwise_join_expected(inputs, meta, "Max"), 1, "asymmetric pair")

    def test_identical_identity_branches_sub_is_integer_exact(self):
        model = depthwise_join_model("Sub", 1, 1, dense_bias=np.zeros(3, np.float32),
                                     dw_bias=np.zeros(3, np.float32))
        for tensor in model.graph.initializer:
            if tensor.name == "wd":
                identity = np.zeros((3, 3, 1, 1), np.float32)
                for channel in range(3):
                    identity[channel, channel, 0, 0] = 1.0
                tensor.CopyFrom(nh.from_array(identity, "wd"))
            elif tensor.name == "ww":
                tensor.CopyFrom(nh.from_array(np.ones((3, 1, 1, 1), np.float32), "ww"))
        _, meta = compile_depthwise_join(model)
        inputs = sample_grid(61)
        got = depthwise_join_reference(inputs, meta["stem_quantization"],
                                       meta["head_quantization"][0], meta["depthwise_quantization"],
                                       meta["join"])
        self.assert_codes(got, np.zeros((8, 8, 3), np.int64) + meta["output_zero_point"], 0,
                          "identity branches cancel")
        self.assert_codes(got, depthwise_join_expected(inputs, meta, "Sub"), 0, "independent Sub")

    def test_output_override_requires_mul(self):
        with self.assertRaisesRegex(ValueError, "output override requires a Mul join"):
            compile_depthwise_join(depthwise_join_model("Add"),
                                   output_range={"scale": .1, "zero_point": 0})
        override = {"scale": 0.5, "zero_point": 2}
        _, meta = compile_depthwise_join(depthwise_join_model("Mul"), output_range=override)
        self.assertEqual(meta["output_scale"], np.float32(0.5))
        self.assertEqual(meta["output_zero_point"], 2)

    def test_reference_accepts_live_quantization_objects(self):
        model = depthwise_join_model("Add", 1, 5)
        _, meta = compile_depthwise_join(model)
        inputs = sample_grid(67)
        from_dicts = depthwise_join_reference(inputs, meta["stem_quantization"],
                                              meta["head_quantization"][0],
                                              meta["depthwise_quantization"], meta["join"])
        from_objects = depthwise_join_reference(
            inputs, quant(meta["stem_quantization"]), quant(meta["head_quantization"][0]),
            quant(meta["depthwise_quantization"]), meta["join"])
        np.testing.assert_array_equal(from_objects, from_dicts)


class DepthwiseJoinRejectionTests(VariantCase):
    """Every depthwise-join validation branch, paired with a compiling neighbour."""

    def test_parse_rejects_empty_and_non_conv(self):
        self.assertIs(parse_depthwise_join([]), False)
        self.assertIs(parse_depthwise_join([h.make_node("Relu", ["x"], ["y"])]), False)
        nodes = list(depthwise_join_model().graph.node)
        self.assertIs(parse_depthwise_join(nodes), True)

    def test_default_domain_is_required(self):
        def tweak(graph):
            graph.node[0].domain = "com.custom"
        with self.assertRaisesRegex(ValueError, "default-domain nodes"):
            compile_depthwise_join(depthwise_join_model(
                tweak=tweak, extra_opsets=[h.make_opsetid("com.custom", 1)]))

    def test_structure_is_checked(self):
        def extra_node(graph):
            graph.node.append(h.make_node("Relu", ["output"], ["final"]))
            graph.output[0].name = "final"
        with self.assertRaisesRegex(ValueError, "stem\\[,Relu\\], dense Conv"):
            compile_depthwise_join(depthwise_join_model(tweak=extra_node))

    def test_join_kind_is_checked(self):
        def tweak(graph):
            graph.node[4].op_type = "Div"
        with self.assertRaisesRegex(ValueError, "dense Conv, a depthwise Conv and Add/Mul/Sub/Max"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_join_must_not_carry_attributes(self):
        def tweak(graph):
            graph.node[4].attribute.append(h.make_attribute("axis", 0))
        model = depthwise_join_model(tweak=tweak)
        with mock.patch("onnx.checker.check_model"):
            with self.assertRaisesRegex(ValueError, "must not carry attributes"):
                compile_depthwise_join(model)

    def test_branches_must_read_the_shared_stem(self):
        def tweak(graph):
            graph.node[3].input[0] = "dense"
        with self.assertRaisesRegex(ValueError, "consume the shared stem output"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_join_operand_order_is_checked(self):
        def tweak(graph):
            node = graph.node[4]
            node.input[0], node.input[1] = node.input[1], node.input[0]
        with self.assertRaisesRegex(ValueError, "dense branch then the depthwise branch"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_one_input_and_one_output_are_required(self):
        def tweak(graph):
            graph.output.append(h.make_tensor_value_info("dense", 1, [1, 3, 8, 8]))
        with self.assertRaisesRegex(ValueError, "one input and one output"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_external_tensors_must_be_float32(self):
        def tweak(graph):
            graph.input[0].type.tensor_type.elem_type = 2
        with self.assertRaisesRegex(ValueError, "external tensors must be float32"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_convolution_constants_are_required(self):
        def tweak(graph):
            del graph.node[2].input[2]
        with self.assertRaisesRegex(ValueError, "constant float32 weights and bias"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_convolution_constants_must_be_float32(self):
        def tweak(graph):
            set_initializer(graph, "bd", np.zeros(3, np.float64))
        with self.assertRaisesRegex(ValueError, "constants must be float32"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_stem_attributes_are_checked(self):
        def tweak(graph):
            graph.node[0].attribute.append(h.make_attribute("strides", [2, 2]))
        with self.assertRaisesRegex(ValueError, "unsupported stem attributes"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_stem_relu_must_consume_the_stem(self):
        def tweak(graph):
            graph.node[1].input[0] = "input"
        with self.assertRaisesRegex(ValueError, "stem Relu must consume the stem output"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_dense_branch_shape_is_checked(self):
        def tweak(graph):
            set_initializer(graph, "wd", np.zeros((4, 3, 1, 1), np.float32))
        with self.assertRaisesRegex(ValueError, "3-output 1x1/3x3 Conv"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_dense_branch_attributes_are_checked(self):
        def tweak(graph):
            graph.node[2].attribute.append(h.make_attribute("strides", [2, 2]))
        with self.assertRaisesRegex(ValueError, "unsupported dense branch attributes"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_dense_3x3_requires_symmetric_pad1(self):
        def tweak(graph):
            node = graph.node[2]
            node.attribute.remove(next(a for a in node.attribute if a.name == "pads"))
        with self.assertRaisesRegex(ValueError, "symmetric pad1"):
            compile_depthwise_join(depthwise_join_model(dense_kernel=3, tweak=tweak))

    def test_depthwise_branch_shape_is_checked(self):
        def tweak(graph):
            set_initializer(graph, "ww", np.zeros((3, 1, 4, 4), np.float32))
        with self.assertRaisesRegex(ValueError, "group3 depthwise 1x1/3x3/5x5"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_depthwise_branch_attributes_are_checked(self):
        def tweak(graph):
            graph.node[3].attribute.append(h.make_attribute("strides", [2, 2]))
        with self.assertRaisesRegex(ValueError, "unsupported depthwise branch attributes"):
            compile_depthwise_join(depthwise_join_model(tweak=tweak))

    def test_zero_centered_operands_are_required(self):
        with self.assertRaisesRegex(ValueError, "zero-centered join operands"):
            compile_depthwise_join(depthwise_join_model(), operand_zero_points=(1, 0))
        _, meta = compile_depthwise_join(depthwise_join_model())
        self.assertEqual(meta["profile"], "depthwise-join")

    def test_external_overlap_guard_is_reachable(self):
        """`plan` never overlaps an external today; the guard still fires if it did."""
        def overlapping_plan(bindings, sizes, start=0, defined=()):
            return (list(range(len(bindings))),
                    {name: (-1, -1) for name in sizes},
                    {name: start - 128 for name in sizes})
        model = depthwise_join_model()
        with mock.patch.object(depthwise_join_module, "plan", overlapping_plan):
            with self.assertRaisesRegex(ValueError, "external tensor .* overlaps"):
                compile_depthwise_join(model)


# ---------------------------------------------------------------------------
# pooled_branches: two or three pooled Conv chains folded by one or two joins
# ---------------------------------------------------------------------------


def pooled_branches_model(branches, joins=("Add",), hidden=3, pool="MaxPool", stem_relu=False,
                          tweak=None, extra_opsets=()):
    rng = np.random.default_rng(len(branches) * 17 + sum(len(b) for b in branches))
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "bs_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "bs_stem"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    channels = {source: hidden}
    pooled = []
    biases = []
    for branch_index, layers in enumerate(branches):
        previous = source
        branch_biases = []
        for layer_index, (kind, in_channels, out_channels, kernel) in enumerate(layers):
            shape = ((out_channels, 1, kernel, kernel) if kind == DEPTHWISE
                     else (out_channels, in_channels, kernel, kernel))
            weight = rng.uniform(.02, .06, shape).astype(np.float32)
            bias = rng.uniform(-1, 1, (out_channels,)).astype(np.float32)
            constants += [nh.from_array(weight, "w%d_%d" % (branch_index, layer_index)),
                          nh.from_array(bias, "bs%d_%d" % (branch_index, layer_index))]
            attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
            if kind == DEPTHWISE:
                attributes["group"] = out_channels
            name = "b%d_%d" % (branch_index, layer_index)
            nodes.append(h.make_node("Conv", [previous, "w%d_%d" % (branch_index, layer_index),
                                              "bs%d_%d" % (branch_index, layer_index)],
                                     [name], **attributes))
            previous = name
            channels[name] = out_channels
            branch_biases.append(bias)
        biases.append(branch_biases)
        pooled_name = "p%d" % branch_index
        nodes.append(h.make_node(pool, [previous], [pooled_name], kernel_shape=[2, 2], strides=[2, 2]))
        pooled.append(pooled_name)
    last = None
    for position, kind in enumerate(joins):
        first = pooled[0] if position == 0 else last
        nodes.append(h.make_node(kind, [first, pooled[position + 1]], ["j%d" % position]))
        last = "j%d" % position
    graph = h.make_graph(nodes, "pooled_branches",
                         [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info(last, 1, [1, 3, 4, 4])], constants)
    for name, count in channels.items():
        value_info(graph, name, (1, count, 8, 8))
    for position in range(len(branches)):
        value_info(graph, "p%d" % position, (1, 3, 4, 4))
    for position in range(len(joins)):
        value_info(graph, "j%d" % position, (1, 3, 4, 4))
    if tweak is not None:
        tweak(graph)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)] + list(extra_opsets))
    model.ir_version = 8
    return model, dict(biases=biases, stem_bias=nh.to_array(constants[1]))


def pooled_branches_expected(inputs, meta, pool, info):
    stem = quant(meta["stem_quantization"])
    stem_real = conv2d(to_real(inputs, meta["input_scale"], meta["input_zero_point"]),
                       dequant_weights(stem), np.asarray(info["stem_bias"], np.float64),
                       (0, 0, 0, 0))
    stem_codes = requantize(stem_real, stem.output_scale, stem.output_zero_point)
    pooled = []
    for branch_index, names in enumerate(meta["branch_names"]):
        grid = to_real(stem_codes, stem.output_scale, stem.output_zero_point)
        for position, _ in enumerate(names):
            q = quant(meta["branch_quantization"][branch_index][position])
            real = conv2d(grid, dequant_weights(q),
                          np.asarray(info["biases"][branch_index][position], np.float64),
                          (q.kernel_size // 2,) * 4)
            grid = to_real(requantize(real, q.output_scale, q.output_zero_point),
                           q.output_scale, q.output_zero_point)
        pooled.append(pool_real(grid, pool))
    values = {meta["pooled_names"][index]: pooled[index] for index in range(len(pooled))}
    for step in meta["join_expression"]:
        values[step["name"]] = join_real(step["kind"], values[step["inputs"][0]],
                                         values[step["inputs"][1]])
    last = meta["join_expression"][-1]["name"]
    return requantize(values[last], meta["output_scale"], meta["output_zero_point"])


class PooledBranchesSemanticsTests(VariantCase):
    """Accepted pooled-branch variants and the independent float64 output."""

    def test_two_branch_chain_and_join_metadata(self):
        model, info = pooled_branches_model([[(DENSE, 3, 5, 1), (DENSE, 5, 3, 1)],
                                             [(DENSE, 3, 3, 3)]])
        _, meta = compile_pooled_branches(model)
        self.assertEqual(meta["profile"], "pooled-branches")
        self.assertEqual(meta["pool"], "MaxPool")
        self.assertEqual(meta["branch_names"], [["b0_0", "b0_1"], ["b1_0"]])
        self.assertEqual(meta["pooled_names"], ["p0", "p1"])
        self.assertEqual(meta["head_kinds"], ["dense", "dense"])
        self.assertEqual(meta["output_shape_nhwc"], [1, 4, 4, 3])
        self.assertEqual(meta["join_expression"],
                         [dict(name="j0", kind="Add", inputs=["p0", "p1"])])
        self.assertEqual(meta["schedule"][-1], "output")
        inputs = sample_grid(71)
        got = pooled_branches_reference(inputs, meta["stem_quantization"], meta["branch_names"],
                                        meta["branch_quantization"], meta["join_expression"],
                                        meta["pool"], pooled_names=meta["pooled_names"])
        self.assert_codes(got, pooled_branches_expected(inputs, meta, "MaxPool", info), 1,
                          "two-branch chain")

    def test_three_branches_with_two_joins_and_a_chained_depthwise_layer(self):
        model, info = pooled_branches_model(
            [[(DENSE, 3, 3, 1), (DEPTHWISE, 3, 3, 3)],
             [(DENSE, 3, 3, 1)],
             [(DENSE, 3, 4, 3), (DENSE, 4, 3, 1)]],
            joins=("Add", "Max"))
        _, meta = compile_pooled_branches(model)
        self.assertEqual(len(meta["branch_names"]), 3)
        self.assertEqual(meta["join_expression"],
                         [dict(name="j0", kind="Add", inputs=["p0", "p1"]),
                          dict(name="j1", kind="Max", inputs=["j0", "p2"])])
        self.assertEqual(meta["branch_names"][0], ["b0_0", "b0_1"])
        self.assertTrue(meta["branch_quantization"][0][1])  # a live band for the expanded layer
        inputs = sample_grid(73)
        got = pooled_branches_reference(inputs, meta["stem_quantization"], meta["branch_names"],
                                        meta["branch_quantization"], meta["join_expression"],
                                        meta["pool"], pooled_names=meta["pooled_names"])
        self.assert_codes(got, pooled_branches_expected(inputs, meta, "MaxPool", info), 1,
                          "three-branch two-join")

    def test_average_pool_and_mul_join(self):
        model, info = pooled_branches_model([[(DENSE, 3, 3, 1), (DENSE, 3, 3, 1)],
                                             [(DENSE, 3, 3, 1)]], joins=("Mul",), pool="AveragePool")
        _, meta = compile_pooled_branches(model)
        self.assertEqual(meta["pool"], "AveragePool")
        self.assertEqual(meta["join_expression"][0]["kind"], "Mul")
        inputs = sample_grid(79)
        got = pooled_branches_reference(inputs, meta["stem_quantization"], meta["branch_names"],
                                        meta["branch_quantization"], meta["join_expression"],
                                        meta["pool"], pooled_names=meta["pooled_names"])
        self.assert_codes(got, pooled_branches_expected(inputs, meta, "AveragePool", info), 1,
                          "average pool mul join")

    def test_every_internal_has_a_fresh_slot(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]])
        _, meta = compile_pooled_branches(model)
        offsets = {name: value for name, value in meta["tensor_offsets"].items()
                   if name not in ("input0", "output")}
        self.assertEqual(len(set(offsets.values())), len(offsets))

    def test_reference_accepts_live_quantization_objects(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]])
        _, meta = compile_pooled_branches(model)
        inputs = sample_grid(83)
        from_dicts = pooled_branches_reference(inputs, meta["stem_quantization"],
                                               meta["branch_names"], meta["branch_quantization"],
                                               meta["join_expression"], meta["pool"],
                                               pooled_names=meta["pooled_names"])
        from_objects = pooled_branches_reference(
            inputs, quant(meta["stem_quantization"]), meta["branch_names"],
            [[quant(entry) for entry in branch] for branch in meta["branch_quantization"]],
            meta["join_expression"], meta["pool"], pooled_names=meta["pooled_names"])
        np.testing.assert_array_equal(from_objects, from_dicts)


def parse_node(op_type, inputs, outputs, domain="", **attributes):
    return h.make_node(op_type, list(inputs), list(outputs), domain=domain, **attributes)


class PooledBranchesParseTests(VariantCase):
    """The structural rejections `parse_pooled_branches` must return `None` for."""

    def test_valid_neighbour_parses(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]])
        self.assertIsNotNone(parse_pooled_branches(list(model.graph.node)))

    def test_too_few_nodes(self):
        self.assertIsNone(parse_pooled_branches([parse_node("Conv", ["a", "w", "b"], ["x"])]))

    def test_a_conv_after_the_branch_pool_is_rejected(self):
        nodes = [parse_node("Conv", ["image", "w", "b"], ["s"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["s", "w", "b"], ["a"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["a", "w", "b"], ["b"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["b"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Conv", ["a", "w", "b"], ["late"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["s", "w", "b"], ["d"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["d"], ["p1"], kernel_shape=[2, 2], strides=[2, 2])]
        self.assertIsNone(parse_pooled_branches(nodes))

    def test_pool_must_read_the_branch_tail(self):
        nodes = [parse_node("Conv", ["image", "w", "b"], ["s"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["s", "w", "b"], ["a"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["a", "w", "b"], ["b"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["a"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Conv", ["s", "w", "b"], ["d"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["d"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Add", ["p0", "p1"], ["j"])]
        self.assertIsNone(parse_pooled_branches(nodes))

    def test_join_shape_and_operands(self):
        base = [parse_node("Conv", ["image", "w", "b"], ["s"], kernel_shape=[1, 1]),
                parse_node("Conv", ["s", "w", "b"], ["a"], kernel_shape=[1, 1]),
                parse_node("Conv", ["a", "w", "b"], ["b"], kernel_shape=[1, 1]),
                parse_node("MaxPool", ["b"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
                parse_node("Conv", ["s", "w", "b"], ["d"], kernel_shape=[1, 1]),
                parse_node("MaxPool", ["d"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
                parse_node("Add", ["p0", "p1"], ["j"])]
        with_attribute = list(base)
        with_attribute[-1] = parse_node("Add", ["p0", "p1"], ["j"], axis=0)
        self.assertIsNone(parse_pooled_branches(with_attribute))
        same_operand = list(base)
        same_operand[-1] = parse_node("Add", ["p0", "p0"], ["j"])
        self.assertIsNone(parse_pooled_branches(same_operand))
        unknown = list(base)
        unknown[-1] = parse_node("Add", ["p0", "missing"], ["j"])
        self.assertIsNone(parse_pooled_branches(unknown))

    def test_join_count_must_be_branches_minus_one(self):
        nodes = [parse_node("Conv", ["image", "w", "b"], ["s"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["s", "w", "b"], ["a"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["a"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Conv", ["s", "w", "b"], ["b"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["b"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Conv", ["s", "w", "b"], ["c"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["c"], ["p2"], kernel_shape=[2, 2], strides=[2, 2])]
        self.assertIsNone(parse_pooled_branches(nodes))

    def test_branch_length_is_capped_at_three(self):
        nodes = [parse_node("Conv", ["image", "w", "b"], ["s"], kernel_shape=[1, 1])]
        previous = "s"
        for index in range(4):
            nodes.append(parse_node("Conv", [previous, "w", "b"], ["a%d" % index], kernel_shape=[1, 1]))
            previous = "a%d" % index
        nodes += [parse_node("MaxPool", [previous], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
                  parse_node("Conv", ["s", "w", "b"], ["d"], kernel_shape=[1, 1]),
                  parse_node("MaxPool", ["d"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
                  parse_node("Add", ["p0", "p1"], ["j"])]
        self.assertIsNone(parse_pooled_branches(nodes))

    def test_duplicate_producer_names_are_rejected(self):
        nodes = [parse_node("Conv", ["image", "w", "b"], ["s"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["s", "w", "b"], ["x"], kernel_shape=[1, 1]),
                 parse_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["y"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Conv", ["s", "w", "b"], ["x"], kernel_shape=[1, 1]),
                 parse_node("MaxPool", ["x"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
                 parse_node("Add", ["p0", "p1"], ["j"])]
        self.assertIsNone(parse_pooled_branches(nodes))


class PooledBranchesRejectionTests(VariantCase):
    """Every pooled-branches validation branch, paired with a compiling neighbour."""

    def test_unparseable_graph_is_rejected(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]])
        compile_pooled_branches(model)
        broken = h.make_model(
            h.make_graph([h.make_node("Conv", ["image", "w", "b"], ["output"], kernel_shape=[1, 1])],
                         "m", [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                         [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                          nh.from_array(np.zeros(3, np.float32), "b")]),
            opset_imports=[h.make_opsetid("", 13)])
        broken.ir_version = 8
        with self.assertRaisesRegex(ValueError, "1x1 stem, two or three"):
            compile_pooled_branches(broken)

    def test_default_domain_is_required(self):
        def tweak(graph):
            graph.node[0].domain = "com.custom"
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak,
                                         extra_opsets=[h.make_opsetid("com.custom", 1)])
        with self.assertRaisesRegex(ValueError, "default-domain nodes"):
            compile_pooled_branches(model)

    def test_external_io_is_checked(self):
        def tweak(graph):
            graph.output.append(h.make_tensor_value_info("stem0", 1, [1, 3, 8, 8]))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "one input, one output and a final join"):
            compile_pooled_branches(model)

    def test_external_tensors_must_be_float32(self):
        def tweak(graph):
            graph.input[0].type.tensor_type.elem_type = 2
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "external tensors must be float32"):
            compile_pooled_branches(model)

    def test_branch_tensor_shapes_are_checked(self):
        def tweak(graph):
            for info in graph.value_info:
                if info.name == "b0_0":
                    info.type.tensor_type.shape.dim[2].dim_value = 7
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "branch tensors must be float32 1xCx8x8"):
            compile_pooled_branches(model)

    def test_pool_attributes_are_checked(self):
        def tweak(graph):
            pool = next(node for node in graph.node if node.op_type == "MaxPool")
            pool.attribute.remove(next(a for a in pool.attribute if a.name == "kernel_shape"))
            pool.attribute.append(h.make_attribute("kernel_shape", [3, 3]))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "2x2 stride-2 MaxPool/AveragePool"):
            compile_pooled_branches(model)

    def test_stem_shape_is_checked(self):
        def tweak(graph):
            set_initializer(graph, "w_stem", np.zeros((3, 1, 1, 1), np.float32))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "hidden channels 3..16"):
            compile_pooled_branches(model)

    def test_stem_attributes_are_checked(self):
        def tweak(graph):
            graph.node[0].attribute.append(h.make_attribute("strides", [2, 2]))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported stem attributes"):
            compile_pooled_branches(model)

    def test_stem_relu_must_consume_the_stem(self):
        def tweak(graph):
            graph.node[0].output[0] = "c1"
            graph.node[1].input[0] = "image"
            value_info(graph, "c1", (1, 3, 8, 8))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], stem_relu=True, tweak=tweak)
        with self.assertRaisesRegex(ValueError, "stem Relu must consume the stem output"):
            compile_pooled_branches(model)

    def test_layer_chaining_is_checked(self):
        def tweak(graph):
            index = next(i for i, node in enumerate(graph.node) if node.output[0] == "b0_0")
            graph.initializer.append(nh.from_array(np.zeros((4, 4, 1, 1), np.float32), "wx"))
            graph.initializer.append(nh.from_array(np.zeros(4, np.float32), "bsx"))
            graph.node.insert(index + 1, h.make_node(
                "Conv", ["b0_0", "wx", "bsx"], ["b0_x"], kernel_shape=[1, 1]))
            value_info(graph, "b0_x", (1, 4, 8, 8))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "chain off the previous layer"):
            compile_pooled_branches(model)

    def test_constant_float32_layers_are_required(self):
        def tweak(graph):
            set_initializer(graph, "w0_0", np.zeros((4, 3, 1, 1), np.float64))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "constant float32 weights and bias"):
            compile_pooled_branches(model)

    def test_chained_depthwise_layers_are_checked(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 3, 1), (DEPTHWISE, 3, 3, 2)],
                                          [(DENSE, 3, 3, 1)]])
        with self.assertRaisesRegex(ValueError, "chained depthwise layers need group C"):
            compile_pooled_branches(model)

    def test_dense_layer_shape_is_checked(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 3, 2), (DENSE, 3, 3, 1)],
                                          [(DENSE, 3, 3, 1)]])
        with self.assertRaisesRegex(ValueError, "C1..16 1x1/3x3"):
            compile_pooled_branches(model)

    def test_layer_attributes_are_checked(self):
        def tweak(graph):
            graph.node[1].attribute.append(h.make_attribute("strides", [2, 2]))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported layer attributes"):
            compile_pooled_branches(model)

    def test_group_is_dense_or_matching_depthwise(self):
        def tweak(graph):
            graph.node[1].attribute.append(h.make_attribute("group", 2))
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "dense or depthwise group C"):
            compile_pooled_branches(model)

    def test_branch_must_end_with_three_channels(self):
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1)],
                                          [(DENSE, 3, 3, 1), (DENSE, 3, 3, 1)]])
        with self.assertRaisesRegex(ValueError, "end each chain with three channels"):
            compile_pooled_branches(model)

    def test_join_operand_order_is_checked(self):
        def tweak(graph):
            join = graph.node[-1]
            join.input[0], join.input[1] = join.input[1], join.input[0]
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]], tweak=tweak)
        with self.assertRaisesRegex(ValueError, "fold the pooled branches in order"):
            compile_pooled_branches(model)

    def test_mixed_pool_kinds_guard_is_reachable(self):
        """`parse_pooled_branches` rejects mixed pools; the later guard still fires."""
        model, _ = pooled_branches_model([[(DENSE, 3, 4, 1), (DENSE, 4, 3, 1)],
                                          [(DENSE, 3, 3, 1)]])
        spec = parse_pooled_branches(list(model.graph.node))
        mixed = dict(spec)
        pools = list(spec["pools"])
        tail = pools[1]
        pools[1] = h.make_node("AveragePool", list(tail.input), list(tail.output),
                               kernel_shape=[2, 2], strides=[2, 2])
        mixed["pools"] = pools
        with mock.patch.object(pooled_branches_module, "parse_pooled_branches",
                               lambda nodes: mixed):
            with self.assertRaisesRegex(ValueError, "matching pool kinds"):
                compile_pooled_branches(model)


# ---------------------------------------------------------------------------
# chain: the legacy two-layer Conv -> Relu -> Conv container
# ---------------------------------------------------------------------------


def chain_model(hidden=4, k1=1, k2=1, seed=12, tweak=None, extra_opsets=(), relu=True):
    rng = np.random.default_rng(hidden * 3 + k1 * 5 + k2 * 7 + seed)
    constants = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, k1, k1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b1"),
                 nh.from_array(rng.uniform(-.7, .8, (3, hidden, k2, k2)).astype(np.float32), "w2"),
                 nh.from_array(rng.uniform(-2, 2, (3,)).astype(np.float32), "b2")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[k1, k1],
                         pads=[k1 // 2] * 4)]
    source = "c1"
    if relu:
        nodes.append(h.make_node("Relu", ["c1"], ["r"]))
        source = "r"
    nodes.append(h.make_node("Conv", [source, "w2", "b2"], ["output"], kernel_shape=[k2, k2],
                             pads=[k2 // 2] * 4))
    graph = h.make_graph(nodes, "chain", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    if tweak is not None:
        tweak(graph)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)] + list(extra_opsets))
    model.ir_version = 8
    return model, dict(w1=nh.to_array(constants[0]), b1=nh.to_array(constants[1]),
                       w2=nh.to_array(constants[2]), b2=nh.to_array(constants[3]))


def chain_expected(inputs, meta, params, k1, k2):
    """Float64 two-layer chain: the container pads an internal grid with code -128."""
    first = quant(meta["first"])
    second = quant(meta["second"])
    real = conv2d(to_real(inputs, 1.0, 0), dequant_weights(first),
                  np.asarray(params["b1"], np.float64), (k1 // 2,) * 4)
    real = np.maximum(real, 0.0)
    codes = requantize(real, first.output_scale, first.output_zero_point)
    border = (-128 - first.output_zero_point) * first.output_scale
    second_real = conv2d_code_border(to_real(codes, first.output_scale, first.output_zero_point),
                                     dequant_weights(second), np.asarray(params["b2"], np.float64),
                                     k2 // 2, border)
    return requantize(second_real, second.output_scale, second.output_zero_point)


class ChainSemanticsTests(VariantCase):
    """Accepted two-layer chains, band reuse and the independent float64 output."""

    def test_kernel_matrix(self):
        for k1, k2 in ((1, 1), (1, 3), (3, 1), (3, 3)):
            with self.subTest(k1=k1, k2=k2):
                model, params = chain_model(hidden=5, k1=k1, k2=k2)
                _, meta = compile_sequence_model(model)
                self.assertEqual(meta["profile"], 2)
                self.assertEqual(meta["sequence_profile"], "Conv-Relu-Conv")
                self.assertEqual(meta["shape_nhwc"], [1, 8, 8, 3])
                self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 3])
                self.assertEqual(len(meta["first"]["quantization"]["weights"]), 5)
                inputs = sample_grid(89 + k1 + k2)
                got = chain_n_reference(inputs, [meta["first"], meta["second"]])
                self.assert_codes(got, chain_expected(inputs, meta, params, k1, k2), 1,
                                  "chain %d %d" % (k1, k2))
                self.assert_non_degenerate(got, "chain")

    def test_identity_chain_is_integer_exact(self):
        model, params = chain_model(hidden=3, k1=1, k2=1)
        identity = np.zeros((3, 3, 1, 1), np.float32)
        for channel in range(3):
            identity[channel, channel, 0, 0] = 1.0
        for graph_tensor, array in (("w1", identity), ("w2", identity),
                                    ("b1", np.zeros(3, np.float32)),
                                    ("b2", np.zeros(3, np.float32))):
            set_initializer(model.graph, graph_tensor, array)
        params.update(w1=identity, b1=np.zeros(3, np.float32),
                      w2=identity, b2=np.zeros(3, np.float32))
        _, meta = compile_sequence_model(model)
        inputs = sample_grid(97)
        got = chain_n_reference(inputs, [meta["first"], meta["second"]])
        expected = np.clip(inputs.astype(np.int64) - 128, -128, 127)
        self.assert_codes(got, expected, 0, "double identity")
        self.assert_codes(got, chain_expected(inputs, meta, params, 1, 1), 0, "independent identity")

    def test_weight_scale_reuse_for_a_zero_channel(self):
        """A bias-only output channel reuses the maximum weight scale (line 43)."""
        weights = np.array([[[[0.0]]], [[[0.5]]], [[[-0.25]]]], np.float32)
        bias = np.array([1.0, 0.5, -0.25], np.float32)
        q = native_quantize(weights, bias, 1.0, 0, None)
        maximum = float(np.max(q.weight_scales))
        self.assertEqual(float(q.weight_scales[0]), maximum)
        self.assertEqual(int(q.channel_multipliers[0]), 16384)
        np.testing.assert_array_equal(q.weights[0], np.zeros(1, np.int64) + int(q.weight_zero_points[0]))
        # The same reuse survives a compiled chain whose head has a dead output channel.
        model, _ = chain_model(hidden=4, k1=1, k2=1)
        for tensor in model.graph.initializer:
            if tensor.name == "w2":
                head = np.array(nh.to_array(tensor))
                head[0] = 0.0
                tensor.CopyFrom(nh.from_array(head, "w2"))
        _, meta = compile_sequence_model(model)
        scales = np.asarray(meta["second"]["weight_scales"], np.float32)
        self.assertEqual(float(scales[0]), float(np.max(scales)))
        self.assertEqual(int(np.asarray(meta["second"]["channel_multipliers"])[0]), 16384)

    def test_internal_padding_code_is_minus_128(self):
        """The first layer's Relu band always has zero point -128, so padding is real zero."""
        model, params = chain_model(hidden=6, k1=3, k2=3)
        _, meta = compile_sequence_model(model)
        self.assertEqual(meta["first"]["output_zero_point"], -128)
        inputs = sample_grid(101)
        got = chain_n_reference(inputs, [meta["first"], meta["second"]])
        self.assert_codes(got, chain_expected(inputs, meta, params, 3, 3), 1, "internal pad1")


class ChainRejectionTests(VariantCase):
    """`native_quantize` and `compile_chain` rejection branches."""

    def test_native_quantize_parameter_rejections(self):
        unit = np.ones((1, 1, 1, 1), np.float32)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            native_quantize(unit, [0.0], 0.0, 0)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            native_quantize(unit, [0.0], float("nan"), 0)
        # The valid neighbour with healthy parameters compiles.
        self.assertGreater(native_quantize(np.ones((2, 3, 1, 1), np.float32),
                                           np.zeros(2, np.float32), 1.0, 0).output_scale, 0)

    def test_native_quantize_bias_overflow(self):
        with self.assertRaisesRegex(ValueError, "native INT32 bias overflow"):
            native_quantize(np.zeros((1, 1, 1, 1), np.float32), [1e6], 1.0, 0)
        self.assertLess(abs(int(native_quantize(np.zeros((1, 1, 1, 1), np.float32),
                                                [1.0], 1.0, 0).biases[0])), 1 << 31)

    def test_native_quantize_accumulator_guard(self):
        weights = np.full((1, 3, 3, 3), -1.0, np.float32)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            native_quantize(weights, [2145983648 / 255.0], 1.0, 255)
        self.assertLess(abs(int(native_quantize(weights, [0.0], 1.0, 255).biases[0])), 1 << 31)

    def test_native_quantize_output_scale_guard(self):
        with np.errstate(over="ignore"):
            with self.assertRaisesRegex(ValueError, "invalid native output scale"):
                native_quantize(np.full((1, 1, 1, 1), 3e38, np.float32), [0.0], 1.0, 0)
        self.assertGreater(native_quantize(np.ones((1, 1, 1, 1), np.float32),
                                           [0.0], 1.0, 0).output_scale, 0)

    def test_native_quantize_shift_guard(self):
        weights = np.random.default_rng(0).uniform(-1, 1, (1, 3, 1, 1)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            native_quantize(weights, [0.0], 1.0, 0, {"scale": 1e-7, "zero_point": 0})
        self.assertGreater(native_quantize(weights, [0.0], 1.0, 0).output_scale, 0)

    def test_node_list_must_be_conv_relu_conv(self):
        def tweak(graph):
            graph.node[1].op_type = "Conv"
            graph.node[1].input[:] = ["c1", "w2", "b2"]
            graph.node[1].output[0] = "r"
        model, _ = chain_model(tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_default_domain_is_required(self):
        def tweak(graph):
            graph.node[0].domain = "com.custom"
        model, _ = chain_model(tweak=tweak, extra_opsets=[h.make_opsetid("com.custom", 1)])
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_external_io_and_relu_attributes_are_checked(self):
        def two_outputs(graph):
            graph.output.append(h.make_tensor_value_info("r", 1, [1, 4, 8, 8]))
        model, _ = chain_model(tweak=two_outputs)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_activation_wiring_is_checked(self):
        def tweak(graph):
            graph.node[2].input[0] = "c1"
        model, _ = chain_model(tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_external_names_are_checked(self):
        def tweak(graph):
            graph.output[0].name = "r"
        model, _ = chain_model(tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_bias_is_required_on_both_layers(self):
        def tweak(graph):
            del graph.node[2].input[2]
        model, _ = chain_model(tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_hidden_channels_are_bounded(self):
        model, _ = chain_model(hidden=2)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_constants_must_be_float32(self):
        def tweak(graph):
            set_initializer(graph, "b2", np.zeros(3, np.float64))
        model, _ = chain_model(tweak=tweak)
        with self.assertRaisesRegex(ValueError, "unsupported two-layer graph"):
            compile_chain_model(model)

    def test_calibration_and_output_override_cannot_be_combined(self):
        model, _ = chain_model()
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            compile_chain_model(model, calibration_ranges={"r": {"scale": 0.1, "zero_point": 0}},
                                output_range={"scale": 0.2, "zero_point": 0})
        # Each override alone is accepted.
        payload, meta = compile_chain_model(model, output_range={"scale": 0.2, "zero_point": 0})
        self.assertEqual(meta["output_scale"], np.float32(0.2))
        self.assertEqual(len(payload), 8192)


if __name__ == "__main__":
    unittest.main()
