"""SPDX-License-Identifier: MIT

Host tests for the two front-end lowerings in `open_rknpu.normalize`:

* F1 - a dense `MatMul`/`Gemm` becomes the verified 1x1 `Conv` (`_lower_dense_layers`);
* F2 - a rank-3 `[N, C, L]` convolution graph becomes its `H=1` 2-D form
  (`_promote_rank_three`).

Both are pure ONNX rewrites into shapes the emitters already accept, so they are pinned
the same way the rest of the front end is: the lowered model must compile to *the same
bytes* as the hand-written equivalent, the profile's own integer reference must reproduce
the container's output, and an unsupported form must keep the scheduler's explicit
`ValueError` rather than being lowered halfway.
"""
import re
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.native import native_input_reference
from open_rknpu.normalize import normalize_model
from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

# The scheduler's rejection for a graph whose first node is not a Conv. Both lowerings are
# front-end rewrites, so every unsupported dense/1-D form must still reach this message.
SEQUENCE_REJECTION = "sequence lowering requires one input, one output, and an initial Conv"


def build(nodes, inputs, outputs, initializers, name="lowering"):
    graph = h.make_graph(nodes, name, inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def conv_model(weights, input_shape, output_shape, bias=None):
    """The hand-written 1x1 Conv a lowered dense layer must reproduce byte for byte."""
    inputs = ["input", "w"] + (["b"] if bias is not None else [])
    initializers = [nh.from_array(weights, "w")]
    if bias is not None:
        initializers.append(nh.from_array(bias, "b"))
    return build([h.make_node("Conv", inputs, ["output"], kernel_shape=[1, 1])],
                 [h.make_tensor_value_info("input", 1, list(input_shape))],
                 [h.make_tensor_value_info("output", 1, list(output_shape))], initializers)


def quantized(meta):
    return Quantization(**{key: (np.array(value) if isinstance(value, list) else value)
                           for key, value in meta["quantization"].items()})


def round_output(acc, quant):
    """The profile's two-stage channel/global rounding, written out for the test oracle."""
    if quant.relu:
        acc = np.maximum(acc, 0)
    product = acc * quant.channel_multipliers
    scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    product = scaled * quant.multiplier
    if quant.shift:
        product += quant.output_zero_point << quant.shift
        result = (product + (1 << (quant.shift - 1)) - 1 + ((product >> quant.shift) & 1)) >> quant.shift
    else:
        result = product + quant.output_zero_point
    return np.clip(result, -128, 127).astype(np.int8)


def dense_oracle(case, quant):
    """`sum_c x[c] * W[k, c] + b[k]` over one [H, W, C] case, from the container's numbers."""
    x = case.reshape(-1, case.shape[-1]).astype(np.int64) - 128
    centered = quant.weights - quant.weight_zero_points[:, None]
    return round_output(x @ centered.T + quant.biases, quant).reshape(
        case.shape[0], case.shape[1], -1)


def conv1d_oracle(case, quant):
    """The promoted 1-D window arithmetic, independent of `native_input_reference`.

    The H=1 2-D form pads `[0, 1, 0, 1]` and its square-embedded kernel keeps the taps in
    row 0, so `out[k, l] = sum_t sum_c x[c, l + t - 1] * kernel[k, c, 0, t]`.
    """
    length, channels = case.shape[1], case.shape[2]
    kernel = quant.kernel_size
    centered = (quant.weights.reshape(-1, channels, kernel, kernel)
                - quant.weight_zero_points[:, None, None, None])
    # The border injects the input zero point, exactly like `native_input_reference`'s
    # `constant_values=zero_point-128`; the raw image origin would be a different value.
    edge = np.full((1, channels), quant.input_zero_point - 128, np.int64)
    data = case.reshape(length, channels).astype(np.int64) - 128
    padded = np.concatenate([edge, data, edge], axis=0)
    taps = np.stack([padded[0:length], padded[1:length + 1], padded[2:length + 2]], axis=1)
    acc = np.einsum("ltc,kct->lk", taps, centered[:, :, 0, :]) + quant.biases
    return round_output(acc, quant).reshape(1, length, -1)


def deterministic_cases(shape):
    """At least two fixed cases: all-zero, all-255 and a ramp."""
    count = int(np.prod(shape))
    ramp = (np.arange(count, dtype=np.int64) * 37 % 256).astype(np.uint8).reshape(shape)
    return [np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8), ramp]


class DenseLoweringTests(unittest.TestCase):
    """F1: `MatMul`/`Gemm` lowered to the verified 1x1 Conv path."""

    def setUp(self):
        rng = np.random.default_rng(110361)
        self.matrix = rng.uniform(-.8, .8, (4, 8, 1, 1)).astype(np.float32)
        self.operand = self.matrix[:, :, 0, 0].T.copy()          # [C, K] = [8, 4]
        self.bias = rng.uniform(-1, 1, 4).astype(np.float32)
        self.conv = conv_model(self.matrix, [1, 8, 1, 1], [1, 4, 1, 1])
        self.conv_bias = conv_model(self.matrix, [1, 8, 1, 1], [1, 4, 1, 1], self.bias)

    def flattened(self):
        return build([h.make_node("Flatten", ["input"], ["flat"], axis=1),
                      h.make_node("MatMul", ["flat", "b"], ["output"])],
                     [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                     [h.make_tensor_value_info("output", 1, [1, 4])],
                     [nh.from_array(self.operand, "b")], name="flatten_matmul")

    def lowerings(self):
        """Every supported dense spelling, with the hand-written Conv it must equal."""
        reshaped = build([h.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
                          h.make_node("MatMul", ["flat", "b"], ["output"])],
                         [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                         [h.make_tensor_value_info("output", 1, [1, 4])],
                         [nh.from_array(self.operand, "b"),
                          nh.from_array(np.array([0, -1], np.int64), "flat_shape")], name="reshape_matmul")
        literal = build([h.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
                         h.make_node("MatMul", ["flat", "b"], ["output"])],
                        [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                        [h.make_tensor_value_info("output", 1, [1, 4])],
                        [nh.from_array(self.operand, "b"),
                         nh.from_array(np.array([1, 8], np.int64), "flat_shape")], name="literal_matmul")
        return [
            ("MatMul on the [N, C, 1, 1] map", build(
                [h.make_node("MatMul", ["input", "b"], ["output"])],
                [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                [nh.from_array(self.operand, "b")]), self.conv),
            ("Flatten(axis=1) -> MatMul", self.flattened(), self.conv),
            ("constant Reshape [0, -1] -> MatMul", reshaped, self.conv),
            ("constant Reshape [1, 8] -> MatMul", literal, self.conv),
            ("Gemm with bias", build(
                [h.make_node("Gemm", ["input", "b", "bias"], ["output"])],
                [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                [nh.from_array(self.operand, "b"), nh.from_array(self.bias, "bias")]), self.conv_bias),
            ("Flatten -> Gemm transB=1 with bias", build(
                [h.make_node("Flatten", ["input"], ["flat"], axis=1),
                 h.make_node("Gemm", ["flat", "bt", "bias"], ["output"], transB=1)],
                [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                [h.make_tensor_value_info("output", 1, [1, 4])],
                [nh.from_array(self.operand.T.copy(), "bt"),
                 nh.from_array(self.bias, "bias")]), self.conv_bias),
            ("Reshape -> Gemm without bias", build(
                [h.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
                 h.make_node("Gemm", ["flat", "b"], ["output"])],
                [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                [h.make_tensor_value_info("output", 1, [1, 4])],
                [nh.from_array(self.operand, "b"),
                 nh.from_array(np.array([1, 8], np.int64), "flat_shape")]), self.conv),
        ]

    def test_dense_forms_compile_to_the_verified_conv(self):
        for label, dense, reference in self.lowerings():
            with self.subTest(form=label):
                binary, meta = compile_sequence(dense)
                info = decode_sequence(binary)
                self.assertEqual(meta["profile"], "native16-input")
                self.assertEqual(info["shape_nhwc"], [1, 1, 1, 8])
                self.assertEqual(info["output_shape_nhwc"], [1, 1, 1, 4])
                self.assertEqual(len(binary), 1328)   # the maintainer's verified probe
                self.assertEqual(binary, compile_sequence(reference)[0])

    def test_lowering_drops_the_flatten_and_leaves_one_conv(self):
        normalized = normalize_model(self.flattened())
        self.assertEqual([node.op_type for node in normalized.graph.node], ["Conv"])
        self.assertEqual(normalized.graph.node[0].input[0], "input")
        self.assertEqual([d.dim_value for d in normalized.graph.output[0].type.tensor_type.shape.dim],
                         [1, 4, 1, 1])
        weights = {tensor.name: nh.to_array(tensor) for tensor in normalized.graph.initializer}
        np.testing.assert_array_equal(weights[normalized.graph.node[0].input[1]], self.matrix)

    def test_dense_forms_accept_terminal_rank_two_outputs(self):
        # The valid ONNX MatMul output is [N, K]; the same bytes are the Conv's [N, K, 1, 1].
        for label, dense, reference in self.lowerings():
            with self.subTest(form=label):
                declared = dense.graph.output[0].type.tensor_type.shape
                self.assertIn(len(declared.dim), (2, 4))
                self.assertEqual(compile_sequence(dense)[0], compile_sequence(reference)[0])

    def test_integer_reference_reproduces_the_container(self):
        for label, dense, _ in self.lowerings():
            with self.subTest(form=label):
                _, meta = compile_sequence(dense)
                quant = quantized(meta)
                cases = deterministic_cases((1, 1, 8))
                self.assertGreaterEqual(len(cases), 2)
                for case in cases:
                    expected = native_input_reference(
                        case, quant, meta["input_zero_point"], pads=meta["conv_pads"],
                        strides=tuple(meta["conv_strides"]),
                        dilations=tuple(meta["conv_dilations"]))
                    self.assertEqual(expected.shape, (1, 1, 4))
                    np.testing.assert_array_equal(expected, dense_oracle(case, quant))

    def test_weight_and_bias_names_avoid_initializer_collisions(self):
        colliding = build([h.make_node("Gemm", ["input", "b", "bias"], ["output"])],
                          [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                          [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                          [nh.from_array(self.operand, "b"), nh.from_array(self.bias, "bias"),
                           nh.from_array(np.zeros(1, np.float32), "output_dense_weight"),
                           nh.from_array(np.zeros(1, np.float32), "output_dense_bias")],
                          name="collide")
        self.assertEqual(compile_sequence(colliding)[0], compile_sequence(self.conv_bias)[0])
        names = {tensor.name for tensor in normalize_model(colliding).graph.initializer}
        self.assertIn("output_dense_weight_", names)
        self.assertIn("output_dense_bias_", names)

    def test_unsupported_dense_forms_keep_the_explicit_rejection(self):
        alpha = build([h.make_node("Gemm", ["input", "b"], ["output"], alpha=2.0)],
                      [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                      [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                      [nh.from_array(self.operand, "b")], name="alpha")
        trans_a = build([h.make_node("Gemm", ["input", "b"], ["output"], transA=1)],
                        [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                        [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                        [nh.from_array(self.operand, "b")], name="trans_a")
        external = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                         [h.make_tensor_value_info("input", 1, [1, 8, 1, 1]),
                          h.make_tensor_value_info("b", 1, [8, 4])],
                         [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])], [], name="external")
        spatial = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                        [h.make_tensor_value_info("input", 1, [1, 8, 2, 2])],
                        [h.make_tensor_value_info("output", 1, [1, 4, 2, 2])],
                        [nh.from_array(self.operand, "b")], name="spatial")
        rank_two = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                         [h.make_tensor_value_info("input", 1, [1, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 4])],
                         [nh.from_array(self.operand, "b")], name="rank_two")
        rank_three_b = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                             [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                             [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                             [nh.from_array(np.zeros((8, 4, 1), np.float32), "b")], name="rank_three_b")
        double = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                       [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                       [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                       [nh.from_array(np.zeros((8, 4), np.float64), "b")], name="float64")
        bad_reshape = build([h.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
                             h.make_node("MatMul", ["flat", "b"], ["output"])],
                            [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                            [h.make_tensor_value_info("output", 1, [1, 4])],
                            [nh.from_array(self.operand, "b"),
                             nh.from_array(np.array([8, 1], np.int64), "flat_shape")], name="bad_reshape")
        two_minus_one = build([h.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
                               h.make_node("MatMul", ["flat", "b"], ["output"])],
                              [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                              [h.make_tensor_value_info("output", 1, [1, 4])],
                              [nh.from_array(self.operand, "b"),
                               nh.from_array(np.array([-1, -1], np.int64), "flat_shape")], name="two_minus_one")
        shared = build([h.make_node("Flatten", ["input"], ["flat"], axis=1),
                        h.make_node("MatMul", ["flat", "b"], ["output"]),
                        h.make_node("Identity", ["flat"], ["also"])],
                       [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                       [h.make_tensor_value_info("output", 1, [1, 4]),
                        h.make_tensor_value_info("also", 1, [1, 8])],
                       [nh.from_array(self.operand, "b")], name="shared")
        external_bias = build([h.make_node("Gemm", ["input", "b", "bias"], ["output"])],
                              [h.make_tensor_value_info("input", 1, [1, 8, 1, 1]),
                               h.make_tensor_value_info("bias", 1, [4])],
                              [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                              [nh.from_array(self.operand, "b")], name="external_bias")
        mismatch = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                         [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                         [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                         [nh.from_array(np.zeros((4, 4), np.float32), "b")], name="mismatch")
        non_terminal = build([h.make_node("MatMul", ["input", "b"], ["hidden"]),
                              h.make_node("Relu", ["hidden"], ["output"])],
                             [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                             [h.make_tensor_value_info("output", 1, [1, 4, 1, 1])],
                             [nh.from_array(self.operand, "b")], name="non_terminal")
        bad_output = build([h.make_node("MatMul", ["input", "b"], ["output"])],
                           [h.make_tensor_value_info("input", 1, [1, 8, 1, 1])],
                           [h.make_tensor_value_info("output", 1, [1, 5, 1, 1])],
                           [nh.from_array(self.operand, "b")], name="bad_output")
        spatial_flatten = build([h.make_node("Flatten", ["input"], ["flat"], axis=1),
                                 h.make_node("MatMul", ["flat", "b"], ["output"])],
                                [h.make_tensor_value_info("input", 1, [1, 3, 2, 2])],
                                [h.make_tensor_value_info("output", 1, [1, 4])],
                                [nh.from_array(np.zeros((12, 4), np.float32), "b")],
                                name="spatial_flatten")
        dynamic_shape = build([h.make_node("Reshape", ["input", "shape"], ["flat"]),
                               h.make_node("MatMul", ["flat", "b"], ["output"])],
                              [h.make_tensor_value_info("input", 1, [1, 8, 1, 1]),
                               h.make_tensor_value_info("shape", onnx.TensorProto.INT64, [2])],
                              [h.make_tensor_value_info("output", 1, [1, 4])],
                              [nh.from_array(self.operand, "b")], name="dynamic_shape")
        for label, dense in [("external MatMul operand", external),
                             ("spatial activation", spatial),
                             ("rank-2 activation without a flatten", rank_two),
                             ("rank-3 operand", rank_three_b),
                             ("non-float32 operand", double),
                             ("channel count that does not match the layer", mismatch),
                             ("a dense layer that is not the graph output", non_terminal),
                             ("declared output shape that does not match", bad_output),
                             ("Gemm alpha != 1", alpha),
                             ("Gemm transA != 0", trans_a),
                             ("external Gemm bias", external_bias),
                             ("Reshape target that is not [N, C]", bad_reshape),
                             ("Reshape with two inferred dimensions", two_minus_one),
                             ("Reshape with a dynamic shape tensor", dynamic_shape),
                             ("flatten of a spatial feature map", spatial_flatten),
                             ("flatten with a second consumer", shared)]:
            with self.subTest(form=label):
                with self.assertRaisesRegex(ValueError, re.escape(SEQUENCE_REJECTION)):
                    compile_sequence(dense)


class Conv1DLoweringTests(unittest.TestCase):
    """F2: a rank-3 `[N, C, L]` Conv graph promoted to the `H=1` 2-D form."""

    def setUp(self):
        rng = np.random.default_rng(110362)
        self.weights_1d = rng.uniform(-.8, .8, (4, 3, 3)).astype(np.float32)
        self.weights_2d = self.weights_1d.reshape(4, 3, 1, 3)

    def rank_three(self, **attrs):
        return build([h.make_node("Conv", ["input", "w"], ["output"], **attrs)],
                     [h.make_tensor_value_info("input", 1, [1, 3, 16])],
                     [h.make_tensor_value_info("output", 1, [1, 4, 16])],
                     [nh.from_array(self.weights_1d, "w")], name="conv1d")

    def rank_two(self):
        return build([h.make_node("Conv", ["input", "w"], ["output"],
                                  kernel_shape=[1, 3], pads=[0, 1, 0, 1])],
                     [h.make_tensor_value_info("input", 1, [1, 3, 1, 16])],
                     [h.make_tensor_value_info("output", 1, [1, 4, 1, 16])],
                     [nh.from_array(self.weights_2d, "w")], name="conv2d")

    def test_rank_three_conv_is_byte_identical_to_its_two_dimensional_form(self):
        binary, meta = compile_sequence(self.rank_three(kernel_shape=[3], pads=[1, 1]))
        info = decode_sequence(binary)
        self.assertEqual(meta["profile"], "native16-input")
        self.assertEqual(info["shape_nhwc"], [1, 1, 16, 3])
        self.assertEqual(info["output_shape_nhwc"], [1, 1, 16, 4])
        self.assertEqual(len(binary), 1840)                        # the maintainer's verified probe
        self.assertEqual(meta["conv_pads"], [0, 1, 2, 1])          # [0, a, 0, b], then square embedded
        self.assertEqual(binary, compile_sequence(self.rank_two())[0])

    def test_promotion_rewrites_every_rank_one_axis(self):
        graph = build([h.make_node("Conv", ["input", "w"], ["conv"], kernel_shape=[3], pads=[1, 1]),
                       h.make_node("Relu", ["conv"], ["relu"]),
                       h.make_node("MaxPool", ["relu"], ["pool"], kernel_shape=[2], strides=[2],
                                   pads=[0, 0], dilations=[1], auto_pad="NOTSET")],
                      [h.make_tensor_value_info("input", 1, [1, 3, 16])],
                      [h.make_tensor_value_info("pool", 1, [1, 4, 8])],
                      [nh.from_array(self.weights_1d, "w")], name="promote")
        normalized = normalize_model(graph)
        input_dims = [d.dim_value for d in normalized.graph.input[0].type.tensor_type.shape.dim]
        output_dims = [d.dim_value for d in normalized.graph.output[0].type.tensor_type.shape.dim]
        self.assertEqual(input_dims, [1, 3, 1, 16])
        self.assertEqual(output_dims, [1, 4, 1, 8])
        conv, pool = normalized.graph.node[0], normalized.graph.node[2]
        # `_promote_rank_three` writes the [1, 3] kernel and the [0, 1, 0, 1] pads; the
        # rest of `normalize_model` then square-embeds the rectangular 1x3 kernel exactly
        # as it does for a hand-written rank-4 Conv, leaving `w` as the promoted [4, 3, 1, 3].
        self.assertEqual({a.name: h.get_attribute_value(a) for a in conv.attribute},
                         {"kernel_shape": [3, 3], "pads": [0, 1, 2, 1]})
        self.assertEqual({a.name: h.get_attribute_value(a) for a in pool.attribute},
                         {"kernel_shape": [1, 2], "strides": [1, 2], "pads": [0, 0, 0, 0],
                          "dilations": [1, 1], "auto_pad": b"NOTSET"})
        weights = {tensor.name: nh.to_array(tensor) for tensor in normalized.graph.initializer}
        np.testing.assert_array_equal(weights["w"], self.weights_2d)

    def test_a_symbolic_output_extent_is_carried_through_the_promotion(self):
        graph = build([h.make_node("Conv", ["input", "w"], ["output"], kernel_shape=[3], pads=[1, 1])],
                      [h.make_tensor_value_info("input", 1, [1, 3, 16])],
                      [h.make_tensor_value_info("output", 1, [1, 4, "L"])],
                      [nh.from_array(self.weights_1d, "w")], name="symbolic")
        dims = normalize_model(graph).graph.output[0].type.tensor_type.shape.dim
        # The promotion inserts the unit height axis and keeps the symbolic extent; the
        # final `infer_shapes` then resolves `L` to the Conv's 16.
        self.assertEqual([d.dim_value for d in dims], [1, 4, 1, 16])
        self.assertEqual([d.dim_param for d in dims], ["", "", "", ""])

    def test_integer_reference_reproduces_the_container(self):
        _, meta = compile_sequence(self.rank_three(kernel_shape=[3], pads=[1, 1]))
        quant = quantized(meta)
        cases = deterministic_cases((1, 16, 3))
        self.assertGreaterEqual(len(cases), 2)
        for case in cases:
            expected = native_input_reference(case, quant, meta["input_zero_point"],
                                              pads=meta["conv_pads"],
                                              strides=tuple(meta["conv_strides"]),
                                              dilations=tuple(meta["conv_dilations"]))
            self.assertEqual(expected.shape, (1, 16, 4))
            np.testing.assert_array_equal(expected, conv1d_oracle(case, quant))

    def test_unpromotable_rank_three_graphs_are_rejected_by_name(self):
        reshaped = build([h.make_node("Conv", ["input", "w"], ["conv"], kernel_shape=[3], pads=[1, 1]),
                          h.make_node("Reshape", ["conv", "shape"], ["output"], name="flat")],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 4, 8])],
                         [nh.from_array(self.weights_1d, "w"),
                          nh.from_array(np.array([1, 4, 8], np.int64), "shape")], name="reshape")
        transposed = build([h.make_node("Conv", ["input", "w"], ["conv"], kernel_shape=[3], pads=[1, 1]),
                            h.make_node("Transpose", ["conv"], ["output"], perm=[0, 2, 1], name="swap")],
                           [h.make_tensor_value_info("input", 1, [1, 3, 8])],
                           [h.make_tensor_value_info("output", 1, [1, 8, 4])],
                           [nh.from_array(self.weights_1d, "w")], name="transpose")
        rank_two_output = build([h.make_node("Conv", ["input", "w"], ["output"],
                                             kernel_shape=[3], pads=[1, 1])],
                                [h.make_tensor_value_info("input", 1, [1, 3, 8])],
                                [h.make_tensor_value_info("output", 1, [4, 8])],
                                [nh.from_array(self.weights_1d, "w")], name="rank_two_output")
        computed_weight = build([h.make_node("Identity", ["w0"], ["w"]),
                                 h.make_node("Conv", ["input", "w"], ["output"],
                                             kernel_shape=[3], pads=[1, 1])],
                                [h.make_tensor_value_info("input", 1, [1, 3, 8])],
                                [h.make_tensor_value_info("output", 1, [1, 4, 8])],
                                [nh.from_array(self.weights_1d, "w0")], name="computed_weight")
        for label, graph, message in [
                ("a 1-D Reshape", reshaped, "1-D rank promotion cannot rewrite Reshape node 'flat'"),
                ("a Transpose", transposed, "1-D rank promotion cannot rewrite Transpose node 'swap'"),
                ("a rank-2 output", rank_two_output,
                 "1-D rank promotion requires rank-3 outputs; 'output' is rank 2"),
                ("a Conv whose weights are not a constant",
                 computed_weight,
                 "1-D rank promotion requires constant rank-3 weights for Conv node 'output'")]:
            with self.subTest(form=label):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    compile_sequence(graph)

    def test_rank_three_graphs_without_a_conv_keep_the_existing_rejection(self):
        softmax = build([h.make_node("Softmax", ["input"], ["output"])],
                        [h.make_tensor_value_info("input", 1, [1, 3, 8])],
                        [h.make_tensor_value_info("output", 1, [1, 3, 8])], [], name="softmax")
        with self.assertRaisesRegex(ValueError, re.escape(SEQUENCE_REJECTION)):
            compile_sequence(softmax)
        # `normalize_model` leaves the graph alone, so the scheduler still names the shape.
        normalized = normalize_model(softmax)
        self.assertEqual([d.dim_value for d in normalized.graph.input[0].type.tensor_type.shape.dim],
                         [1, 3, 8])


if __name__ == "__main__":
    unittest.main()
