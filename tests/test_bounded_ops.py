"""SPDX-License-Identifier: MIT

Checklist F6/F7 bounded front-end ops: the subset the hardware can actually express.

Three groups:

* **F6a - spatial mean.** A terminal `GlobalAveragePool` or `ReduceMean(axes=[2,3],
  keepdims=1)` on `[N,C,8,8]` is rewritten to the three-stage 8->4->2->1 average
  reduction the pooling engine already runs (legacy profiles 5/6). The emitted container
  must be byte-identical to the hand-written three-pool graph, and the reduction
  reference must replay the suite's expected bytes. The rewrite is deliberately
  approximate: each stage rounds to INT8 separately, so it is not an unrounded global
  mean.
* **F6b - sibling-Conv `Concat`.** `Concat(axis=1)` of Conv branches reading one shared
  graph input with identical geometry is a single wide Conv (stacked weights/biases).
  The emitted container must be byte-identical to the hand-written wide Conv.
* **F6c/F7 - measured negatives and the padding bound.** `Softmax`, `Slice` and `Resize`
  after a Conv keep their exact rejection; asymmetric/one-sided pads and rectangular
  kernels are already accepted by the geometry emitter and are pinned here against an
  independent explicit-pad oracle, while `pads >= K` and `K > 31` stay rejected.

No board, no network: the container's own quantization parameters and the recorded
`inputNNN.u8`/`expectedNNN.i8` are all the replay needs.
"""
from pathlib import Path
import json
import re
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.native import native_input_reference
from open_rknpu.normalize import normalize_model
from open_rknpu.quantization import Quantization, reference
from open_rknpu.reduction import reduction_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"

SEQUENCE_REJECTION = "sequence lowering currently supports Conv[/Relu] followed by 2x2 pooling"
POOL_REJECTION = "unsupported pooling attributes, shape, or graph connections"
NATIVE_GEOMETRY = "invalid native Conv output geometry"
NATIVE_KERNEL = ("native Conv supports odd K1..31, explicit padding, stride 1..4, "
                 "input C1..16352/output C1..8192")
BOUND = ("lowering supports exactly a terminal static [N,C,8,8] tensor with one input, "
         "one output and no other attributes")
CONCAT_BOUND = ("Concat lowering requires a terminal axis-1 Concat of two or more Conv "
                "branches reading one shared graph input with identical "
                "kernel/strides/pads/dilations/group")


def build(nodes, inputs, outputs, initializers, name="bounded"):
    graph = h.make_graph(nodes, name, inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def conv_stem(ic=3, oc=3, kernel=1, seed=110501):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.8, .8, (oc, ic, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-1, 1, oc).astype(np.float32)
    return weights, bias


def quantized(params):
    """A live `Quantization` from a container's own `quantization` metadata dict."""
    params = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                "channel_multipliers"):
        params[key] = np.array(params[key])
    return Quantization(**params)


def deterministic_cases(shape):
    count = int(np.prod(shape))
    ramp = (np.arange(count, dtype=np.int64) * 37 % 256).astype(np.uint8).reshape(shape)
    return [np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8), ramp]


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


def padded_conv_oracle(case, quant, pads, strides=(1, 1), dilations=(1, 1)):
    """The explicit-pad window arithmetic, independent of `native_input_reference`.

    `quantization.reference` uses same-padding, so the asymmetric case is written out
    here: the border injects the input zero-point code on each side and the window only
    slides over the padded plane.
    """
    pt, pl, pb, pr = pads
    kernel = quant.kernel_size
    ekh = (kernel - 1) * dilations[0] + 1
    ekw = (kernel - 1) * dilations[1] + 1
    edge = quant.input_zero_point - 128
    plane = np.pad(case.astype(np.int64) - 128, ((pt, pb), (pl, pr), (0, 0)),
                   constant_values=edge)
    patches = np.lib.stride_tricks.sliding_window_view(plane, (ekh, ekw), axis=(0, 1))
    patches = patches[::strides[0], ::strides[1], :, ::dilations[0], ::dilations[1]]
    centered = quant.weights - quant.weight_zero_points[:, None]
    acc = np.einsum("hwc,oc->hwo", patches.reshape(*patches.shape[:2], -1), centered)
    return round_output(acc + quant.biases, quant)


def pooled_oracle(case, quant, levels=3):
    """`levels` 2x2 average stages, spelled out independently of `pooling.pool_reference`."""
    codes = reference(case, quant).astype(np.int64)
    for _ in range(levels):
        height, width = codes.shape[0] // 2, codes.shape[1] // 2
        blocks = codes[:height * 2, :width * 2].reshape(height, 2, width, 2, codes.shape[2])
        codes = np.clip(np.rint(blocks.sum(axis=(1, 3)) / 4.0), -128, 127).astype(np.int8)
    return codes


def stem_quantization(meta):
    """The quantization the emitted stem task uses, from either container spelling."""
    if "quantization" in meta:
        return meta["quantization"]
    return meta["quantizations"][0]


# ---------------------------------------------------------------------------
# F6a: GlobalAveragePool / ReduceMean over the spatial axes
# ---------------------------------------------------------------------------


def global_pool_model(op="GlobalAveragePool", ic=3, oc=3, kernel=1, relu=False, seed=110501):
    weights, bias = conv_stem(ic, oc, kernel, seed)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    previous = "conv"
    if relu:
        nodes.append(h.make_node("Relu", [previous], ["relu"]))
        previous = "relu"
    attrs = {} if op == "GlobalAveragePool" else dict(axes=[2, 3], keepdims=1)
    nodes.append(h.make_node(op, [previous], ["output"], **attrs))
    return build(nodes, [h.make_tensor_value_info("input", 1, [1, ic, 8, 8])],
                 [h.make_tensor_value_info("output", 1, [1, oc, 1, 1])],
                 [nh.from_array(weights, "w"), nh.from_array(bias, "b")])


def three_pool_model(ic=3, oc=3, kernel=1, relu=False, seed=110501, kind="AveragePool"):
    weights, bias = conv_stem(ic, oc, kernel, seed)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    previous = "conv"
    if relu:
        nodes.append(h.make_node("Relu", [previous], ["relu"]))
        previous = "relu"
    for stage in range(3):
        output = "output" if stage == 2 else "pool%d" % stage
        nodes.append(h.make_node(kind, [previous], [output], kernel_shape=[2, 2],
                                 strides=[2, 2]))
        previous = output
    return build(nodes, [h.make_tensor_value_info("input", 1, [1, ic, 8, 8])],
                 [h.make_tensor_value_info("output", 1, [1, oc, 1, 1])],
                 [nh.from_array(weights, "w"), nh.from_array(bias, "b")])


def saved(model, folder, name):
    path = Path(folder) / ("%s.onnx" % name)
    onnx.save(model, path)
    return path


class GlobalPoolLoweringTests(unittest.TestCase):
    """F6a: an 8x8 spatial mean is the verified three-stage average reduction."""

    def test_global_average_pool_and_reduce_mean_are_byte_identical_to_three_pools(self):
        for op, relu in (("GlobalAveragePool", False), ("GlobalAveragePool", True),
                         ("ReduceMean", False), ("ReduceMean", True)):
            with self.subTest(op=op, relu=relu):
                rewritten, _ = compile_sequence(global_pool_model(op, relu=relu))
                hand, _ = compile_sequence(three_pool_model(relu=relu))
                self.assertEqual(rewritten, hand)

    def test_the_rewrite_emits_three_chained_average_pools(self):
        normalized = normalize_model(global_pool_model("ReduceMean"))
        self.assertEqual([node.op_type for node in normalized.graph.node],
                         ["Conv", "AveragePool", "AveragePool", "AveragePool"])
        for node in normalized.graph.node[1:]:
            self.assertEqual({a.name: h.get_attribute_value(a) for a in node.attribute},
                             {"kernel_shape": [2, 2], "strides": [2, 2]})
        self.assertEqual(len(normalized.graph.output), 1)

    def test_the_reduction_profile_emitter_accepts_the_rewritten_graph(self):
        # The legacy profile 5/6 emitter is the board-verified three-stage reduction, so
        # the rewritten graph must compile through it identically to a hand-written one.
        rewritten = normalize_model(global_pool_model("GlobalAveragePool"))
        hand = three_pool_model()
        with tempfile.TemporaryDirectory() as folder:
            first, first_meta = compile_model(saved(rewritten, folder, "rewritten"))
            second, second_meta = compile_model(saved(hand, folder, "hand"))
        self.assertEqual(first_meta["profile"], 6)
        self.assertEqual(second_meta["profile"], 6)
        self.assertEqual(encode(first, first_meta), encode(second, second_meta))

    def test_the_reduction_reference_matches_the_independent_stage_oracle(self):
        _, meta = compile_sequence(global_pool_model("ReduceMean", oc=4, kernel=3, relu=True))
        quantization = quantized(stem_quantization(meta))
        for case in deterministic_cases((8, 8, 3)):
            produced = reduction_reference(case, quantization, "AveragePool", 3)
            self.assertEqual(produced.tobytes(), pooled_oracle(case, quantization).tobytes())

    def test_out_of_envelope_spatial_means_name_the_node_and_the_bound(self):
        wrong_size = global_pool_model()
        wrong_size.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 16
        wrong_size.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 16
        with self.assertRaisesRegex(ValueError, re.escape(
                "GlobalAveragePool " + BOUND + "; node 'output' reads [1, 3, 16, 16]")):
            compile_sequence(wrong_size)

        axes = global_pool_model("ReduceMean")
        del axes.graph.node[1].attribute[:]
        axes.graph.node[1].attribute.extend([h.make_attribute("axes", [1, 2]),
                                             h.make_attribute("keepdims", 1)])
        with self.assertRaisesRegex(ValueError, re.escape(
                "ReduceMean " + BOUND + "; node 'output' reduces axes [1, 2]")):
            compile_sequence(axes)

        keepdims = global_pool_model("ReduceMean")
        del keepdims.graph.node[1].attribute[:]
        keepdims.graph.node[1].attribute.extend([h.make_attribute("axes", [2, 3]),
                                                 h.make_attribute("keepdims", 0)])
        with self.assertRaisesRegex(ValueError, re.escape(
                "ReduceMean " + BOUND + "; node 'output' does not keep the reduced dimensions")):
            compile_sequence(keepdims)

        non_terminal = global_pool_model()
        del non_terminal.graph.node[1:]
        non_terminal.graph.node.append(h.make_node("GlobalAveragePool", ["conv"], ["mean"]))
        non_terminal.graph.node.append(h.make_node("Identity", ["mean"], ["output"]))
        with self.assertRaisesRegex(ValueError, re.escape(
                "GlobalAveragePool " + BOUND + "; node 'mean' is not the terminal node")):
            compile_sequence(non_terminal)

    def test_a_three_channel_five_tap_stem_keeps_the_walk_kernel_bound(self):
        """A K5 3-channel stem reaches the op-level chain walk, which accepts K1/K3 only.

        The 1-channel form instead takes the scheduler's pooling sequence, whose legacy
        Conv bound includes K5, so the bound is per input layout rather than per mean.
        """
        with self.assertRaisesRegex(
                ValueError, re.escape("walk Conv supports square K1/K3 kernels")):
            compile_sequence(global_pool_model("GlobalAveragePool", ic=3, oc=4, kernel=5))


# ---------------------------------------------------------------------------
# F6b: Concat of sibling Conv branches
# ---------------------------------------------------------------------------


def branch_parts(splits, ic=3, kernel=3, seed=110511):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-.8, .8, (oc, ic, kernel, kernel)).astype(np.float32),
             rng.uniform(-1, 1, oc).astype(np.float32)) for oc in splits]


def concat_model(splits=(10, 10), ic=3, kernel=3, biases=True, seed=110511):
    parts = branch_parts(splits, ic, kernel, seed)
    nodes, initializers, concat_inputs = [], [], []
    for index, (oc, (weights, bias)) in enumerate(zip(splits, parts)):
        inputs = ["input", "w%d" % index] + (["b%d" % index] if biases else [])
        nodes.append(h.make_node("Conv", inputs, ["c%d" % index],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        initializers.append(nh.from_array(weights, "w%d" % index))
        if biases:
            initializers.append(nh.from_array(bias, "b%d" % index))
        concat_inputs.append("c%d" % index)
    nodes.append(h.make_node("Concat", concat_inputs, ["output"], axis=1))
    return build(nodes, [h.make_tensor_value_info("input", 1, [1, ic, 8, 8])],
                 [h.make_tensor_value_info("output", 1, [1, sum(splits), 8, 8])],
                 initializers, name="concat")


def wide_conv_model(splits=(10, 10), ic=3, kernel=3, biases=True, seed=110511):
    parts = branch_parts(splits, ic, kernel, seed)
    weights = np.concatenate([value for value, _ in parts], axis=0)
    initializers = [nh.from_array(weights, "w")]
    inputs = ["input", "w"]
    if biases:
        initializers.append(nh.from_array(np.concatenate([b for _, b in parts]), "b"))
        inputs.append("b")
    nodes = [h.make_node("Conv", inputs, ["output"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    return build(nodes, [h.make_tensor_value_info("input", 1, [1, ic, 8, 8])],
                 [h.make_tensor_value_info("output", 1, [1, sum(splits), 8, 8])],
                 initializers, name="wide")


class ConcatLoweringTests(unittest.TestCase):
    """F6b: sibling Conv branches concatenated on the channel axis become one wide Conv."""

    def test_two_and_three_branches_are_byte_identical_to_the_wide_conv(self):
        for splits, ic, kernel, biases in (((10, 10), 3, 1, False),
                                           ((10, 10), 3, 3, True),
                                           ((8, 8, 8), 3, 1, True),
                                           ((8, 8, 8), 1, 3, True),
                                           ((6, 8, 10), 3, 3, True)):
            with self.subTest(splits=splits, ic=ic, kernel=kernel):
                rewritten, _ = compile_sequence(concat_model(splits, ic, kernel, biases=biases))
                hand, _ = compile_sequence(wide_conv_model(splits, ic, kernel, biases=biases))
                self.assertEqual(rewritten, hand)

    def test_the_rewrite_leaves_one_conv_with_stacked_weights(self):
        normalized = normalize_model(concat_model((10, 10)))
        self.assertEqual([node.op_type for node in normalized.graph.node], ["Conv"])
        constants = {tensor.name: nh.to_array(tensor) for tensor in normalized.graph.initializer}
        node = normalized.graph.node[0]
        parts = branch_parts((10, 10))
        np.testing.assert_array_equal(constants[node.input[1]],
                                      np.concatenate([w for w, _ in parts], axis=0))
        np.testing.assert_array_equal(constants[node.input[2]],
                                      np.concatenate([b for _, b in parts]))

    def test_a_branch_with_a_directly_following_relu_is_not_rewritten(self):
        model = concat_model()
        model.graph.node[1].output[0] = "relu"
        model.graph.node.insert(2, h.make_node("Relu", ["relu"], ["c1"]))
        model.graph.node[-1].input[1] = "c1"
        with self.assertRaisesRegex(
                ValueError, re.escape("depthwise sequence requires Conv[/Relu] -> depthwise Conv")):
            compile_sequence(model)

    def test_out_of_envelope_concats_name_the_node_and_the_bound(self):
        wrong_axis = concat_model()
        del wrong_axis.graph.node[-1].attribute[:]
        wrong_axis.graph.node[-1].attribute.append(h.make_attribute("axis", 0))
        with self.assertRaisesRegex(ValueError, re.escape(
                CONCAT_BOUND + "; node 'output' is not a Concat on axis 1")):
            compile_sequence(wrong_axis)

        # A different pad split with the same total keeps the output extent, so the model
        # is shape-consistent; the rewrite still refuses to stack the two geometries.
        varying = concat_model((10, 10))
        for attribute in varying.graph.node[1].attribute:
            if attribute.name == "pads":
                del attribute.ints[:]
                attribute.ints.extend([2, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, re.escape(
                CONCAT_BOUND + "; node 'c1' has different kernel/strides/pads/dilations/group")):
            compile_sequence(varying)

        second_input = concat_model()
        second_input.graph.input.append(
            h.make_tensor_value_info("other", 1, [1, 3, 8, 8]))
        second_input.graph.node[1].input[0] = "other"
        with self.assertRaisesRegex(ValueError, re.escape(
                CONCAT_BOUND + "; node 'output' is in a graph without exactly one input")):
            compile_sequence(second_input)

        non_terminal = concat_model()
        non_terminal.graph.node.append(h.make_node("Identity", ["output"], ["also"]))
        non_terminal.graph.output.append(
            h.make_tensor_value_info("also", 1, [1, 20, 8, 8]))
        with self.assertRaisesRegex(ValueError, re.escape(
                CONCAT_BOUND + "; node 'output' is not the terminal node producing the single output")):
            compile_sequence(non_terminal)

        extra_consumer = concat_model()
        extra_consumer.graph.node.insert(2, h.make_node("Identity", ["c0"], ["dead"]))
        with self.assertRaisesRegex(ValueError, re.escape(
                CONCAT_BOUND + "; node 'c0' has another consumer or is a graph output")):
            compile_sequence(extra_consumer)

    def test_an_op_level_concat_keeps_its_downstream_rejection(self):
        # Not sibling Conv branches (one external input, one constant operand): the
        # rewrite must not touch it, so the ordinary sequence rejection stays.
        model = build([h.make_node("Concat", ["input", "operand"], ["output"], axis=1)],
                      [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                      [h.make_tensor_value_info("output", 1, [1, 6, 8, 8])],
                      [nh.from_array(np.zeros((1, 3, 8, 8), np.float32), "operand")])
        with self.assertRaisesRegex(
                ValueError, re.escape("sequence lowering requires one input, one output, "
                                      "and an initial Conv")):
            compile_sequence(model)


# ---------------------------------------------------------------------------
# F6c: measured negatives
# ---------------------------------------------------------------------------


def conv_then(op_type, output_shape, inputs, attrs, initializers=()):
    weights, bias = conv_stem(3, 3, 1, seed=110521)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
             h.make_node(op_type, ["conv", *inputs], ["output"], **attrs)]
    values = [nh.from_array(weights, "w"), nh.from_array(bias, "b")]
    values.extend(nh.from_array(value, name) for name, value in initializers)
    return build(nodes, [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                 [h.make_tensor_value_info("output", 1, list(output_shape))], values)


class MeasuredNegativeTests(unittest.TestCase):
    """F6c: the ops with no verified register program keep their exact rejection."""

    def test_softmax_slice_and_resize_after_a_conv_stay_rejected(self):
        models = {
            "Softmax": conv_then("Softmax", (1, 3, 8, 8), [], dict(axis=1)),
            "Slice": conv_then("Slice", (1, 3, 8, 4), ["starts", "ends", "axes"], {},
                               [("starts", np.array([0], np.int64)),
                                ("ends", np.array([4], np.int64)),
                                ("axes", np.array([3], np.int64))]),
            "Resize": conv_then("Resize", (1, 3, 16, 16), ["roi", "scales"],
                                dict(mode="nearest"),
                                [("roi", np.array([], np.float32)),
                                 ("scales", np.array([1., 1., 2., 2.], np.float32))]),
        }
        for label, model in models.items():
            with self.subTest(op=label):
                with self.assertRaisesRegex(ValueError, re.escape(SEQUENCE_REJECTION)):
                    compile_sequence(model)

    def test_eight_by_eight_average_pool_keeps_its_rejection(self):
        # Only the GlobalAveragePool/ReduceMean rewrite generates 2x2 stages; an explicit
        # 8x8 kernel average has no verified emitter and stays rejected.
        model = conv_then("AveragePool", (1, 3, 1, 1), [], dict(kernel_shape=[8, 8],
                                                                strides=[8, 8]))
        with self.assertRaisesRegex(ValueError, re.escape(POOL_REJECTION)):
            compile_sequence(model)


# ---------------------------------------------------------------------------
# F7: asymmetric / one-sided padding is already expressible
# ---------------------------------------------------------------------------


def pad_model(kernel, pads, ic=3, oc=6, relu=False, seed=110531):
    kh, kw = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.8, .8, (oc, ic, kh, kw)).astype(np.float32)
    bias = rng.uniform(-1, 1, oc).astype(np.float32)
    out_h = 8 + pads[0] + pads[2] - kh + 1
    out_w = 8 + pads[1] + pads[3] - kw + 1
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"],
                         kernel_shape=[kh, kw], pads=list(pads))]
    output = "conv"
    if relu:
        nodes.append(h.make_node("Relu", ["conv"], ["output"]))
        output = "output"
    return build(nodes, [h.make_tensor_value_info("input", 1, [1, ic, 8, 8])],
                 [h.make_tensor_value_info(output, 1, [1, oc, out_h, out_w])],
                 [nh.from_array(weights, "w"), nh.from_array(bias, "b")])


class AsymmetricPaddingTests(unittest.TestCase):
    """F7: the geometry emitter already accepts one-sided/asymmetric pads."""

    CONFIGS = ((3, (1, 1, 1, 1), False), (3, (0, 1, 0, 1), False),
               (3, (1, 0, 1, 0), True), (3, (0, 0, 0, 1), False),
               (3, (1, 1, 0, 0), True), (3, (2, 0, 0, 0), False),
               (5, (1, 0, 2, 3), False), (5, (0, 2, 1, 0), True),
               (5, (4, 0, 0, 0), False), ((1, 3), (0, 1, 0, 1), False),
               ((1, 3), (0, 0, 0, 1), True))

    def test_accepted_geometries_compile_and_match_the_explicit_pad_oracle(self):
        for kernel, pads, relu in self.CONFIGS:
            with self.subTest(kernel=kernel, pads=pads, relu=relu):
                model = pad_model(kernel, pads, relu=relu)
                binary, meta = compile_sequence(model)
                info = decode_sequence(binary)
                declared = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
                self.assertEqual(info["output_shape_nhwc"],
                                 [declared[0], declared[2], declared[3], declared[1]])
                quantization = quantized(meta["quantization"])
                effective = tuple(meta.get("conv_pads")
                                  or (quantization.kernel_size // 2,) * 4)
                strides = tuple(meta.get("conv_strides", (1, 1)))
                dilations = tuple(meta.get("conv_dilations", (1, 1)))
                for case in deterministic_cases((8, 8, 3)):
                    expected = native_input_reference(
                        case, quantization, meta["input_zero_point"], pads=meta.get("conv_pads"),
                        strides=strides, dilations=dilations)
                    self.assertEqual(
                        expected.tobytes(),
                        padded_conv_oracle(case, quantization, effective, strides,
                                           dilations).tobytes())

    def test_padding_at_or_beyond_the_kernel_is_rejected(self):
        # A wide output channel count routes to the native16 profile, whose explicit-pad
        # path is the one that names the geometry bound.
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_GEOMETRY)):
            compile_sequence(pad_model(3, (3, 0, 3, 0), oc=20))

    def test_kernel_larger_than_31_keeps_its_bound(self):
        with self.assertRaisesRegex(ValueError, re.escape(NATIVE_KERNEL)):
            compile_sequence(pad_model(33, (16, 16, 16, 16), oc=3))


# ---------------------------------------------------------------------------
# Suite replay: the emitted expected bytes are exactly the profile reference
# ---------------------------------------------------------------------------


class SuiteReferenceReplayTests(unittest.TestCase):
    """The new suites' expected bytes are exactly the profile's own integer reference."""

    def _entries(self, suite):
        return json.loads((suite / "manifest.json").read_text())

    def test_global_pool_suites_replay_through_the_reduction_reference(self):
        for name in ("global_pool_suite", "global_pool_reduce_suite"):
            suite = RESEARCH / name
            for entry in self._entries(suite):
                index = entry["index"]
                with self.subTest(suite=name, model="%03d" % index):
                    binary, meta = compile_sequence(suite / ("model%03d.onnx" % index))
                    self.assertEqual(binary, (suite / ("model%03d.bin" % index)).read_bytes())
                    quantization = quantized(stem_quantization(meta))
                    cases = np.fromfile(suite / ("input%03d.u8" % index), np.uint8).reshape(
                        -1, 8, 8, entry["input_channels"])
                    expected = np.fromfile(suite / ("expected%03d.i8" % index), np.int8).reshape(
                        -1, 1, 1, entry["output_channels"])
                    produced = np.stack([reduction_reference(case, quantization,
                                                             "AveragePool", 3)
                                         for case in cases])
                    self.assertEqual(produced.tobytes(), expected.tobytes())

    def test_the_global_pool_suites_are_each_one_container_format(self):
        """The split is pinned: the board runner stages a suite as a single format.

        The first cut kept both spellings in one directory - 3-channel graphs take the
        op-level chain walk (v5), 1-channel graphs take the scheduler's pooling-sequence
        branch (v3) - and the v5 runner refused the v3 model as "not a named-tensor
        executable". `global_pool_suite` is v5 (run_v5_suite.py) and
        `global_pool_reduce_suite` is v3 (run_profile_suite.py).
        """
        expectations = {"global_pool_suite": (5, "chain-walk"),
                        "global_pool_reduce_suite": (3, "pooling-sequence")}
        for name, (version, route) in expectations.items():
            suite = RESEARCH / name
            versions = set()
            for entry in self._entries(suite):
                index = entry["index"]
                with self.subTest(suite=name, model="%03d" % index):
                    binary, meta = compile_sequence(suite / ("model%03d.onnx" % index))
                    info = decode_sequence(binary)
                    versions.add(info["format_version"])
                    self.assertEqual(info["format_version"], version)
                    self.assertEqual(int(entry["format_version"]), version)
                    if route == "chain-walk":
                        self.assertEqual(meta.get("profile"), "chain-walk")
                    else:
                        self.assertEqual(meta.get("pool_stages"), ["AveragePool"] * 3)
            self.assertEqual(versions, {version}, "a suite mixes container formats")

    def test_wide_concat_suite_replays_through_the_native_reference(self):
        suite = RESEARCH / "wide_concat_suite"
        for entry in self._entries(suite):
            index = entry["index"]
            with self.subTest(model="%03d" % index):
                binary, meta = compile_sequence(suite / ("model%03d.onnx" % index))
                self.assertEqual(binary, (suite / ("model%03d.bin" % index)).read_bytes())
                quantization = quantized(meta["quantization"])
                cases = np.fromfile(suite / ("input%03d.u8" % index), np.uint8).reshape(
                    -1, 8, 8, entry["input_channels"])
                expected = np.fromfile(suite / ("expected%03d.i8" % index), np.int8)
                produced = np.stack([
                    native_input_reference(case, quantization, meta["input_zero_point"],
                                           pads=meta["conv_pads"],
                                           strides=tuple(meta["conv_strides"]),
                                           dilations=tuple(meta.get("conv_dilations", (1, 1))))
                    for case in cases])
                self.assertEqual(produced.tobytes(), expected.tobytes())

    def test_rect_pad_suite_replays_through_the_native_reference(self):
        suite = RESEARCH / "rect_pad_suite"
        for entry in self._entries(suite):
            index = entry["index"]
            with self.subTest(model="%03d" % index):
                binary, meta = compile_sequence(suite / ("model%03d.onnx" % index))
                self.assertEqual(binary, (suite / ("model%03d.bin" % index)).read_bytes())
                quantization = quantized(meta["quantization"])
                cases = np.fromfile(suite / ("input%03d.u8" % index), np.uint8).reshape(
                    -1, 8, 8, entry["input_channels"])
                expected = np.fromfile(suite / ("expected%03d.i8" % index), np.int8).reshape(
                    -1, entry["output_height"], entry["output_width"], entry["output_channels"])
                produced = np.stack([
                    native_input_reference(case, quantization, meta["input_zero_point"],
                                           pads=meta.get("conv_pads"),
                                           strides=tuple(meta.get("conv_strides", (1, 1))),
                                           dilations=tuple(meta.get("conv_dilations", (1, 1))))
                    for case in cases])
                self.assertEqual(produced.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
