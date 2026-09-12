# SPDX-License-Identifier: MIT
"""Pins the emitter rejection paths that no accepted graph can reach.

Every scheduled family validates its ONNX graph before it emits: the elementwise
DAGs, the bounded LUT profile, the native input profile, the staged reduction,
the single pool, the strided geometry profile, the tiled chain and the op-level
walk. Most of those guards sit inside a validator that only a malformed graph
reaches, so the accepted suites never execute them. This module drives each
guard with the smallest graph (or the smallest direct call) that reaches it and
asserts the *exact* message, so a refactor that silently drops a rejection fails
here instead of on a board run.

Two guards are provably dead and are documented rather than tested:

* ``elementwise.py``'s ``per-channel constant Mul requires a broadcastable
  constant`` is shadowed by ``numpy.broadcast_to``: for a rank-four target,
  ``broadcast_to(factor, shape)`` succeeding implies
  ``broadcast_shapes(factor.shape, shape) == shape``, so the guard can never
  fire (brute-forced over ranks 0..6 in the change notes).
* ``elementwise_chain.py``'s ``payload.extend(bytes(program_offset -
  len(payload)))`` only runs when the program cursor is ahead of the payload,
  but ``programs_offset`` starts at ``len(payload)`` and every stage extends the
  payload by exactly the next stage's stride, so the lengths are always equal.

Where a guard is shadowed by an *earlier validator* the test patches that one
collaborator so the guard itself executes: ``tiled_chain`` re-validates through
``chain_n._validate`` (which already rejects the kernel, shape and channel
cases) and ``walk.parse_chain`` guarantees ``ops[0]`` is a Conv and the external
input has three channels. Each patch is noted at the call site.

No board, network or vendor artifact is used; every container is decoded from
the bytes the emitter just produced.
"""
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.chain_n import _validate as validate_chain_n
from open_rknpu.elementwise_chain import chain_reference, compile_elementwise_dag
from open_rknpu.elementwise_multi import compile_multi_input_dag
from open_rknpu.lut import compile_lut
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.pooling import compile_pool
from open_rknpu.quantization import Quantization, receptive_fields
from open_rknpu.reduction import compile_reduction
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.tiled_chain import compile_tiled_chain
from open_rknpu.walk import compile_chain_walk

RGB = [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]), h.make_tensor_value_info("b", 1, [1, 3, 8, 8])]
OUT = [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])]


def model_of(nodes, inputs, outputs, initializers=(), opset=13, extra_opsets=(), infer=True):
    graph = h.make_graph(nodes, "rejections", list(inputs), list(outputs), list(initializers))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", opset), *extra_opsets])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model) if infer else model


def save(model):
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "model.onnx"
    onnx.save(model, path)
    return folder, path


def branch_pair(seed=31):
    rng = np.random.default_rng(seed)
    wa = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
    ba = rng.uniform(-2, 2, (3,)).astype(np.float32)
    wb = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
    bb = rng.uniform(-2, 2, (3,)).astype(np.float32)
    return [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"), nh.from_array(wb, "wb"), nh.from_array(bb, "bb")]


def conv(source, weight, bias, destination, **attrs):
    return h.make_node("Conv", [source, weight, bias], [destination], kernel_shape=[1, 1], **attrs)


def chain_model(first="Add"):
    """Conv, Conv, first, Mul: the smallest accepted two-input elementwise DAG."""
    return model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                     h.make_node(first, ["A", "B"], ["C"]), h.make_node("Mul", ["C", "A"], ["out"])],
                    RGB, OUT, branch_pair())


def multi_model():
    """Conv, Conv, Add, Conv, Mul: the smallest accepted multi-input v5 DAG."""
    nodes = [conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"), h.make_node("Add", ["A", "B"], ["C"]),
             conv("c", "wa", "ba", "D"), h.make_node("Mul", ["C", "D"], ["out"])]
    return model_of(nodes, RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT, branch_pair())


_FIRST_STAGE = None


def first_stage_artifacts():
    """A real (binary, meta) for the three-node Conv,Conv,Add first-stage sub-graph."""
    global _FIRST_STAGE
    if _FIRST_STAGE is None:
        sub = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                        h.make_node("Add", ["A", "B"], ["C"])],
                       RGB, [h.make_tensor_value_info("C", 1, [1, 3, 8, 8])], branch_pair())
        folder, path = save(sub)
        try:
            _FIRST_STAGE = compile_sequence(path)
        finally:
            folder.cleanup()
    return _FIRST_STAGE


def lut_model(weights, bias, kind="Sigmoid"):
    weights = np.asarray(weights, np.float32).reshape(3, 3, 1, 1)
    bias = np.asarray(bias, np.float32)
    return model_of([h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
                     h.make_node(kind, ["conv"], ["output"])],
                    [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                    [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                    [nh.from_array(weights, "w"), nh.from_array(bias, "b")])


def native_conv(activation=None, weight=0.4):
    weights = np.full((3, 3, 1, 1), weight, np.float32)
    bias = np.zeros(3, np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[1, 1])]
    output = "c"
    if activation is not None:
        nodes.append(h.make_node(activation, ["c"], ["r"]))
        output = "r"
    return model_of(nodes, [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                    [h.make_tensor_value_info(output, 1, [1, 3, 8, 8])],
                    [nh.from_array(weights, "w"), nh.from_array(bias, "b")])


def lut_table(binary):
    """Decode the two 513-entry LUT banks the emitter writes (negative first)."""
    info = decode_sequence(binary)
    payload = binary[96 + 16 * info["task_count"]:]
    setup = info["tasks"][0]
    banks = {0x20000: [], 0x30000: []}
    bank = None
    for i in range(setup["register_count"]):
        word = struct.unpack_from("<Q", payload, setup["command_offset"] + i * 8)[0]
        register, value = word & 0xffff, word >> 16 & 0xffffffff
        if register == 0x4100 and value in banks:
            bank = value
        elif register == 0x4104 and bank is not None:
            banks[bank].append(value - 65536 if value > 32767 else value)
    return banks


class RejectionTestCase(unittest.TestCase):
    def assert_rejects(self, message, function, *args, **kwargs):
        """Assert the function fails with exactly this message (not a prefix)."""
        with self.assertRaisesRegex(ValueError, "^" + re.escape(message) + "$"):
            function(*args, **kwargs)

    def compile_saved(self, model):
        folder, path = save(model)
        try:
            return compile_sequence(path)
        finally:
            folder.cleanup()


class ElementwiseChainRejectionTests(RejectionTestCase):
    """`elementwise_chain._validate`/`compile_elementwise_dag` guards (lines 53-107)."""

    def test_rejects_a_non_default_operator_domain(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"], domain="com.example"),
                          h.make_node("Mul", ["C", "A"], ["out"])],
                         RGB, OUT, branch_pair(), extra_opsets=[h.make_opsetid("com.example", 1)])
        self.assert_rejects("unsupported operator domain", self.compile_saved, model)

    def test_rejects_a_join_that_is_not_the_graph_output(self):
        # `Mul` produces the unused "joined"; the graph output is the earlier "C".
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"]), h.make_node("Mul", ["C", "A"], ["joined"])],
                         RGB, [h.make_tensor_value_info("C", 1, [1, 3, 8, 8])], branch_pair())
        self.assert_rejects("the last stage must produce the graph output", self.compile_saved, model)

    def test_rejects_operator_attributes(self):
        # Opset 6 is the last that gives Add a legal attribute, so the checker passes
        # and the DAG's own "no attributes" rule is what rejects the graph.
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"], broadcast=0, axis=1),
                          h.make_node("Mul", ["C", "A"], ["out"])],
                         RGB, OUT, branch_pair(), opset=6)
        self.assert_rejects("elementwise DAG operators take no attributes", self.compile_saved, model)

    def test_rejects_a_branch_without_bias(self):
        model = model_of([h.make_node("Conv", ["a", "wa"], ["A"], kernel_shape=[1, 1]),
                          conv("b", "wb", "bb", "B"), h.make_node("Add", ["A", "B"], ["C"]),
                          h.make_node("Mul", ["C", "A"], ["out"])],
                         RGB, OUT, branch_pair())
        self.assert_rejects("elementwise DAG branches require constant weights and bias", self.compile_saved, model)

    def test_rejects_a_first_stage_without_a_static_shape(self):
        # No shape inference, so "C" has no value_info and the sub-graph output cannot
        # be described. The scheduler would re-infer shapes, so call the emitter itself.
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"]), h.make_node("Mul", ["C", "A"], ["out"])],
                         RGB, OUT, branch_pair(), infer=False)
        self.assert_rejects("missing static first-stage output shape", compile_elementwise_dag, model)

    def test_rejects_an_unexpected_first_stage_container(self):
        # The four-task container is a valid sequence, but the first stage must be three.
        with mock.patch("open_rknpu.elementwise_chain.decode_sequence",
                        return_value={"task_count": 4, "output_shape_nhwc": [1, 8, 8, 3]}):
            self.assert_rejects("unexpected first-stage container", compile_elementwise_dag, chain_model())

    def test_rejects_a_nonzero_zero_point_first_stage(self):
        binary, meta = first_stage_artifacts()
        with mock.patch("open_rknpu.scheduler.compile_sequence",
                        return_value=(binary, dict(meta, output_zero_point=7))):
            self.assert_rejects("elementwise DAG requires zero-point-zero stages",
                                compile_elementwise_dag, chain_model())

    def test_rejects_unequal_first_stage_scales(self):
        # Add/Sub/Max require the first-stage band to be exactly twice a branch band.
        binary, meta = first_stage_artifacts()
        branches = [dict(branch) for branch in meta["branches"]]
        branches[0]["output_scale"] = branches[0]["output_scale"] / 2
        with mock.patch("open_rknpu.scheduler.compile_sequence",
                        return_value=(binary, dict(meta, branches=branches))):
            self.assert_rejects("elementwise DAG requires the verified equal-scale Add",
                                compile_elementwise_dag, chain_model())

    def test_chain_reference_rejects_an_unknown_first_stage(self):
        binary, meta = self.compile_saved(chain_model())
        self.assertEqual(decode_sequence(binary)["task_count"], 4)
        rng = np.random.default_rng(7)
        a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        self.assert_rejects("unknown first stage: Bogus", chain_reference, a, b, dict(meta, first_stage="Bogus"))


class ElementwiseMultiRejectionTests(RejectionTestCase):
    """`elementwise_multi._validate`/`compile_multi_input_dag` guards (lines 46-119)."""

    def test_rejects_a_non_conv_mul_extra(self):
        # The extra pair is (Relu, Mul); the scheduler would not dispatch this shape,
        # so the validator is called directly.
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"]), h.make_node("Relu", ["C"], ["R"]),
                          h.make_node("Mul", ["R", "R"], ["out"])], RGB, OUT, branch_pair())
        self.assert_rejects("each extra input requires a Conv then a Mul", compile_multi_input_dag, model)

    def test_rejects_a_non_default_operator_domain(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"], domain="com.example"),
                          conv("c", "wa", "ba", "D"), h.make_node("Mul", ["C", "D"], ["out"])],
                         RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT, branch_pair(),
                         extra_opsets=[h.make_opsetid("com.example", 1)])
        self.assert_rejects("unsupported operator domain", compile_multi_input_dag, model)

    def test_rejects_a_first_stage_not_combining_the_branches(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "A"], ["C"]), conv("c", "wa", "ba", "D"),
                          h.make_node("Mul", ["C", "D"], ["out"])],
                         RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT, branch_pair())
        self.assert_rejects("the first stage must combine the two Conv branches", self.compile_saved, model)

    def test_rejects_a_mul_that_swaps_its_operands(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"]), conv("c", "wa", "ba", "D"),
                          h.make_node("Mul", ["D", "C"], ["out"])],
                         RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT, branch_pair())
        self.assert_rejects("each extra stage must multiply the previous result by its Conv", self.compile_saved, model)

    def test_rejects_a_join_that_is_not_the_graph_output(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"]), conv("c", "wa", "ba", "D"),
                          h.make_node("Mul", ["C", "D"], ["joined"])],
                         RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("C", 1, [1, 3, 8, 8])], branch_pair())
        self.assert_rejects("the last stage must produce the graph output", self.compile_saved, model)

    def test_rejects_operator_attributes(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"], broadcast=0, axis=1),
                          conv("c", "wa", "ba", "D"), h.make_node("Mul", ["C", "D"], ["out"])],
                         RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT, branch_pair(), opset=6)
        self.assert_rejects("elementwise DAG operators take no attributes", self.compile_saved, model)

    def test_rejects_a_branch_without_bias(self):
        model = model_of([h.make_node("Conv", ["a", "wa"], ["A"], kernel_shape=[1, 1]),
                          conv("b", "wb", "bb", "B"), h.make_node("Add", ["A", "B"], ["C"]),
                          conv("c", "wa", "ba", "D"), h.make_node("Mul", ["C", "D"], ["out"])],
                         RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT, branch_pair())
        self.assert_rejects("elementwise DAG branches require constant weights and bias", self.compile_saved, model)

    def test_rejects_a_mismatched_external_tensor(self):
        model = model_of([conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"),
                          h.make_node("Add", ["A", "B"], ["C"]), conv("c", "wa", "ba", "D"),
                          h.make_node("Mul", ["C", "D"], ["out"])],
                         [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]),
                          h.make_tensor_value_info("b", 1, [1, 3, 8, 8]),
                          h.make_tensor_value_info("c", 1, [1, 3, 8, 4])], OUT, branch_pair())
        self.assert_rejects("elementwise DAG external tensors must be float32 [1,3,8,8]",
                            compile_multi_input_dag, model)

    def test_rejects_a_first_stage_without_a_static_shape(self):
        nodes = [conv("a", "wa", "ba", "A"), conv("b", "wb", "bb", "B"), h.make_node("Add", ["A", "B"], ["C"]),
                 conv("c", "wa", "ba", "D"), h.make_node("Mul", ["C", "D"], ["out"])]
        model = model_of(nodes, RGB + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])], OUT,
                         branch_pair(), infer=False)
        self.assert_rejects("missing static first-stage output shape", compile_multi_input_dag, model)

    def test_rejects_an_unexpected_first_stage_container(self):
        with mock.patch("open_rknpu.elementwise_multi.decode_sequence",
                        return_value={"task_count": 4, "output_shape_nhwc": [1, 8, 8, 3]}):
            self.assert_rejects("unexpected first-stage container", compile_multi_input_dag, multi_model())

    def test_rejects_a_nonzero_zero_point_first_stage(self):
        binary, meta = first_stage_artifacts()
        with mock.patch("open_rknpu.scheduler.compile_sequence",
                        return_value=(binary, dict(meta, output_zero_point=7))):
            self.assert_rejects("multi-input DAG requires zero-point-zero stages",
                                compile_multi_input_dag, multi_model())

    def test_rejects_unequal_first_stage_scales(self):
        binary, meta = first_stage_artifacts()
        branches = [dict(branch) for branch in meta["branches"]]
        branches[0]["output_scale"] = branches[0]["output_scale"] / 2
        with mock.patch("open_rknpu.scheduler.compile_sequence",
                        return_value=(binary, dict(meta, branches=branches))):
            self.assert_rejects("multi-input DAG requires the verified equal-scale Add",
                                compile_multi_input_dag, multi_model())


class LutRejectionTests(RejectionTestCase):
    """`lut.compile_lut` guards (lines 73, 116, 126)."""

    def test_rejects_a_stem_that_is_not_conv_plus_activation(self):
        model = model_of([h.make_node("Relu", ["input"], ["r"]), h.make_node("Sigmoid", ["r"], ["output"])],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])])
        self.assert_rejects("LUT profile requires Conv followed by Sigmoid or Tanh", self.compile_saved, model)

    def test_rejects_channels_in_different_gain_bands(self):
        # Mixed stems are normally refused earlier, so allow_mixed_stems exposes the
        # gain-band guard: channel 0 has ratio 1, channels 1 and 2 have ratio 2.
        weights = np.array([[1 / 32, 0, 0], [1 / 64, 0, 0], [1 / 64, 0, 0]], np.float32)
        self.assert_rejects("LUT stem channels must share one negative-half gain band; got [1.0, 2.0, 2.0]",
                            compile_lut, lut_model(weights, np.zeros(3)), 1, 128, allow_mixed_stems=True)

    def test_negative_gain_override_changes_the_negative_table_half(self):
        weights = np.eye(3) / 32
        default, default_meta = compile_lut(lut_model(weights, np.zeros(3)), 1, 128)
        overridden, overridden_meta = compile_lut(lut_model(weights, np.zeros(3)), 1, 128,
                                                  negative_gain_override=2.0)
        self.assertEqual(default_meta["negative_gain"], 1.0)
        self.assertEqual(overridden_meta["negative_gain"], 2.0)
        self.assertNotEqual(default, overridden)
        # The override scales the negative half's argument grid by two, exactly as the
        # emitted table does; the positive half is untouched.
        banks = lut_table(overridden)
        arguments = (np.arange(513) - 512) / 64 * 2.0
        expected = np.clip(np.rint((1 / (1 + np.exp(-arguments))) * 32768), -32768, 32767)
        np.testing.assert_array_equal(banks[0x20000], expected)
        # The positive half does not depend on the negative gain.
        np.testing.assert_array_equal(lut_table(default)[0x30000], banks[0x30000])
        self.assertNotEqual(lut_table(default)[0x20000], banks[0x20000])


class NativeRejectionTests(RejectionTestCase):
    """`native.compile_native_input` and `native_input_reference` branches (lines 37, 117)."""

    def test_native_input_reference_with_a_zero_shift(self):
        # An output scale 20000x below the natural one drives factor = 20000, so the
        # container carries shift 0 and the reference takes its no-shift branch.
        maximum = float(np.float32(0.4 / 255))
        scale = maximum / 20000.0
        binary, meta = compile_native_input(native_conv(), output_range={"scale": scale, "zero_point": 0})
        self.assertEqual(decode_sequence(binary)["task_count"], 1)
        self.assertEqual(meta["quantization"]["shift"], 0)
        q = Quantization(**{key: (np.array(value) if isinstance(value, list) else value)
                            for key, value in meta["quantization"].items()})
        inputs = (np.arange(8 * 8 * 3) % 256).astype(np.uint8).reshape(8, 8, 3)
        got = native_input_reference(inputs, q, 0)
        patches = receptive_fields(inputs.astype(np.int64) - 128, q.kernel_size, -128)
        acc = np.einsum("hwc,oc->hwo", patches, q.weights - q.weight_zero_points[:, None]) + q.biases
        product = acc * q.channel_multipliers
        scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
        expected = np.clip(scaled * q.multiplier + q.output_zero_point, -128, 127).astype(np.int8)
        self.assertEqual(got.dtype, np.int8)
        self.assertEqual(got.shape, (8, 8, 3))
        np.testing.assert_array_equal(got, expected)

    def test_rejects_prequantized_output_override(self):
        # A prequantized Conv already fixes its output band, so a fused activation or
        # an output-range override is ambiguous.
        _, meta = compile_native_input(native_conv())
        q = Quantization(**{key: (np.array(value) if isinstance(value, list) else value)
                            for key, value in meta["quantization"].items()})
        self.assert_rejects("prequantized native Conv already carries output quantization",
                            compile_native_input, native_conv(activation="Relu"), quantization=q)


class ReductionRejectionTests(RejectionTestCase):
    """`reduction.compile_reduction` guards (lines 23, 33). The staged profile is not
    reachable through `compiler.compile_model` for these malformed shapes, so the
    function is called directly on a saved path."""

    def compile_path(self, model):
        folder, path = save(model)
        try:
            return compile_reduction(path)
        finally:
            folder.cleanup()

    def test_rejects_a_non_pool_third_from_last_node(self):
        # Five nodes keep the length legal, but the last three are Relus, not pools.
        nodes = [conv("input", "w", "b", "c0")]
        nodes += [h.make_node("Relu", [f"c{i}"], [f"c{i + 1}"]) for i in range(4)]
        model = model_of(nodes, [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("c4", 1, [1, 3, 8, 8])],
                         [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                          nh.from_array(np.zeros(3, np.float32), "b")])
        self.assert_rejects("unsupported staged pooling graph", self.compile_path, model)

    def test_rejects_a_mismatched_graph_output(self):
        # The three pools chain to c3, but the graph output is the Conv's c0.
        nodes = [conv("input", "w", "b", "c0")]
        nodes += [h.make_node("MaxPool", [f"c{i}"], [f"c{i + 1}"], kernel_shape=[2, 2], strides=[2, 2])
                  for i in range(3)]
        model = model_of(nodes, [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("c0", 1, [1, 3, 8, 8])],
                         [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                          nh.from_array(np.zeros(3, np.float32), "b")])
        self.assert_rejects("unsupported staged pooling graph", self.compile_path, model)


class PoolingRejectionTests(RejectionTestCase):
    """`pooling.compile_pool` guards (lines 43, 45, 50, 62)."""

    def compile_path(self, model):
        folder, path = save(model)
        try:
            return compile_pool(path)
        finally:
            folder.cleanup()

    def conv_inputs(self, kernel=1):
        weights = np.full((3, 3, kernel, kernel), 0.4, np.float32)
        return [nh.from_array(weights, "w"), nh.from_array(np.zeros(3, np.float32), "b")]

    def test_rejects_a_non_default_pool_domain(self):
        model = model_of([conv("input", "w", "b", "c0"),
                          h.make_node("MaxPool", ["c0"], ["output"], kernel_shape=[2, 2], strides=[2, 2],
                                      domain="com.example")],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                         self.conv_inputs(), extra_opsets=[h.make_opsetid("com.example", 1)])
        self.assert_rejects("unsupported pooling graph", self.compile_path, model)

    def test_rejects_a_disconnected_pool(self):
        # The pool reads the graph input instead of the Relu that precedes it.
        model = model_of([conv("input", "w", "b", "c0"), h.make_node("Relu", ["c0"], ["r0"]),
                          h.make_node("MaxPool", ["input"], ["output"], kernel_shape=[2, 2], strides=[2, 2])],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])], self.conv_inputs())
        self.assert_rejects("unsupported pooling graph", self.compile_path, model)

    def test_rejects_a_pool_that_is_not_the_graph_output(self):
        model = model_of([conv("input", "w", "b", "c0"),
                          h.make_node("MaxPool", ["c0"], ["p0"], kernel_shape=[2, 2], strides=[2, 2])],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("c0", 1, [1, 3, 8, 8])], self.conv_inputs())
        self.assert_rejects("unsupported pooling graph", self.compile_path, model)

    def test_rejects_a_non_unit_kernel_stem(self):
        # A 3x3 stem lowers to a valid Conv container, but the pool wrapper only
        # composes the single-Conv (1x1) profile.
        model = model_of([h.make_node("Conv", ["input", "w", "b"], ["c0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                          h.make_node("MaxPool", ["c0"], ["output"], kernel_shape=[2, 2], strides=[2, 2])],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])], self.conv_inputs(kernel=3))
        self.assert_rejects("unsupported pooling graph", self.compile_path, model)


class StridedRejectionTests(RejectionTestCase):
    """`strided.compile_strided` guard (line 15). The scheduler routes a strided Conv
    with extra nodes here because the legacy pool profile only accepts Conv[/Relu]."""

    def test_rejects_extra_nodes(self):
        weights = np.full((3, 3, 3, 3), 0.4, np.float32)
        model = model_of([h.make_node("Conv", ["input", "w", "b"], ["c0"], kernel_shape=[3, 3],
                                      strides=[2, 2], pads=[1, 1, 1, 1]),
                          h.make_node("Relu", ["c0"], ["r0"]), h.make_node("Relu", ["r0"], ["output"])],
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                         [nh.from_array(weights, "w"), nh.from_array(np.zeros(3, np.float32), "b")])
        self.assert_rejects("geometry profile requires Conv with optional Relu", self.compile_saved, model)


class TiledChainGuardTests(RejectionTestCase):
    """`tiled_chain.compile_tiled_chain` guards (lines 53, 55, 57).

    `compile_tiled_chain` re-validates through `chain_n._validate`, which already
    rejects these graphs with its own message, so the collaborator is patched to
    return the parsed model while leaving the guard under test in place.
    """

    def valid_chain(self):
        rng = np.random.default_rng(3)
        nodes, initializers, source, channels = [], [], "input", 3
        for index, (out_channels, activation) in enumerate(((8, "Relu"), (3, None))):
            nodes.append(h.make_node("Conv", [source, f"w{index}", f"b{index}"], [f"c{index}"],
                                     kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
            initializers += [nh.from_array(rng.uniform(-.7, .7, (out_channels, channels, 1, 1)).astype(np.float32),
                                           f"w{index}"),
                             nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32), f"b{index}")]
            source, channels = f"c{index}", out_channels
            if activation:
                nodes.append(h.make_node("Relu", [source], [f"r{index}"]))
                source = f"r{index}"
        return model_of(nodes, [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                        [h.make_tensor_value_info(source, 1, [1, 3, 8, 8])], initializers)

    def test_rejects_a_kernel_outside_the_family(self):
        model = self.valid_chain()
        graph, nodes, convs, layers, constants = validate_chain_n(model)
        widened = [(w, b, 5) for w, b, _ in layers]
        with mock.patch("open_rknpu.tiled_chain._validate", return_value=(graph, nodes, convs, widened, constants)):
            self.assert_rejects("tiled chain supports the chain family's 1x1 and 3x3 kernels",
                                compile_tiled_chain, model)

    def test_rejects_a_non_rgb_8x8_input(self):
        model = self.valid_chain()
        graph, nodes, convs, layers, constants = validate_chain_n(model)
        other = model_of([conv("input", "w", "b", "c")], [h.make_tensor_value_info("input", 1, [1, 3, 8, 4])],
                         [h.make_tensor_value_info("c", 1, [1, 3, 8, 4])],
                         [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                          nh.from_array(np.zeros(3, np.float32), "b")])
        with mock.patch("open_rknpu.tiled_chain._validate",
                        return_value=(other.graph, nodes, convs, layers, constants)):
            self.assert_rejects("tiled chain requires the 8x8 C3 input", compile_tiled_chain, model)

    def test_rejects_wide_hidden_layers(self):
        model = self.valid_chain()
        graph, nodes, convs, layers, constants = validate_chain_n(model)
        wide = [(np.zeros((17, 3, 1, 1), np.float32), b, k) for w, b, k in layers]
        with mock.patch("open_rknpu.tiled_chain._validate", return_value=(graph, nodes, convs, wide, constants)):
            self.assert_rejects("tiled chain supports up to 16 channels per layer", compile_tiled_chain, model)


class WalkGuardTests(RejectionTestCase):
    """`walk.compile_chain_walk` guards (lines 271, 282).

    Both are shadowed by `walk.parse_chain`: it returns None unless `ops[0]` is a
    Conv and the external input is RGB (3 channels), so no real graph can reach
    them. The parser is patched with the description it would have returned.
    """

    def simple_model(self):
        return model_of([conv("input", "w", "b", "output")],
                        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                        [nh.from_array(np.zeros((3, 3, 1, 1), np.float32), "w"),
                         nh.from_array(np.zeros(3, np.float32), "b")])

    def test_walk_requires_the_chain_to_start_with_a_conv(self):
        plan = {"ops": [{"kind": "pool"}, {"kind": "conv"}]}
        with mock.patch("open_rknpu.walk.parse_chain", return_value=plan):
            self.assert_rejects("walk chain must start with a Conv", compile_chain_walk, self.simple_model())

    def test_walk_rejects_a_native_input_over_sixteen_channels(self):
        plan = {"ops": [{"kind": "conv", "native_input": True}], "input_shape": (1, 20, 8, 8),
                "output_shape": (1, 3, 8, 8), "output_channels": 3, "channels": 3}
        with mock.patch("open_rknpu.walk.parse_chain", return_value=plan):
            self.assert_rejects("walk native input supports up to 16 channels",
                                compile_chain_walk, self.simple_model())


if __name__ == "__main__":
    unittest.main()
