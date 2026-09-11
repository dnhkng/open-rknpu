"""SPDX-License-Identifier: MIT

Why this file exists
--------------------

`graph.py` is the bounded DAG emitter module and the largest remaining coverage
gap. The published suites reach it almost entirely through the scheduler's
op-level `walk`, which is byte-identical for the shapes it covers, so a whole
tier of emitter code — `compile_diamond`'s accepted body, `compile_two_head`'s
validation branches and the `compile_join_chain` rejection guards — was never
called directly. This module calls those emitters with their own models.

What it pins
------------

* Every rejection branch of `compile_two_head`, `compile_diamond`,
  `compile_join_chain` and `parse_join_chain` together with the message it
  raises, each guarded by a valid neighbour that compiles, so the boundary is
  proven rather than just the failure. (Two guards, noted inline, are defensive
  code whose precondition the classifier already enforces.)
* The accepted emitter paths: two-head fan-out with two outputs, diamond
  joins in all four kinds, the `[Conv, Relu]* Conv` tail with its per-layer
  activation, dense and group-3 depthwise heads, 3-5 head chains, mixed joins,
  the runtime per-channel scale and the runtime residual tail.
* The decoded v5 container for each accepted case: profile, task count, tensor
  roles, shapes, output band and head metadata.
* `diamond_reference`, `join_chain_scale_reference`,
  `join_chain_residual_reference` and the two-head composition against an
  **independent float64 implementation written here**: the container bands are
  dequantized, the graph arithmetic runs in float64 and the result is requantized
  with round-half-to-even. The emitter's own reference is never used to build the
  expected side. Where the emitter's per-channel multiplier approximation and
  the ideal band arithmetic disagree at a rounding boundary the test allows
  1 LSB; everything else is exact.

Deterministic seeds, tempfile only, offline.
"""
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu.graph import (
    compile_diamond,
    compile_join_chain,
    compile_two_head,
    diamond_reference,
    join_chain_residual_reference,
    join_chain_scale_reference,
    parse_join_chain,
)
from open_rknpu.quantization import Quantization
from open_rknpu.sequence import decode_sequence

OPERAND_SCALE = 1 / 127
SPATIAL = [1, 3, 8, 8]
# The ideal float64 band arithmetic can differ from the emitter's integer
# per-channel conversion by one LSB at a rounding boundary; an extra LSB can
# survive a join that folds two already-requantized operands.
LSB_TOLERANCE = 1


# ---------------------------------------------------------------------------
# Model construction and mutation helpers
# ---------------------------------------------------------------------------


def _value(name, shape, elem_type=1):
    return h.make_tensor_value_info(name, elem_type, list(shape))


def _initializer(name, array):
    return nh.from_array(np.asarray(array), name)


def _finish(nodes, inputs, outputs, initializers, value_info=(), infer=True):
    graph = h.make_graph(nodes, "graph", inputs, outputs, initializers, value_info=list(value_info))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model) if infer else model


def _copy(model):
    return onnx.load_from_string(model.SerializeToString())


def _add_opset(model, domain):
    model.opset_import.append(h.make_opsetid(domain, 1))


def _set_attr(node, name, value):
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(h.make_attribute(name, value))


def _drop_attr(node, name):
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


def _set_initializer(model, name, array):
    for index, entry in enumerate(model.graph.initializer):
        if entry.name == name:
            model.graph.initializer[index].CopyFrom(nh.from_array(np.asarray(array), name))
            return
    raise KeyError(name)


def _set_value_info(model, name, shape):
    for index, entry in enumerate(model.graph.value_info):
        if entry.name == name:
            model.graph.value_info[index].CopyFrom(_value(name, shape))
            return
    model.graph.value_info.append(_value(name, shape))


def _replace_node(model, index, node):
    del model.graph.node[index]
    model.graph.node.insert(index, node)


@contextlib.contextmanager
def _checker_disabled():
    """Skip `onnx.checker` so a guard the ONNX schema itself rejects is still pinned.

    Relu and Add carry no schema attributes, so a model with one is refused by
    `onnx.checker` before the emitter can see it. The emitter guard is still real
    defence for callers that supply a hand-built protobuf, so it is exercised
    here with the checker stubbed out.
    """
    with mock.patch.object(onnx.checker, "check_model", lambda *args, **kwargs: None):
        yield


def _weights(out_channels, in_channels, kernel, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.7, 0.8, (out_channels, in_channels, kernel, kernel)).astype(np.float32)


def _bias(size, seed):
    return np.random.default_rng(seed).uniform(-4, 4, (size,)).astype(np.float32)


# ---------------------------------------------------------------------------
# Two-head fan-out
# ---------------------------------------------------------------------------


def two_head_model(hidden=5, kernel_a=1, kernel_b=3, seed=7, infer=True):
    """`input -> Conv -> Relu -> t -> {Conv a -> A, Conv b -> B}` with two outputs."""
    nodes = [
        h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
        h.make_node("Relu", ["stem"], ["relu1"]),
        h.make_node("Conv", ["relu1", "wa", "ba"], ["outputA"],
                    kernel_shape=[kernel_a, kernel_a], pads=[kernel_a // 2] * 4),
        h.make_node("Conv", ["relu1", "wb", "bb"], ["outputB"],
                    kernel_shape=[kernel_b, kernel_b], pads=[kernel_b // 2] * 4),
    ]
    initializers = [
        _initializer("w1", _weights(hidden, 3, 1, seed)),
        _initializer("b1", _bias(hidden, seed + 1)),
        _initializer("wa", _weights(3, hidden, kernel_a, seed + 2)),
        _initializer("ba", _bias(3, seed + 3)),
        _initializer("wb", _weights(3, hidden, kernel_b, seed + 4)),
        _initializer("bb", _bias(3, seed + 5)),
    ]
    value_info = [_value("relu1", [1, hidden, 8, 8])]
    return _finish(nodes, [_value("input", SPATIAL)],
                   [_value("outputA", SPATIAL), _value("outputB", SPATIAL)],
                   initializers, value_info, infer)


class TwoHeadRejectionTests(unittest.TestCase):
    """Each guard of `compile_two_head` with the exact message and a valid neighbour."""

    MESSAGE = {
        "shape": "two-head profile requires Conv, Relu, Conv, Conv",
        "relu": "two-head profile requires a valid stem Relu",
        "io": "two-head profile requires one input and two outputs",
        "outputs": "two-head outputs must be the two head outputs in order",
        "external": "two-head external tensors must be float32 .1,3,8,8.",
        "constants": "two-head convolutions require constant float32 weights and bias",
        "dtype": "two-head constants must be float32",
        "stem": "two-head stem must be a 1x1 Conv with hidden channels 3..16",
        "stem_attrs": "unsupported stem attributes",
        "head": "two-head heads must be dense 3-output 1x1/3x3 Conv with hidden inputs",
        "head_attrs": "unsupported head attributes",
        "pads": "two-head 3x3 heads require symmetric pad1",
    }

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def _check(self, model, key, name="reject.onnx"):
        path = Path(self.folder.name) / name
        onnx.save(model, path)
        with self.assertRaisesRegex(ValueError, self.MESSAGE[key]):
            compile_two_head(model)

    def test_two_head_accepts_hidden_channels_and_kernels(self):
        for hidden in (3, 8, 16):
            for kernel_a, kernel_b in ((1, 1), (1, 3), (3, 1), (3, 3)):
                with self.subTest(hidden=hidden, kernels=(kernel_a, kernel_b)):
                    binary, meta = compile_two_head(
                        two_head_model(hidden=hidden, kernel_a=kernel_a, kernel_b=kernel_b))
                    self.assertEqual(meta["profile"], "two-head-fanout")
                    self.assertEqual(meta["hidden_channels"], hidden)
                    self.assertEqual(meta["head_kernels"], [kernel_a, kernel_b])
                    self.assertEqual(meta["shape_nhwc"], [1, 8, 8, 3])
                    self.assertEqual(meta["output_tensors"], ["output_a", "output_b"])
                    info = decode_sequence(binary)
                    self.assertEqual(info["task_count"], 3)
                    self.assertEqual(info["input_tensor_count"], 1)
                    self.assertEqual(info["output_tensor_count"], 2)
                    self.assertEqual([t["index"] for t in info["output_tensors"]], [0, 1])
                    roles = {t["name"]: t["role_name"] for t in info["tensors"]}
                    self.assertEqual(roles, {"input0": "input", "stem": "internal",
                                             "output_a": "output", "output_b": "output"})
                    stem = next(t for t in info["tensors"] if t["name"] == "stem")
                    self.assertEqual((stem["layout_name"], stem["channels"]),
                                     ("native16", hidden))
                    self.assertAlmostEqual(info["output_scale"], meta["output_scales"][0], places=6)
                    self.assertEqual(info["output_zero_point"], meta["output_zero_points"][0])

    def test_two_head_rejects_bad_node_sequence(self):
        model = _copy(two_head_model())
        model.graph.node.append(h.make_node("Relu", ["outputA"], ["extra"]))
        self._check(model, "shape")

    def test_two_head_rejects_broken_stem_relu(self):
        model = _copy(two_head_model())
        model.graph.node[1].input[0] = "input"
        self._check(model, "relu")

    def test_two_head_requires_two_outputs(self):
        model = _copy(two_head_model())
        del model.graph.output[1]
        self._check(model, "io")

    def test_two_head_outputs_must_be_the_head_outputs(self):
        model = _copy(two_head_model())
        model.graph.output[1].CopyFrom(_value("relu1", [1, 5, 8, 8]))
        self._check(model, "outputs")

    def test_two_head_external_tensors_must_be_float32_8x8(self):
        model = _copy(two_head_model())
        model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = 4
        self._check(model, "external")

    def test_two_head_convolutions_require_constants(self):
        model = _copy(two_head_model())
        del model.graph.node[2].input[2]
        self._check(model, "constants")

    def test_two_head_constants_must_be_float32(self):
        model = _copy(two_head_model())
        _set_initializer(model, "wa", _weights(3, 5, 1, 3).astype(np.float64))
        self._check(model, "dtype")

    def test_two_head_stem_shape_is_bounded(self):
        model = _copy(two_head_model(hidden=2))
        self._check(model, "stem")

    def test_two_head_stem_attributes_are_bounded(self):
        model = _copy(two_head_model())
        _set_attr(model.graph.node[0], "strides", [2, 2])
        self._check(model, "stem_attrs")

    def test_two_head_heads_must_be_dense(self):
        model = _copy(two_head_model())
        _set_initializer(model, "wa", _weights(3, 5, 5, 9))
        _set_attr(model.graph.node[2], "kernel_shape", [5, 5])
        _set_attr(model.graph.node[2], "pads", [2, 2, 2, 2])
        self._check(model, "head")

    def test_two_head_head_attributes_are_bounded(self):
        model = _copy(two_head_model())
        _set_attr(model.graph.node[2], "strides", [2, 2])
        self._check(model, "head_attrs")

    def test_two_head_3x3_heads_require_pad1(self):
        model = _copy(two_head_model(kernel_a=3))
        _drop_attr(model.graph.node[2], "pads")
        self._check(model, "pads")


# ---------------------------------------------------------------------------
# Diamond: shared stem, two heads, one join, optional tail
# ---------------------------------------------------------------------------


def diamond_model(hidden=8, kernel_a=1, kernel_b=3, kind="Add", seed=3, relu=True, tail=(), infer=True,
                  output_shape=SPATIAL):
    """`input -> Conv[, Relu] -> t -> {head_a, head_b} -> join -> [Conv, Relu]* Conv`."""
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1])]
    source = "stem"
    if relu:
        nodes.append(h.make_node("Relu", ["stem"], ["relu1"]))
        source = "relu1"
    nodes.append(h.make_node("Conv", [source, "wa", "ba"], ["head_a"],
                             kernel_shape=[kernel_a, kernel_a], pads=[kernel_a // 2] * 4))
    nodes.append(h.make_node("Conv", [source, "wb", "bb"], ["head_b"],
                             kernel_shape=[kernel_b, kernel_b], pads=[kernel_b // 2] * 4))
    nodes.append(h.make_node(kind, ["head_a", "head_b"], ["joined"]))
    initializers = [
        _initializer("w1", _weights(hidden, 3, 1, seed)),
        _initializer("b1", _bias(hidden, seed + 1)),
        _initializer("wa", _weights(3, hidden, kernel_a, seed + 2)),
        _initializer("ba", _bias(3, seed + 3)),
        _initializer("wb", _weights(3, hidden, kernel_b, seed + 4)),
        _initializer("bb", _bias(3, seed + 5)),
    ]
    source = "joined"
    for index, kernel in enumerate(tail):
        final = index == len(tail) - 1
        target = "output" if final else f"tail{index}"
        initializers += [_initializer(f"wt{index}", _weights(3, 3, kernel, seed + 10 + index)),
                         _initializer(f"bt{index}", _bias(3, seed + 20 + index))]
        nodes.append(h.make_node("Conv", [source, f"wt{index}", f"bt{index}"], [target],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        if not final:
            nodes.append(h.make_node("Relu", [target], [f"tail{index}r"]))
            source = f"tail{index}r"
        else:
            source = target
    if not tail:
        nodes[-1].output[0] = "output"
    value_info = []
    if relu:
        value_info.append(_value("relu1", [1, hidden, 8, 8]))
    else:
        value_info.append(_value("stem", [1, hidden, 8, 8]))
    return _finish(nodes, [_value("input", SPATIAL)], [_value("output", output_shape)],
                   initializers, value_info, infer)


class DiamondRejectionTests(unittest.TestCase):
    """Every `compile_diamond` guard with its message; the unmutated model compiles."""

    MESSAGE = {
        "domain": "diamond profile requires default-domain nodes",
        "join": "diamond profile requires stem.,Relu., two heads and one join",
        "attributes": "diamond join must be Add, Mul, Sub or Max without attributes",
        "tail": "diamond tail must be .Conv, Relu.\\* Conv",
        "tail_relu": "diamond tail Relu must not carry attributes",
        "stem": "diamond stem must be Conv or Conv,Relu",
        "heads": "diamond heads must be Conv",
        "stem_relu": "diamond stem Relu must consume the stem output",
        "head_input": "diamond heads must consume the stem output",
        "join_order": "diamond join must consume both head outputs in order",
        "tail_input": "diamond tail must consume the join output",
        "tail_order": "diamond tail nodes must be connected in order",
        "io": "diamond profile requires one input and one output",
        "external": "diamond external tensors must be float32 .1,3,8,8.",
        "head_shape": "diamond heads must produce float32 .1,3,8,8.",
        "tail_shape": "diamond tail must produce float32 .1,3,8,8.",
        "constants": "diamond convolutions require constant float32 weights and bias",
        "dtype": "diamond constants must be float32",
        "stem_shape": "diamond stem must be a 1x1 Conv with hidden channels 3..16",
        "stem_attrs": "unsupported stem attributes",
        "head_dense": "diamond heads must be dense 3-output 1x1/3x3 Conv with hidden inputs",
        "head_attrs": "unsupported head attributes",
        "pads": "diamond 3x3 heads require symmetric pad1",
        "operands": "Mul operand zero points must be two INT8 values",
        "operand_kind": "operand zero points apply only to the Mul join",
        "override": "the diamond output override requires the Mul join or a Conv tail",
        "tail_dense": "diamond tail layers must be dense 3-output 1x1/3x3 Conv with three inputs",
        "tail_attrs": "unsupported tail attributes",
        "tail_pads": "diamond 3x3 tail layers require symmetric pad1",
    }

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def _check(self, model, key, expect=None, **kwargs):
        if expect is None:
            expect = self.MESSAGE[key]
        with self.assertRaisesRegex(ValueError, expect):
            compile_diamond(model, **kwargs)

    def test_diamond_valid_neighbours_compile(self):
        # Prove the boundary before the rejection subtests below rely on the
        # shape being otherwise legal.
        for kind in ("Add", "Mul", "Sub", "Max"):
            with self.subTest(kind=kind):
                binary, meta = compile_diamond(diamond_model(kind=kind))
                self.assertEqual(meta["profile"], "diamond-join")
                self.assertEqual(meta["join"], kind)
                self.assertEqual(decode_sequence(binary)["task_count"], 4)

    def test_diamond_rejects_foreign_domain(self):
        model = _copy(diamond_model())
        model.graph.node[0].domain = "com.example"
        _add_opset(model, "com.example")
        self._check(model, "domain")

    def test_diamond_requires_a_join(self):
        model = _finish([h.make_node("Add", ["input", "input"], ["output"])],
                        [_value("input", SPATIAL)], [_value("output", SPATIAL)], [])
        self._check(model, "join")

    def test_diamond_join_must_be_attribute_free(self):
        model = _copy(diamond_model())
        _set_attr(model.graph.node[4], "axis", 1)
        with _checker_disabled():
            self._check(model, "attributes")

    def test_diamond_tail_shape_alternates(self):
        # A second Conv appended after the single Conv tail makes the tail list
        # even, so the `[Conv, Relu]* Conv` guard rejects it.
        model = _copy(diamond_model(tail=(1,)))
        model.graph.node.append(h.make_node("Conv", ["output", "wt0", "bt0"], ["output2"],
                                            kernel_shape=[1, 1]))
        model.graph.output[0].CopyFrom(_value("output2", SPATIAL))
        self._check(model, "tail")

    def test_diamond_tail_relu_must_be_attribute_free(self):
        model = _copy(diamond_model(tail=(1, 1, 1)))
        _set_attr(model.graph.node[6], "alpha", 0.5)
        with _checker_disabled():
            self._check(model, "tail_relu")

    def test_diamond_stem_shape_is_bounded(self):
        model = _copy(diamond_model())
        model.graph.node.insert(1, h.make_node("Conv", ["stem", "w1", "b1"], ["stem2"],
                                               kernel_shape=[1, 1]))
        self._check(model, "stem")

    def test_diamond_heads_must_be_conv(self):
        model = _copy(diamond_model())
        _replace_node(model, 3, h.make_node("Relu", ["relu1"], ["head_b"]))
        self._check(model, "heads")

    def test_diamond_stem_relu_must_consume_the_stem(self):
        model = _copy(diamond_model())
        model.graph.node[1].input[0] = "input"
        self._check(model, "stem_relu")

    def test_diamond_heads_must_consume_the_stem(self):
        model = _copy(diamond_model())
        model.graph.node[3].input[0] = "stem"
        self._check(model, "head_input")

    def test_diamond_join_operand_order(self):
        model = _copy(diamond_model())
        join = model.graph.node[4]
        join.input[0], join.input[1] = join.input[1], join.input[0]
        self._check(model, "join_order")

    def test_diamond_tail_must_consume_the_join(self):
        model = _copy(diamond_model(tail=(3,)))
        model.graph.node[5].input[0] = "head_a"
        self._check(model, "tail_input")

    def test_diamond_tail_nodes_must_be_connected(self):
        model = _copy(diamond_model(tail=(1, 1, 1)))
        model.graph.node[6].input[0] = "head_a"
        self._check(model, "tail_order")

    def test_diamond_requires_one_input_and_output(self):
        model = _copy(diamond_model())
        model.graph.input.append(_value("extra", SPATIAL))
        self._check(model, "io")

    def test_diamond_external_tensors_must_be_float32_8x8(self):
        model = _copy(diamond_model())
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 4
        self._check(model, "external")

    def test_diamond_head_output_shape_is_checked(self):
        model = _copy(diamond_model())
        _set_value_info(model, "head_a", [1, 3, 4, 4])
        self._check(model, "head_shape")

    def test_diamond_tail_output_shape_is_checked(self):
        model = _copy(diamond_model(tail=(1, 1, 1)))
        _set_value_info(model, "tail0", [1, 3, 4, 4])
        self._check(model, "tail_shape")

    def test_diamond_convolutions_require_constants(self):
        model = _copy(diamond_model())
        del model.graph.node[2].input[2]
        self._check(model, "constants")

    def test_diamond_constants_must_be_float32(self):
        model = _copy(diamond_model())
        _set_initializer(model, "wa", _weights(3, 8, 1, 3).astype(np.float64))
        self._check(model, "dtype")

    def test_diamond_stem_must_be_1x1(self):
        model = _copy(diamond_model())
        _set_initializer(model, "w1", _weights(8, 3, 3, 2))
        _set_attr(model.graph.node[0], "kernel_shape", [3, 3])
        _set_attr(model.graph.node[0], "pads", [1, 1, 1, 1])
        self._check(model, "stem_shape")

    def test_diamond_stem_attributes_are_bounded(self):
        model = _copy(diamond_model())
        _set_attr(model.graph.node[0], "strides", [2, 2])
        self._check(model, "stem_attrs")

    def test_diamond_heads_must_be_dense(self):
        model = _copy(diamond_model())
        _set_initializer(model, "wa", _weights(3, 8, 5, 3))
        _set_attr(model.graph.node[2], "kernel_shape", [5, 5])
        _set_attr(model.graph.node[2], "pads", [2, 2, 2, 2])
        self._check(model, "head_dense")

    def test_diamond_head_attributes_are_bounded(self):
        model = _copy(diamond_model())
        _set_attr(model.graph.node[2], "strides", [2, 2])
        self._check(model, "head_attrs")

    def test_diamond_3x3_heads_require_pad1(self):
        model = _copy(diamond_model(kernel_b=3))
        _drop_attr(model.graph.node[3], "pads")
        self._check(model, "pads")

    def test_diamond_operand_zero_points_are_bounded(self):
        model = diamond_model(kind="Mul")
        self._check(model, "operands", operand_zero_points=(200, 0))
        self._check(model, "operands", operand_zero_points=(0, 0, 0))

    def test_diamond_operand_zero_points_only_for_mul(self):
        self._check(diamond_model(kind="Add"), "operand_kind", operand_zero_points=(1, 0))

    def test_diamond_output_override_needs_mul_or_tail(self):
        self._check(diamond_model(kind="Add"), "override",
                    output_range={"scale": 0.5, "zero_point": 0})

    def test_diamond_tail_layers_must_be_dense(self):
        model = _copy(diamond_model(tail=(3,)))
        _set_initializer(model, "wt0", _weights(3, 3, 5, 12))
        _set_attr(model.graph.node[5], "kernel_shape", [5, 5])
        _set_attr(model.graph.node[5], "pads", [2, 2, 2, 2])
        self._check(model, "tail_dense")

    def test_diamond_tail_attributes_are_bounded(self):
        model = _copy(diamond_model(tail=(3,)))
        _set_attr(model.graph.node[5], "strides", [2, 2])
        self._check(model, "tail_attrs")

    def test_diamond_3x3_tail_requires_pad1(self):
        model = _copy(diamond_model(tail=(3,)))
        _drop_attr(model.graph.node[5], "pads")
        self._check(model, "tail_pads")


class DiamondAcceptanceTests(unittest.TestCase):
    """The accepted `compile_diamond` bodies and the container they declare."""

    def test_diamond_join_kinds_and_tails_compile(self):
        cases = (
            ("Add", (), "diamond-join"),
            ("Sub", (), "diamond-join"),
            ("Max", (), "diamond-join"),
            ("Mul", (), "diamond-join"),
            ("Add", (3,), "diamond-tail"),
            ("Mul", (1, 1, 1), "diamond-tail"),
        )
        for kind, tail, profile in cases:
            with self.subTest(kind=kind, tail=tail):
                binary, meta = compile_diamond(diamond_model(kind=kind, tail=tail))
                self.assertEqual(meta["profile"], profile)
                self.assertEqual(meta["join"], kind)
                self.assertEqual(meta["tail_kernels"], list(tail))
                self.assertEqual(meta["shape_nhwc"], [1, 8, 8, 3])
                self.assertEqual(meta["output_tensors"], ["output"])
                info = decode_sequence(binary)
                # stem + two heads + join + one task per tail layer
                self.assertEqual(info["task_count"], 4 + len(tail))
                self.assertEqual(info["output_tensor_count"], 1)
                self.assertAlmostEqual(info["output_scale"], meta["output_scale"], places=6)
                self.assertEqual(info["output_zero_point"], meta["output_zero_point"])

    def test_diamond_tail_layers_carry_their_activation(self):
        _, meta = compile_diamond(diamond_model(tail=(1, 1, 1)))
        self.assertEqual([entry["relu"] for entry in meta["tail_quantization"]], [True, True, False])
        self.assertEqual(meta["tail_kernels"], [1, 1, 1])

    def test_diamond_mul_operand_zero_points_reach_the_join(self):
        # Non-zero operand zero points add the extra conversion register pair.
        binary, meta = compile_diamond(diamond_model(kind="Mul"), operand_zero_points=(5, -3))
        self.assertEqual(meta["operand_zero_points"], [5, -3])
        self.assertEqual(decode_sequence(binary)["task_count"], 4)
        # Non-zero zero points are only legal for the Mul join.
        with self.assertRaisesRegex(ValueError, "operand zero points apply only to the Mul join"):
            compile_diamond(diamond_model(kind="Add"), operand_zero_points=(5, -3))

    def test_diamond_output_overrides_resolve_the_band(self):
        model = diamond_model(kind="Mul")
        binary, meta = compile_diamond(model, output_range={"scale": 0.4, "zero_point": 11})
        self.assertAlmostEqual(meta["output_scale"], 0.4, places=6)
        self.assertEqual(meta["output_zero_point"], 11)
        self.assertAlmostEqual(decode_sequence(binary)["output_scale"], 0.4, places=6)
        # A Conv tail carries the override instead of the join.
        binary, meta = compile_diamond(diamond_model(kind="Add", tail=(1,)),
                                       output_range={"scale": 0.2, "zero_point": -4})
        self.assertAlmostEqual(meta["output_scale"], 0.2, places=6)
        self.assertEqual(meta["output_zero_point"], -4)
        self.assertAlmostEqual(decode_sequence(binary)["output_scale"], 0.2, places=6)


# ---------------------------------------------------------------------------
# Join chain: 3..8 heads folded left to right, optional tail and runtime tails
# ---------------------------------------------------------------------------


def chain_model(heads=3, kinds=("Add", "Add"), hidden=8, stem_relu=False, kernels=None, groups=None,
                tail=(), runtime=None, operand_shape=(1, 3, 1, 1), image_shape=SPATIAL,
                output_shape=SPATIAL, infer=True, stem_dtype=np.float32, head_dtypes=None,
                value_info=(), extra_input=None):
    """`stem -> n heads -> n-1 chained joins -> [tail] -> [runtime tail] -> output`."""
    kinds = tuple(kinds)
    kernels = list(kernels) if kernels is not None else [1] * heads
    groups = list(groups) if groups is not None else [1] * heads
    head_dtypes = list(head_dtypes) if head_dtypes is not None else [np.float32] * heads
    initializers = [_initializer("w1", _weights(hidden, 3, 1, 11).astype(stem_dtype)),
                    _initializer("b1", _bias(hidden, 12))]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1])]
    source = "stem"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem"], ["stem_relu"]))
        source = "stem_relu"
    for index in range(heads):
        kernel, group = kernels[index], groups[index]
        in_channels = hidden // group
        initializers += [_initializer(f"wh{index}",
                                      _weights(3, in_channels, kernel, 21 + index).astype(head_dtypes[index])),
                         _initializer(f"bh{index}", _bias(3, 31 + index))]
        attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        if group != 1:
            attributes["group"] = group
        nodes.append(h.make_node("Conv", [source, f"wh{index}", f"bh{index}"], [f"h{index}"],
                                 **attributes))
    previous = "h0"
    for position, kind in enumerate(kinds):
        nodes.append(h.make_node(kind, [previous, f"h{position + 1}"], [f"j{position}"]))
        previous = f"j{position}"
    source = previous
    for index, kernel in enumerate(tail):
        target = f"tail{index}"
        initializers += [_initializer(f"wt{index}", _weights(3, 3, kernel, 41 + index)),
                         _initializer(f"bt{index}", _bias(3, 51 + index))]
        nodes.append(h.make_node("Conv", [source, f"wt{index}", f"bt{index}"], [target],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        if index < len(tail) - 1:
            nodes.append(h.make_node("Relu", [target], [f"tail{index}r"]))
            source = f"tail{index}r"
        else:
            source = target
    inputs = [_value("input", image_shape)]
    if runtime is not None:
        external, shape, op = ("scale", operand_shape, "Mul") if runtime == "scale" else \
            ("residual", SPATIAL, runtime)
        inputs.append(_value(external, shape))
        nodes.append(h.make_node(op, [source, external], ["output"]))
    else:
        nodes[-1].output[0] = "output"
    if extra_input is not None:
        inputs.append(_value(extra_input, SPATIAL))
    info = list(value_info) if value_info else [_value("stem" if not stem_relu else "stem_relu",
                                                       [1, hidden, 8, 8])]
    return _finish(nodes, inputs, [_value("output", output_shape)], initializers, info, infer)


class JoinChainRecognitionTests(unittest.TestCase):
    """`parse_join_chain` / `_classify_join_chain` boundaries."""

    def test_parse_join_chain_rejects_scale_before_a_tail(self):
        # A runtime per-channel scale followed by a Conv tail: the classifier
        # drops the whole graph rather than guessing which tail the scale is.
        model = _copy(chain_model(heads=3, kinds=("Add", "Add"), runtime="scale"))
        model.graph.initializer.extend([_initializer("wt0", _weights(3, 3, 1, 45)),
                                        _initializer("bt0", _bias(3, 55))])
        model.graph.node.append(h.make_node("Conv", ["output", "wt0", "bt0"], ["output2"],
                                            kernel_shape=[1, 1]))
        model.graph.output[0].CopyFrom(_value("output2", SPATIAL))
        nodes = list(model.graph.node)
        externals = {value.name for value in model.graph.input}
        self.assertIsNone(parse_join_chain(nodes, externals))
        with self.assertRaisesRegex(ValueError, "join chain requires stem"):
            compile_join_chain(model)

    def test_parse_join_chain_reports_the_left_fold(self):
        model = chain_model(heads=4, kinds=("Add", "Mul", "Max"), kernels=(1, 3, 1, 1))
        nodes = list(model.graph.node)
        externals = {value.name for value in model.graph.input}
        parts = parse_join_chain(nodes, externals)
        self.assertEqual(parts["head_count"], 4)
        self.assertEqual(parts["join_count"], 3)
        self.assertTrue(parts["stem_end"] in (1, 2))
        self.assertFalse(parts["runtime_scale"])


class JoinChainRejectionTests(unittest.TestCase):
    """Every `compile_join_chain` guard with its message and a compiling neighbour."""

    MESSAGE = {
        "domain": "join chain requires default-domain nodes",
        "classify": "join chain requires stem",
        "operands": "chained Mul joins require zero-centered operands",
        "stem_relu": "join chain stem Relu must consume the stem output",
        "heads": "join chain heads must be Conv nodes reading the stem output",
        "joins": "join chain joins must be attribute-free Add/Mul/Sub/Max nodes",
        "order": "join chain joins must consume the previous result and the next head in order",
        "tail_relu": "join chain tail Relu must not carry attributes",
        "tail_input": "join chain tail must consume the last join output",
        "tail_order": "join chain tail nodes must be connected in order",
        "runtime": "join chain runtime tail must combine the last join result with one external input",
        "io": "join chain requires one image input and one output",
        "external": "join chain external tensors must be float32 .1,3,8,8.",
        "head_shape": "join chain heads must produce float32 .1,3,8,8.",
        "tail_shape": "join chain tail must produce float32 .1,3,8,8.",
        "constants": "join chain convolutions require constant float32 weights and bias",
        "dtype": "join chain constants must be float32",
        "stem_shape": "join chain stem must be a 1x1 Conv with hidden channels 3..16",
        "stem_attrs": "unsupported stem attributes",
        "head_dense": "join chain heads must be dense 3-output 1x1/3x3 Conv with hidden inputs",
        "head_attrs": "unsupported head attributes",
        "pads": "join chain 3x3 heads require symmetric pad1",
        "depthwise": "join chain depthwise heads require a three-channel stem",
        "depthwise_attrs": "unsupported depthwise head attributes",
        "tail_dense": "join chain tail layers must be dense 3-output 1x1/3x3 Conv with three inputs",
        "tail_attrs": "unsupported tail attributes",
        "tail_pads": "join chain 3x3 tail layers require symmetric pad1",
    }

    def _check(self, model, key, **kwargs):
        with self.assertRaisesRegex(ValueError, self.MESSAGE[key]):
            compile_join_chain(model, **kwargs)

    def test_join_chain_valid_neighbours_compile(self):
        for model in (chain_model(), chain_model(heads=3, kinds=("Add", "Add"), tail=(1, 1, 1)),
                      chain_model(heads=3, kinds=("Add", "Add"), runtime="scale")):
            with self.subTest(profile=compile_join_chain(model)[1]["profile"]):
                binary, meta = compile_join_chain(model)
                self.assertEqual(meta["head_count"], 3)
                self.assertEqual(decode_sequence(binary)["format_version"], 5)

    def test_join_chain_rejects_foreign_domain(self):
        model = _copy(chain_model())
        model.graph.node[0].domain = "com.example"
        _add_opset(model, "com.example")
        self._check(model, "domain")

    def test_join_chain_requires_a_recognisable_chain(self):
        model = chain_model(heads=2, kinds=("Add",))
        self._check(model, "classify")

    def test_join_chain_requires_zero_centered_operands(self):
        self._check(chain_model(), "operands", operand_zero_points=(1, 0))

    def test_join_chain_stem_relu_must_consume_the_stem(self):
        model = _copy(chain_model(stem_relu=True))
        model.graph.node[1].input[0] = "input"
        self._check(model, "stem_relu")

    def test_join_chain_heads_must_read_the_stem(self):
        model = _copy(chain_model())
        model.graph.node[1].input[0] = "input"
        # A head that does not read the stem is not classified as a head at all:
        # the classifier drops the graph before the guard can see it.
        self._check(model, "classify")

    def test_join_chain_joins_must_be_attribute_free(self):
        model = _copy(chain_model())
        _set_attr(model.graph.node[4], "axis", 1)
        with _checker_disabled():
            self._check(model, "joins")

    def test_join_chain_joins_must_fold_left_to_right(self):
        # The last join reuses head0 instead of head2: the fold order is wrong.
        model = _copy(chain_model())
        model.graph.node[5].input[1] = "h0"
        self._check(model, "order")

    def test_join_chain_tail_relu_must_be_attribute_free(self):
        model = _copy(chain_model(tail=(1, 1, 1)))
        _set_attr(model.graph.node[7], "alpha", 0.5)
        with _checker_disabled():
            self._check(model, "tail_relu")

    def test_join_chain_tail_must_consume_the_last_join(self):
        model = _copy(chain_model(tail=(1,)))
        model.graph.node[6].input[0] = "h0"
        self._check(model, "tail_input")

    def test_join_chain_tail_nodes_must_be_connected(self):
        model = _copy(chain_model(tail=(1, 1, 1)))
        model.graph.node[7].input[0] = "h0"
        self._check(model, "tail_order")

    def test_join_chain_runtime_tail_must_use_the_join_result(self):
        model = _copy(chain_model(runtime="scale"))
        model.graph.node[6].input[0] = "h0"
        self._check(model, "runtime")

    def test_join_chain_requires_one_image_input_and_one_output(self):
        model = chain_model(extra_input="extra")
        self._check(model, "io")

    def test_join_chain_external_tensors_must_be_float32_8x8(self):
        model = chain_model(output_shape=[1, 3, 4, 4], infer=False)
        self._check(model, "external")

    def test_join_chain_head_output_shape_is_checked(self):
        model = _copy(chain_model())
        _set_value_info(model, "h1", [1, 3, 4, 4])
        self._check(model, "head_shape")

    def test_join_chain_tail_output_shape_is_checked(self):
        # With no inferred value_info the tail's intermediate Conv output has no
        # declared shape, which the emitter refuses to guess.
        model = chain_model(tail=(1, 1, 1), infer=False,
                            value_info=[_value("stem", [1, 8, 8, 8]), _value("h0", SPATIAL),
                                        _value("h1", SPATIAL), _value("h2", SPATIAL)])
        self._check(model, "tail_shape")

    def test_join_chain_convolutions_require_constants(self):
        model = _copy(chain_model())
        del model.graph.node[2].input[2]
        self._check(model, "constants")

    def test_join_chain_constants_must_be_float32(self):
        model = _copy(chain_model())
        _set_initializer(model, "wh0", _weights(3, 8, 1, 22).astype(np.float64))
        self._check(model, "dtype")

    def test_join_chain_stem_must_be_1x1(self):
        model = _copy(chain_model())
        _set_initializer(model, "w1", _weights(8, 3, 3, 14))
        _set_attr(model.graph.node[0], "kernel_shape", [3, 3])
        _set_attr(model.graph.node[0], "pads", [1, 1, 1, 1])
        self._check(model, "stem_shape")

    def test_join_chain_stem_attributes_are_bounded(self):
        model = _copy(chain_model())
        _set_attr(model.graph.node[0], "strides", [2, 2])
        self._check(model, "stem_attrs")

    def test_join_chain_heads_must_be_dense(self):
        model = _copy(chain_model())
        _set_initializer(model, "wh0", _weights(3, 8, 5, 24))
        _set_attr(model.graph.node[1], "kernel_shape", [5, 5])
        _set_attr(model.graph.node[1], "pads", [2, 2, 2, 2])
        self._check(model, "head_dense")

    def test_join_chain_head_attributes_are_bounded(self):
        model = _copy(chain_model())
        _set_attr(model.graph.node[1], "strides", [2, 2])
        self._check(model, "head_attrs")

    def test_join_chain_3x3_heads_require_pad1(self):
        model = _copy(chain_model(kernels=(3, 1, 1)))
        _drop_attr(model.graph.node[1], "pads")
        self._check(model, "pads")

    def test_join_chain_depthwise_heads_require_a_three_channel_stem(self):
        model = chain_model(hidden=8, kernels=(3, 1, 1), groups=(3, 1, 1))
        self._check(model, "depthwise")

    def test_join_chain_depthwise_attributes_are_bounded(self):
        model = _copy(chain_model(hidden=3, kernels=(3, 1, 1), groups=(3, 1, 1)))
        _set_attr(model.graph.node[1], "strides", [2, 2])
        self._check(model, "depthwise_attrs")

    def test_join_chain_tail_layers_must_be_dense(self):
        model = _copy(chain_model(tail=(3,)))
        _set_initializer(model, "wt0", _weights(3, 3, 5, 44))
        _set_attr(model.graph.node[6], "kernel_shape", [5, 5])
        _set_attr(model.graph.node[6], "pads", [2, 2, 2, 2])
        self._check(model, "tail_dense")

    def test_join_chain_tail_attributes_are_bounded(self):
        model = _copy(chain_model(tail=(3,)))
        _set_attr(model.graph.node[6], "strides", [2, 2])
        self._check(model, "tail_attrs")

    def test_join_chain_3x3_tail_requires_pad1(self):
        model = _copy(chain_model(tail=(3,)))
        _drop_attr(model.graph.node[6], "pads")
        self._check(model, "tail_pads")


class JoinChainAcceptanceTests(unittest.TestCase):
    """The accepted chain bodies: head counts, mixed joins, tails and depthwise heads."""

    def test_three_to_five_heads_with_mixed_joins(self):
        cases = (
            (3, ("Add", "Add"), (1, 1, 1), None),
            (4, ("Mul", "Add", "Max"), (1, 3, 1, 1), None),
            (5, ("Add", "Mul", "Sub", "Max"), (1, 3, 1, 3, 1), None),
        )
        for heads, kinds, kernels, tail in cases:
            with self.subTest(heads=heads, kinds=kinds):
                binary, meta = compile_join_chain(chain_model(heads=heads, kinds=kinds,
                                                              kernels=kernels, tail=tail or ()))
                self.assertEqual(meta["profile"], "join-chain")
                self.assertEqual(meta["head_count"], heads)
                self.assertEqual(meta["join_count"], heads - 1)
                self.assertEqual(meta["join_kinds"], list(kinds))
                self.assertEqual(meta["head_kinds"], ["dense"] * heads)
                info = decode_sequence(binary)
                self.assertEqual(info["task_count"], 1 + heads + (heads - 1))
                self.assertAlmostEqual(info["output_scale"], meta["output_scale"], places=6)
                self.assertEqual(info["output_zero_point"], meta["output_zero_point"])

    def test_chain_tail_carries_its_activation(self):
        binary, meta = compile_join_chain(chain_model(tail=(1, 3, 1)))
        self.assertEqual(meta["profile"], "join-chain-tail")
        self.assertEqual(meta["tail_kernels"], [1, 3, 1])
        self.assertEqual([entry["relu"] for entry in meta["tail_quantization"]], [True, True, False])
        self.assertEqual(decode_sequence(binary)["task_count"], 1 + 3 + 2 + 3)

    def test_depthwise_heads_compose_with_dense_heads(self):
        binary, meta = compile_join_chain(
            chain_model(hidden=3, kinds=("Add", "Mul"), kernels=(3, 1, 1), groups=(3, 1, 1)))
        self.assertEqual(meta["head_kinds"], ["depthwise", "dense", "dense"])
        self.assertEqual(meta["profile"], "join-chain")
        info = decode_sequence(binary)
        self.assertEqual(info["task_count"], 1 + 3 + 2)
        self.assertEqual(info["output_tensor_count"], 1)
        self.assertIn("depthwise", [entry for entry in meta["head_kinds"]])

    def test_runtime_per_channel_scale_tail(self):
        binary, meta = compile_join_chain(chain_model(runtime="scale"))
        self.assertEqual(meta["profile"], "join-chain-scale")
        self.assertEqual(meta["runtime_tail"], {"kind": "Mul", "operand_scale": OPERAND_SCALE,
                                                "conversion": meta["runtime_scale"]["conversion"],
                                                "shift": meta["runtime_scale"]["shift"]})
        self.assertIsNotNone(meta["runtime_scale"])
        info = decode_sequence(binary)
        self.assertEqual(info["input_tensor_count"], 2)
        self.assertEqual(info["task_count"], 1 + 3 + 2 + 1)
        scale = next(t for t in info["tensors"] if t["name"] == "scale")
        self.assertEqual(scale["shape_nhwc"] if "shape_nhwc" in scale else
                         (scale["batch"], scale["height"], scale["width"], scale["channels"]),
                         (1, 1, 1, 3))

    def test_runtime_residual_tails(self):
        for kind in ("Add", "Sub", "Max"):
            with self.subTest(kind=kind):
                binary, meta = compile_join_chain(chain_model(runtime=kind))
                self.assertEqual(meta["profile"], "join-chain-residual")
                self.assertEqual(meta["runtime_tail"], {"kind": kind, "residual": True})
                info = decode_sequence(binary)
                self.assertEqual(info["input_tensor_count"], 2)
                self.assertEqual(info["task_count"], 1 + 3 + 2 + 1)
                residual = next(t for t in info["tensors"] if t["name"] == "scale")
                self.assertEqual((residual["height"], residual["width"]), (8, 8))


# ---------------------------------------------------------------------------
# Independent float64 parity for the composed references
# ---------------------------------------------------------------------------


def _float_conv(volume, weights, bias, pad):
    out_channels, _, kernel, _ = weights.shape
    padded = np.pad(volume, ((pad, pad), (pad, pad), (0, 0))) if pad else volume
    height, width, _ = volume.shape
    accumulator = np.zeros((height, width, out_channels), np.float64)
    for kh in range(kernel):
        for kw in range(kernel):
            accumulator += padded[kh:kh + height, kw:kw + width, :] @ weights[:, :, kh, kw].T
    return accumulator + bias


def _requantize(real, scale, zero_point):
    return np.clip(np.rint(real / np.float64(scale)) + zero_point, -128, 127).astype(np.int64)


def _dequantize(codes, scale, zero_point):
    return (codes.astype(np.float64) - zero_point) * np.float64(scale)


_JOIN_OPS = {"Add": np.add, "Sub": np.subtract, "Max": np.maximum, "Mul": np.multiply}


def _independent_head_grids(image, meta):
    """Stem plus every dense head, dequantized/requantized from the declared bands."""
    stem_band = meta["stem_quantization"]
    stem = _float_conv(image.astype(np.float64), np.asarray(meta["stem_weights"], np.float64),
                       np.asarray(meta["stem_bias"], np.float64), 0)
    if stem_band.get("relu"):
        stem = np.maximum(stem, 0.0)
    stem_codes = _requantize(stem, stem_band["output_scale"], stem_band["output_zero_point"])
    stem_volume = _dequantize(stem_codes, stem_band["output_scale"], stem_band["output_zero_point"])
    grids = []
    for index, weights in enumerate(meta["weights"]):
        band = meta["head_quantization"][index]
        kernel = meta["head_kernels"][index]
        real = _float_conv(stem_volume, np.asarray(weights, np.float64),
                           np.asarray(meta["bias"][index], np.float64), kernel // 2)
        grids.append(_requantize(real, band["output_scale"], band["output_zero_point"]))
    return grids


def _independent_dense_pipeline(image, meta):
    """Return the joined codes on the final join band for a diamond or chain."""
    grids = _independent_head_grids(image, meta)
    kinds = meta["join_kinds"] if "join_kinds" in meta else [meta["join"]]
    value = _dequantize(grids[0], meta["head_scales"][0], meta["head_zero_points"][0])
    for position, kind in enumerate(kinds):
        other = _dequantize(grids[position + 1], meta["head_scales"][position + 1],
                            meta["head_zero_points"][position + 1])
        value = _JOIN_OPS[kind](value, other)
        if "join_scales" in meta:
            scale = meta["join_scales"][position]
            zero = meta["join_zero_points"][position]
        else:
            scale = float(np.float32(128 * meta["head_scales"][0] * meta["head_scales"][1])) \
                if kind == "Mul" else float(np.float32(2 * meta["head_scales"][0]))
            zero = meta["join_zero_point"]
        value = _dequantize(_requantize(value, scale, zero), scale, zero)
    return _requantize(value, scale, zero), scale, zero


def _independent_tails(codes, scale, zero, meta):
    for index, band in enumerate(meta["tail_quantization"]):
        kernel = meta["tail_kernels"][index]
        real = _float_conv(_dequantize(codes, scale, zero),
                           np.asarray(meta["tail_weights"][index], np.float64),
                           np.asarray(meta["tail_bias"][index], np.float64), kernel // 2)
        if band.get("relu"):
            real = np.maximum(real, 0.0)
        codes = _requantize(real, band["output_scale"], band["output_zero_point"])
        scale, zero = band["output_scale"], band["output_zero_point"]
    return _requantize(_dequantize(codes, scale, zero),
                       meta["output_scale"], meta["output_zero_point"])


def _quantization_object(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


def _sample_images(count=16, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (count, 8, 8, 3), dtype=np.uint8)


def _assert_within_lsb(case, got, expected, tolerance=LSB_TOLERANCE):
    difference = np.abs(got.astype(np.int64) - expected.astype(np.int64)).max()
    case.assertLessEqual(int(difference), tolerance,
                         f"independent float64 parity exceeded {tolerance} LSB")


class ReferenceParityTests(unittest.TestCase):
    """Compose the emitter's references against arithmetic written in this file."""

    def test_diamond_reference_matches_independent_float64(self):
        for kind in ("Add", "Sub", "Max", "Mul"):
            with self.subTest(kind=kind):
                _, meta = compile_diamond(diamond_model(kind=kind))
                images = _sample_images()
                expected = np.stack([_independent_dense_pipeline(image, meta)[0]
                                     for image in images])
                got = np.stack([diamond_reference(image, meta["stem_quantization"],
                                                  meta["head_quantization"], meta["join"])
                                for image in images])
                _assert_within_lsb(self, got, expected)

    def test_diamond_reference_with_tail_matches_independent_float64(self):
        _, meta = compile_diamond(diamond_model(kind="Add", tail=(1, 3, 1)))
        images = _sample_images(8, seed=4)
        expected = np.stack([_independent_tails(*_independent_dense_pipeline(image, meta),
                                                meta) for image in images])
        got = np.stack([diamond_reference(image, meta["stem_quantization"],
                                          meta["head_quantization"], meta["join"],
                                          meta["tail_quantization"], meta["join_zero_point"])
                        for image in images])
        _assert_within_lsb(self, got, expected)

    def test_diamond_reference_accepts_quantization_objects(self):
        _, meta = compile_diamond(diamond_model())
        image = _sample_images(1)[0]
        stem = _quantization_object(meta["stem_quantization"])
        heads = [_quantization_object(params) for params in meta["head_quantization"]]
        baseline = diamond_reference(image, meta["stem_quantization"], meta["head_quantization"],
                                     meta["join"])
        self.assertTrue(np.array_equal(diamond_reference(image, stem, heads, meta["join"]),
                                       baseline))

    def test_join_chain_reference_matches_independent_float64(self):
        for kinds, tail in ((("Add", "Add"), ()), (("Mul", "Add", "Max"), ()),
                            (("Add", "Mul"), (1, 3, 1))):
            with self.subTest(kinds=kinds, tail=tail):
                _, meta = compile_join_chain(chain_model(heads=len(kinds) + 1, kinds=kinds,
                                                         tail=tail))
                images = _sample_images()
                expected = []
                for image in images:
                    codes, scale, zero = _independent_dense_pipeline(image, meta)
                    expected.append(_independent_tails(codes, scale, zero, meta))
                got = np.stack([diamond_reference(image, meta["stem_quantization"],
                                                  meta["head_quantization"], meta["join_kinds"],
                                                  meta["tail_quantization"], meta["join_zero_point"])
                                for image in images])
                _assert_within_lsb(self, got, np.stack(expected))

    def test_join_chain_scale_reference_matches_independent_float64(self):
        # The runtime tail folds the join band with a fixed operand band; the
        # ideal float64 product and the container's multiplier/shift agree to
        # within one LSB (the head grid is also allowed one).
        _, meta = compile_join_chain(chain_model(runtime="scale"))
        codes = np.asarray([9, -14, 127], np.int32)
        images = _sample_images(8)
        expected = []
        for image in images:
            grid, scale, zero = _independent_dense_pipeline(image, meta)
            real = _dequantize(grid, scale, zero) * _dequantize(codes.reshape(1, 1, 3),
                                                                OPERAND_SCALE, 0)
            output_scale = float(np.float32(128 * scale * OPERAND_SCALE))
            expected.append(_requantize(real, output_scale, meta["output_zero_point"]))
        got = np.stack([join_chain_scale_reference(image, meta["stem_quantization"],
                                                   meta["head_quantization"], meta["join_kinds"],
                                                   codes, runtime_scale=meta["runtime_scale"],
                                                   output_zero_point=meta["output_zero_point"])
                        for image in images])
        _assert_within_lsb(self, got, np.stack(expected))

    def test_join_chain_scale_reference_without_conversion(self):
        # `runtime_scale=None` takes the ideal `mul_reference` path of the helper.
        _, meta = compile_join_chain(chain_model(runtime="scale"))
        codes = np.asarray([-7, 0, 31], np.int32)
        image = _sample_images(1, seed=9)[0]
        grid, scale, zero = _independent_dense_pipeline(image, meta)
        expected = _requantize(_dequantize(grid, scale, zero) * _dequantize(codes.reshape(1, 1, 3),
                                                                           OPERAND_SCALE, 0),
                               float(np.float32(128 * scale * OPERAND_SCALE)), 0)
        got = join_chain_scale_reference(image, meta["stem_quantization"],
                                         meta["head_quantization"], meta["join_kinds"], codes,
                                         runtime_scale=None)
        _assert_within_lsb(self, got, expected)

    def test_join_chain_residual_reference_matches_independent_float64(self):
        for kind in ("Add", "Sub", "Max"):
            with self.subTest(kind=kind):
                _, meta = compile_join_chain(chain_model(runtime=kind))
                images = _sample_images(8, seed=11)
                residual = np.random.default_rng(13).integers(-128, 128, (8, 8, 3), dtype=np.int64)
                expected = []
                for image in images:
                    grid, scale, zero = _independent_dense_pipeline(image, meta)
                    real = _JOIN_OPS[kind](_dequantize(grid, scale, zero),
                                           _dequantize(residual, scale, 0))
                    expected.append(_requantize(real, float(np.float32(2 * scale)), 0))
                got = np.stack([join_chain_residual_reference(
                    image, meta["stem_quantization"], meta["head_quantization"], meta["join_kinds"],
                    kind, residual) for image in images])
                _assert_within_lsb(self, got, np.stack(expected))

    def test_two_head_profile_matches_the_recorded_board_evidence(self):
        import json

        root = Path(__file__).resolve().parents[1] / "research" / "two_head_suite"
        manifest = json.loads((root / "manifest.json").read_text())
        checked = 0
        for entry in manifest[:4]:
            index = entry["index"]
            _, meta = compile_two_head(onnx.load(root / f"model{index:03}.onnx"))
            images = np.fromfile(root / f"input{index:03}.u8", dtype=np.uint8)
            images = images.reshape(entry["cases"], 8, 8, 3)
            expected = np.fromfile(root / f"expected{index:03}.i8", dtype=np.int8)
            expected = expected.reshape(entry["cases"], 2, 8, 8, 3)
            for case_index, image in enumerate(images):
                grids = _independent_head_grids(image, meta)
                for head in range(2):
                    _assert_within_lsb(self, grids[head], expected[case_index, head])
                    checked += 1
        self.assertGreaterEqual(checked, 128)


if __name__ == "__main__":
    unittest.main()
