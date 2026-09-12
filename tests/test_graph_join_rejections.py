# SPDX-License-Identifier: MIT
"""Pins the join/transposed/depthwise lines the rest of the suite never runs.

The scheduler's bounded profiles reject malformed graphs with an exact message, and
a few of those messages (and two emitter *success* branches) were left uncovered by
the family suites.  This module reaches each one with the smallest hand-built ONNX
graph that gets there, or - where the scheduler's dispatch cannot route to the guard -
by calling the emitter's own entry point with the crafted model.  Every rejection
asserts the *exact* string, so a reworded or removed guard fails here.

Covered here:

* ``join_dag._prepare``'s committed-band path (``join_dag.py:342``): two branch
  intermediates built from one shared first layer share one band, so an ``Add`` over
  them takes the already-committed band instead of re-quantizing, and the DAG still
  emits a decodable container;
* the dense ConvTranspose geometry derivers (``transposed.py:28/29/30`` from a
  static ``output_shape`` and ``transposed.py:32`` from ``SAME_UPPER``/``SAME_LOWER``)
  plus their bounded-profile rejections (``transposed.py:19/27/121/134``);
* the depthwise stem/channel/pointwise guards (``depthwise.py:67/74/136``).

Not reachable by construction, and therefore *not* faked with a substituted planner
(the concrete reason is in each comment where the guard lives):

* ``graph.py:473``, ``graph.py:846``, ``join_dag.py:471``, ``compose.py:250`` - the
  placement policy always separates externals from internals and gives the diamond's
  two head buffers adjacent slots, so those overlap/adjacency tests cannot fire;
* ``graph.py:598`` - ``_classify_join_chain`` only ever puts a ``Conv`` reading the
  stem output into ``heads``, which is exactly what the guard re-checks;
* ``join_dag.py:347/350`` - a join operand that is not committed is always an
  uncommitted branch final, which is always a ``by_name`` key, and the "free" operand
  of the one-sided branch is never itself committed;
* ``join_dag.py:529`` - ``_prepare`` unconditionally recompiles every branch final and a
  depthwise entry can only be a single-layer branch final, so ``entry['recompiled']`` is
  never ``None``;
* ``liveness.py:165`` - the candidate at the end of the highest placed tensor can never
  overlap any placed tensor, so first-fit always finds a slot.

No vendor artifact, board or network is read; every compile is host-only and
deterministic (fixed-weight builders, no RNG).
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.depthwise import compile_depthwise_pointwise
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.transposed import compile_transposed


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def compile_file(model, function=compile_sequence):
    """Save `model` and run `function` on the file, as the family suites do."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return function(path)


def join_dag_shared_band_model():
    """Three dense branches, two Adds; branches 0 and 1 share their first layer.

    Sharing the first layer makes the two *intermediate* tensors ``h0_0`` and ``h1_0``
    carry identical quantization, so the first ``Add`` (which folds two already
    committed intermediates) can take the committed band directly instead of raising
    "needs both operands on one scale" - the ``join_dag.py:342`` path.
    """
    constants = [nh.from_array(np.full((3, 3, 1, 1), 0.05, np.float32), "w_stem"),
                 nh.from_array(np.zeros(3, np.float32), "b_stem"),
                 nh.from_array(np.full((3, 3, 1, 1), 0.03, np.float32), "w_first"),
                 nh.from_array(np.full((3,), 0.5, np.float32), "b_first")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1])]
    for branch in range(2):
        constants += [nh.from_array(np.full((3, 3, 1, 1), 0.02 + 0.01 * branch, np.float32),
                                    "w_second%d" % branch),
                      nh.from_array(np.full((3,), 0.1 * branch, np.float32), "b_second%d" % branch)]
        nodes.append(h.make_node("Conv", ["stem", "w_first", "b_first"], ["h%d_0" % branch],
                                 kernel_shape=[1, 1]))
        nodes.append(h.make_node("Conv", ["h%d_0" % branch, "w_second%d" % branch,
                                          "b_second%d" % branch], ["h%d_1" % branch],
                                 kernel_shape=[1, 1]))
    constants += [nh.from_array(np.full((3, 3, 1, 1), 0.04, np.float32), "w_third"),
                  nh.from_array(np.zeros(3, np.float32), "b_third")]
    nodes.append(h.make_node("Conv", ["stem", "w_third", "b_third"], ["h2_0"], kernel_shape=[1, 1]))
    nodes.append(h.make_node("Add", ["h0_0", "h1_0"], ["j0"]))
    nodes.append(h.make_node("Add", ["j0", "h2_0"], ["j1"]))
    graph = h.make_graph(nodes, "join_dag_shared_band",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("j1", 1, [1, 3, 8, 8])], constants)
    for name in ("h0_0", "h0_1", "h1_0", "h1_1", "h2_0", "j0", "j1"):
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def dense_transposed_model(attributes, stem_output=(1, 3, 8, 8), output=(1, 3, 8, 8)):
    """A 1x1 dense stem followed by one dense (group 1) K3 ConvTranspose."""
    nodes = [h.make_node("Conv", ["input", "stem_w", "stem_b"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("ConvTranspose", ["stem", "w", "b"], ["output"], group=1, **attributes)]
    graph = h.make_graph(nodes, "dense_transposed",
        [h.make_tensor_value_info("input", 1, [1, 3, stem_output[2], stem_output[3]])],
        [h.make_tensor_value_info("output", 1, list(output))],
        [nh.from_array(np.full((3, 3, 1, 1), 0.1, np.float32), "stem_w"),
         nh.from_array(np.zeros(3, np.float32), "stem_b"),
         nh.from_array(np.full((3, 3, 3, 3), 0.1, np.float32), "w"),
         nh.from_array(np.full((3,), 0.25, np.float32), "b")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, list(stem_output)))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def single_transposed_model(weight_shape, attributes):
    """The smallest model the K3-dilation2/rectangular rewrites read: one node."""
    nodes = [h.make_node("ConvTranspose", ["input", "w"], ["output"], **attributes)]
    graph = h.make_graph(nodes, "transposed_rewrite",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(np.full(weight_shape, 0.1, np.float32), "w")])
    graph.value_info.append(h.make_tensor_value_info("input", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def depthwise_model(stem_kernel, dw_group, stem_hidden=3):
    """A Conv stem followed by one grouped Conv, for the depthwise stem guards."""
    nodes = [h.make_node("Conv", ["input", "stem_w", "stem_b"], ["stem"],
                         kernel_shape=[stem_kernel, stem_kernel],
                         pads=[stem_kernel // 2] * 4),
             h.make_node("Conv", ["stem", "dw_w", "dw_b"], ["output"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=dw_group)]
    graph = h.make_graph(nodes, "depthwise_guard",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, dw_group, 8, 8])],
        [nh.from_array(np.full((stem_hidden, 3, stem_kernel, stem_kernel), 0.1, np.float32), "stem_w"),
         nh.from_array(np.zeros(stem_hidden, np.float32), "stem_b"),
         nh.from_array(np.full((dw_group, 1, 3, 3), 0.1, np.float32), "dw_w"),
         nh.from_array(np.zeros(dw_group, np.float32), "dw_b")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, stem_hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def depthwise_pointwise_two_node_model():
    """Two nodes where the pointwise emitter needs three."""
    nodes = [h.make_node("Conv", ["input", "stem_w", "stem_b"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["stem", "dw_w", "dw_b"], ["output"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=3)]
    graph = h.make_graph(nodes, "depthwise_pointwise_guard",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(np.full((3, 3, 1, 1), 0.1, np.float32), "stem_w"),
         nh.from_array(np.zeros(3, np.float32), "stem_b"),
         nh.from_array(np.full((3, 1, 3, 3), 0.1, np.float32), "dw_w"),
         nh.from_array(np.zeros(3, np.float32), "dw_b")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class JoinDagBandTests(unittest.TestCase):
    def test_reused_intermediates_take_their_committed_band(self):
        """Two intermediates on one shared band: the Add takes it (join_dag.py:342)."""
        binary, meta = compile_file(join_dag_shared_band_model())
        self.assertEqual(meta["profile"], "join-dag")
        info = decode_sequence(binary)
        # stem + two 2-layer branches + one 1-layer branch + two joins.
        self.assertEqual(info["task_count"], 8)
        # The Add folded two tensors that already shared the first layer's band, so it
        # did not re-quantize either: the join scale is exactly twice that band.
        intermediate = meta["branch_quantization"][0][0]["output_scale"]
        self.assertEqual(meta["join_scales"][0], float(np.float32(2 * intermediate)))
        self.assertEqual(meta["head_names"], ["h0_1", "h1_1", "h2_0"])


class DenseTransposedGeometryTests(unittest.TestCase):
    def test_output_shape_derives_bounded_pads(self):
        """`output_shape=[8,8]` becomes pads[1,1,1,1] (transposed.py:28/29/30)."""
        binary, meta = compile_file(dense_transposed_model(
            dict(kernel_shape=[3, 3], strides=[1, 1], output_shape=[8, 8])))
        info = decode_sequence(binary)
        self.assertEqual(info["task_count"], 2)
        self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 3])
        self.assertEqual(meta["transposed_profile"],
                         "8x8-dense-c3-to-c3-k3-s1x1-pads[1, 1, 1, 1]")
        # The derived pads must equal the pads the same emitter gets when the host
        # spells them out: the geometry is the module's own reference.
        explicit, _ = compile_file(dense_transposed_model(
            dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])))
        self.assertEqual(binary, explicit)

    def test_same_auto_pad_derives_the_same_pads(self):
        """`SAME_UPPER`/`SAME_LOWER` derive pads[1,1,1,1] (transposed.py:32)."""
        explicit, _ = compile_file(dense_transposed_model(
            dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])))
        for auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
            with self.subTest(auto_pad=auto_pad):
                binary, meta = compile_file(dense_transposed_model(
                    dict(kernel_shape=[3, 3], strides=[1, 1], auto_pad=auto_pad)))
                self.assertEqual(decode_sequence(binary)["task_count"], 2)
                self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 3])
                self.assertEqual(binary, explicit)

    def test_dense_requires_an_eight_by_eight_stem(self):
        model = dense_transposed_model(dict(kernel_shape=[3, 3], strides=[1, 1],
                                            pads=[1, 1, 1, 1]),
                                       stem_output=(1, 3, 7, 7), output=(1, 3, 7, 7))
        with self.assertRaises(ValueError) as caught:
            compile_transposed(model)
        self.assertEqual(str(caught.exception),
                         "dense ConvTranspose requires Conv[/Relu] stem with static 8x8 C1..16 output")

    def test_output_shape_needs_two_dimensions(self):
        model = dense_transposed_model(dict(kernel_shape=[3, 3], strides=[1, 1],
                                            output_shape=[8, 8, 8]))
        with self.assertRaises(ValueError) as caught:
            compile_transposed(model)
        self.assertEqual(str(caught.exception),
                         "dense ConvTranspose output_shape requires two dimensions and no explicit pads")

    def test_k3_dilation2_needs_square_three_by_three_weights(self):
        model = single_transposed_model((3, 1, 3, 5),
                                        dict(group=3, kernel_shape=[3, 3], strides=[1, 1],
                                             dilations=[2, 2]))
        with self.assertRaises(ValueError) as caught:
            compile_transposed(model)
        self.assertEqual(str(caught.exception),
                         "K3 dilation2 ConvTranspose requires constant float32 "
                         "(C_in,C_out/group,3,3) weights")

    def test_rectangular_rewrite_needs_padding_below_its_kernel(self):
        model = single_transposed_model((3, 1, 1, 3),
                                        dict(group=3, kernel_shape=[1, 3], strides=[1, 1],
                                             pads=[0, 0, 0, 3]))
        with self.assertRaises(ValueError) as caught:
            compile_transposed(model)
        self.assertEqual(str(caught.exception),
                         "rectangular ConvTranspose rewrite requires per-axis padding below its kernel")


class DepthwiseGuardTests(unittest.TestCase):
    def test_input_must_match_the_group_count(self):
        with self.assertRaises(ValueError) as caught:
            compile_file(depthwise_model(stem_kernel=1, dw_group=2))
        self.assertEqual(str(caught.exception), "depthwise input must match group count at 8x8")

    def test_stem_must_be_one_by_one_or_three_by_three(self):
        with self.assertRaises(ValueError) as caught:
            compile_file(depthwise_model(stem_kernel=5, dw_group=3))
        self.assertEqual(str(caught.exception),
                         "depthwise profile supports only 1x1 or 3x3 stem convolution")

    def test_pointwise_needs_a_three_node_chain(self):
        with self.assertRaises(ValueError) as caught:
            compile_depthwise_pointwise(depthwise_pointwise_two_node_model())
        self.assertEqual(str(caught.exception),
                         "depthwise-pointwise requires Conv[/Relu] -> depthwise Conv -> pointwise Conv")


if __name__ == "__main__":
    unittest.main()
