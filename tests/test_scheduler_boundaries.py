# SPDX-License-Identifier: MIT
"""Pins the scheduler's profile-selection boundary rejections.

`open_rknpu.scheduler` dispatches a normalized ONNX graph to one emitter only after a
cheap structural match, and every profile that consumes an already-quantized DAG or a
pooled/branched shape then re-checks the *quantization boundary* the emitters were
measured against: input UINT8 scale 1.0 / zero point 0, no operand zero points unless
the join is a Mul, and no output/calibration override the profile cannot honour.  Those
guards sit at the exact lines a wrong argument must stop at, and an accidental fall
through would silently compile a graph with a band the retained board evidence does not
cover.

This module builds the smallest graph that matches each profile and calls
`compile_sequence` with one wrong argument, asserting the *exact* rejection string so a
renamed or reordered guard is caught.  It also pins `batched_supported`'s serial-container
reason and the successful single-program batched path.

Note on the single-program batched case: no emitter reachable through
`compile_sequence` returns a legacy ``ORNPUBIN`` container any more (all of them wrap
their payload in an ``ORNPUSEQ`` sequence), so the ``else`` branch of the batched
post-pass is exercised by patching the `_compile_sequence` seam with a genuine legacy
container.  The test pins that the container is passed through untouched and reported
as one engine run.

No board, no network and no sleep: every case is a fresh in-memory model compiled in a
temporary directory.
"""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.compiler import compile_model
from open_rknpu.model import decode, encode
from open_rknpu.scheduler import batched_supported, compile_sequence
from open_rknpu.sequence import decode_sequence

D, W = "dense", "depthwise"


def compile_graph(model, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path, **kwargs)


def finish(graph, name):
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def conv(weights, bias, source, out, kernel=1, group=1):
    attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
    if group != 1:
        attributes["group"] = group
    return h.make_node("Conv", [source, weights, bias], [out], **attributes)


def conv_model(seed=0):
    """A single 1x1 Conv, the smallest sequence container (one program)."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.4, 0.4, (3, 3, 1, 1)).astype(np.float32)
    bias = rng.uniform(-1, 1, (3,)).astype(np.float32)
    graph = h.make_graph(
        [conv("w", "b", "input", "output")], "conv",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    return finish(graph, "conv")


def join_dag_model(seed=0):
    """Stem plus three heads folded by two joins, the general (non-chain) join DAG."""
    rng = np.random.default_rng(seed)
    hidden = 3
    initializers = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                    nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1])]
    for position, kernel in enumerate((1, 3, 1)):
        initializers += [nh.from_array(rng.uniform(.02, .06, (3, hidden, kernel, kernel)).astype(np.float32),
                                       f"wh{position}"),
                         nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{position}")]
        nodes.append(conv(f"wh{position}", f"bh{position}", "stem", f"h{position}", kernel))
    # The second join reads h0 again, so the left-fold chain parser declines and the
    # general join DAG (which tolerates reused tensors) takes the graph.
    nodes.append(h.make_node("Add", ["h0", "h1"], ["j0"]))
    nodes.append(h.make_node("Mul", ["j0", "h0"], ["j1"]))
    graph = h.make_graph(
        nodes, "join_dag",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("j1", 1, [1, 3, 8, 8])], initializers)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, hidden, 8, 8]))
    return finish(graph, "join_dag")


def pooled_branches_model(joins=("Add",), hidden=3, pool="MaxPool"):
    """A stem, two pooled Conv-chain branches and one or two joins.

    Each branch carries two Conv layers so the shape is longer than the five-node
    `pool_join` profile and is claimed by the pooled-branches parser instead.
    """
    branches = [[(D, 3, 3, 1), (D, 3, 3, 1)], [(D, 3, 3, 1)]]
    rng = np.random.default_rng(len(branches) * 17 + sum(len(b) for b in branches))
    initializers = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                    nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "bs_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "bs_stem"], ["stem"], kernel_shape=[1, 1])]
    channels_of = {"stem": hidden}
    pooled = []
    for branch_index, layers in enumerate(branches):
        previous = "stem"
        for layer_index, (kind, in_channels, out_channels, kernel) in enumerate(layers):
            shape = (out_channels, 1, kernel, kernel) if kind == W else (out_channels, in_channels, kernel, kernel)
            initializers += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32),
                                           f"w{branch_index}_{layer_index}"),
                             nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32),
                                           f"bs{branch_index}_{layer_index}")]
            name = f"b{branch_index}_{layer_index}"
            nodes.append(conv(f"w{branch_index}_{layer_index}", f"bs{branch_index}_{layer_index}",
                              previous, name, kernel, group=out_channels if kind == W else 1))
            previous = name
            channels_of[name] = out_channels
        pool_name = f"p{branch_index}"
        nodes.append(h.make_node(pool, [previous], [pool_name],
                                 kernel_shape=[2, 2], strides=[2, 2]))
        pooled.append(pool_name)
    last = None
    for position, kind in enumerate(joins):
        first = pooled[0] if position == 0 else last
        nodes.append(h.make_node(kind, [first, pooled[position + 1]], [f"j{position}"]))
        last = f"j{position}"
    graph = h.make_graph(
        nodes, "pooled_branches",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(last, 1, [1, 3, 4, 4])], initializers)
    for name, channels in channels_of.items():
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, channels, 8, 8]))
    for position in range(len(branches)):
        graph.value_info.append(h.make_tensor_value_info(f"p{position}", 1, [1, 3, 4, 4]))
    for position in range(len(joins)):
        graph.value_info.append(h.make_tensor_value_info(f"j{position}", 1, [1, 3, 4, 4]))
    return finish(graph, "pooled_branches")


def diamond_model(seed=0):
    """Stem plus two Conv heads and an Add join.

    The second head is written as ``Conv -> Mul(const)`` so the *un-normalized* graph
    contains a Mul (passing the early "operand zero points need a Mul" pre-check) while
    `normalize_model` folds the constant Mul into the Conv, leaving a clean diamond whose
    join is Add.
    """
    rng = np.random.default_rng(seed)
    weights = {name: rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
               for name in ("ws", "wa", "wb")}
    biases = {name: rng.uniform(-4, 4, (3,)).astype(np.float32) for name in ("bs", "ba", "bb")}
    nodes = [conv("ws", "bs", "input", "stem"),
             conv("wa", "ba", "stem", "head_a"),
             conv("wb", "bb", "stem", "mid"),
             h.make_node("Mul", ["mid", "factor"], ["head_b"]),
             h.make_node("Add", ["head_a", "head_b"], ["output"])]
    initializers = [nh.from_array(weights[name], name) for name in ("ws", "wa", "wb")]
    initializers += [nh.from_array(biases[name], name) for name in ("bs", "ba", "bb")]
    initializers.append(nh.from_array(np.float32(1.5), "factor"))
    graph = h.make_graph(
        nodes, "diamond",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], initializers)
    for name in ("stem", "head_a", "head_b", "mid"):
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, 3, 8, 8]))
    return finish(graph, "diamond")


def chain_walk_model(seed=0):
    """A linear Conv chain with an interior pool and a folded constant Mul.

    `parse_chain` refuses any node that is not a Conv or a pool, so the graph needs a
    Mul for the early operand-zero-point pre-check and no surviving Mul for the chain
    walk: `Conv -> Mul(const)` is folded into one Conv by `normalize_model`, leaving
    ``Conv -> Relu -> MaxPool -> Conv`` with the pool before the last node.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
    b1 = rng.uniform(-4, 4, (3,)).astype(np.float32)
    w2 = rng.uniform(-.7, .8, (3, 3, 3, 3)).astype(np.float32)
    b2 = rng.uniform(-4, 4, (3,)).astype(np.float32)
    nodes = [conv("w1", "b1", "input", "c1"),
             h.make_node("Mul", ["c1", "factor"], ["m1"]),
             h.make_node("Relu", ["m1"], ["r1"]),
             h.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
             conv("w2", "b2", "p1", "output", kernel=3)]
    initializers = [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"),
                    nh.from_array(w2, "w2"), nh.from_array(b2, "b2"),
                    nh.from_array(np.float32(1.5), "factor")]
    graph = h.make_graph(
        nodes, "chain_walk",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])], initializers)
    for name, channels, size in (("c1", 3, 8), ("m1", 3, 8), ("r1", 3, 8), ("p1", 3, 4)):
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, channels, size, size]))
    return finish(graph, "chain_walk")


def two_head_model(hidden=5, kernel_a=1, kernel_b=3, seed=7):
    """Two Conv heads over one Relu'd stem, both exposed as graph outputs."""
    rng = np.random.default_rng(seed)
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-4, 4, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, kernel_a, kernel_a)).astype(np.float32), "wa"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "ba"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, kernel_b, kernel_b)).astype(np.float32), "wb"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "bb")]
    nodes = [conv("w1", "b1", "input", "stem"),
             h.make_node("Relu", ["stem"], ["relu1"]),
             conv("wa", "ba", "relu1", "outputA", kernel_a),
             conv("wb", "bb", "relu1", "outputB", kernel_b)]
    graph = h.make_graph(
        nodes, "two_head",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("outputA", 1, [1, 3, 8, 8]),
         h.make_tensor_value_info("outputB", 1, [1, 3, 8, 8])], initializers)
    graph.value_info.append(h.make_tensor_value_info("relu1", 1, [1, hidden, 8, 8]))
    return finish(graph, "two_head")


def multi_input_model(seed=13):
    """Two Conv branches, one join, then one Conv/Mul pair over a third external input."""
    rng = np.random.default_rng(seed)

    def branch():
        return (rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32),
                rng.uniform(-2, 2, (3,)).astype(np.float32))

    wa, ba = branch()
    wb, bb = branch()
    wc, bc = branch()
    initializers = [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"),
                    nh.from_array(wb, "wb"), nh.from_array(bb, "bb"),
                    nh.from_array(wc, "wc"), nh.from_array(bc, "bc")]
    nodes = [conv("wa", "ba", "a", "A"),
             conv("wb", "bb", "b", "B"),
             h.make_node("Add", ["A", "B"], ["C"]),
             conv("wc", "bc", "c", "D"),
             h.make_node("Mul", ["C", "D"], ["out"])]
    graph = h.make_graph(
        nodes, "multi_input",
        [h.make_tensor_value_info(name, 1, [1, 3, 8, 8]) for name in ("a", "b", "c")],
        [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])], initializers)
    return finish(graph, "multi_input")


def elementwise_dag_model(seed=31):
    """Two Conv branches into Add, then one Mul: the elementwise-chain DAG."""
    rng = np.random.default_rng(seed)
    wa = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
    ba = rng.uniform(-2, 2, (3,)).astype(np.float32)
    wb = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
    bb = rng.uniform(-2, 2, (3,)).astype(np.float32)
    nodes = [conv("wa", "ba", "a", "A"),
             conv("wb", "bb", "b", "B"),
             h.make_node("Add", ["A", "B"], ["C"]),
             h.make_node("Mul", ["C", "A"], ["out"])]
    graph = h.make_graph(
        nodes, "elementwise_chain",
        [h.make_tensor_value_info(name, 1, [1, 3, 8, 8]) for name in ("a", "b")],
        [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])],
        [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"),
         nh.from_array(wb, "wb"), nh.from_array(bb, "bb")])
    return finish(graph, "elementwise_chain")


class BoundaryCase(unittest.TestCase):
    """Assert a compile is refused with one exact scheduler message."""

    def assert_rejected(self, model, message, **kwargs):
        with self.assertRaises(ValueError) as caught:
            compile_graph(model, **kwargs)
        self.assertEqual(str(caught.exception), message)


class BatchedSubmissionBoundaryTests(unittest.TestCase):
    def test_batched_supported_rejects_a_serial_container(self):
        # A freshly compiled container carries terminal tails (`serial=True`), so its
        # submission shape is already one job per task.
        binary, _ = compile_graph(conv_model())
        info = decode_sequence(binary)
        self.assertTrue(info["serial"])
        with self.assertRaises(ValueError) as caught:
            batched_supported(binary, info)
        self.assertEqual(str(caught.exception),
                         "batched submission rejected: the container is serial (one task per submission)")

    def test_legacy_single_program_container_is_submitted_as_one_run(self):
        # No emitter reachable through `compile_sequence` returns the legacy one-program
        # ORNPUBIN container, so the batched post-pass's legacy branch is pinned by
        # supplying a genuine legacy container at the compiler seam.
        model = conv_model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "conv.onnx"
            onnx.save(model, path)
            payload, metadata = compile_model(path)
        legacy = encode(payload, metadata)
        self.assertEqual(legacy[:8], b"ORNPUBIN")
        with mock.patch("open_rknpu.scheduler._compile_sequence",
                        return_value=(legacy, dict(metadata))):
            binary, meta = compile_sequence(model, submission="batched")
        self.assertEqual(binary, legacy)
        self.assertEqual(meta["engine_runs"], [1])
        self.assertEqual(meta["submission"], "batched")
        info = decode(binary)
        self.assertEqual(info["task_count"], 1)
        self.assertEqual(info["format_version"], 1)


class JoinDagBoundaryTests(BoundaryCase):
    def test_join_dag_requires_the_uint8_boundary(self):
        self.assert_rejected(join_dag_model(),
                             "join DAG requires the established UINT8 scale1/zero-point0 boundary",
                             input_scale=0.5, input_zero_point=128)


class PooledBranchesBoundaryTests(BoundaryCase):
    def test_pooled_branches_requires_the_uint8_boundary(self):
        self.assert_rejected(pooled_branches_model(),
                             "pooled branches require the established UINT8 scale1/zero-point0 boundary",
                             input_scale=0.5, input_zero_point=128)

    def test_pooled_branches_rejects_calibration(self):
        self.assert_rejected(pooled_branches_model(),
                             "calibration is unsupported for pooled branches",
                             calibration_ranges={"output": {"scale": 1.0, "zero_point": 0}})

    def test_pooled_branches_rejects_mul_operand_zero_points(self):
        # The Mul join keeps the operand zero points plausible, so the pre-check passes
        # and the profile guard is what refuses the request.
        self.assert_rejected(pooled_branches_model(joins=("Mul",)),
                             "Mul operand zero points require a Mul profile",
                             mul_operand_zero_points=(1, 0))


class TwoHeadBoundaryTests(BoundaryCase):
    def test_two_head_requires_the_uint8_boundary(self):
        self.assert_rejected(two_head_model(),
                             "two-head profile requires the established UINT8 scale1/zero-point0 boundary",
                             input_scale=0.5, input_zero_point=128)


class DiamondBoundaryTests(BoundaryCase):
    def test_diamond_requires_the_uint8_boundary(self):
        self.assert_rejected(diamond_model(),
                             "diamond profile requires the established UINT8 scale1/zero-point0 boundary",
                             input_scale=0.5, input_zero_point=128)

    def test_diamond_rejects_mul_operand_zero_points(self):
        # The folded constant Mul keeps the early pre-check satisfied; the surviving
        # diamond's join is an Add, so operand zero points have no Mul to live on.
        self.assert_rejected(diamond_model(),
                             "Mul operand zero points require a Mul profile",
                             mul_operand_zero_points=(1, 0))


class ElementwiseDagBoundaryTests(BoundaryCase):
    def test_multi_input_dag_requires_the_uint8_boundary(self):
        self.assert_rejected(multi_input_model(),
                             "multi-input elementwise DAG requires the established UINT8 scale1/zero-point0 boundary",
                             input_scale=0.5, input_zero_point=128)

    def test_multi_input_dag_rejects_output_override(self):
        self.assert_rejected(multi_input_model(),
                             "output override unsupported for the multi-input elementwise DAG",
                             output_range={"scale": 1.0, "zero_point": 0})

    def test_elementwise_dag_requires_the_uint8_boundary(self):
        self.assert_rejected(elementwise_dag_model(),
                             "elementwise DAG requires the established UINT8 scale1/zero-point0 boundary",
                             input_scale=0.5, input_zero_point=128)


class ChainWalkBoundaryTests(BoundaryCase):
    def test_chain_walk_rejects_mul_operand_zero_points(self):
        # The interior pool rules out every profile above the chain walk; the folded
        # constant Mul satisfies the early pre-check without surviving in the parsed chain.
        self.assert_rejected(chain_walk_model(),
                             "Mul operand zero points require a Mul profile",
                             mul_operand_zero_points=(1, 0))


if __name__ == "__main__":
    unittest.main()
