"""Diamond DAG profile: shared stem, two consuming heads, one elementwise join.

The board suite that pins exact bytes is `research/diamond_suite/`; this module
checks the container, the lifetime plan and the composed integer reference
against the float ONNX model.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.graph import diamond_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence


def diamond(hidden=8, kernel_a=1, kernel_b=3, kind="Add", seed=3, relu=True):
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1])]
    source = "stem"
    if relu:
        nodes.append(h.make_node("Relu", ["stem"], ["relu1"]))
        source = "relu1"
    nodes.append(h.make_node("Conv", [source, "wa", "ba"], ["head_a"],
                             kernel_shape=[kernel_a, kernel_a], pads=[kernel_a // 2] * 4))
    nodes.append(h.make_node("Conv", [source, "wb", "bb"], ["head_b"],
                             kernel_shape=[kernel_b, kernel_b], pads=[kernel_b // 2] * 4))
    nodes.append(h.make_node(kind, ["head_a", "head_b"], ["output"]))
    initializers = [
        nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
        nh.from_array(rng.uniform(-4, 4, (hidden,)).astype(np.float32), "b1"),
        nh.from_array(rng.uniform(-.7, .8, (3, hidden, kernel_a, kernel_a)).astype(np.float32), "wa"),
        nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "ba"),
        nh.from_array(rng.uniform(-.7, .8, (3, hidden, kernel_b, kernel_b)).astype(np.float32), "wb"),
        nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "bb"),
    ]
    graph = h.make_graph(nodes, "diamond",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], initializers)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class DiamondTests(unittest.TestCase):
    def compile(self, model, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path, **kwargs)

    def test_container_schedule_and_lifetimes(self):
        binary, meta = self.compile(diamond())
        info = decode_sequence(binary)
        # The op-level walk dispatches this class and is byte-identical to the
        # diamond emitter (tests/test_walk.py); the profile name follows the dispatch.
        self.assertEqual(meta["profile"], "walk-join")
        self.assertEqual(info["task_count"], 4)
        self.assertEqual([t["name"] for t in info["tensors"]],
                         ["input0", "stem", "head_a", "head_b", "output"])
        self.assertEqual([t["role_name"] for t in info["tensors"]],
                         ["input", "internal", "internal", "internal", "output"])
        self.assertEqual(meta["schedule"], ["stem", "head_a", "head_b", "output"])
        self.assertEqual(meta["tensor_lifetimes"], {"input0": [-1, 0], "stem": [0, 2],
                                                    "head_a": [1, 3], "head_b": [2, 3], "output": [3, 3]})
        offsets = meta["tensor_offsets"]
        self.assertEqual(offsets["head_b"] - offsets["head_a"], 8 * 8 * 16)
        self.assertEqual(meta["allocated_bytes"], 3 * 8 * 8 * 16)
        self.assertEqual(meta["lifetime_bytes"], 3 * 8 * 8 * 16)
        externals = [t for t in info["tensors"] if t["role_name"] != "internal"]
        internals = [t for t in info["tensors"] if t["role_name"] == "internal"]
        for a in externals:
            for b in internals:
                self.assertFalse(a["byte_offset"] < b["byte_offset"] + b["bytes"]
                                 and b["byte_offset"] < a["byte_offset"] + a["bytes"])

    def test_reference_tracks_the_float_graph(self):
        rng = np.random.default_rng(11)
        joins = {"Add": 1, "Mul": 2, "Sub": 3, "Max": 4}
        for kind, offset in joins.items():
            for hidden, kernel_a, kernel_b in ((8, 1, 3), (3, 3, 1), (16, 1, 1)):
                with self.subTest(kind=kind, hidden=hidden):
                    model = diamond(hidden, kernel_a, kernel_b, kind, seed=101 * hidden + offset)
                    _, meta = self.compile(model)
                    self.assertEqual([int(z) for z in meta["head_zero_points"]], [0, 0])
                    cases = rng.integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
                    reference = np.stack([diamond_reference(case, meta["stem_quantization"],
                                                            meta["head_quantization"], kind)
                                          for case in cases])
                    value = reference.astype(np.float64) * meta["output_scale"]
                    expected = np.stack([
                        ReferenceEvaluator(model).run(None, {"input": case[None].transpose(0, 3, 1, 2).astype(np.float32)})[0][0].transpose(1, 2, 0)
                        for case in cases])
                    error = float(np.abs(value - expected).max())
                    if kind == "Mul":
                        # The product of two INT8 operands can cancel almost exactly,
                        # so the span is not a fair denominator. The error is bounded
                        # by the propagated head errors plus half an output step, and
                        # a factor-of-128 scale mistake would exceed it by 64x.
                        bound = (127 * meta["head_scales"][0] * meta["head_scales"][1] * 2
                                 + 0.5 * meta["output_scale"])
                        self.assertLess(error, 2 * meta["output_scale"])
                        self.assertLess(error, bound)
                    else:
                        span = float(expected.max() - expected.min())
                        self.assertLess(error, 0.25 * span)

    def test_shared_scale_join_uses_one_grid_per_branch(self):
        _, meta = self.compile(diamond(kind="Add"))
        self.assertEqual([q["output_zero_point"] for q in meta["head_quantization"]], [0, 0])
        self.assertEqual(meta["head_scales"][0], meta["head_scales"][1])
        self.assertAlmostEqual(meta["output_scale"], 2 * meta["head_scales"][0], places=4)
        _, mul_meta = self.compile(diamond(kind="Mul"))
        self.assertAlmostEqual(mul_meta["output_scale"], 128 * mul_meta["head_scales"][0] * mul_meta["head_scales"][1], places=1)

    def test_optional_stem_relu_and_other_profiles_are_not_hijacked(self):
        _, meta = self.compile(diamond(relu=True))
        # The op-level walk dispatches this class and is byte-identical to the
        # diamond emitter (tests/test_walk.py); the profile name follows the dispatch.
        self.assertEqual(meta["profile"], "walk-join")
        # A stem without Relu is supported by the diamond profile.
        _, no_relu = self.compile(diamond(relu=False))
        self.assertEqual(no_relu["profile"], "walk-join")
        # Two independent branches that both read the graph input are the verified
        # elementwise profile, not a diamond (regression for the dispatch guard).
        for name in ("model000", "model003"):
            path = Path(__file__).resolve().parents[1] / "research" / "mul_after_relu_suite" / f"{name}.onnx"
            if not path.exists():
                continue
            with self.subTest(name=name):
                _, other = compile_sequence(path)
                self.assertIsNone(other.get("profile"))
                self.assertTrue(other["elementwise_profile"].startswith("mul-"))

    def test_rejections(self):
        model = diamond()
        model.graph.node[-1].op_type = "Div"
        with self.assertRaises(ValueError):
            self.compile(model)
        model = diamond()
        model.graph.node[-2].input[0] = "head_a"
        with self.assertRaises(ValueError):
            self.compile(model)
        model = diamond()
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 4
        with self.assertRaises(ValueError):
            self.compile(model)
        model = diamond(kind="Add")
        model.graph.node[-1].input.reverse()
        with self.assertRaises(ValueError):
            self.compile(model)
        with self.assertRaises(ValueError):
            self.compile(diamond(kind="Add"), output_range=dict(scale=1.0, zero_point=0))
        with self.assertRaises(ValueError):
            self.compile(diamond(kind="Add"), mul_operand_zero_points=(1, 0))


if __name__ == "__main__":
    unittest.main()
