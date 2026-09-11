"""MIT. Pooled-branch join regression: shared stem, two Conv+pool branches."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.pool_join import pool_join_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "pool_join_suite"


def build(kind="Add", pool="MaxPool", kernels=(1, 3), hidden=8, stem_relu=True,
          mismatch_pool=False, pool_kernel=2):
    """A shared 1x1 stem with two Conv+pool branches feeding one join."""
    rng = np.random.default_rng(sum(kernels) * 5 + hidden + pool_kernel)
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    for name, kernel in (("a", kernels[0]), ("b", kernels[1])):
        constants += [nh.from_array(rng.uniform(.02, .06, (3, hidden, kernel, kernel)).astype(np.float32), f"w{name}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"b{name}")]
        nodes.append(h.make_node("Conv", [source, f"w{name}", f"b{name}"], [f"head_{name}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        kind_b = "AveragePool" if (mismatch_pool and name == "b") else pool
        nodes.append(h.make_node(kind_b, [f"head_{name}"], [f"pool_{name}"],
                                 kernel_shape=[pool_kernel, pool_kernel], strides=[2, 2]))
    nodes.append(h.make_node(kind, ["pool_a", "pool_b"], ["output"]))
    graph = h.make_graph(nodes, "pool_join",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
    for name in ("head_a", "head_b"):
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, 3, 8, 8]))
    for name in ("pool_a", "pool_b"):
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, 3, 4, 4]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class PoolJoinTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path)
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "pool-join")
                self.assertEqual(meta["join"], entry["join"])
                self.assertEqual(meta["pool"], entry["pool"])
                self.assertEqual(len(meta["schedule"]), 6)
                self.assertEqual(meta["schedule"][-1], "output")

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 4, 4, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx")
            got = np.stack([pool_join_reference(case, meta["stem_quantization"],
                                                meta["head_quantization"], meta["join"], meta["pool"])
                            for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] * 9 // 10)

    def test_dense_diamond_is_not_hijacked(self):
        rng = np.random.default_rng(5)
        constants = [nh.from_array(rng.uniform(-.7, .8, (8, 3, 1, 1)).astype(np.float32), "w1"),
                     nh.from_array(rng.uniform(-4, 4, (8,)).astype(np.float32), "b1")]
        nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
                 h.make_node("Relu", ["stem"], ["relu1"])]
        for name in ("a", "b"):
            constants += [nh.from_array(rng.uniform(-.7, .8, (3, 8, 1, 1)).astype(np.float32), f"w{name}"),
                          nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), f"b{name}")]
            nodes.append(h.make_node("Conv", ["relu1", f"w{name}", f"b{name}"], [f"head_{name}"], kernel_shape=[1, 1]))
        nodes.append(h.make_node("Add", ["head_a", "head_b"], ["output"]))
        graph = h.make_graph(nodes, "diamond",
            [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
            [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
        graph.value_info.append(h.make_tensor_value_info("relu1", 1, [1, 8, 8, 8]))
        graph.value_info.append(h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]))
        graph.value_info.append(h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8]))
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        _, meta = compile_sequence(onnx.shape_inference.infer_shapes(model))
        # The op-level walk dispatches the dense diamond class now; the pool-join
        # profile still owns the pooled-branch graphs above.
        self.assertEqual(meta["profile"], "walk-join")

    def test_rejections(self):
        with self.assertRaises(ValueError):
            compile_sequence(build(kind="Add"), output_range={"scale": .1, "zero_point": 0})
        with self.assertRaises(ValueError):
            compile_sequence(build(kind="Mul"), mul_operand_zero_points=(0, 1))
        # A mismatched pool pair is no longer an error: `parse_pooled_branches`
        # declines it and the op-level walk lowers the graph instead
        # (`research/walk_join_suite/`).
        _, mismatched = compile_sequence(build(mismatch_pool=True))
        self.assertEqual(mismatched["profile"], "walk-join")
        with self.assertRaises(ValueError):
            compile_sequence(build(pool_kernel=3))
        with self.assertRaises(ValueError):
            compile_sequence(build(hidden=1))
        with self.assertRaises(ValueError):
            compile_sequence(build(), input_scale=2.0)
        with self.assertRaises(ValueError):
            compile_sequence(build(), calibration_ranges={})

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 384)
        self.assertEqual(sum(entry["bytes"] for entry in results), 18432)


if __name__ == "__main__":
    unittest.main()
