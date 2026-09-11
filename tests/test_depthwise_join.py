"""MIT. Depthwise-branch join regression: shared stem, dense branch, depthwise branch."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.depthwise_join import depthwise_join_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "depthwise_join_suite"


def build(kind="Add", dense_kernel=1, dw_kernel=3, group=3, stem_relu=True,
          dense_heads=1, hidden=3, depthwise=True):
    """A shared 1x1 stem with a dense branch and a depthwise branch feeding one join."""
    rng = np.random.default_rng(dense_kernel * 7 + dw_kernel * 13 + group)
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    branches = []
    for index in range(dense_heads):
        constants += [nh.from_array(rng.uniform(.02, .06, (3, hidden, dense_kernel, dense_kernel)).astype(np.float32),
                                   f"wd{index}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bd{index}")]
        nodes.append(h.make_node("Conv", [source, f"wd{index}", f"bd{index}"], [f"dense{index}"],
                                 kernel_shape=[dense_kernel] * 2, pads=[dense_kernel // 2] * 4))
        branches.append(f"dense{index}")
    if depthwise:
        constants += [nh.from_array(rng.uniform(.02, .06, (group, 1, dw_kernel, dw_kernel)).astype(np.float32), "ww"),
                      nh.from_array(rng.uniform(-1, 1, (group,)).astype(np.float32), "bw")]
        nodes.append(h.make_node("Conv", [source, "ww", "bw"], ["depthwise"],
                                 kernel_shape=[dw_kernel] * 2, pads=[dw_kernel // 2] * 4, group=group))
        branches.append("depthwise")
    if len(branches) == 2:
        nodes.append(h.make_node(kind, branches, ["output"]))
    else:
        previous = branches[0]
        for position, name in enumerate(branches[1:]):
            nodes.append(h.make_node(kind, [previous, name], [f"j{position}"]))
            previous = f"j{position}"
        nodes[-1].output[0] = "output"
    graph = h.make_graph(nodes, "depthwise_join",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class DepthwiseJoinTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path, asymmetric_depthwise=entry["asymmetric_depthwise"])
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "depthwise-join")
                self.assertEqual(meta["join"], entry["join"])
                self.assertEqual(meta["dense_kernel"], entry["dense_kernel"])
                self.assertEqual(meta["depthwise_kernel"], entry["depthwise_kernel"])
                self.assertEqual(meta["schedule"], ["stem", "dense", "depthwise", "output"])

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 8, 8, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx",
                                       asymmetric_depthwise=entry["asymmetric_depthwise"])
            got = np.stack([depthwise_join_reference(case, meta["stem_quantization"],
                                                     meta["head_quantization"][0],
                                                     meta["depthwise_quantization"], meta["join"])
                            for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] // 2)

    def test_dense_diamond_is_not_hijacked(self):
        _, meta = compile_sequence(build(dense_heads=2, depthwise=False, kind="Add"))
        self.assertEqual(meta["profile"], "walk-join")
        self.assertNotIn("depthwise_quantization", meta)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            compile_sequence(build(kind="Add"), output_range={"scale": .1, "zero_point": 0})
        with self.assertRaises(ValueError):
            compile_sequence(build(kind="Mul"), mul_operand_zero_points=(0, 1))
        with self.assertRaises(ValueError):
            compile_sequence(build(group=2))
        with self.assertRaises(ValueError):
            compile_sequence(build(hidden=4))
        with self.assertRaises(ValueError):
            compile_sequence(build(dw_kernel=2))
        with self.assertRaises(ValueError):
            compile_sequence(build(), input_scale=2.0)
        with self.assertRaises(ValueError):
            compile_sequence(build(), calibration_ranges={})

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 384)
        self.assertEqual(sum(entry["bytes"] for entry in results), 73728)


if __name__ == "__main__":
    unittest.main()
