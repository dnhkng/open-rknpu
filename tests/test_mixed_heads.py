"""MIT. Mixed dense/depthwise head fan-out regression (chained joins)."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.graph import diamond_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "mixed_head_suite"


def build(kinds, kernels, joins, hidden=3, tail=None, stem_relu=True, depthwise_group=3):
    """A shared stem with dense and/or depthwise heads folded by chained joins."""
    rng = np.random.default_rng(len(kinds) * 3 + sum(kernels) + hidden)
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["input", "w_stem", "b_stem"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    heads = []
    for position, (kind, kernel) in enumerate(zip(kinds, kernels)):
        shape = (3, 1, kernel, kernel) if kind == "depthwise" else (3, hidden, kernel, kernel)
        constants += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32), f"wh{position}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{position}")]
        attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        if kind == "depthwise":
            attributes["group"] = depthwise_group
        nodes.append(h.make_node("Conv", [source, f"wh{position}", f"bh{position}"], [f"h{position}"],
                                 **attributes))
        heads.append(f"h{position}")
    previous = heads[0]
    for position, join in enumerate(joins):
        nodes.append(h.make_node(join, [previous, heads[position + 1]], [f"j{position}"]))
        previous = f"j{position}"
    if tail is not None:
        constants += [nh.from_array(rng.uniform(.02, .06, (3, 3, tail, tail)).astype(np.float32), "wt"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bt")]
        nodes.append(h.make_node("Conv", [previous, "wt", "bt"], ["output"],
                                 kernel_shape=[tail, tail], pads=[tail // 2] * 4))
    else:
        nodes[-1].output[0] = "output"
    graph = h.make_graph(nodes, "mixed_heads",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class MixedHeadTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path, asymmetric_depthwise=entry["asymmetric_depthwise"])
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "join-chain" if not entry["tail_kernel"] else "join-chain-tail")
                self.assertEqual(meta["head_kinds"], entry["head_kinds"])
                self.assertEqual(meta["join_kinds"], entry["join_kinds"])

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 8, 8, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx",
                                       asymmetric_depthwise=entry["asymmetric_depthwise"])
            got = np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                              meta["join_kinds"], meta["tail_quantization"],
                                              meta["join_zero_point"], head_kinds=meta["head_kinds"],
                                              depthwise_quantizations=meta["depthwise_quantization"])
                            for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] * 2 // 3)

    def test_mixed_head_scales_are_tracked(self):
        _, meta = compile_sequence(build(("dense", "depthwise", "dense"), (1, 3, 1), ("Add", "Mul")))
        self.assertEqual(meta["head_kinds"], ["dense", "depthwise", "dense"])
        self.assertIsNone(meta["depthwise_quantization"][0])
        self.assertIsNotNone(meta["depthwise_quantization"][1])
        self.assertIsNone(meta["depthwise_quantization"][2])
        self.assertEqual(len(meta["head_scales"]), 3)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            compile_sequence(build(("dense", "depthwise", "dense"), (1, 3, 1), ("Add", "Mul"), hidden=8))
        with self.assertRaises(ValueError):
            compile_sequence(build(("dense", "depthwise", "dense"), (1, 3, 1), ("Add", "Mul"),
                                   depthwise_group=2))
        with self.assertRaises(ValueError):
            compile_sequence(build(("dense", "depthwise", "dense"), (1, 7, 1), ("Add", "Mul")))
        with self.assertRaises(ValueError):
            compile_sequence(build(("dense", "depthwise", "dense"), (1, 3, 1), ("Add", "Mul")),
                             mul_operand_zero_points=(1, 0))
        with self.assertRaises(ValueError):
            compile_sequence(build(("dense", "depthwise", "dense"), (1, 3, 1), ("Add", "Mul")),
                             calibration_ranges={})

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 384)
        self.assertEqual(sum(entry["bytes"] for entry in results), 73728)


if __name__ == "__main__":
    unittest.main()
