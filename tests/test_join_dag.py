"""MIT. General join-expression DAG regression (internal grids reused across joins)."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.join_dag import join_dag_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "join_dag_suite"


def build(kinds=("dense", "dense", "dense"), kernels=(1, 3, 1),
          expression=(("Add", "h0", "h1"), ("Mul", "j0", "h0")), hidden=3):
    """A stem, dense/depthwise heads and joins over produced tensor names."""
    rng = np.random.default_rng(len(kinds) * 11 + sum(kernels) + len(expression))
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1])]
    for position, (kind, kernel) in enumerate(zip(kinds, kernels)):
        shape = (3, 1, kernel, kernel) if kind == "depthwise" else (3, hidden, kernel, kernel)
        constants += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32), f"wh{position}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{position}")]
        attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        if kind == "depthwise":
            attributes["group"] = 3
        nodes.append(h.make_node("Conv", ["stem", f"wh{position}", f"bh{position}"], [f"h{position}"],
                                 **attributes))
    last = None
    for position, (kind, first, second) in enumerate(expression):
        nodes.append(h.make_node(kind, [first, second], [f"j{position}"]))
        last = f"j{position}"
    graph = h.make_graph(nodes, "join_dag",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(last, 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class JoinDagTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path)
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "join-dag")
                self.assertEqual([[s["kind"], s["inputs"][0], s["inputs"][1]]
                                  for s in meta["join_expression"]], entry["expression"])
                self.assertEqual(meta["schedule"][-1], "output")

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 8, 8, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx")
            got = np.stack([join_dag_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                               meta["head_names"], meta["join_expression"],
                                               head_kinds=meta["head_kinds"],
                                               depthwise_quantizations=meta["depthwise_quantization"])
                            for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] // 4)

    def test_left_fold_chain_still_uses_the_chain_emitter(self):
        _, meta = compile_sequence(build(expression=(("Add", "h0", "h1"), ("Mul", "j0", "h2"))))
        self.assertEqual(meta["profile"], "join-chain")
        self.assertNotIn("join_expression", meta)

    def test_reused_tensor_keeps_one_lifetime(self):
        _, meta = compile_sequence(build(expression=(("Add", "h0", "h1"), ("Mul", "j0", "h0"))))
        lifetime = meta["tensor_lifetimes"]["h0"]
        self.assertLessEqual(lifetime[0], 4)
        self.assertEqual(lifetime[1], 5)  # h0 is still read by the second join

    def test_rejections(self):
        # h0 is committed to the Add band, so a later Add/Sub/Max against another
        # band cannot re-quantize it.
        with self.assertRaises(ValueError):
            compile_sequence(build(expression=(("Mul", "h0", "h1"), ("Add", "j0", "h2"),
                                               ("Max", "j1", "h0"))))
        with self.assertRaises(ValueError):
            compile_sequence(build(expression=(("Add", "h0", "h1"), ("Mul", "j0", "h0"))),
                             mul_operand_zero_points=(1, 0))
        with self.assertRaises(ValueError):
            compile_sequence(build(expression=(("Add", "h0", "h1"), ("Mul", "j0", "h0"))),
                             calibration_ranges={})
        with self.assertRaises(ValueError):
            compile_sequence(build(kinds=("dense", "depthwise", "dense"), kernels=(1, 3, 1),
                                   hidden=8))
        with self.assertRaises(ValueError):
            compile_sequence(build(expression=(("Add", "h0", "h0"), ("Mul", "j0", "h1"))))

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 384)
        self.assertEqual(sum(entry["bytes"] for entry in results), 73728)


if __name__ == "__main__":
    unittest.main()
