"""MIT. Pooled-DAG regression: a join expression with a terminal pooling stage."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.join_dag import join_dag_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "pooled_dag_suite"
D, W = "dense", "depthwise"


def build(branches, expression, hidden=3, pool="MaxPool", no_pool=False, pool_kernel=2):
    """A stem, branch layer chains, joins over produced tensor names and a terminal pool."""
    rng = np.random.default_rng(len(branches) * 13 + sum(len(b) for b in branches))
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "bs_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "bs_stem"], ["stem"], kernel_shape=[1, 1])]
    names = ["stem"]
    channels_of = {"stem": hidden}
    for branch_index, layers in enumerate(branches):
        previous = "stem"
        for layer_index, (kind, in_channels, out_channels, kernel) in enumerate(layers):
            shape = (out_channels, 1, kernel, kernel) if kind == W else (out_channels, in_channels, kernel, kernel)
            constants += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32),
                                       f"w{branch_index}_{layer_index}"),
                          nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32),
                                        f"bs{branch_index}_{layer_index}")]
            attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
            if kind == W:
                attributes["group"] = out_channels
            name = f"b{branch_index}_{layer_index}"
            nodes.append(h.make_node("Conv", [previous, f"w{branch_index}_{layer_index}",
                                              f"bs{branch_index}_{layer_index}"], [name], **attributes))
            previous = name
            names.append(name)
            channels_of[name] = out_channels
    last = None
    for position, (kind, first, second) in enumerate(expression):
        nodes.append(h.make_node(kind, [first, second], [f"j{position}"]))
        last = f"j{position}"
    if no_pool:
        output_shape = [1, 3, 8, 8]
    else:
        nodes.append(h.make_node(pool, [last], ["output"],
                                 kernel_shape=[pool_kernel, pool_kernel], strides=[2, 2]))
        output_shape = [1, 3, 4, 4]
    graph = h.make_graph(nodes, "pooled_dag",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, output_shape)], constants)
    for name in names:
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, channels_of[name], 8, 8]))
    for position in range(len(expression)):
        graph.value_info.append(h.make_tensor_value_info(f"j{position}", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class PooledDagTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path)
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "join-dag")
                self.assertEqual(meta["branch_names"], entry["branch_names"])
                self.assertEqual(meta["pool"], entry["pool"])
                self.assertEqual(meta["output_shape_nhwc"], [1, 4, 4, 3])

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 4, 4, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx")
            got = np.stack([join_dag_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                               meta["head_names"], meta["join_expression"],
                                               head_kinds=meta["head_kinds"],
                                               depthwise_quantizations=meta["depthwise_quantization"],
                                               branch_names=meta["branch_names"],
                                               branch_quantization=meta["branch_quantization"],
                                               pool=meta["pool"])
                            for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] * 3 // 4)

    def test_every_internal_has_a_fresh_slot(self):
        _, meta = compile_sequence(SUITE / "model000.onnx")
        offsets = {name: value for name, value in meta["tensor_offsets"].items()
                   if name not in ("input0", "output")}
        self.assertEqual(len(set(offsets.values())), len(offsets))

    def test_pool_task_is_recorded(self):
        _, meta = compile_sequence(SUITE / "model000.onnx")
        self.assertEqual(meta["pool"], "MaxPool")
        self.assertEqual(meta["schedule"][-1], "output")
        self.assertEqual(meta["output_shape_nhwc"], [1, 4, 4, 3])

    def test_rejections(self):
        # A pool kernel other than 2x2 is rejected.
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
                                   [("Add", "b0_0", "b1_0"), ("Mul", "j0", "b2_0")],
                                   pool_kernel=3))
        # A join operand with more than three channels is rejected.
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 8, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
                                   [("Add", "b0_0", "b1_0"), ("Mul", "j0", "b2_0")]))
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
                                   [("Add", "b0_0", "b1_0"), ("Mul", "j0", "b2_0")]),
                             calibration_ranges={})
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
                                   [("Add", "b0_0", "b1_0"), ("Mul", "j0", "b2_0")]),
                             mul_operand_zero_points=(1, 0))

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 384)
        self.assertEqual(sum(entry["bytes"] for entry in results), 18432)


if __name__ == "__main__":
    unittest.main()
