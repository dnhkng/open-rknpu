"""MIT. Pooled-branches regression: multi-layer branch chains that pool before joining."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.pooled_branches import pooled_branches_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "pooled_branches_suite"
D, W = "dense", "depthwise"


def build(branches, joins=("Add",), hidden=3, pool="MaxPool", pool_kernel=2):
    """A stem, branch layer chains each followed by a pool, then joins over the pools."""
    rng = np.random.default_rng(len(branches) * 17 + sum(len(b) for b in branches))
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "bs_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "bs_stem"], ["stem"], kernel_shape=[1, 1])]
    channels_of = {"stem": hidden}
    pooled = []
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
            channels_of[name] = out_channels
        pool_name = f"p{branch_index}"
        nodes.append(h.make_node(pool, [previous], [pool_name],
                                 kernel_shape=[pool_kernel, pool_kernel], strides=[2, 2]))
        pooled.append(pool_name)
    last = None
    for position, kind in enumerate(joins):
        first = pooled[0] if position == 0 else last
        nodes.append(h.make_node(kind, [first, pooled[position + 1]], [f"j{position}"]))
        last = f"j{position}"
    graph = h.make_graph(nodes, "pooled_branches",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(last, 1, [1, 3, 4, 4])], constants)
    for name, channels in channels_of.items():
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, channels, 8, 8]))
    for position in range(len(branches)):
        graph.value_info.append(h.make_tensor_value_info(f"p{position}", 1, [1, 3, 4, 4]))
    for position in range(len(joins)):
        graph.value_info.append(h.make_tensor_value_info(f"j{position}", 1, [1, 3, 4, 4]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class PooledBranchesTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 13)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path)
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "pooled-branches")
                self.assertEqual(meta["branch_names"], entry["branch_names"])
                self.assertEqual(meta["pool"], entry["pool"])
                self.assertEqual(meta["pooled_names"], entry["pooled_names"])
                self.assertEqual(meta["output_shape_nhwc"], [1, 4, 4, 3])

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 4, 4, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx")
            got = np.stack([pooled_branches_reference(case, meta["stem_quantization"],
                                                      meta["branch_names"], meta["branch_quantization"],
                                                      meta["join_expression"], meta["pool"],
                                                      pooled_names=meta["pooled_names"])
                            for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] // 2)

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
            compile_sequence(build([[(D, 3, 3, 1)], [(D, 3, 3, 1)]], pool_kernel=3))
        # Four branches exceed the two-or-three-branch profile.
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 3, 1)]] * 4, joins=("Add", "Add", "Add")))
        # A branch chain that does not end in three channels cannot be pooled-and-joined.
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 8, 1)], [(D, 3, 3, 1)]]))
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 3, 1)], [(D, 3, 3, 1)]]), calibration_ranges={})
        with self.assertRaises(ValueError):
            compile_sequence(build([[(D, 3, 3, 1)], [(D, 3, 3, 1)]]), mul_operand_zero_points=(1, 0))

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 13)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 416)
        self.assertEqual(sum(entry["bytes"] for entry in results), 19968)


if __name__ == "__main__":
    unittest.main()
