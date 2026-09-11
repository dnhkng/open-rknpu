"""MIT. Join-chain fan-out regression: 3..5 heads folded by mixed elementwise joins."""
from pathlib import Path
import json
import struct
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.graph import diamond_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "join_chain_suite"


def build(kinds, hidden=8, heads=None, kernels=None, tail=None, reuse_head=False):
    """A stem fan-out with the requested join kinds and optional Conv tail."""
    head_count = heads if heads is not None else len(kinds) + 1
    kernels = kernels or [1] * head_count
    rng = np.random.default_rng(len(kinds) * 13 + head_count)
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1])]
    for head in range(head_count):
        kernel = kernels[head]
        constants += [nh.from_array(rng.uniform(.02, .06, (3, hidden, kernel, kernel)).astype(np.float32), f"wh{head}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{head}")]
        nodes.append(h.make_node("Conv", ["stem", f"wh{head}", f"bh{head}"], [f"h{head}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
    previous = "h0"
    for position, kind in enumerate(kinds):
        if reuse_head and position == len(kinds) - 1:
            nodes.append(h.make_node(kind, [previous, "h0"], [f"j{position}"]))
        else:
            nodes.append(h.make_node(kind, [previous, f"h{position + 1}"], [f"j{position}"]))
        previous = f"j{position}"
    if tail is not None:
        constants += [nh.from_array(rng.uniform(.02, .06, (3, 3, tail, tail)).astype(np.float32), "wt"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bt")]
        nodes.append(h.make_node("Conv", [previous, "wt", "bt"], ["output"],
                                 kernel_shape=[tail, tail], pads=[tail // 2] * 4))
    else:
        nodes[-1].output[0] = "output"
    graph = h.make_graph(nodes, "join_chain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class JoinChainTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path)
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["head_count"], entry["head_count"])
                self.assertEqual(meta["join_kinds"], entry["join_kinds"])
                # stem task + one per head + one per join + one per tail layer
                self.assertEqual(len(meta["schedule"]), 1 + entry["head_count"] + entry["join_count"]
                                 + (1 if entry["tail_kernel"] else 0))
                self.assertEqual(meta["schedule"][0], "stem")
                self.assertEqual(meta["schedule"][-1], "output")

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            inputs = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 8, 8, 3)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 8, 8, 3)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx")
            got = np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                              meta["join_kinds"], meta["tail_quantization"],
                                              meta["join_zero_point"]) for case in inputs])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] // 2)

    def test_two_heads_still_take_the_diamond_path(self):
        for kinds in (("Add",), ("Mul",), ("Sub",), ("Max",)):
            with self.subTest(kind=kinds[0]):
                _, meta = compile_sequence(build(list(kinds)))
                self.assertEqual(meta["profile"], "walk-join")
                self.assertEqual(meta["head_count"], 2)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            compile_sequence(build(["Mul", "Add"]), mul_operand_zero_points=(1, 0))
        with self.assertRaises(ValueError):
            compile_sequence(build(["Add", "Add"]), output_range={"scale": .1, "zero_point": 0})
        with self.assertRaises(ValueError):
            compile_sequence(build(["Add", "Add"], reuse_head=True))
        with self.assertRaises(ValueError):
            compile_sequence(build(["Add"] * 8, heads=9))
        with self.assertRaises(ValueError):
            compile_sequence(build(["Add", "Add"]), input_scale=2.0)
        with self.assertRaises(ValueError):
            compile_sequence(build(["Add", "Add"]), calibration_ranges={})

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 384)
        self.assertEqual(sum(entry["bytes"] for entry in results), 73728)


if __name__ == "__main__":
    unittest.main()


class JoinChainTailActivationTests(unittest.TestCase):
    """A hidden tail layer carries its Relu in its own program (and in the reference).

    The join-chain tail is `[Conv, Relu]* Conv`; until 2026-09-11 the emitter never set
    `q.relu`, so a tail Relu was dropped from the container *and* from the reference.
    No board suite had such a tail, so this pins the fixed path at register level.
    """

    def build_tail(self, height=8):
        rng = np.random.default_rng(31)
        constants = [nh.from_array(rng.uniform(.02, .06, (8, 3, 1, 1)).astype(np.float32), "w1"),
                     nh.from_array(rng.uniform(-1, 1, (8,)).astype(np.float32), "b1")]
        nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1])]
        for head in range(3):
            constants += [nh.from_array(rng.uniform(.02, .06, (3, 8, 1, 1)).astype(np.float32), f"wh{head}"),
                          nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{head}")]
            nodes.append(h.make_node("Conv", ["stem", f"wh{head}", f"bh{head}"], [f"h{head}"],
                                     kernel_shape=[1, 1]))
        nodes.append(h.make_node("Add", ["h0", "h1"], ["j0"]))
        nodes.append(h.make_node("Add", ["j0", "h2"], ["j1"]))
        constants += [nh.from_array(rng.uniform(.02, .06, (3, 3, 1, 1)).astype(np.float32), "wt0"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bt0"),
                      nh.from_array(rng.uniform(.02, .06, (3, 3, 1, 1)).astype(np.float32), "wt1"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bt1")]
        nodes.append(h.make_node("Conv", ["j1", "wt0", "bt0"], ["t0"], kernel_shape=[1, 1]))
        nodes.append(h.make_node("Relu", ["t0"], ["t0r"]))
        nodes.append(h.make_node("Conv", ["t0r", "wt1", "bt1"], ["output"], kernel_shape=[1, 1]))
        graph = h.make_graph(nodes, "join_chain_tail",
            [h.make_tensor_value_info("input", 1, [1, 3, height, 8])],
            [h.make_tensor_value_info("output", 1, [1, 3, height, 8])], constants)
        graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, 8, height, 8]))
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        return onnx.shape_inference.infer_shapes(model)

    def test_hidden_tail_layer_applies_its_activation(self):
        model = self.build_tail()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            binary, meta = compile_sequence(path)
        self.assertEqual(meta["profile"], "join-chain-tail")
        self.assertEqual([entry["relu"] for entry in meta["tail_quantization"]], [True, False])
        info = decode_sequence(binary)
        header = 112 + 16 * info["task_count"] + 64 * info["tensor_count"]
        registers = []
        for task in info["tasks"]:
            words = {}
            for index in range(task["register_count"]):
                word = struct.unpack_from("<Q", binary, header + task["command_offset"] + index * 8)[0]
                words[word & 0xFFFF] = (word >> 16) & 0xFFFFFFFF
            registers.append(words)
        # The last two tasks are the tail layers: hidden applies Relu, final does not.
        self.assertEqual(registers[-2][0x4060], 0x12)
        self.assertEqual(registers[-2][0x406C], 0)
        self.assertEqual(registers[-2][0x40E0], 0)
        self.assertEqual(registers[-1][0x4060], 0x13)
        self.assertEqual(registers[-1][0x406C], 0x80000000)
        self.assertEqual(registers[-1][0x40E0], 0x80000000)


if __name__ == "__main__":
    unittest.main()
