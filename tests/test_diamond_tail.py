"""Diamond + Conv tail: the join output consumed by name through the v5 table.

The board suite is `research/diamond_tail_suite/` (12 models, 192 inferences,
36,864 exact bytes). This composition is the first that binds the elementwise
join emitter to the native Conv emitter by tensor name instead of matching a
single whole-graph pattern.
"""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.graph import diamond_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "diamond_tail_suite"


def build(kind="Add", tails=(1,), hidden=8, seed=5):
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["stem"], ["relu1"]),
             h.make_node("Conv", ["relu1", "wa", "ba"], ["head_a"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
             h.make_node("Conv", ["relu1", "wb", "bb"], ["head_b"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             h.make_node(kind, ["head_a", "head_b"], ["join_out"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-4, 4, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, 1, 1)).astype(np.float32), "wa"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "ba"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, 3, 3)).astype(np.float32), "wb"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "bb")]
    values = [h.make_tensor_value_info("relu1", 1, [1, hidden, 8, 8]),
              h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]),
              h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8]),
              h.make_tensor_value_info("join_out", 1, [1, 3, 8, 8])]
    source = "join_out"
    for index, kernel in enumerate(tails):
        last = index == len(tails) - 1
        output = "output" if last else f"tail_conv{index}"
        initializers.extend([nh.from_array(rng.uniform(-.5, .5, (3, 3, kernel, kernel)).astype(np.float32), f"tw{index}"),
                             nh.from_array(rng.uniform(-2, 2, (3,)).astype(np.float32), f"tb{index}")])
        nodes.append(h.make_node("Conv", [source, f"tw{index}", f"tb{index}"], [output],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        if last:
            break
        relu = output + "_relu"
        nodes.append(h.make_node("Relu", [output], [relu]))
        values.extend([h.make_tensor_value_info(output, 1, [1, 3, 8, 8]),
                       h.make_tensor_value_info(relu, 1, [1, 3, 8, 8])])
        source = relu
    graph = h.make_graph(nodes, "diamond_tail",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], initializers)
    for value in values:
        graph.value_info.append(value)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class DiamondTailTests(unittest.TestCase):
    def compile(self, model):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path)

    def test_name_bound_composition_and_lifetimes(self):
        binary, meta = self.compile(build("Add", (1, 3)))
        info = decode_sequence(binary)
        self.assertEqual(meta["profile"], "walk-join")
        self.assertEqual(meta["schedule"], ["stem", "head_a", "head_b", "join", "tail0", "output"])
        self.assertEqual([t["name"] for t in info["tensors"]],
                         ["input0", "stem", "head_a", "head_b", "join", "tail0", "output"])
        self.assertEqual(info["task_count"], 6)
        self.assertEqual(meta["tensor_lifetimes"]["join"], [3, 4])
        self.assertEqual(meta["tensor_lifetimes"]["tail0"], [4, 5])
        self.assertEqual(meta["tensor_lifetimes"]["output"], [5, 5])
        # The join output is internal and the tail layer reads it by name.
        join = next(t for t in info["tensors"] if t["name"] == "join")
        self.assertEqual(join["role_name"], "internal")
        self.assertNotEqual(join["byte_offset"], info["output_offset"])
        externals = [t for t in info["tensors"] if t["role_name"] != "internal"]
        for external in externals:
            for other in info["tensors"]:
                if other["name"] == external["name"]:
                    continue
                self.assertFalse(external["byte_offset"] < other["byte_offset"] + other["bytes"]
                                 and other["byte_offset"] < external["byte_offset"] + external["bytes"])

    def test_tail_declares_the_input_grid_it_reads(self):
        for kind in ("Add", "Mul"):
            with self.subTest(kind=kind):
                binary, meta = self.compile(build(kind, (3,)))
                info = decode_sequence(binary)
                start = 96 + 16 + 16 * info["task_count"] + 64 * info["tensor_count"]
                task = info["tasks"][-1]
                registers = {}
                for index in range(task["register_count"]):
                    word = struct.unpack_from("<Q", binary[start:], task["command_offset"] + index * 8)[0]
                    registers[word & 0xffff] = (word >> 16) & 0xffffffff
                expected_zero = meta["join_zero_point"] if kind == "Mul" else 0
                self.assertEqual(registers[0x1184], expected_zero & 0xffff)
                self.assertEqual(registers[0x1038], 0x3030003)
                self.assertEqual(registers[0x1070], meta["tensor_offsets"]["join"])
                self.assertEqual(registers[0x4020], meta["tensor_offsets"]["output"])

    def test_reference_reproduces_the_board_verified_bytes(self):
        # The stored expected files are exactly what the board produced, so this is
        # the strongest host check for the composed reference.
        for path in sorted(SUITE.glob("model*.onnx")):
            index = path.stem[5:]
            with self.subTest(model=index):
                binary, meta = compile_sequence(path)
                cases = np.frombuffer((SUITE / f"input{index}.u8").read_bytes(), np.uint8).reshape(-1, 8, 8, 3)
                expected = np.frombuffer((SUITE / f"expected{index}.i8").read_bytes(), np.int8).reshape(-1, 8, 8, 3)
                got = np.stack([diamond_reference(case, meta["stem_quantization"],
                                                  meta["head_quantization"], meta["join"],
                                                  meta["tail_quantization"], meta["join_zero_point"])
                                for case in cases])
                np.testing.assert_array_equal(got, expected)

    def test_shallow_tail_tracks_the_float_graph(self):
        # Deep tails run with coarse analytic grids, so only the single-layer tail is
        # compared with the float model; the deep cases are pinned by board bytes.
        rng = np.random.default_rng(3)
        for kind in ("Add", "Mul", "Sub", "Max"):
            with self.subTest(kind=kind):
                model = build(kind, (1,), seed=7)
                _, meta = self.compile(model)
                cases = rng.integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
                got = np.stack([diamond_reference(case, meta["stem_quantization"],
                                                  meta["head_quantization"], kind,
                                                  meta["tail_quantization"], meta["join_zero_point"])
                                for case in cases])
                value = (got.astype(np.float64) - meta["output_zero_point"]) * meta["output_scale"]
                expected = np.stack([ReferenceEvaluator(model).run(
                    None, {"input": case[None].transpose(0, 3, 1, 2).astype(np.float32)})[0][0].transpose(1, 2, 0)
                    for case in cases])
                self.assertGreater(float(np.corrcoef(value.ravel(), expected.ravel())[0, 1]), 0.8)

    def test_rejections(self):
        model = build("Add", (1,))
        model.graph.node[-1].input[0] = "head_a"
        with self.assertRaises(ValueError):
            self.compile(model)
        model = build("Add", (1,))
        model.graph.node[-1].op_type = "Relu"
        with self.assertRaises((ValueError, onnx.checker.ValidationError)):
            self.compile(model)
        model = build("Add", (1,))
        model.graph.node[-1].attribute.extend([h.make_attribute("group", 3)])
        with self.assertRaises(ValueError):
            self.compile(model)

    def test_board_suite_is_recorded(self):
        import json
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 12)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 192)
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual({entry["join"] for entry in manifest}, {"Add", "Mul", "Sub", "Max"})
        self.assertEqual(sorted({tuple(entry["tail_kernels"]) for entry in manifest}),
                         [(1,), (1, 3), (3,)])


if __name__ == "__main__":
    unittest.main()
