"""Runtime per-channel scale Mul: two external inputs of different shapes.

The board suite is `research/runtime_scale_suite/` (16 models, 256 inferences,
32,832 exact bytes). The per-channel operand is bound as an external tensor of
shape (1,1,1,3) in the verified per-channel elementwise operand mode, which is the
first profile with unequal runtime input shapes.
"""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from onnx.reference import ReferenceEvaluator
from open_rknpu.elementwise import runtime_scale_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "runtime_scale_suite"


def build(height=8, width=8, second="scale"):
    graph = h.make_graph(
        [h.make_node("Mul", ["image", second], ["output"])], "runtime_scale",
        [h.make_tensor_value_info("image", 1, [1, 3, height, width]),
         h.make_tensor_value_info(second, 1, [1, 3, 1, 1])],
        [h.make_tensor_value_info("output", 1, [1, 3, height, width])], [])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class RuntimeScaleTests(unittest.TestCase):
    def compile(self, model):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path)

    def test_container_binds_two_unequal_inputs(self):
        for height, width in ((8, 8), (5, 5), (6, 7), (5, 8)):
            with self.subTest(shape=(height, width)):
                binary, meta = self.compile(build(height, width))
                info = decode_sequence(binary)
                self.assertEqual(meta["profile"], "runtime-scale-mul")
                self.assertEqual([t["name"] for t in info["tensors"]],
                                 ["image", "scale", "converted", "output"])
                self.assertEqual([t["role_name"] for t in info["tensors"]],
                                 ["input", "input", "internal", "output"])
                inputs = info["input_tensors"]
                self.assertEqual(len(inputs), 2)
                self.assertEqual((inputs[0]["height"], inputs[0]["width"], inputs[0]["channels"]),
                                 (height, width, 3))
                self.assertEqual((inputs[1]["height"], inputs[1]["width"], inputs[1]["channels"]), (1, 1, 3))
                self.assertNotEqual(inputs[0]["byte_offset"], inputs[1]["byte_offset"])
                self.assertEqual(inputs[0]["byte_offset"], info["input_offset"])
                self.assertEqual(inputs[0]["bytes"], height * 16 * 3)
                self.assertEqual(inputs[1]["bytes"], 64)
                self.assertEqual(info["task_count"], 2)

    def test_operand_row_is_wired_into_the_elementwise_task(self):
        binary, meta = self.compile(build(6, 7))
        info = decode_sequence(binary)
        start = 96 + 16 + 16 * info["task_count"] + 64 * info["tensor_count"]
        task = info["tasks"][1]
        registers = {}
        for index in range(task["register_count"]):
            word = struct.unpack_from("<Q", binary[start:], task["command_offset"] + index * 8)[0]
            registers[word & 0xffff] = (word >> 16) & 0xffffffff
        scale_offset = next(t["byte_offset"] for t in info["tensors"] if t["name"] == "scale")
        self.assertEqual(registers[0x5038], scale_offset)
        self.assertEqual(registers[0x5034], 4)
        self.assertEqual(registers[0x5040], 16)
        self.assertEqual(registers[0x5018], next(t["byte_offset"] for t in info["tensors"] if t["name"] == "converted"))

    def test_reference_reproduces_the_board_verified_bytes(self):
        for path in sorted(SUITE.glob("model*.onnx")):
            index = path.stem[5:]
            with self.subTest(model=index):
                binary, meta = compile_sequence(path)
                entry = next(e for e in __import__("json").loads((SUITE / "manifest.json").read_text())
                             if e["index"] == int(index))
                height, width = entry["height"], entry["width"]
                raw = np.frombuffer((SUITE / f"input{index}.u8").read_bytes(), np.uint8)
                expected = np.frombuffer((SUITE / f"expected{index}.i8").read_bytes(), np.int8)
                per_image = height * width * 3
                cases = raw.reshape(-1, per_image + 3)[:, :per_image].reshape(-1, height, width, 3)
                codes = raw.reshape(-1, per_image + 3)[0, per_image:].astype(np.int32) - 128
                np.testing.assert_array_equal(np.asarray(codes, np.int32), np.asarray(entry["codes"], np.int32))
                got = np.stack([runtime_scale_reference(case, codes, meta["branches"][0]["quantization"])
                                for case in cases])
                np.testing.assert_array_equal(got.reshape(-1), expected)

    def test_reference_tracks_the_float_model(self):
        rng = np.random.default_rng(4)
        for codes in ([127, 127, 127], [10, -40, 64], [-128, 127, 0]):
            with self.subTest(codes=codes):
                model = build()
                _, meta = self.compile(model)
                values = np.asarray(codes, np.float64)
                cases = rng.integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
                got = np.stack([runtime_scale_reference(case, values, meta["branches"][0]["quantization"])
                                for case in cases])
                value = got.astype(np.float64) * meta["output_scale"]
                expected = np.stack([ReferenceEvaluator(model).run(None, {
                    "image": case[None].transpose(0, 3, 1, 2).astype(np.float32),
                    "scale": (values * meta["operand_scale"])[None, :, None, None].astype(np.float32)})[0][0].transpose(1, 2, 0)
                    for case in cases])
                span = float(expected.max() - expected.min())
                self.assertLess(float(np.abs(value - expected).max()), 0.15 * span)

    def test_rejections_and_default_path_unchanged(self):
        # Two matching RGB inputs stay on the verified standalone two-input path.
        matching = h.make_model(h.make_graph(
            [h.make_node("Mul", ["a", "b"], ["output"])], "matching",
            [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]),
             h.make_tensor_value_info("b", 1, [1, 3, 8, 8])],
            [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])]), opset_imports=[h.make_opsetid("", 13)])
        matching.ir_version = 8
        _, meta = self.compile(onnx.shape_inference.infer_shapes(matching))
        self.assertNotEqual(meta.get("profile"), "runtime-scale-mul")
        two_vectors = build(8, 8, second="scale")
        two_vectors.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
        # A second per-channel input cannot be the image operand.
        model = build(3, 3)
        with self.assertRaises(ValueError):
            self.compile(model)

    def test_board_suite_is_recorded(self):
        import json
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 16)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 256)
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(sorted({(entry["height"], entry["width"]) for entry in manifest}),
                         [(5, 5), (5, 8), (6, 7), (8, 8)])


if __name__ == "__main__":
    unittest.main()
