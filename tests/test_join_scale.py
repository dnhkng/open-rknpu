"""MIT. Join-chain runtime per-channel scale regression (two external inputs)."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.graph import join_chain_scale_reference
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "join_scale_suite"


def build(kinds=("dense", "dense", "dense"), kernels=(1, 3, 1), joins=("Add", "Mul"),
          scale_shape=(1, 3, 1, 1), extra_output=False, no_scale=False, hidden=3):
    """A stem fan-out with chained joins and an optional runtime scale input."""
    rng = np.random.default_rng(len(kinds) * 5 + sum(kernels))
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1])]
    heads = []
    for position, (kind, kernel) in enumerate(zip(kinds, kernels)):
        shape = (3, 1, kernel, kernel) if kind == "depthwise" else (3, hidden, kernel, kernel)
        constants += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32), f"wh{position}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{position}")]
        attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        if kind == "depthwise":
            attributes["group"] = 3
        nodes.append(h.make_node("Conv", ["stem", f"wh{position}", f"bh{position}"], [f"h{position}"],
                                 **attributes))
        heads.append(f"h{position}")
    previous = heads[0]
    for position, join in enumerate(joins):
        nodes.append(h.make_node(join, [previous, heads[position + 1]], [f"j{position}"]))
        previous = f"j{position}"
    inputs = [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])]
    if not no_scale:
        inputs.append(h.make_tensor_value_info("scale", 1, list(scale_shape)))
        nodes.append(h.make_node("Mul", [previous, "scale"], ["output"]))
    else:
        nodes[-1].output[0] = "output"
    outputs = [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])]
    if extra_output:
        outputs.append(h.make_tensor_value_info("extra", 1, [1, 3, 8, 8]))
        nodes[-1].output[0] = "extra"
        nodes.append(h.make_node("Identity", ["extra"], ["output"]))
    graph = h.make_graph(nodes, "join_scale", inputs, outputs, constants)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class JoinScaleTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 12)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path, asymmetric_depthwise=entry["asymmetric_depthwise"])
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"], "join-chain-scale")
                self.assertEqual(meta["schedule"][-1], "output")
                self.assertEqual(meta["tensor_offsets"]["scale"], 64 * (meta["tensor_offsets"]["scale"] // 64))

    def test_reference_reproduces_recorded_expected(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            packed = np.fromfile(SUITE / f"input{index:03}.u8", dtype=np.uint8).reshape(-1, 195)
            expected = np.fromfile(SUITE / f"expected{index:03}.i8", dtype=np.int8).reshape(-1, 8, 8, 3)
            codes = np.asarray(entry["codes"], np.int32)
            _, meta = compile_sequence(SUITE / f"model{index:03}.onnx",
                                       asymmetric_depthwise=entry["asymmetric_depthwise"])
            got = np.stack([join_chain_scale_reference(case[:192].reshape(8, 8, 3),
                                                       meta["stem_quantization"], meta["head_quantization"],
                                                       meta["join_kinds"], codes,
                                                       head_kinds=meta["head_kinds"],
                                                       depthwise_quantizations=meta["depthwise_quantization"],
                                                       runtime_scale=meta["runtime_scale"],
                                                       output_zero_point=meta["output_zero_point"])
                            for case in packed])
            with self.subTest(model=index):
                self.assertTrue(np.array_equal(got, expected))

    def test_suite_outputs_are_not_degenerate(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            with self.subTest(model=entry["index"]):
                self.assertGreater(entry["expected_nonzero"], entry["expected_values"] // 5)

    def test_scale_primary_gets_a_fresh_slot(self):
        _, meta = compile_sequence(build(joins=("Add", "Mul")))
        offsets = meta["tensor_offsets"]
        primary = offsets[meta["schedule"][-2]]
        internals = {name: value for name, value in offsets.items()
                     if name not in ("input0", "scale", "output")}
        self.assertEqual(sum(value == primary for value in internals.values()), 1)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            compile_sequence(build(scale_shape=(1, 3, 8, 8)))
        with self.assertRaises(ValueError):
            compile_sequence(build(no_scale=True, joins=("Add", "Add")),
                             output_range={"scale": .1, "zero_point": 0})
        with self.assertRaises(ValueError):
            compile_sequence(build(), mul_operand_zero_points=(1, 0))
        with self.assertRaises(ValueError):
            compile_sequence(build(hidden=8, kinds=("dense", "depthwise", "dense")))
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
