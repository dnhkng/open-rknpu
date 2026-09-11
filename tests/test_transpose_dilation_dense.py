"""MIT. Dense K3-dilation2 ConvTranspose via the sparse-K5 rewrite (PIPELINING_PLAN P3).

`transposed.compile_transposed` zero-stuffs a dense `(C_in, C_out, 3, 3)` dilated kernel
to `(C_in, C_out, 5, 5)` and emits the dense-K5 form; the dense emitter derives its kernel
size from the weights, and the dispatch sends dilated nodes through the rewrite before the
dense path. These tests pin the rewrite, its host guards and the retained board evidence.
"""
from pathlib import Path
import json
import unittest

import numpy as np
import onnx

from open_rknpu.quantization import Quantization, reference  # noqa: F401
from open_rknpu.scheduler import compile_sequence

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "transpose_dilation_dense_suite"


class DenseDilationTests(unittest.TestCase):
    def test_dense_dilated_graphs_rewrite_to_sparse_k5(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 4)
        for entry in manifest:
            model = onnx.load(SUITE / f"model{entry['index']:03}.onnx")
            with self.subTest(index=entry["index"], channels=(entry["input_channels"],
                                                              entry["output_channels"])):
                binary, meta = compile_sequence(SUITE / f"model{entry['index']:03}.onnx")
                self.assertEqual(meta["transposed_rewrite"], "K3 dilation2 expanded to sparse K5")
                self.assertIn("k5", meta["transposed_profile"])
                self.assertEqual(binary, (SUITE / f"model{entry['index']:03}.bin").read_bytes())
                node = model.graph.node[-1]
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                self.assertEqual(attrs["dilations"], [2, 2])
                self.assertEqual(attrs["kernel_shape"], [3, 3])

    def test_existing_transposed_suites_are_byte_identical(self):
        # The K5 generalization must not move the verified K3, K2-dilation and K5 output.
        for suite in ("transpose_dense_suite", "transpose_dilation_suite",
                      "transpose_k5_suite"):
            for path in sorted((ROOT / "research" / suite).glob("model*.onnx")):
                with self.subTest(suite=suite, model=path.name):
                    self.assertEqual(compile_sequence(path)[0],
                                     path.with_suffix(".bin").read_bytes())

    def test_dense_dilation_bounds_are_enforced(self):
        import copy
        from onnx import helper as h, numpy_helper as nh
        model = onnx.load(SUITE / "model000.onnx")
        node = model.graph.node[-1]
        # an even kernel has no zero-stuffable centre
        broken = copy.deepcopy(model)
        weights = next(v for v in broken.graph.initializer if v.name == node.input[1])
        raw = nh.to_array(weights)
        weights.CopyFrom(nh.from_array(np.zeros((raw.shape[0], raw.shape[1], 2, 2), np.float32),
                                       weights.name))
        del broken.graph.node[-1].attribute[:]
        broken.graph.node[-1].attribute.extend([
            h.make_attribute("group", 1), h.make_attribute("kernel_shape", [2, 2]),
            h.make_attribute("strides", [1, 1]), h.make_attribute("pads", [0, 0, 0, 0]),
            h.make_attribute("dilations", [2, 2]), h.make_attribute("output_padding", [0, 0])])
        for dim, value in zip(broken.graph.output[0].type.tensor_type.shape.dim[2:], [9, 9]):
            dim.dim_value = value
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.onnx"
            onnx.save(broken, path)
            with self.assertRaises(ValueError):
                compile_sequence(path)

    def test_board_evidence_is_retained(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 4)
        for record in results:
            with self.subTest(model=record["model"]):
                self.assertTrue(record["passed"])
                self.assertEqual(record["inferences"], 8)
        summary = (SUITE / "board_summary.txt").read_text()
        self.assertIn("32 inferences", summary)
        self.assertIn("18952", summary)


if __name__ == "__main__":
    unittest.main()
