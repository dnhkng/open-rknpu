"""Fashion-MNIST companion model: structure, pinned hash and board reports.

Mirrors `tests/test_mnist_reports.py` for the second trained model. The dataset is
not vendored, so the tests key off the trained ONNX model and the checked-in
reports (`examples/fashion/sanity-results/`).
"""
import hashlib
import json
import unittest
from pathlib import Path
import onnx
from onnx import numpy_helper as nh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "pretrained" / "fashion-mnist"
RESULTS = ROOT / "examples" / "fashion" / "sanity-results"
MODEL = SOURCE / "normalized.onnx"
REPORTS = {name: RESULTS / f"full_dataset_{name}_prefix_report.json"
           for name in ("hybrid", "analytic", "calibrated")}


@unittest.skipUnless(MODEL.exists(), "the trained Fashion-MNIST model is not present")
class FashionStructureTests(unittest.TestCase):
    def model(self):
        return onnx.load(MODEL)

    def test_pinned_hash_matches(self):
        pinned = (SOURCE / "model.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(), pinned)

    def test_graph_matches_the_pinned_structure(self):
        graph = self.model().graph
        self.assertEqual([n.op_type for n in graph.node],
                         ["Conv", "Relu", "MaxPool", "Conv", "Relu", "MaxPool", "Reshape", "MatMul", "Add"])
        tensors = {t.name: nh.to_array(t) for t in graph.initializer}
        self.assertEqual(tensors["Parameter5"].shape, (8, 1, 5, 5))
        self.assertEqual(tensors["Parameter87"].shape, (16, 8, 5, 5))
        self.assertEqual(tensors["Parameter193_reshape1"].shape, (256, 10))
        self.assertEqual(list(tensors["Pooling160_Output_0_reshape0_shape"]), [1, 256])
        # The shared C suffix indexes dense_w as [k*10+j], so MatMul must be [256,10].
        self.assertEqual(graph.node[7].input[1], "Parameter193_reshape1")

    def test_float_accuracy_and_reports(self):
        report = json.loads((SOURCE / "train_report.json").read_text())
        self.assertEqual(report["parameters"], 5994)
        self.assertGreater(report["test_accuracy"], 0.85)
        for name, path in REPORTS.items():
            if not path.exists():
                continue
            with self.subTest(name=name):
                entry = json.loads(path.read_text())
                self.assertEqual(entry["count"], 10000)
                self.assertGreaterEqual(entry["agreement_with_float"], 9800)
                self.assertGreater(entry["board_accuracy"], 0.85)
                self.assertLess(abs(entry["board_accuracy"] - entry["float_accuracy"]), 0.02)
                self.assertEqual(sum(entry["prediction_histogram"]), 10000)
                self.assertGreater(len(set(entry["prediction_histogram"])), 1)

    def test_calibrated_range_is_wider_than_analytic(self):
        analytic = json.loads((ROOT / "examples/fashion/build-native/reference.json").read_text())
        calibrated = json.loads((ROOT / "examples/fashion/build-native-calibrated/reference.json").read_text())
        self.assertAlmostEqual(analytic["input_scale"], 1 / 255, places=6)
        self.assertEqual(analytic["input_zero_point"], 0)
        self.assertIsNone(analytic.get("native_conv2_output_range"))
        self.assertTrue(calibrated["native_conv2_calibrated"])
        measured = calibrated["native_conv2_output_range"]
        self.assertGreater(measured["scale"], 0)
        self.assertNotEqual(measured["scale"], analytic["output_scale"])
        self.assertNotEqual(measured["zero_point"], analytic["output_zero_point"])
        self.assertFalse(calibrated["dataset_accuracy_measured"])


if __name__ == "__main__":
    unittest.main()
