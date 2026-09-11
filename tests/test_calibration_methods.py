"""SPDX-License-Identifier: MIT

Measurement-level regression tests for `open_rknpu.calibration`.

The calibration pass is the only place a *trained* model's real activation range
reaches the compiler, so this module pins the three selectors against data whose
distribution is known rather than against a smoke check:

* `minmax` must return the observed extremes (including the pass's documented zero
  clamp for a band that never crosses zero), for both accepted `.npy` dtypes;
* `percentile` must clip the upper tail while keeping the observed minimum, and must
  move monotonically with the requested percentile;
* `kl` must return a finite threshold inside the observed range for every bin count
  and must be deterministic for fixed input data;
* every rejection the pass documents is exercised: missing/empty directory, wrong
  shape or dtype, values outside [0,255], non-finite values, unknown method, and a
  percentile outside (0,100];
* the report's measured band must reach the container through
  `compile_sequence(..., calibration_ranges=...)` and `compile_model`, and the
  documented "cannot be combined" rejection against `output_range` must hold.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.calibration import (METHODS, _histogram, kl_range, measure,
                                    measured_tensor_names, percentile_range,
                                    range_to_quantization)
from open_rknpu.compiler import compile_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence


def value_info(graph, name, shape):
    graph.value_info.append(h.make_tensor_value_info(name, 1, shape))


def make_model(nodes, initializers, input_shape, output_name, output_shape, value_infos=()):
    graph = h.make_graph(nodes, "calibration_graph",
                         [h.make_tensor_value_info("input", 1, list(input_shape))],
                         [h.make_tensor_value_info(output_name, 1, list(output_shape))],
                         list(initializers))
    for name, shape in value_infos:
        value_info(graph, name, shape)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def signed_conv_pair(seed=17, hidden=4):
    """Conv -> Conv with no activation: activations cross zero in both directions."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["c1", "w2", "b2"], ["output"], kernel_shape=[1, 1])]
    initializers = [nh.from_array(rng.uniform(-.8, .9, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-3, 3, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.8, .9, (hidden, hidden, 1, 1)).astype(np.float32), "w2"),
                    nh.from_array(rng.uniform(-3, 3, (hidden,)).astype(np.float32), "b2")]
    return make_model(nodes, initializers, [1, 3, 8, 8], "output", [1, hidden, 8, 8],
                      value_infos=(("c1", [1, hidden, 8, 8]),))


def single_conv_graph(seed=6, channels=4):
    """One Conv: the profile `compile_model` accepts calibration ranges for."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.6, .7, (channels, 3, 1, 1)).astype(np.float32)
    bias = rng.uniform(-2, 2, (channels,)).astype(np.float32)
    node = h.make_node("Conv", ["input", "weights", "bias"], ["output"], kernel_shape=[1, 1])
    return make_model([node],
                      [nh.from_array(weights, "weights"), nh.from_array(bias, "bias")],
                      [1, 3, 8, 8], "output", [1, channels, 8, 8])


def positive_spike_graph():
    """Conv -> Relu -> Conv with a small positive bulk and a long outlier tail.

    Weights are all exactly 0.1, so a sample with one 255 channel produces a 25.5
    activation while the bulk stays under 1: min/max spans the tail, percentile clips
    it, and the KL search must find a threshold below it.
    """
    w1 = np.full((4, 3, 1, 1), 0.1, np.float32)
    b1 = np.zeros(4, np.float32)
    w2 = np.full((3, 4, 1, 1), 0.1, np.float32)
    b2 = np.zeros(3, np.float32)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c1"], ["r1"]),
             h.make_node("Conv", ["r1", "w2", "b2"], ["output"], kernel_shape=[1, 1])]
    initializers = [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"),
                    nh.from_array(w2, "w2"), nh.from_array(b2, "b2")]
    return make_model(nodes, initializers, [1, 3, 8, 8], "output", [1, 3, 8, 8],
                      value_infos=(("c1", [1, 4, 8, 8]), ("r1", [1, 4, 8, 8])))


def walk_chain_graph(seed=3, hidden=5):
    """Conv -> Relu -> MaxPool -> Conv -> Relu, the op-level chain walk."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c1"], ["r1"]),
             h.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
             h.make_node("Conv", ["p1", "w2", "b2"], ["c2"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c2"], ["output"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (hidden, hidden, 1, 1)).astype(np.float32), "w2"),
                    nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b2")]
    return make_model(nodes, initializers, [1, 3, 8, 8], "output", [1, hidden, 4, 4],
                      value_infos=(("c1", [1, hidden, 8, 8]), ("r1", [1, hidden, 8, 8]),
                                   ("p1", [1, hidden, 4, 4]), ("c2", [1, hidden, 4, 4])))


def save(model):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "model.onnx"
    onnx.save(model, path)
    return path


def observed_extremes(path, batches):
    """Per measured tensor, the (min, max) an independent ONNX run observes."""
    model = onnx.load(path)
    evaluator = ReferenceEvaluator(model)
    samples = [sample for batch in batches for sample in batch]
    totals = {}
    for name in measured_tensor_names(model):
        values = [evaluator.run([name], {"input": sample[None].astype(np.float32)})[0] for sample in samples]
        stacked = np.concatenate([value.ravel() for value in values])
        totals[name] = (float(stacked.min()), float(stacked.max()))
    return totals


class MinMaxMeasurementTests(unittest.TestCase):
    """`minmax` reproduces exactly what one pass observes."""

    def test_minmax_reproduces_the_observed_extremes(self):
        path = save(signed_conv_pair())
        rng = np.random.default_rng(2)
        batches = [rng.integers(0, 256, (3, 3, 8, 8), dtype=np.uint8),
                   rng.integers(0, 256, (2, 3, 8, 8), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as folder:
            for index, batch in enumerate(batches):
                np.save(Path(folder) / f"batch{index}.npy", batch)
            report = measure(path, folder)
        self.assertEqual(report["samples"], 5)
        self.assertEqual(report["method"], "minmax")
        self.assertEqual(report["input_layout"], "NCHW")
        self.assertEqual((report["input_scale"], report["input_zero_point"]), (1, 0))
        self.assertEqual(len(report["files"]), 2)
        extremes = observed_extremes(path, batches)
        self.assertEqual(sorted(report["ranges"]), sorted(extremes))
        for name, (low, high) in extremes.items():
            entry = report["ranges"][name]
            # The pass seeds the search at zero, so it reports min(0, observed) and
            # max(0, observed); this graph crosses zero, so the extremes are exact.
            self.assertLess(low, 0.0)
            self.assertGreater(high, 0.0)
            self.assertEqual((entry["min"], entry["max"]), (low, high))
            self.assertEqual(entry["scale"], float(np.float32((high - low) / 255)) or 1.0)
            self.assertEqual(entry["zero_point"],
                             int(np.clip(np.rint(-128 - low / entry["scale"]), -128, 127)))

    def test_minmax_accepts_uint8_and_float32_in_the_unit_range(self):
        path = save(positive_spike_graph())
        rng = np.random.default_rng(4)
        integer = rng.integers(0, 20, (4, 3, 8, 8), dtype=np.uint8)
        floating = rng.uniform(0.0, 20.0, (2, 3, 8, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder) / "integer.npy", integer)
            np.save(Path(folder) / "floating.npy", floating)
            report = measure(path, folder)
        self.assertEqual(report["samples"], 6)
        extremes = observed_extremes(path, [integer, floating])
        for name, (low, high) in extremes.items():
            self.assertEqual((report["ranges"][name]["min"], report["ranges"][name]["max"]),
                             (min(0.0, low), max(0.0, high)))

    def test_all_positive_band_is_clamped_at_zero(self):
        path = save(positive_spike_graph())
        rng = np.random.default_rng(5)
        batch = rng.integers(0, 10, (3, 3, 8, 8), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder) / "batch.npy", batch)
            report = measure(path, folder)
        for entry in report["ranges"].values():
            self.assertEqual(entry["min"], 0.0)

    def test_range_to_quantization_is_the_documented_closed_form(self):
        entry = {"min": -2.0, "max": 6.0}
        scale, zero_point = range_to_quantization(entry)
        self.assertEqual(scale, float(np.float32(8.0 / 255)))
        self.assertEqual(zero_point, int(np.clip(np.rint(-128 - entry["min"] / scale), -128, 127)))
        # A degenerate band falls back to unit scale (`... or 1.0`) rather than raising.
        self.assertEqual(range_to_quantization({"min": 0.0, "max": 0.0})[0], 1.0)
        for bad in ({"min": 0.0, "max": float("inf")}, {"min": float("nan"), "max": 1.0},
                    {"min": 1.0, "max": float("-inf")}):
            with self.subTest(entry=bad):
                with self.assertRaisesRegex(ValueError, "invalid calibration scale"):
                    range_to_quantization(bad)


class PercentileAndKlTests(unittest.TestCase):
    """Tail clipping and the KL saturation search over a known distribution."""

    def setUp(self):
        self.path = save(positive_spike_graph())
        rng = np.random.default_rng(7)
        batch = rng.integers(0, 20, (8, 3, 8, 8), dtype=np.uint8)
        batch[0, 0, 0, 0] = 255
        batch[1, 1, 2, 2] = 255
        batch[2, 2, 3, 3] = 255
        self.folder = tempfile.mkdtemp()
        np.save(Path(self.folder) / "batch.npy", batch)
        self.minmax = measure(self.path, self.folder)

    def test_percentile_clips_the_upper_tail_and_keeps_the_minimum(self):
        report = measure(self.path, self.folder, method="percentile", percentile=99.0)
        self.assertEqual(report["method"], "percentile")
        self.assertEqual(report["percentile"], 99.0)
        for name, entry in report["ranges"].items():
            self.assertEqual(entry["min"], self.minmax["ranges"][name]["min"])
            self.assertLess(entry["max"], self.minmax["ranges"][name]["max"])
        self.assertLess(report["ranges"]["output"]["max"], self.minmax["ranges"]["output"]["max"])

    def test_percentile_is_monotone_and_saturates_at_the_observed_maximum(self):
        previous = -np.inf
        for percentile in (10.0, 50.0, 90.0, 99.0, 100.0):
            report = measure(self.path, self.folder, method="percentile", percentile=percentile)
            high = report["ranges"]["output"]["max"]
            self.assertGreaterEqual(high, previous)
            previous = high
        full = measure(self.path, self.folder, method="percentile", percentile=100.0)
        self.assertEqual(full["ranges"]["output"]["max"], self.minmax["ranges"]["output"]["max"])

    def test_kl_is_finite_inside_the_range_for_several_bin_counts(self):
        clipped = False
        for bins in (256, 512, 1024, 2048, 4096):
            with self.subTest(bins=bins):
                report = measure(self.path, self.folder, method="kl", bins=bins)
                self.assertEqual(report["bins"], bins)
                entry = report["ranges"]["output"]
                self.assertTrue(np.isfinite(entry["max"]))
                self.assertGreaterEqual(entry["max"], entry["min"])
                self.assertLessEqual(entry["max"], self.minmax["ranges"]["output"]["max"])
                self.assertGreaterEqual(entry["max"], 0.0)
                if entry["max"] < self.minmax["ranges"]["output"]["max"]:
                    clipped = True
        self.assertTrue(clipped, "the KL search must clip the spike for at least one bin count")

    def test_kl_is_deterministic_for_fixed_data(self):
        first = measure(self.path, self.folder, method="kl")
        second = measure(self.path, self.folder, method="kl")
        self.assertEqual(first["ranges"], second["ranges"])
        rng = np.random.default_rng(4)
        values = np.concatenate([rng.uniform(0, 10, 100000), np.full(50, 100.0)])
        histogram, edges = _histogram(values, 2048, 0, 100)
        self.assertEqual(kl_range(histogram, edges), kl_range(histogram, edges))

    def test_histogram_helpers_clip_a_spike_and_keep_the_observed_minimum(self):
        rng = np.random.default_rng(4)
        values = np.concatenate([rng.uniform(0, 10, 100000), np.full(50, 100.0)])
        histogram, edges = _histogram(values, 2048, 0, 100)
        self.assertEqual(percentile_range(histogram, edges, 100.0), (0.0, 100.0))
        low, high = percentile_range(histogram, edges, 99.9)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 11.0)
        kl_low, kl_high = kl_range(histogram, edges)
        self.assertEqual(kl_low, 0.0)
        self.assertLess(kl_high, 11.0)
        self.assertTrue(np.isfinite(kl_high))


class CalibrationErrorTests(unittest.TestCase):
    """Every documented rejection, with its message."""

    def setUp(self):
        self.path = save(positive_spike_graph())
        self.rng = np.random.default_rng(11)
        self.good = self.rng.integers(0, 20, (2, 3, 8, 8), dtype=np.uint8)

    def test_missing_and_empty_directories_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"must contain \.npy arrays"):
            measure(self.path, "/nonexistent/calibration/folder")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, r"must contain \.npy arrays"):
                measure(self.path, folder)

    def test_wrong_shape_and_dtype_are_rejected(self):
        bad = [np.zeros((8, 8, 3), np.uint8), np.zeros((1, 8, 8, 3), np.uint8),
               np.zeros((2, 3, 8, 8), np.int8), np.zeros((2, 3, 8, 8), np.float64),
               np.zeros((0, 3, 8, 8), np.uint8)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.npy"
            for data in bad:
                with self.subTest(shape=data.shape, dtype=data.dtype):
                    np.save(path, data)
                    with self.assertRaisesRegex(ValueError, "expected uint8/float32"):
                        measure(self.path, folder)

    def test_values_outside_the_unit_range_and_nonfinite_values_are_rejected(self):
        bad = [np.full((1, 3, 8, 8), 256.0, np.float32),
               np.full((1, 3, 8, 8), -1.0, np.float32),
               np.full((1, 3, 8, 8), np.nan, np.float32),
               np.full((1, 3, 8, 8), np.inf, np.float32)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.npy"
            for data in bad:
                with self.subTest(value=data.ravel()[0]):
                    np.save(path, data)
                    with self.assertRaisesRegex(ValueError, r"finite values in \[0,255\]"):
                        measure(self.path, folder)

    def test_unknown_method_and_out_of_range_percentile_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder) / "batch.npy", self.good)
            with self.assertRaisesRegex(ValueError, "method must be one of"):
                measure(self.path, folder, method="entropy")
            self.assertEqual(METHODS, ("minmax", "percentile", "kl"))
            for percentile in (0.0, -1.0, 100.0001):
                with self.subTest(percentile=percentile):
                    with self.assertRaisesRegex(ValueError, r"percentile must be in \(0,100\]"):
                        measure(self.path, folder, method="percentile", percentile=percentile)
            self.assertEqual(measure(self.path, folder, method="percentile",
                                     percentile=100.0)["percentile"], 100.0)

    def test_graph_shape_requirements_are_rejected(self):
        no_conv = make_model([h.make_node("Relu", ["input"], ["output"])], [],
                             [1, 3, 8, 8], "output", [1, 3, 8, 8])
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder) / "batch.npy", self.good)
            with self.assertRaisesRegex(ValueError, "at least one Conv"):
                measure(save(no_conv), folder)
        dynamic = onnx.load(self.path)
        dynamic.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder) / "batch.npy", self.good)
            with self.assertRaisesRegex(ValueError, "static rank-four NCHW input"):
                measure(save(dynamic), folder)


class CalibrationIntegrationTests(unittest.TestCase):
    """The measured band must survive into the compiled container."""

    def measured_report(self, path, method="minmax"):
        rng = np.random.default_rng(11)
        batch = rng.integers(0, 20, (8, 3, 8, 8), dtype=np.uint8)
        batch[0, 0, 0, 0] = 255
        batch[1, 1, 2, 2] = 255
        folder = tempfile.mkdtemp()
        np.save(Path(folder) / "batch.npy", batch)
        return measure(path, folder, method=method)

    def test_measured_band_reaches_a_walk_chain_container(self):
        path = save(walk_chain_graph())
        report = self.measured_report(path)
        analytic, _ = compile_sequence(path)
        binary, meta = compile_sequence(path, calibration_ranges=report["ranges"])
        entry = report["ranges"]["output"]
        self.assertEqual(meta["profile"], "chain-walk")
        self.assertEqual((meta["output_scale"], meta["output_zero_point"]),
                         (entry["scale"], entry["zero_point"]))
        final = meta["quantizations"][-1]
        self.assertEqual((final["output_scale"], final["output_zero_point"]),
                         (entry["scale"], entry["zero_point"]))
        self.assertEqual(decode_sequence(binary)["output_scale"], entry["scale"])
        self.assertNotEqual(decode_sequence(binary)["output_scale"],
                            decode_sequence(analytic)["output_scale"])

    def test_measured_band_reaches_the_single_conv_quantization(self):
        path = save(single_conv_graph())
        report = self.measured_report(path)
        _, meta = compile_model(path, calibration_ranges=report["ranges"])
        entry = report["ranges"]["output"]
        self.assertEqual((meta["quantization"]["output_scale"],
                          meta["quantization"]["output_zero_point"]),
                         (entry["scale"], entry["zero_point"]))
        self.assertEqual(meta["output_scale"], entry["scale"])
        self.assertEqual(meta["output_zero_point"], entry["zero_point"])

    def test_calibration_and_output_range_cannot_be_combined(self):
        path = save(walk_chain_graph())
        report = self.measured_report(path)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            compile_sequence(path, output_range={"scale": 1.0, "zero_point": 0},
                             calibration_ranges=report["ranges"])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            compile_model(path, 1.0, 0, calibration_ranges=report["ranges"])

    def test_calibration_ranges_must_cover_every_measured_tensor(self):
        path = save(walk_chain_graph())
        with self.assertRaisesRegex(ValueError, "lack tensor"):
            compile_sequence(path, calibration_ranges={"output": {"scale": 1.0, "zero_point": 0}})


if __name__ == "__main__":
    unittest.main()
