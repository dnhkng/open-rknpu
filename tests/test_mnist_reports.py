"""Full-dataset MNIST reports: calibrated beats uncalibrated, float is the reference."""
import json
import unittest
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "examples/mnist/sanity-results"
CALIBRATED = RESULTS / "full_dataset_calibrated_prefix_report.json"
UNCALIBRATED = RESULTS / "full_dataset_native_prefix_report.json"


@unittest.skipUnless(CALIBRATED.exists() and UNCALIBRATED.exists(),
                     "full-dataset board reports are not present")
class FullDatasetReportTests(unittest.TestCase):
    def load(self, path):
        return json.loads(path.read_text())

    def test_calibrated_beats_uncalibrated_and_tracks_float(self):
        calibrated = self.load(CALIBRATED)
        uncalibrated = self.load(UNCALIBRATED)
        self.assertEqual(calibrated["count"], 10000)
        self.assertEqual(uncalibrated["count"], 10000)
        self.assertGreater(calibrated["board_accuracy"], 0.9)
        self.assertLess(uncalibrated["board_accuracy"], 0.2)
        self.assertGreater(calibrated["board_accuracy"], uncalibrated["board_accuracy"])
        # The calibrated both-Conv network tracks the float model closely.
        self.assertLess(abs(calibrated["float_accuracy"] - calibrated["board_accuracy"]), 0.05)
        self.assertGreaterEqual(calibrated["agreement_with_float"], 9000)
        # The uncalibrated variant collapses to a single class.
        self.assertEqual(max(uncalibrated["prediction_histogram"]), 10000)


if __name__ == "__main__":
    unittest.main()
