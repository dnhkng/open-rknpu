"""SPDX-License-Identifier: MIT

The cost-model baseline is evidence: it must describe exactly the pinned model set, the
comparison must call a growth a regression and a shrink an improvement, and the live tree
must still measure the pinned structure (task counts, engine blocks, registers, arena).

`research/perf_regression.py` is the harness; these tests are what stop it from rotting into
a file nobody re-pins, and they run the real compile in a few hundred milliseconds.
"""
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import perf_regression  # noqa: E402  (the harness under test)

BASELINE = ROOT / "research" / "perf_baseline.json"


def cost(**overrides):
    """A minimal measurement record, so a test can perturb exactly one field."""
    record = {"bytes": 100, "tasks": 2, "blocks": 4, "registers": 10,
              "arena_bytes": 8, "payload_bytes": 9}
    record.update(overrides)
    return record


class PerfRegressionTests(unittest.TestCase):
    def test_baseline_covers_exactly_the_pinned_model_set(self):
        baseline = json.loads(BASELINE.read_text())
        expected = {str(model.relative_to(ROOT / "research"))
                    for model in perf_regression.selected_models()}
        self.assertEqual(set(baseline), expected)
        self.assertGreaterEqual(len(baseline), 60)
        self.assertGreater(sum(1 for value in baseline.values() if "error" not in value), 60)

    def test_growth_in_any_structural_field_is_a_regression(self):
        baseline = {"a/b.onnx": cost()}
        for field in ("tasks", "blocks", "registers", "arena_bytes", "payload_bytes"):
            with self.subTest(field=field):
                current = {"a/b.onnx": cost(**{field: cost()[field] + 1})}
                _window, failures, improvements = perf_regression.compare(baseline, current)
                self.assertTrue(any(field in line for line in failures), failures)
                self.assertFalse(improvements)

    def test_bytes_have_a_small_tolerance_but_only_upwards(self):
        baseline = {"a/b.onnx": cost()}
        allowed = cost(bytes=100 + 16)
        _window, failures, _ = perf_regression.compare(baseline, {"a/b.onnx": allowed})
        self.assertFalse(failures, failures)
        too_big = cost(bytes=100 + 17)
        _window, failures, _ = perf_regression.compare(baseline, {"a/b.onnx": too_big})
        self.assertTrue(any("bytes" in line for line in failures), failures)
        smaller = cost(bytes=99)
        _window, failures, improvements = perf_regression.compare(baseline, {"a/b.onnx": smaller})
        self.assertFalse(failures)
        self.assertTrue(any("bytes" in line for line in improvements), improvements)

    def test_shrinking_work_is_reported_not_rejected(self):
        baseline = {"a/b.onnx": cost()}
        current = {"a/b.onnx": cost(tasks=1, registers=9, arena_bytes=7)}
        _window, failures, improvements = perf_regression.compare(baseline, current)
        self.assertFalse(failures, failures)
        self.assertEqual(len(improvements), 3, improvements)

    def test_a_changed_compile_outcome_is_a_regression(self):
        baseline = {"a/b.onnx": {"error": "ValueError"}}
        current = {"a/b.onnx": cost()}
        _window, failures, _ = perf_regression.compare(baseline, current)
        self.assertTrue(failures)
        baseline = {"a/b.onnx": cost()}
        current = {"a/b.onnx": {"error": "ValueError"}}
        _window, failures, _ = perf_regression.compare(baseline, current)
        self.assertTrue(failures)

    def test_live_tree_matches_the_pinned_cost_model(self):
        baseline = json.loads(BASELINE.read_text())
        current = perf_regression.collect()
        self.assertEqual(len(current), len(baseline))
        _window, failures, _improvements = perf_regression.compare(baseline, current)
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
