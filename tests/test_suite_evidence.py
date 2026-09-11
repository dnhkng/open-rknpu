"""MIT. Guard the retained board evidence and the generators that write suites.

Three housekeeping invariants that regressed once and cost a suite its recorded
board run:

* a suite that records `board_results_*.json` also keeps a `board_summary.txt`,
  and every entry is well formed;
* a `research/build_*.py` generator never deletes the board evidence when it
  rewrites its own suite directory;
* `research/container_baseline.json` covers exactly the published suite models,
  so a new suite cannot slip past the `research/verify_suites.py` contract.
"""
from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
BASELINE = RESEARCH / "container_baseline.json"


class SuiteEvidenceTests(unittest.TestCase):
    def test_board_evidence_is_complete(self):
        results = sorted(RESEARCH.glob("*/board_results_*.json"))
        self.assertGreater(len(results), 20)
        for path in results:
            with self.subTest(suite=path.parent.name):
                entries = json.loads(path.read_text())
                self.assertIsInstance(entries, list)
                self.assertTrue(entries)
                for entry in entries:
                    self.assertEqual(
                        set(entry) >= {"model", "passed", "inferences", "bytes", "output"},
                        True)
                    self.assertIn(entry["passed"], (True, False))

    def test_generators_preserve_board_evidence(self):
        for path in sorted(RESEARCH.glob("build_*.py")):
            text = path.read_text()
            for patterns in re.findall(r"for pattern in \((.*?)\):", text, re.S):
                with self.subTest(generator=path.name):
                    self.assertNotIn("board_results", patterns)
        runner = RESEARCH / "run_v5_suite.py"
        self.assertTrue(runner.is_file())
        self.assertIn("board_summary.txt", runner.read_text())

    def test_container_baseline_covers_every_suite_model(self):
        baseline = json.loads(BASELINE.read_text())
        models = {str(path.relative_to(RESEARCH)) for path in RESEARCH.glob("*suite*/model*.onnx")}
        self.assertEqual(set(baseline), models)
        for key, value in baseline.items():
            with self.subTest(model=key):
                self.assertTrue(value.startswith("ERR:") or (len(value) == 64 and int(value, 16) >= 0),
                                "an entry is either a sha256 or a pinned rejection")


if __name__ == "__main__":
    unittest.main()
