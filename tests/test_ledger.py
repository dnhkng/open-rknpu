"""MIT. Ledger integrity: every row is checked against its suite artifacts.

The board ledger in `research/COVERAGE_EXPANSION_RESULTS.md` is the project's evidence
index, so it is verified like code:

* every relative link in the ledger resolves;
* the totals row equals the sum of the rows;
* for every row whose suite keeps board evidence, the **union of per-model passing
  entries across all `board_results_*.json` files** reproduces the row's model count,
  inference count and exact byte count (a suite debugged over several board runs is
  therefore still checked as a whole);
* where a manifest records `cases` and `output_bytes`, those also reproduce the row;
* the number of rows with reproducible evidence cannot silently shrink.
"""
from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "research" / "COVERAGE_EXPANSION_RESULTS.md"
ROW = re.compile(r"^\| \[([^\]]+)\]\(([^)]*)\) \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \|$", re.M)
MIN_EVIDENCE_ROWS = 100
MIN_MANIFEST_ROWS = 11


def _number(text):
    return int(text.replace(",", ""))


def _passing_by_model(suite):
    """Union of passing per-model results across every board evidence file."""
    best = {}
    for path in sorted(suite.glob("board_results_*.json")):
        for entry in json.loads(path.read_text()):
            model = entry.get("model")
            if model is None:
                continue
            key = str(model)
            previous = best.get(key)
            if previous is None or (entry.get("passed") and not previous[0]):
                best[key] = (bool(entry.get("passed")), int(entry.get("inferences", 0)),
                             int(entry.get("bytes", 0)))
    return {key: value for key, value in best.items() if value[0]}


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = LEDGER.read_text()
        cls.rows = [(name, target, _number(models), _number(inferences), _number(byte_count))
                    for name, target, models, inferences, byte_count in ROW.findall(cls.text)]
        total = re.search(r"\*\*Total\*\* \| \*\*([\d,]+)\*\* \| \*\*([\d,]+)\*\* \| \*\*([\d,]+)\*\*",
                          cls.text)
        cls.total = tuple(_number(value) for value in total.groups())

    def test_rows_sum_to_the_total(self):
        self.assertEqual(len(self.rows), 129)
        self.assertEqual(tuple(sum(row[index] for row in self.rows) for index in (2, 3, 4)),
                         self.total)

    def test_every_ledger_link_resolves(self):
        links = re.findall(r"\[([^\]]+)\]\(([^)#]+)\)", self.text)
        self.assertGreater(len(links), 100)
        for name, target in links:
            if target.startswith(("http", "mailto")):
                continue
            with self.subTest(link=name):
                self.assertTrue((LEDGER.parent / target).resolve().exists(), target)

    def test_board_evidence_reproduces_every_row(self):
        checked = 0
        for name, target, models, inferences, byte_count in self.rows:
            suite = (LEDGER.parent / target).resolve()
            if not suite.is_dir():
                continue
            passing = _passing_by_model(suite)
            if not passing:
                continue
            checked += 1
            with self.subTest(suite=name):
                self.assertEqual(len(passing), models)
                self.assertEqual(sum(value[1] for value in passing.values()), inferences)
                self.assertEqual(sum(value[2] for value in passing.values()), byte_count)
        self.assertGreaterEqual(checked, MIN_EVIDENCE_ROWS)

    def test_manifests_reproduce_their_rows(self):
        checked = 0
        for name, target, models, inferences, byte_count in self.rows:
            suite = (LEDGER.parent / target).resolve()
            manifest = suite / "manifest.json"
            if not manifest.is_file():
                continue
            entries = json.loads(manifest.read_text())
            if not entries or not all("cases" in entry and "output_bytes" in entry for entry in entries):
                continue
            checked += 1
            with self.subTest(suite=name):
                self.assertEqual(len(entries), models)
                self.assertEqual(sum(int(entry["cases"]) for entry in entries), inferences)
                self.assertEqual(sum(int(entry["cases"]) * int(entry["output_bytes"]) for entry in entries),
                                 byte_count)
        self.assertGreaterEqual(checked, MIN_MANIFEST_ROWS)

    def test_campaign_suites_are_fully_evidenced(self):
        campaign = {"runtime_scale_suite", "join_chain_suite", "depthwise_join_suite",
                    "pool_join_suite", "mixed_head_suite", "join_scale_suite",
                    "join_residual_suite", "join_dag_suite", "branch_join_suite",
                    "transpose_k5_dilation_suite", "diamond_tail_suite",
                    "depthwise_chain_suite", "pooled_dag_suite", "pooled_branches_suite"}
        models = inferences = bytes_total = 0
        seen = set()
        for name, target, row_models, row_inferences, row_bytes in self.rows:
            suite = (LEDGER.parent / target).resolve()
            if suite.name not in campaign:
                continue
            seen.add(suite.name)
            models += row_models
            inferences += row_inferences
            bytes_total += row_bytes
        self.assertEqual(seen, campaign)
        self.assertEqual((models, inferences, bytes_total), (169, 4_832, 798_784))


if __name__ == "__main__":
    unittest.main()
