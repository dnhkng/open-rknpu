"""SPDX-License-Identifier: MIT

E9: `examples/benchmark/bench.py`'s host mode is a documented, reusable sizing tool,
so the numbers it prints must be the container's own decoded fields.

The harness is run exactly as `examples/benchmark/README.md` documents - over a few
published suites, into a temporary markdown table - and the test checks that every
published container appears exactly once and that its `bytes`, `task_count` and
`engine_runs` cells agree with `open_rknpu.sequence.decode_sequence` and
`open_rknpu.compose.engine_runs` for the same file. Compile time is deliberately not
asserted: it is wall-clock and machine dependent, and the README says so.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "benchmark" / "bench.py"
SUITES = ("pool_join_suite", "diamond_tail_suite", "join_dag_suite")
ROW = re.compile(r"^\|\s*(research/\S+\.bin)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
                 r"\s*([\d.]+|-)\s*\|$")
BUDGET_SECONDS = 10.0


def suite_containers(suite):
    return sorted(path.relative_to(ROOT).as_posix()
                  for path in (ROOT / "research" / suite).glob("model*.bin"))


class BenchmarkHarnessTest(unittest.TestCase):
    """Run the documented host command and re-derive every structural cell."""

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp(prefix="rknpu-benchmark-"))
        cls.output = cls.folder / "table.md"
        command = [sys.executable, str(BENCH)]
        for suite in SUITES:
            command += ["--suite", suite]
        command += ["--markdown", "--output", str(cls.output)]
        started = time.monotonic()
        cls.process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                     env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                                     timeout=120)
        cls.elapsed = time.monotonic() - started
        cls.text = cls.output.read_text() if cls.output.is_file() else ""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def rows(self):
        parsed = []
        for line in self.text.splitlines():
            match = ROW.match(line)
            if match:
                parsed.append((match.group(1), int(match.group(2)), int(match.group(3)),
                               int(match.group(4)), match.group(5)))
        return parsed

    def test_host_mode_exits_zero_and_writes_the_requested_file(self):
        self.assertEqual(self.process.returncode, 0, self.process.stderr[-2000:])
        self.assertTrue(self.output.is_file(), "no table was written to --output")

    def test_every_published_container_has_exactly_one_row(self):
        expected = sorted(name for suite in SUITES for name in suite_containers(suite))
        self.assertEqual([row[0] for row in self.rows()], expected)

    def test_structural_cells_agree_with_the_decoders(self):
        from open_rknpu.compose import engine_runs
        from open_rknpu.sequence import decode_sequence
        for model, size, tasks, runs, _compile in self.rows():
            with self.subTest(model=model):
                data = (ROOT / model).read_bytes()
                info = decode_sequence(data)
                self.assertEqual(size, len(data), "bytes cell")
                self.assertEqual(tasks, info["task_count"], "task_count cell")
                self.assertEqual(runs, len(engine_runs(data, info)), "engine_runs cell")

    def test_the_documented_command_stays_under_its_budget(self):
        self.assertLess(self.elapsed, BUDGET_SECONDS,
                        "the host benchmark table took %.1f s" % self.elapsed)


if __name__ == "__main__":
    unittest.main()
