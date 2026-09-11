"""MIT. Cross-job pipelining evidence and primitive regression (PIPELINING_PLAN S4).

The runtime exposes experimental job-flag submission (`ornpu_submit_flags`),
completion polling (`ornpu_wait_fence`), output cache maintenance
(`ornpu_sync_outputs`) and single-input packing (`ornpu_set_input`). Board evidence in
`research/async_probe/` records what the attached kernel allows: NONBLOCK is accepted
and a queue-then-drain pipeline is 1.5-2.25x faster than synchronous submission, while
FENCE_OUT/FENCE_IN return -EINVAL because the kernel lacks
`CONFIG_ROCKCHIP_RKNPU_FENCE`.
"""
from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "async_probe"
EVIDENCE = PROBE / "board_results.txt"


class AsyncPipeliningTests(unittest.TestCase):
    def test_runtime_exposes_the_pipeline_primitives(self):
        header = (ROOT / "runtime" / "open_rknpu.h").read_text()
        for symbol in ("ornpu_submit_flags", "ornpu_wait_fence", "ornpu_sync_outputs",
                       "ornpu_set_input"):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, header)
        source = (ROOT / "runtime" / "open_rknpu.c").read_text()
        self.assertIn("submit_tasks_flags", source)
        self.assertIn("poll(&entry,1,timeout_ms)", source)

    def test_evidence_records_the_measured_rules(self):
        self.assertTrue(EVIDENCE.is_file(), EVIDENCE)
        text = EVIDENCE.read_text()
        self.assertEqual(text.count("nonblock: rc=0"), 2)
        self.assertEqual(text.count("fence_out: rc=-22"), 2)
        ratios = [float(value) for value in re.findall(r"ratio=([0-9.]+)", text)]
        self.assertEqual(len(ratios), 2)
        for ratio in ratios:
            self.assertGreater(ratio, 1.2)
        self.assertEqual(text.count("outputs_ok=yes"), 2)
        self.assertEqual(text.count("expected_match=yes"), 2)
        self.assertTrue((PROBE / "README.md").is_file())
        harness = (ROOT / "tests" / "board_async.c").read_text()
        for symbol in ("JOB_NONBLOCK", "JOB_FENCE_OUT", "JOB_FENCE_IN"):
            self.assertIn(symbol, harness)


    def test_barrier_probe_records_fence_free_completion(self):
        probe = ROOT / "research" / "barrier_probe"
        results = json.loads((probe / "board_results.json").read_text())
        containers = [record for record in results if "completed_median_us" in record]
        self.assertEqual(len(containers), 2)
        for record in containers:
            with self.subTest(suite=record["suite"], model=record["model"]):
                self.assertEqual(record["sync_mismatches"], 0)
                self.assertEqual(record["completed_mismatches"], 0)
                self.assertGreater(record["sync_median_us"], 0)
                self.assertGreater(record["barrier_median_us"], 0)
        # Queueing a serial list and waiting on a barrier replaces N ioctl round trips,
        # so it is faster there; a container already submitted as one job pays the
        # barrier's own cost instead. Both read the output with lag 0.
        serial, batched = containers
        self.assertEqual(serial["mode"], "serial")
        self.assertLess(serial["overhead_us"], 0)
        self.assertEqual(batched["mode"], "batched")
        self.assertGreater(batched["overhead_us"], 0)
        fence = [record for record in results if "fence_out_rc" in record]
        self.assertEqual(len(fence), 1)
        self.assertEqual(fence[0]["fence_out_rc"], -22)     # still -EINVAL
        self.assertGreater(fence[0]["pipeline_ratio"], 1.0)
        self.assertIn("barrier-completed", (probe / "board_results.txt").read_text())
        header = (ROOT / "runtime" / "open_rknpu.h").read_text()
        for symbol in ("ORNPU_JOB_NONBLOCK", "ORNPU_JOB_FENCE_OUT", "ornpu_sync_outputs"):
            self.assertIn(symbol, header)
        self.assertTrue((ROOT / "tests" / "board_barrier.c").is_file())

if __name__ == "__main__":
    unittest.main()
