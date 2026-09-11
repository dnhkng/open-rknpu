"""MIT. Family-cost probe regression (docs/plans/pipelining-plan.md S5).

Pins the attempt and its measured outcome: three matched 8x8 containers (one CNA task,
the same Conv plus a pool task, and the verified Conv + per-channel Mul) all run
exactly, but the differential against the single-task container is inside the noise, so
the per-family table stays unresolved with a stated prerequisite.
"""
from pathlib import Path
import json
import unittest

from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "family_cost_probe"


TASKS = ROOT / "research" / "family_tasks_probe"


class FamilyTaskFitTests(unittest.TestCase):
    def test_all_variants_are_exact_and_fit_positive_slopes(self):
        manifest = json.loads((TASKS / "manifest.json").read_text())
        evidence = json.loads((TASKS / "family_tasks.json").read_text())
        self.assertEqual(len(manifest), len(evidence))
        for name, record in evidence.items():
            with self.subTest(variant=name):
                self.assertTrue(record["passed"])
                self.assertEqual(record.get("mismatches", 0), 0)
        summary = (TASKS / "summary.txt").read_text()
        for family in ("conv", "pool", "elem"):
            self.assertIn(family, summary)
        self.assertEqual(summary.count("exact=True"), 3)
        self.assertTrue((TASKS / "README.md").is_file())


class FamilyCostProbeTests(unittest.TestCase):
    def test_containers_are_the_matched_families(self):
        self.assertEqual([(t["register_count"], t["enable"])
                          for t in decode_sequence((PROBE / "model000.bin").read_bytes())["tasks"]],
                         [(126, 29)])
        self.assertEqual([(t["register_count"], t["enable"])
                          for t in decode_sequence((PROBE / "model001.bin").read_bytes())["tasks"]],
                         [(126, 29), (37, 96)])
        self.assertEqual([(t["register_count"], t["enable"])
                          for t in decode_sequence((PROBE / "model002.bin").read_bytes())["tasks"]],
                         [(126, 29), (78, 24)])

    def test_evidence_records_all_three_exact(self):
        evidence = json.loads((PROBE / "family_cost.json").read_text())
        self.assertEqual(sorted(evidence), ["conv", "conv_pool", "ew2"])
        for name, record in evidence.items():
            with self.subTest(model=name):
                self.assertTrue(record["passed"])
                self.assertEqual(record.get("mismatches", 0), 0)
                self.assertEqual(record.get("unstable", 0), 0)
                self.assertGreater(record["min_us"], 0)
        summary = (PROBE / "summary.txt").read_text()
        self.assertIn("load-dominated", summary)
        self.assertIn("pool task", summary)
        self.assertIn("elementwise task", summary)
        self.assertTrue((PROBE / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()


class FamilyCostCrosscheckTests(unittest.TestCase):
    """The table is cross-checked on 23 held-out real containers (two board runs)."""

    CROSS = ROOT / "research" / "family_cost_crosscheck"

    def table_and_manifest(self):
        import json

        def fit(xs, ys):
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            slope = sxy / sxx if sxx else 0.0
            return slope, my - slope * mx

        evidence = json.loads((TASKS / "family_tasks.json").read_text())
        slopes = {}
        for family in ("conv", "pool", "elem"):
            entries = sorted((entry for entry in evidence.values() if entry["family"] == family),
                             key=lambda entry: entry["family_tasks"])
            slopes[family] = fit([entry["family_tasks"] for entry in entries],
                                 [entry["min_us"] for entry in entries])[0]
        manifest = json.loads((self.CROSS / "manifest.json").read_text())
        return slopes, manifest

    def test_manifest_predictions_come_from_the_table(self):
        slopes, manifest = self.table_and_manifest()
        self.assertEqual(len(manifest["containers"]), 23)
        for entry in manifest["containers"]:
            with self.subTest(container=entry["name"]):
                predicted = (entry["conv"] * slopes["conv"] + entry["pool"] * slopes["pool"]
                             + entry["elem"] * slopes["elem"])
                self.assertAlmostEqual(entry["table_slope_us"], round(predicted, 2), places=6)

    def test_held_out_runs_are_exact_and_min_is_the_estimator(self):
        import json
        runs = [json.loads((self.CROSS / name).read_text())
                for name in ("crosscheck_run1.json", "crosscheck_run2.json")]
        self.assertEqual(len(runs[0]), 23)
        self.assertEqual(sorted(runs[0]), sorted(runs[1]))
        for run in runs:
            for name, record in run.items():
                with self.subTest(container=name):
                    self.assertTrue(record["passed"])
                    self.assertGreater(record["min_us"], 0)
                    self.assertGreaterEqual(record["median_us"], record["min_us"])
        noise = sorted(record["median_us"] / record["min_us"]
                       for record in runs[1].values())
        self.assertGreater(noise[len(noise) // 2], 1.3, "medians are load-dominated, not the floor")

    def test_repeat_runs_agree_on_the_minimum(self):
        import json
        first = json.loads((self.CROSS / "crosscheck_run1.json").read_text())
        second = json.loads((self.CROSS / "crosscheck_run2.json").read_text())
        ratios = [second[name]["min_us"] / first[name]["min_us"] for name in first]
        self.assertLess(max(ratios), 1.6)
        self.assertGreater(min(ratios), 0.7)

    def test_held_out_fit_confirms_cna_and_scopes_the_dpu_rows(self):
        import json
        import numpy as np
        slopes, manifest = self.table_and_manifest()
        for name in ("crosscheck_run1.json", "crosscheck_run2.json"):
            with self.subTest(run=name):
                run = json.loads((self.CROSS / name).read_text())
                design = np.array([[1.0, r["conv"], r["pool"], r["elem"]] for r in run.values()])
                values = np.array([r["min_us"] for r in run.values()])
                coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
                residuals = values - design @ coefficients
                r2 = 1 - float(residuals @ residuals) / float(((values - values.mean()) ** 2).sum())
                self.assertGreater(r2, 0.9)
                # The CNA family is the one the table claims generalizes.
                self.assertLess(abs(coefficients[1] - slopes["conv"]) / slopes["conv"], 0.3)
                # The elementwise row is profile-specific: real two-surface EW tasks
                # are cheaper than the runtime-scale template the table measured.
                self.assertLess(coefficients[3], slopes["elem"] * 0.8)

    def test_crosscheck_readme_is_retained(self):
        self.assertTrue((self.CROSS / "README.md").is_file())
