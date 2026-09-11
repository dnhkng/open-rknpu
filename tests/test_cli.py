"""MIT. CLI integration for the submission modes (docs/plans/pipelining-plan.md S6/S8).

`open-rknpu compile --sequence --submission batched` links a task list into one job,
writing the measured engine hand-off control per transition (S8), so a mixed-engine
CNA-run-then-DPU-run profile is accepted; the default stays serial and a profile whose
emitter ignores the request is refused with that reason. Runs `cli.main()` in process so
no installed script is needed.
"""
from pathlib import Path
import contextlib
import io
import sys
import unittest

from open_rknpu import cli
from open_rknpu.compose import batched_layout
from open_rknpu.model import decode

ROOT = Path(__file__).resolve().parents[1]


class CliSubmissionTests(unittest.TestCase):
    def compile(self, model, output, *flags):
        argv = ["open-rknpu", "compile", str(model), "--sequence", *flags, "-o", str(output)]
        stdout = io.StringIO()
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(stdout):
                cli.main()
        finally:
            sys.argv = old
        return stdout.getvalue()

    def test_default_submission_is_serial_and_unchanged(self):
        model = ROOT / "research" / "chain_multi_suite" / "model000.onnx"
        output = Path("/tmp/cli_serial_test.bin")
        text = self.compile(model, output, "--expose-intermediates")
        self.assertIn("serial submission", text)
        self.assertEqual(output.read_bytes(), model.with_suffix(".bin").read_bytes())

    def test_batched_submission_emits_a_linked_job(self):
        model = ROOT / "research" / "chain_multi_suite" / "model000.onnx"
        output = Path("/tmp/cli_batched_test.bin")
        text = self.compile(model, output, "--expose-intermediates", "--submission", "batched")
        self.assertIn("batched submission", text)
        data = output.read_bytes()
        info = decode(data)
        self.assertFalse(info["serial"])
        self.assertIsNone(batched_layout(data, info))

    def test_mixed_engine_batched_emits_one_job(self):
        model = ROOT / "research" / "pool_join_suite" / "model000.onnx"
        output = Path("/tmp/cli_mixed_batched.bin")
        text = self.compile(model, output, "--submission", "batched")
        self.assertIn("batched submission", text)
        data = output.read_bytes()
        info = decode(data)
        self.assertFalse(info["serial"])
        self.assertIsNone(batched_layout(data, info))

    def test_mixed_head_is_one_job_via_the_amount_control(self):
        model = ROOT / "research" / "mixed_head_suite" / "model001.onnx"
        output = Path("/tmp/cli_grouped.bin")
        text = self.compile(model, output, "--submission", "batched")
        self.assertIn("batched submission", text)
        data = output.read_bytes()
        info = decode(data)
        self.assertFalse(info["serial"])
        self.assertIsNone(batched_layout(data, info))

    def test_every_profile_accepts_batched_submission(self):
        # Since the post-pass relinks whatever an emitter produced (S10 residual), the
        # submission mode is never refused for a profile the compiler can compile.
        for suite, name in (("add_suite", "model000"), ("application_suite", "model000"),
                            ("mul_batch_suite", "model002")):
            model = ROOT / "research" / suite / f"{name}.onnx"
            output = Path("/tmp/cli_%s_%s.bin" % (suite, name))
            with self.subTest(suite=suite):
                text = self.compile(model, output, "--submission", "batched")
                self.assertIn("batched submission", text)
                data = output.read_bytes()
                info = decode(data)
                self.assertFalse(info["serial"])
                self.assertIsNone(batched_layout(data, info))

if __name__ == "__main__":
    unittest.main()
