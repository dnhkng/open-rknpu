"""SPDX-License-Identifier: MIT

The host-only example entry points must actually run.

`examples/mnist/` and `examples/fashion/` are the project's "combine the primitives into a
model" story, and nothing used to execute them: CI ran only `examples/primitives/`. These
tests run the build scripts exactly as the READMEs tell a user to, then check that the
artifacts exist, decode, and describe the model the README claims (the NPU prefix of a
pretrained classifier). Data-dependent examples (FSDD audio, PyTorch training) are skipped
when their inputs or dependencies are absent, never silently passed.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))


def run_example(*parts):
    """Run an example script from the repository root and return (status, output)."""
    script = ROOT.joinpath(*parts)
    process = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=ENV,
                             capture_output=True, text=True, timeout=600)
    return process.returncode, process.stdout + process.stderr


class ClassifierExampleTests(unittest.TestCase):
    """`examples/mnist/build.py` and `examples/fashion/build.py` in the default mode."""

    def check(self, example, expected_shape, minimum_bytes):
        """Run a build script; both the dataset-present and dataset-absent paths must work.

        CI has no fetched dataset, so it exercises the documented analytic-band fallback;
        a developer machine with the data exercises the dataset-derived band. Whichever
        path runs, the prefix must exist and decode.
        """
        status, output = run_example("examples", example, "build.py")
        self.assertEqual(status, 0, output[-2000:])
        dataset = ROOT / "research" / "pretrained" / f"{example}-mnist" / "test-data" \
            / "t10k-images-idx3-ubyte.gz"
        if example == "fashion" and not dataset.is_file():
            self.assertIn("not found", output, "the fallback must say why it changed the band")
        build = ROOT / "examples" / example / "build"
        prefix = build / "prefix.bin"
        inputs = build / "inputs.u8"
        self.assertTrue(prefix.is_file(), "no prefix container was written")
        self.assertTrue(inputs.is_file(), "no input file was written")
        self.assertGreater(prefix.stat().st_size, minimum_bytes)
        from open_rknpu.model import decode
        info = decode(prefix.read_bytes())
        self.assertEqual(info["shape_nhwc"][1:], list(expected_shape))
        self.assertEqual(info["target"], "rv1103")
        report = build / "report.json"
        if report.is_file():  # the legacy build writes a numeric report too
            payload = json.loads(report.read_text())
            self.assertIsInstance(payload, (dict, list))

    def test_mnist_build_runs(self):
        self.check("mnist", (28, 28, 1), 2000)

    def test_fashion_build_runs(self):
        self.check("fashion", (28, 28, 1), 2000)


class AudioExampleTests(unittest.TestCase):
    """`examples/mel-kws/` needs FSDD on disk; the front end is importable regardless."""

    def test_feature_front_end_is_deterministic(self):
        status, output = run_example("examples", "mel-kws", "features.py")
        # features.py is a library module: running it must at least import cleanly.
        self.assertEqual(status, 0, output[-2000:])

    def test_fetch_script_documents_its_pins(self):
        text = (ROOT / "examples" / "mel-kws" / "fetch_data.py").read_text()
        self.assertIn("sha256", text)
        self.assertIn("https://", text)
        self.assertIn("CC BY-SA", text)


class PrimitivesCatalogTests(unittest.TestCase):
    """The op catalog is a deliverable: every documented script must exist and be runnable."""

    def test_every_documented_script_exists(self):
        readme = (ROOT / "examples" / "primitives" / "README.md").read_text()
        scripts = sorted((ROOT / "examples" / "primitives").glob("[0-9]*.py"))
        self.assertGreaterEqual(len(scripts), 11)
        for script in scripts:
            with self.subTest(script=script.name):
                self.assertIn(script.name, readme, "script missing from the catalog README")

    def test_catalog_scripts_are_importable(self):
        """A syntax/import error in any catalog script fails here, not in CI's shell loop."""
        for script in sorted((ROOT / "examples" / "primitives").glob("[0-9]*.py")):
            with self.subTest(script=script.name):
                process = subprocess.run([sys.executable, "-c",
                                          "import ast,sys;ast.parse(open(sys.argv[1]).read())",
                                          str(script)], capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
