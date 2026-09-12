"""SPDX-License-Identifier: MIT

E7: `examples/depthwise_separable/` builds, decodes and stays byte-repeatable.

`build.py` is run twice as a subprocess into two temporary output directories,
with the process working directory also temporary, so a stray relative write
cannot reach the checkout. The container must decode to the documented
`[1,3,8,8] -> [1,4,4,4]` chain, the fixtures must carry the container's exact
sizes, the written `expected.i8` must equal the `chain-walk` integer reference
recomputed in this process, and both runs must agree byte for byte. Fast,
deterministic, host-only: no board, no network.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "depthwise_separable"
BUILD = EXAMPLE / "build.py"
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
ARTIFACTS = ("prefix.bin", "inputs.u8", "expected.i8", "report.json")


def load_build():
    """Import the example's build.py under a private module name."""
    spec = importlib.util.spec_from_file_location("depthwise_separable_build", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DepthwiseSeparableExampleTests(unittest.TestCase):
    """The published example artifacts and the reference they claim."""

    @classmethod
    def setUpClass(cls):
        cls.build = load_build()
        cls.model = cls.build.build_model()
        cls.binary, cls.meta = cls.build.compile_block(cls.model)
        cls.cases = cls.build.deterministic_cases()
        cls.expected = cls.build.integer_reference(cls.model, cls.meta, cls.cases)

    def run_build(self, output):
        with tempfile.TemporaryDirectory() as cwd:
            process = subprocess.run([sys.executable, str(BUILD), "--out", str(output)],
                                     cwd=cwd, env=ENV, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_build_runs_twice_and_is_byte_repeatable(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.run_build(Path(first))
            self.run_build(Path(second))
            for name in ARTIFACTS:
                self.assertEqual((Path(first) / name).read_bytes(), (Path(second) / name).read_bytes(),
                                 "%s changed between builds" % name)

    def test_container_decodes_to_the_documented_chain(self):
        info = decode_sequence(self.binary)
        self.assertEqual(info["format_version"], 5)
        self.assertEqual(info["task_count"], 4)
        self.assertEqual(list(info["shape_nhwc"]), [1, 8, 8, 3])
        self.assertEqual(list(info["output_shape_nhwc"]), [1, 4, 4, 4])
        self.assertEqual(info["input_bytes"], int(self.cases[0].size))
        self.assertEqual(info["output_bytes"], int(self.expected[0].size))
        self.assertEqual(self.meta["profile"], "chain-walk")

    def test_expected_bytes_are_the_profile_reference(self):
        with tempfile.TemporaryDirectory() as output:
            self.run_build(Path(output))
            written = np.fromfile(Path(output) / "expected.i8", dtype=np.int8).reshape(self.expected.shape)
            np.testing.assert_array_equal(written, self.expected)
            inputs = np.fromfile(Path(output) / "inputs.u8", dtype=np.uint8).reshape(self.cases.shape)
            np.testing.assert_array_equal(inputs, self.cases)
            report = json.loads((Path(output) / "report.json").read_text())
            self.assertEqual(report["profile"], "chain-walk")
            self.assertEqual(report["cases"], self.cases.shape[0])
            self.assertEqual(report["expected_bytes"], int(self.expected.size))
            self.assertEqual(report["expected_sha256"],
                             hashlib.sha256(self.expected.tobytes()).hexdigest())
            self.assertEqual(report["output_shape_nhwc"], [1, 4, 4, 4])
            self.assertGreater(len(np.unique(self.expected)), 1, "the reference is degenerate")


if __name__ == "__main__":
    unittest.main()
