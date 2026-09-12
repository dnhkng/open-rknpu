"""SPDX-License-Identifier: MIT

E13: `examples/multi_model/build.py` compiles two models, exactly and deterministically.

The example's claims are checked here without a board: both published containers
decode as v5 named-tensor executables, each with its own non-zero arena; the two
profile integer references independently reproduce every published `expected*.i8`
byte; and a second build in a second temp directory is byte-identical. The build
is always run in a temporary working directory with `--out`, so the checkout is
never written to (the README's plain command writes to the git-ignored
`examples/multi_model/build/`).
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "examples" / "multi_model" / "build.py"
BOARD = ROOT / "examples" / "multi_model" / "board.c"
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
ARTIFACTS = ("model_a.bin", "model_b.bin", "input_a.u8", "input_b.u8",
             "expected_a.i8", "expected_b.i8", "model_a.onnx", "model_b.onnx", "report.json")
COMPILER = os.environ.get("OPEN_RKNPU_HOST_CC") or shutil.which("cc") or shutil.which("gcc")
HOST_FLAGS = ["-O2", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE",
              "-I%s" % (ROOT / "runtime")]

sys.path.insert(0, str(ROOT / "examples" / "primitives"))

from common import qfrom
from open_rknpu.chain import native_reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence


def run_build(folder):
    """Run the example's build script from `folder` with all artifacts under `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    process = subprocess.run([sys.executable, str(BUILD), "--out", str(folder)],
                             cwd=str(folder), env=ENV, capture_output=True, text=True, timeout=120)
    if process.returncode != 0:
        raise AssertionError("examples/multi_model/build.py failed:\n%s"
                             % (process.stdout + process.stderr)[-2000:])
    return process.stdout + process.stderr


def reference_for(folder, kind):
    """Recompute one model's integer reference from the ONNX graph it was compiled from."""
    row = json.loads((folder / "report.json").read_text())["models"][kind]
    _, meta = compile_sequence(folder / row["onnx"], **row["compile"])
    info = decode_sequence((folder / ("model_%s.bin" % kind)).read_bytes())
    height, width, channels = info["shape_nhwc"][1:]
    cases = np.fromfile(folder / ("input_%s.u8" % kind), dtype=np.uint8)
    cases = cases.reshape(-1, height, width, channels)
    if kind == "a":
        quantization = qfrom(meta["quantization"])
        return np.stack([native_input_reference(case, quantization, int(meta["input_zero_point"]),
                                               pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta["conv_dilations"]))
                         for case in cases])
    stem = qfrom(meta["first"]["quantization"])
    layer = qfrom(meta["depthwise"])
    head = qfrom(meta["pointwise"])
    return np.stack([native_reference(depthwise_reference(reference(case, stem), layer,
                                                         stem.output_zero_point),
                                     head, layer.output_zero_point)
                     for case in cases])


class MultiModelExampleTests(unittest.TestCase):
    """Two containers, each with its own arena, compiled and checked once per class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="rknpu-multi-model-")
        cls.out = Path(cls._tmp.name)
        run_build(cls.out)
        cls.report = json.loads((cls.out / "report.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_two_containers_decode_as_named_tensor_executables(self):
        for kind in ("a", "b"):
            with self.subTest(model=kind):
                info = decode_sequence((self.out / ("model_%s.bin" % kind)).read_bytes())
                self.assertEqual(info["format_version"], 5)
                self.assertEqual(info["tensor_count"], 2)
                self.assertEqual(info["input_tensor_count"], 1)
                self.assertEqual(info["output_tensor_count"], 1)
                self.assertGreater(info["arena_bytes"], 0)
                self.assertEqual(info["arena_bytes"], self.report["models"][kind]["arena_bytes"])
                self.assertEqual(info["task_count"], self.report["models"][kind]["tasks"])

    def test_the_two_models_own_independent_arenas(self):
        arena = self.report["arena"]
        self.assertEqual(arena["combined_arena_bytes"],
                         arena["model_a_bytes"] + arena["model_b_bytes"])
        self.assertEqual(arena["both_models_open_bytes"],
                         arena["combined_arena_bytes"] + 2 * arena["task_buffer_bytes_per_model"])
        self.assertGreater(arena["combined_arena_bytes"], 0)
        self.assertGreater(arena["model_a_bytes"], 0)
        self.assertGreater(arena["model_b_bytes"], 0)

    def test_both_integer_references_reproduce_the_expected_bytes(self):
        for kind in ("a", "b"):
            with self.subTest(model=kind):
                computed = reference_for(self.out, kind)
                published = np.fromfile(self.out / ("expected_%s.i8" % kind), dtype=np.int8)
                self.assertEqual(computed.size, published.size)
                self.assertEqual(computed.astype(np.int8).tobytes(), published.tobytes(),
                                 "model %s reference bytes do not match" % kind)
                self.assertEqual(self.report["models"][kind]["check"], "int8 exact")

    def test_a_second_build_is_byte_identical(self):
        with tempfile.TemporaryDirectory(prefix="rknpu-multi-model-again-") as second:
            run_build(Path(second))
            for name in ARTIFACTS:
                with self.subTest(artifact=name):
                    self.assertEqual((self.out / name).read_bytes(),
                                     (Path(second) / name).read_bytes(),
                                     "%s differs between two builds" % name)

    @unittest.skipUnless(COMPILER, "no host C compiler (cc/gcc) found; board harness check skipped")
    def test_the_board_harness_compiles_against_the_runtime(self):
        """`board.c` uses only the public runtime API; the vendor cross-compile is the same file."""
        with tempfile.TemporaryDirectory(prefix="rknpu-multi-model-cc-") as tmp:
            binary = Path(tmp) / "board_multi"
            process = subprocess.run([COMPILER, *HOST_FLAGS, str(BOARD),
                                      str(ROOT / "runtime" / "open_rknpu.c"), "-o", str(binary)],
                                     cwd=str(ROOT), capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stderr[-2000:])
            self.assertTrue(binary.is_file())


if __name__ == "__main__":
    unittest.main()
