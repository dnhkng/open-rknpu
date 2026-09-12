"""SPDX-License-Identifier: MIT

The per-job timing API (`ornpu_run_timed` / `ornpu_run_io_timed`, checklist F9).

This board has no userspace cycle counter and the driver exposes none, so the supported way
to measure one inference is the runtime's own wall-clock breakdown: pack, submit, readback
and total. The ABI is part of the shipped header, so this module pins

* the header declares `struct ornpu_timing` with the four 64-bit counters and both entry
  points (a renamed field or a dropped declaration is a source-compatibility break);
* `tests/board_timed.c` compiles and links against the runtime with the host compiler
  (`make host-c` builds the same file for the board);
* on a host without `/dev/rknpu` the harness fails cleanly at `ornpu_open` (it never claims
  a timing for a run that did not happen);
* `ornpu_run` and `ornpu_run_io` remain the untimed wrappers, i.e. their prototypes did not
  change.

The measured numbers live in `docs/performance.md` (recorded from the board); a host test
cannot produce them.
"""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
HEADER = RUNTIME / "open_rknpu.h"
HARNESS = ROOT / "tests" / "board_timed.c"
RUNTIME_C = RUNTIME / "open_rknpu.c"
SEED = ROOT / "research" / "walk_chain_suite" / "model000.bin"

COMPILER = os.environ.get("OPEN_RKNPU_HOST_CC") or shutil.which("cc") or shutil.which("gcc")
DEFAULT_FLAGS = ["-O2", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-Iruntime"]
BUILD_FLAGS = shlex.split(os.environ.get("OPEN_RKNPU_HOST_CFLAGS", "")) or DEFAULT_FLAGS


class HeaderContractTests(unittest.TestCase):
    """The timing ABI is source-level: the header is the contract the user compiles against."""

    def setUp(self):
        self.header = HEADER.read_text()

    def test_the_timing_struct_has_the_four_counters(self):
        self.assertIn("struct ornpu_timing {", self.header)
        body = self.header.split("struct ornpu_timing {", 1)[1].split("};", 1)[0]
        declared = [part.strip() for part in body.strip().rstrip(";").split(";") if part.strip()]
        self.assertEqual(len(declared), 1, declared)
        line = declared[0]
        self.assertTrue(line.startswith("uint64_t "), line)
        fields = [name.strip() for name in line[len("uint64_t "):].split(",")]
        self.assertEqual(fields, ["pack_ns", "submit_ns", "readback_ns", "total_ns"])

    def test_both_timed_entry_points_are_declared(self):
        self.assertIn("int ornpu_run_timed(ornpu_model *model", self.header)
        self.assertIn("int ornpu_run_io_timed(ornpu_model *model", self.header)
        self.assertIn("struct ornpu_timing *timing);", self.header)

    def test_the_untimed_entry_points_are_unchanged(self):
        self.assertIn("int ornpu_run(ornpu_model *model, const uint8_t *input, size_t input_size,", self.header)
        self.assertIn("int ornpu_run_io(ornpu_model *model, const ornpu_io *inputs, uint32_t input_count,", self.header)

    def test_the_runtime_implements_the_timed_calls(self):
        source = RUNTIME_C.read_text()
        self.assertIn("int ornpu_run_timed(ornpu_model *model", source)
        self.assertIn("int ornpu_run_io_timed(ornpu_model *model", source)
        self.assertIn("return ornpu_run_io_timed(model,&io_in,1,&io_out,1,timing);", source)
        # The untimed wrappers pass NULL, so a caller that does not ask for timing pays
        # nothing but the two branches.
        self.assertIn("return ornpu_run_timed(model,input,input_size,output,output_size,NULL);", source)
        self.assertIn("return ornpu_run_io_timed(model,inputs,input_count,outputs,output_count,NULL);", source)


@unittest.skipUnless(COMPILER, "no host C compiler (cc/gcc) found; timing tests skipped")
class HarnessBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="open-rknpu-timing-")
        cls.binary = Path(cls.tmp.name) / "board_timed"
        build = subprocess.run(
            [COMPILER, *BUILD_FLAGS, str(HARNESS.relative_to(ROOT)),
             str(RUNTIME_C.relative_to(ROOT)), "-o", str(cls.binary)],
            cwd=ROOT, capture_output=True, text=True, timeout=180)
        cls.build = build

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_harness_compiles_and_links_against_the_runtime(self):
        self.assertEqual(self.build.returncode, 0,
                         f"board_timed.c failed to build:\n{self.build.stdout}\n{self.build.stderr}")

    def test_without_a_device_it_fails_before_claiming_a_timing(self):
        if Path("/dev/rknpu").exists():
            self.skipTest("this host has /dev/rknpu; the timing path is the board's")
        result = subprocess.run([str(self.binary), str(SEED), str(SEED), str(SEED), "1"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("open failed", result.stderr)
        self.assertNotIn("SUMMARY", result.stdout, "no timing may be reported for a failed run")

    def test_usage_is_checked(self):
        result = subprocess.run([str(self.binary)], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
