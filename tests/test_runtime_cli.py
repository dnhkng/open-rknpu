"""SPDX-License-Identifier: MIT

The board runner's command line (`runtime/main.c`, checklist T13).

`open-rknpu-run` is the program a user copies to the board, and until now nothing tested
it: the host loader tests cover the container *loader*, not the runner's argument handling,
its host-runnable `--inspect` path, or the input validation and buffer limits it applies
before it ever opens `/dev/rknpu`. This module builds the runner with the system compiler
and drives it as a user would:

* `--inspect` on every published container (2,340 of them across `research/*_suite/`) must
  exit 0 and print *exactly* the geometry, byte counts and quantization numbers the Python
  decoder reports - a cross-implementation check of the C header reader against `model.decode`;
* a missing or malformed container must exit 1 with a `strerror` message on stderr;
* a wrong argument count must print the two-line usage and exit 2;
* a model whose input exceeds the runner's fixed buffers must be refused before any device
  access, and an input file of the wrong length must be refused with the required size;
* the run path on a host without `/dev/rknpu` must fail at `open model`, never write an
  output file and never claim success.

The runner is deterministic, reads only the files named on its command line and skips
cleanly when no host C compiler is installed.
"""
from pathlib import Path
import os
import shlex
import shutil
import struct
import subprocess
import tempfile
import unittest

from open_rknpu.model import checksum, decode as decode_model
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
MAIN_C = RUNTIME / "main.c"
RUNTIME_C = RUNTIME / "open_rknpu.c"
V5_SEED = ROOT / "research" / "walk_chain_suite" / "model000.bin"
LEGACY_SEED = ROOT / "research" / "mnist_first_suite" / "model000.bin"
OVERSIZED_SEED = ROOT / "research" / "native_spatial_tiling_suite" / "model004.bin"
USAGE = ("usage: open-rknpu-run MODEL.bin INPUT.u8 OUTPUT.i8\n"
         "       open-rknpu-run --inspect MODEL.bin\n")
# The runner's fixed stack buffers, from runtime/main.c.
MAX_INPUT_BYTES = 32 * 32 * 32
MAX_OUTPUT_BYTES = 32 * 32 * 64

COMPILER = os.environ.get("OPEN_RKNPU_HOST_CC") or shutil.which("cc") or shutil.which("gcc")
DEFAULT_FLAGS = ["-O2", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-Iruntime"]
BUILD_FLAGS = shlex.split(os.environ.get("OPEN_RKNPU_HOST_CFLAGS", "")) or DEFAULT_FLAGS

# Sequence header is "<8s22I": field index i sits at byte 8 + 4*i. The stored FNV-1a
# checksum covers the container with its own four bytes zeroed (byte 80 for a sequence,
# 84 for a legacy container) - the same layout tests/test_host_loader.py mutates.
SEQ_FIELD = {"version": 8, "header_size": 12, "payload_size": 44, "task_count": 60}
LEGACY_FIELD = {"version": 8, "header_size": 12, "target": 16, "profile": 20}


def expected_inspect(info):
    """The exact stdout `--inspect` must produce for a decoded container."""
    batch, height, width, channels = info["shape_nhwc"]
    out_batch, out_height, out_width, out_channels = info["output_shape_nhwc"]
    return ("input: UINT8 NHWC [%d,%d,%d,%d], %d bytes; scale=%.9g zero_point=%d\n"
            "output: INT8 NHWC [%d,%d,%d,%d], %d bytes; scale=%.9g zero_point=%d\n") % (
        batch, height, width, channels, info["input_bytes"],
        info["input_scale"], info["input_zero_point"],
        out_batch, out_height, out_width, out_channels, info["output_bytes"],
        info["output_scale"], info["output_zero_point"])


def patched(source, field, value):
    """A copy of a container with a header field replaced and the checksum repaired."""
    data = bytearray(source.read_bytes())
    layout = SEQ_FIELD if data[:8] == b"ORNPUSEQ" else LEGACY_FIELD
    struct.pack_into("<I", data, layout[field], value & 0xFFFFFFFF)
    offset = 80 if data[:8] == b"ORNPUSEQ" else 84
    data[offset:offset + 4] = b"\x00" * 4
    struct.pack_into("<I", data, offset, checksum(bytes(data)))
    return bytes(data)


@unittest.skipUnless(COMPILER, "no host C compiler (cc/gcc) found; runtime CLI tests skipped")
@unittest.skipUnless(all(path.is_file() for path in (V5_SEED, LEGACY_SEED, OVERSIZED_SEED)),
                     "research seed containers are missing; runtime CLI tests skipped")
class RuntimeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="open-rknpu-runtime-cli-")
        tmpdir = Path(cls.tmp.name)
        cls.binary = tmpdir / "open-rknpu-run"
        build = subprocess.run(
            [COMPILER, *BUILD_FLAGS, str(MAIN_C.relative_to(ROOT)),
             str(RUNTIME_C.relative_to(ROOT)), "-o", str(cls.binary)],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        if build.returncode != 0:
            raise AssertionError(
                f"runtime/main.c failed to build with {COMPILER}:\n{build.stdout}\n{build.stderr}")
        cls.containers = sorted((ROOT / "research").glob("*_suite/model*.bin"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([str(self.binary), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_inspect_of_a_v5_container_prints_the_decoded_numbers(self):
        for seed in (V5_SEED, LEGACY_SEED):
            with self.subTest(container=seed.name):
                info = decode_sequence(seed.read_bytes()) if seed == V5_SEED \
                    else decode_model(seed.read_bytes())
                result = self.run_cli("--inspect", seed)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected_inspect(info))
                self.assertEqual(len(result.stdout.splitlines()), 2)
                self.assertEqual(result.stderr, "")

    def test_inspect_of_every_published_container_matches_python(self):
        self.assertGreaterEqual(len(self.containers), 2300, "the container tree shrank")
        mismatches, failures = [], []
        for container in self.containers:
            data = container.read_bytes()
            info = decode_sequence(data) if data[:8] == b"ORNPUSEQ" else decode_model(data)
            result = self.run_cli("--inspect", container)
            if result.returncode != 0:
                failures.append(f"{container.relative_to(ROOT)}: rc={result.returncode} "
                                f"{result.stderr.strip()}")
            elif result.stdout != expected_inspect(info):
                mismatches.append(f"{container.relative_to(ROOT)}: {result.stdout!r} != "
                                  f"{expected_inspect(info)!r}")
        self.assertEqual(failures, [], "\n".join(failures[:10]))
        self.assertEqual(mismatches, [], "\n".join(mismatches[:10]))

    def test_usage_and_exit_codes(self):
        cases = [(), ("one-arg",), ("--inspect",), ("--unknown", "model.bin")]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stderr, USAGE)

    def test_missing_container_is_reported(self):
        result = self.run_cli("--inspect", self.tmp.name + "/absent.bin")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertRegex(result.stderr, r"^inspect: (No such file or directory|Invalid argument)")

    def test_malformed_containers_are_rejected(self):
        good = V5_SEED.read_bytes()
        crafted = {
            "bad magic": b"NOPE" + good[4:],
            "bad version": patched(V5_SEED, "version", 99),
            "truncated": good[:64],
            "zero tasks": patched(V5_SEED, "task_count", 0),
            "bad header size": patched(V5_SEED, "header_size", 8),
            "zero payload": patched(V5_SEED, "payload_size", 0),
        }
        for name, data in crafted.items():
            with self.subTest(case=name):
                path = Path(self.tmp.name) / f"{name.replace(' ', '_')}.bin"
                path.write_bytes(data)
                result = self.run_cli("--inspect", path)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("inspect:", result.stderr)
                self.assertNotIn("scale=", result.stdout)

    def test_input_file_must_be_exactly_the_container_input_size(self):
        info = decode_model(V5_SEED.read_bytes())
        for size in (0, info["input_bytes"] - 1, info["input_bytes"] + 1):
            with self.subTest(size=size):
                path = Path(self.tmp.name) / f"input-{size}.u8"
                path.write_bytes(b"\x00" * max(size, 0))
                result = self.run_cli(V5_SEED, path, Path(self.tmp.name) / "out.i8")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertEqual(result.stderr,
                                 f"input must contain exactly {info['input_bytes']} bytes\n")

    def test_oversized_model_is_refused_before_the_device(self):
        info = decode_model(OVERSIZED_SEED.read_bytes())
        self.assertGreater(info["input_bytes"], MAX_INPUT_BYTES)
        path = Path(self.tmp.name) / "big.u8"
        path.write_bytes(b"\x00" * 16)
        result = self.run_cli(OVERSIZED_SEED, path, Path(self.tmp.name) / "big.i8")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "model exceeds runner buffer limits\n")

    def test_run_path_fails_at_the_device_open_and_writes_nothing(self):
        if Path("/dev/rknpu").exists():
            self.skipTest("this host has /dev/rknpu; the device path is not a failure here")
        info = decode_model(V5_SEED.read_bytes())
        source = ROOT / "research" / "walk_chain_suite" / "input000.u8"
        self.assertGreaterEqual(source.stat().st_size, info["input_bytes"])
        input_path = Path(self.tmp.name) / "ok.u8"
        input_path.write_bytes(source.read_bytes()[:info["input_bytes"]])
        output_path = Path(self.tmp.name) / "out.i8"
        result = self.run_cli(V5_SEED, input_path, output_path)
        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stderr, r"^open model: ")
        self.assertEqual(result.stdout, "")
        self.assertFalse(output_path.exists(), "the runner wrote an output without running")


if __name__ == "__main__":
    unittest.main()
