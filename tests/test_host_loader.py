"""SPDX-License-Identifier: MIT

Host C/ABI tests for the runtime container loader (checklist T3).

`runtime/open_rknpu.c` validates a container *before* it opens `/dev/rknpu`, so its
whole rejection surface runs on any x86 host with the system compiler. This module
builds `tests/host_loader.c` against the runtime and drives it over a fixture set:
the two real containers `research/walk_chain_suite/model000.bin` (v5 named-tensor
table) and `research/mnist_first_suite/model000.bin` (legacy `ORNPUBIN` v2), plus
malformed files crafted by copying a real container and editing header, task and
tensor fields (the checksum is recomputed when the mutation must reach a semantic
validator rather than the checksum guard).

It pins, from the C side of the ABI:

* `ornpu_inspect` accepts a valid v5 and a valid legacy container and reports the
  geometry, tensor/task counts and quantization bands the Python decoder reports;
* every malformed container is rejected with the loader's documented `-EINVAL`
  (a missing file with `-ENOENT`);
* `ornpu_open` of a valid container reaches the device open and fails on the host
  with `-ENOENT`/`-ENODEV` (there is no `/dev/rknpu`), returns no partial model and
  leaks no file descriptor.

The module is deterministic, reads no network or board, and skips cleanly when no
host C compiler is installed.
"""
import errno
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

from open_rknpu.model import checksum, decode as decode_model

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
HOST_LOADER_C = ROOT / "tests" / "host_loader.c"
RUNTIME_C = RUNTIME / "open_rknpu.c"
V5_SEED = ROOT / "research" / "walk_chain_suite" / "model000.bin"
LEGACY_SEED = ROOT / "research" / "mnist_first_suite" / "model000.bin"
V4_SEED = ROOT / "research" / "mutable_weights_suite" / "model000.bin"
SEEDS = (V5_SEED, LEGACY_SEED, V4_SEED)

# The flags are overridable so CI can run the very same fixtures under ASan/UBSan:
#   OPEN_RKNPU_HOST_CFLAGS='-O1 -g ... -fsanitize=address,undefined'
COMPILER = os.environ.get("OPEN_RKNPU_HOST_CC") or shutil.which("cc") or shutil.which("gcc")
DEFAULT_FLAGS = ["-O2", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-Iruntime"]
BUILD_FLAGS = shlex.split(os.environ.get("OPEN_RKNPU_HOST_CFLAGS", "")) or DEFAULT_FLAGS

# Sequence header is "<8s22I": field index i sits at byte 8 + 4*i.
SEQ_FIELD = {
    "version": 8, "header_size": 12, "height": 16, "width": 20, "input_channels": 24,
    "output_height": 28, "output_width": 32, "output_channels": 36, "stride": 40,
    "payload_size": 44, "arena": 48, "input_offset": 52, "output_offset": 56,
    "task_count": 60, "flags": 84, "layout": 88, "batch_minus_one": 92,
}
LEGACY_FIELD = {
    "version": 8, "header_size": 12, "target": 16, "profile": 20, "kernel": 68,
}
TENSOR_SIZE = 64
TENSOR_FIELD = {"role": 24, "layout": 28, "index": 32, "offset": 52, "size": 56}
V5_EXTENSION = 96
V5_TASK_TABLE = 112

Case = namedtuple("Case", "name group op expected path checks note")

# The expected ornpu_info for the two real containers, pinned independently of the
# C loader (the container headers are re-decoded by the seed test below).
VALID_V5_CHECKS = {
    "height": 8, "width": 8, "input_channels": 3, "output_channels": 3,
    "input_bytes": 192, "output_bytes": 48, "output_height": 4, "output_width": 4,
    "batch": 1, "task_count": 3, "tensor_count": 4, "constant_count": 0,
    "input_tensor_count": 1, "output_tensor_count": 1, "submission_serial": 1,
    "output_zero_point": 0, "input_zero_point": 0,
    "output_scale": repr(538.692138671875), "input_scale": repr(1.0),
}
VALID_LEGACY_CHECKS = {
    "height": 28, "width": 28, "input_channels": 1, "output_channels": 8,
    "input_bytes": 784, "output_bytes": 6272, "output_height": 28, "output_width": 28,
    "batch": 1, "task_count": 1, "tensor_count": 0, "constant_count": 0,
    "input_tensor_count": 1, "output_tensor_count": 0, "submission_serial": 0,
    "output_zero_point": -128, "input_zero_point": 135,
    "output_scale": repr(1.2440464496612549), "input_scale": repr(0.2710929811000824),
}
MALFORMED = "-EINVAL"


def _patch32(data, offset, value):
    out = bytearray(data)
    struct.pack_into("<I", out, offset, value & 0xFFFFFFFF)
    return bytes(out)


def _repair(data):
    """Recompute the FNV-1a checksum so the next guard, not the checksum, rejects."""
    offset = 80 if data[:8] == b"ORNPUSEQ" else 84
    out = bytearray(data)
    out[offset:offset + 4] = b"\x00" * 4
    struct.pack_into("<I", out, offset, checksum(out))
    return bytes(out)


def _flip_checksum(data):
    """Corrupt the final payload byte without repairing, so the checksum guard fires."""
    out = bytearray(data)
    out[-1] ^= 0xFF
    return bytes(out)


def _append_random(data):
    return data + b"\x00"


def _v5_task_table(data):
    count = struct.unpack_from("<I", data, SEQ_FIELD["task_count"])[0]
    return V5_TASK_TABLE, count


def _v5_tensor_table(data):
    tensor_count = struct.unpack_from("<I", data, V5_EXTENSION)[0]
    _, task_count = _v5_task_table(data)
    start = V5_TASK_TABLE + 16 * task_count
    return [start + i * TENSOR_SIZE for i in range(tensor_count)]


def _format_checks(checks):
    return ",".join(f"{key}={value}" for key, value in checks.items())


def _spec(case):
    fields = [case.name, case.op, case.expected, str(case.path)]
    if case.checks:
        fields.append(_format_checks(case.checks))
    return "|".join(fields)


def build_cases(fixtures):
    """Write every crafted container below `fixtures` and return the case list."""
    v5 = V5_SEED.read_bytes()
    legacy = LEGACY_SEED.read_bytes()
    v4 = V4_SEED.read_bytes()
    v5_tensors = _v5_tensor_table(v5)
    cases = []

    def malformed(name, data, note):
        path = fixtures / f"{name}.bin"
        path.write_bytes(data)
        cases.append(Case(name, "malformed", "inspect", MALFORMED, path, None, note))

    cases.append(Case("valid_v5", "valid", "inspect", "0", V5_SEED, VALID_V5_CHECKS,
                      "research/walk_chain_suite/model000.bin (ORNPUSEQ v5)"))
    cases.append(Case("valid_legacy", "valid", "inspect", "0", LEGACY_SEED, VALID_LEGACY_CHECKS,
                      "research/mnist_first_suite/model000.bin (ORNPUBIN v2)"))

    # Legacy ORNPUBIN: header fields, truncation and checksum.
    malformed("legacy_bad_magic", b"ORNPUBIX" + legacy[8:], "magic is ORNPUBIX")
    malformed("legacy_truncated_header", legacy[:40], "file stops inside the 96-byte header")
    malformed("legacy_bad_header_size", _repair(_patch32(legacy, LEGACY_FIELD["header_size"], 95)),
              "header_size=95")
    malformed("legacy_bad_version", _repair(_patch32(legacy, LEGACY_FIELD["version"], 3)),
              "version=3")
    malformed("legacy_payload_too_short", legacy[:-1], "last payload byte removed")
    malformed("legacy_payload_too_long", _append_random(legacy), "one trailing byte")
    malformed("legacy_checksum_mismatch", _flip_checksum(legacy),
              "payload byte flipped, stored checksum stale")
    malformed("legacy_bad_target", _repair(_patch32(legacy, LEGACY_FIELD["target"], 1126)),
              "target=1126")
    malformed("legacy_bad_kernel", _repair(_patch32(legacy, LEGACY_FIELD["kernel"], 2)),
              "kernel=2")
    malformed("legacy_bad_profile", _repair(_patch32(legacy, LEGACY_FIELD["profile"], 9)),
              "profile=9")
    malformed("legacy_bad_output_scale",
              _repair(_patch32(legacy, 80, struct.unpack("<I", struct.pack("<f", 0.0))[0])),
              "output_scale=0")
    malformed("legacy_bad_output_zero_point", _repair(_patch32(legacy, 76, 200)),
              "output_zero_point=200")
    malformed("legacy_bad_input_zero_point", _repair(_patch32(legacy, 92, 256)),
              "v2 reserved[1] (input_zero_point)=256")

    missing = fixtures / "does_not_exist.bin"
    cases.append(Case("missing_file", "missing", "inspect", "-ENOENT", missing, None,
                      "path does not exist"))

    # ORNPUSEQ v3/v4 (task table, no tensor table).
    malformed("seq_task_count_zero", _repair(_patch32(v4, SEQ_FIELD["task_count"], 0)),
              "task_count=0")
    malformed("seq_task_count_absurd", _repair(_patch32(v4, SEQ_FIELD["task_count"], 65)),
              "task_count=65 (> MAX_TASKS)")
    malformed("seq_task_offset_past_payload", _repair(_patch32(v4, 96, 0xFFFFFF00)),
              "task0 command offset=0xFFFFFF00 past payload")
    malformed("seq_task_bad_amount", _repair(_patch32(v4, 96 + 4, 0)), "task0 amount=0")
    malformed("seq_task_offset_unaligned", _repair(_patch32(v4, 96, 4)),
              "task0 command offset=4, not 8-byte aligned")
    malformed("seq_task_amount_over_max", _repair(_patch32(v4, 96 + 4, 1107)),
              "task0 amount=1107 (> MAX)")
    malformed("seq_bad_stride", _repair(_patch32(v4, SEQ_FIELD["stride"], 999)), "stride=999")
    malformed("seq_bad_header_size", _repair(_patch32(v4, SEQ_FIELD["header_size"], 95)),
              "header_size=95")
    malformed("seq_bad_version", _repair(_patch32(v4, SEQ_FIELD["version"], 6)), "version=6")
    malformed("seq_payload_too_short", v4[:-1], "last payload byte removed")
    malformed("seq_payload_too_long", _append_random(v4), "one trailing byte")
    malformed("seq_checksum_mismatch", _flip_checksum(v4),
              "payload byte flipped, stored checksum stale")

    # ORNPUSEQ v5 (named tensor table).
    constant_count = _patch32(v5, SEQ_FIELD["flags"],
                              struct.unpack_from("<I", v5, SEQ_FIELD["flags"])[0] | 0x100)
    malformed("v5_constant_count_nonzero", _repair(constant_count),
              "v5 flags high byte=1 (constant descriptors)")
    malformed("v5_task_count_zero", _repair(_patch32(v5, SEQ_FIELD["task_count"], 0)),
              "task_count=0")
    malformed("v5_task_offset_past_payload", _repair(_patch32(v5, V5_TASK_TABLE, 0xFFFFFF00)),
              "task0 command offset=0xFFFFFF00 past payload")
    malformed("v5_task_offset_unaligned", _repair(_patch32(v5, V5_TASK_TABLE, 4)),
              "task0 command offset=4, not 8-byte aligned")
    malformed("v5_bad_header_size", _repair(_patch32(v5, SEQ_FIELD["header_size"], 8)),
              "header_size=8 while every field is read at its fixed v5 offset")
    malformed("v5_tensor_role_violation",
              _repair(_patch32(v5, v5_tensors[0] + TENSOR_FIELD["role"], 3)),
              "tensor0 role=3 (no such role)")
    malformed("v5_tensor_layout_violation",
              _repair(_patch32(v5, v5_tensors[0] + TENSOR_FIELD["layout"], 3)),
              "tensor0 layout=3 (no such layout)")
    malformed("v5_tensor_index_violation",
              _repair(_patch32(v5, v5_tensors[0] + TENSOR_FIELD["index"], 5)),
              "primary input index=5, so index 0 is missing")
    original_size = struct.unpack_from("<I", v5, v5_tensors[0] + TENSOR_FIELD["size"])[0]
    malformed("v5_tensor_size_violation",
              _repair(_patch32(v5, v5_tensors[0] + TENSOR_FIELD["size"], original_size + 64)),
              f"tensor0 size={original_size + 64}, not the layout's {original_size}")
    duplicate = bytearray(v5)
    duplicate[v5_tensors[1]:v5_tensors[1] + 24] = v5[v5_tensors[0]:v5_tensors[0] + 24]
    malformed("v5_tensor_duplicate_name", _repair(bytes(duplicate)),
              "tensor1 renames to tensor0's name")
    malformed("v5_arena_smaller_than_tensor", _repair(_patch32(v5, SEQ_FIELD["arena"], 4096)),
              "arena=4096 < primary input offset+size")
    malformed("v5_external_tensors_overlap",
              _repair(_patch32(v5, v5_tensors[3] + TENSOR_FIELD["offset"],
                               struct.unpack_from("<I", v5, v5_tensors[0] + TENSOR_FIELD["offset"])[0])),
              "output tensor moved onto the input tensor")
    malformed("v5_internal_tensor_overlaps_external",
              _repair(_patch32(v5, v5_tensors[1] + TENSOR_FIELD["offset"],
                               struct.unpack_from("<I", v5, v5_tensors[0] + TENSOR_FIELD["offset"])[0] + 128)),
              "internal conv0 tensor moved into the input tensor")
    malformed("v5_output_count_mismatch", _repair(_patch32(v5, V5_EXTENSION + 12, 2)),
              "extension output_count=2 but only one output tensor")
    malformed("v5_bad_extension_tensor_size", _repair(_patch32(v5, V5_EXTENSION + 4, 63)),
              "extension tensor_size=63")
    malformed("v5_payload_too_short", v5[:-1], "last payload byte removed")
    malformed("v5_payload_too_long", _append_random(v5), "one trailing byte")
    malformed("v5_checksum_mismatch", _flip_checksum(v5),
              "payload byte flipped, stored checksum stale")

    # ornpu_open: a valid file stops at the missing device, a malformed one earlier.
    cases.append(Case("open_valid_v5", "open", "open", "-ENOENT,-ENODEV", V5_SEED, None,
                      "valid v5, host has no /dev/rknpu"))
    cases.append(Case("open_valid_legacy", "open", "open", "-ENOENT,-ENODEV", LEGACY_SEED, None,
                      "valid legacy, host has no /dev/rknpu"))
    cases.append(Case("open_malformed_v5_constant_count", "open", "open", MALFORMED,
                      fixtures / "v5_constant_count_nonzero.bin", None,
                      "malformed v5 must fail before the device open"))
    return cases


LINE = re.compile(r"^(ok|FAIL)\s+(\S+)\s+rc=(-?\d+)")


def parse_results(text):
    results = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if match:
            results[match.group(2)] = (match.group(1), int(match.group(3)))
    return results


@unittest.skipUnless(COMPILER, "no host C compiler (cc/gcc) found; host loader tests skipped")
@unittest.skipUnless(all(path.is_file() for path in SEEDS),
                     "research seed containers are missing; host loader tests skipped")
class HostLoaderTests(unittest.TestCase):
    """Build `tests/host_loader.c` once and assert every container outcome."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="open-rknpu-host-loader-")
        tmpdir = Path(cls.tmp.name)
        cls.binary = tmpdir / "host_loader"
        build = subprocess.run(
            [COMPILER, *BUILD_FLAGS, str(HOST_LOADER_C.relative_to(ROOT)),
             str(RUNTIME_C.relative_to(ROOT)), "-o", str(cls.binary)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        if build.returncode != 0:
            raise AssertionError(
                f"host_loader.c failed to build with {COMPILER}:\n{build.stdout}\n{build.stderr}")
        fixtures = tmpdir / "fixtures"
        fixtures.mkdir()
        cls.cases = build_cases(fixtures)
        run = subprocess.run([str(cls.binary), *(_spec(case) for case in cls.cases)],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
        cls.returncode = run.returncode
        cls.output = run.stdout + run.stderr
        cls.results = parse_results(run.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_seed_containers_are_the_pinned_valid_ones(self):
        v5 = decode_model(V5_SEED.read_bytes())
        self.assertEqual(v5["shape_nhwc"], [1, 8, 8, 3])
        self.assertEqual(v5["task_count"], 3)
        self.assertEqual(v5["tensor_count"], 4)
        legacy = decode_model(LEGACY_SEED.read_bytes())
        self.assertEqual(legacy["shape_nhwc"], [1, 28, 28, 1])
        self.assertEqual(legacy["task_count"], 1)

    def test_binary_exits_zero(self):
        self.assertEqual(self.returncode, 0, self.output)

    def test_every_case_reports_ok(self):
        self.assertEqual(set(self.results), {case.name for case in self.cases}, self.output)
        for case in self.cases:
            status, _ = self.results[case.name]
            self.assertEqual(status, "ok", f"{case.name}: {self.output}")

    def test_valid_containers_report_the_expected_info(self):
        cases = [case for case in self.cases if case.group == "valid"]
        self.assertEqual([case.name for case in cases], ["valid_v5", "valid_legacy"])
        for case in cases:
            status, rc = self.results[case.name]
            self.assertEqual((status, rc), ("ok", 0), f"{case.name}: {self.output}")

    def test_malformed_containers_are_rejected_with_einval(self):
        cases = [case for case in self.cases if case.group == "malformed"]
        self.assertGreaterEqual(len(cases), 40)
        for case in cases:
            status, rc = self.results[case.name]
            self.assertEqual(status, "ok", f"{case.name}: {self.output}")
            self.assertEqual(rc, -errno.EINVAL, f"{case.name} ({case.note}) returned {rc}: {self.output}")

    def test_missing_file_reports_enoent(self):
        status, rc = self.results["missing_file"]
        self.assertEqual((status, rc), ("ok", -errno.ENOENT), self.output)

    def test_open_reports_the_missing_device_on_the_host(self):
        for name in ("open_valid_v5", "open_valid_legacy"):
            status, rc = self.results[name]
            self.assertEqual(status, "ok", f"{name}: {self.output}")
            self.assertIn(rc, (-errno.ENOENT, -errno.ENODEV),
                          f"{name} returned {rc}, not -ENOENT/-ENODEV")

    def test_actual_return_codes_are_reported(self):
        if "-v" in sys.argv or "--verbose" in sys.argv:
            print("\nhost_loader container outcomes:")
            for case in self.cases:
                status, rc = self.results[case.name]
                print(f"  {status:4s} {case.name:36s} rc={rc:>4d}  {case.note}")
        summary = [line for line in self.output.splitlines() if line.startswith("SUMMARY")]
        self.assertEqual(summary, [f"SUMMARY total={len(self.cases)} failed=0"], self.output)
