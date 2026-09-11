"""MIT. Retained evidence: per-channel elementwise output conversion hangs the EW path.

Pins the corrected P4 probe: the table address/format was fixed, `OW_SRC=1` still
hangs, and with `OD_BYPASS=1` the same table is ignored (output byte-identical to the
baseline). The builder is deterministic, so the variants are rebuilt and compared with
the retained containers and the retained board records.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import unittest

from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "mul_per_channel_ow_suite"
BUILDER = ROOT / "research" / "check_mul_per_channel_ow.py"
TABLE = 0x1800


def load_builder():
    spec = importlib.util.spec_from_file_location("check_mul_per_channel_ow", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ew_registers(data):
    data = bytes(data)
    info = decode_sequence(data)
    task = info["tasks"][-1]
    start = 96 + 16 * info["task_count"]
    registers = {}
    for i in range(task["register_count"]):
        word = struct.unpack_from("<Q", data, start + task["command_offset"] + i * 8)[0]
        registers[word & 0xFFFF] = (word >> 16) & 0xFFFFFFFF
    return registers, start


class PerChannelOutputConversionTests(unittest.TestCase):
    def test_builder_reproduces_the_retained_variants(self):
        before = {name: hashlib.sha256((SUITE / f"{name}.bin").read_bytes()).hexdigest()
                  for name in ("model000", "model001", "model002", "model003", "model004", "model005")}
        subprocess.run([sys.executable, str(BUILDER)], check=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        for name, digest in before.items():
            with self.subTest(variant=name):
                self.assertEqual(
                    hashlib.sha256((SUITE / f"{name}.bin").read_bytes()).hexdigest(), digest)

    def test_variants_carry_the_designed_registers_and_table(self):
        # name -> (EW 0x4050, where the table lives)
        expected = {"model000": (0x30000002, "none"), "model001": (0x30000001, "file"),
                    "model002": (0x30000001, "payload"), "model003": (0x30000001, "payload"),
                    "model004": (0x30000003, "payload"), "model005": (0x30000002, "payload")}
        for name, (ow_cfg, where) in expected.items():
            with self.subTest(variant=name):
                data = (SUITE / f"{name}.bin").read_bytes()
                registers, start = ew_registers(data)
                self.assertEqual(registers[0x4050], ow_cfg)
                # Bias and weight-zero-point lanes of the decoded block stay zero.
                self.assertEqual(data[start + TABLE:start + TABLE + 24], bytes(24))
                if where == "none":
                    self.assertEqual(registers[0x5020], 0)
                    self.assertEqual(struct.unpack_from("<4H", data, start + TABLE + 24), (0, 0, 0, 0))
                elif where == "payload":
                    self.assertEqual(registers[0x5020], TABLE)
                    self.assertEqual(struct.unpack_from("<4H", data, start + TABLE + 24)[0], 16384)
                else:
                    # The superseded variant wrote its 4xUINT16 table at the file offset,
                    # 0x80 bytes past the payload address 0x5020 names.
                    self.assertEqual(registers[0x5020], TABLE)
                    self.assertEqual(struct.unpack_from("<4H", data, TABLE + 24), (16384, 0, 0, 0))
                    self.assertEqual(struct.unpack_from("<4H", data, start + TABLE + 24), (0, 0, 0, 0))

    def test_retained_board_records_show_the_isolated_field(self):
        results = json.loads((SUITE / "ow_results.json").read_text())
        self.assertEqual(sorted(results), ["model000", "model001", "model002", "model003",
                                           "model004", "model005"])
        for name in ("model000", "model004", "model005"):
            with self.subTest(variant=name):
                record = results[name]
                self.assertTrue(record["passed"])
                self.assertTrue(record["exact"])
                self.assertEqual(record["bytes"], 4608)
        for name in ("model001", "model002", "model003"):
            with self.subTest(variant=name):
                record = results[name]
                self.assertFalse(record["passed"])
                self.assertIn("runs=0 rc=1", record["output"])
                self.assertIn("job timeout", record["dmesg"])
                self.assertIn("soft reset", record["dmesg"])
        summary = (SUITE / "ow_summary.txt").read_text()
        self.assertIn("c1 1536/1536 same", summary)
        self.assertTrue((SUITE / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
