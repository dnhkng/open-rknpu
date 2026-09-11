"""MIT. Retained evidence: a compact EW operand is read linearly, not broadcast.

The ERDMA notch registers (`0x506c` `ew_surf_notch`, `0x5010` `ew_line_notch_addr`) are
decoded from the RK3588 TRM for the same NPU IP. Setting the notch turns the retained
compact-operand hang into a completed run, and the run is *byte for byte the linear
read* of the compact table - recomputed here from the broadcast suite's own reference -
so no native spatial broadcast exists.
"""
from pathlib import Path
import json
import unittest

import numpy as np
import onnx
from onnx import numpy_helper as nh

from open_rknpu.elementwise import mul_reference
from open_rknpu.quantization import Quantization, reference

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "mul_broadcast_notch_suite"
SOURCE = ROOT / "research" / "mul_broadcast_suite"
HEIGHT, WIDTH, CHANNELS = 7, 5, 3
COMPACT = ("compact_stride_plane_notch_w-1", "compact_stride_1_notch_w-1",
           "compact_stride_plane_notch_w", "compact_mode1_surf1_notch_w-1")


def linear_read_prediction():
    """What the board must produce if the compact operand is read linearly."""
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    meta = manifest[2]
    quantization = Quantization(**{key: np.array(value) if isinstance(value, list) else value
                                   for key, value in meta["branches"][0]["quantization"].items()})
    factor = None
    for tensor in onnx.load(str(SOURCE / "model002.onnx")).graph.initializer:
        if tensor.name == "factor":
            factor = nh.to_array(tensor)
    operand = np.clip(np.rint(np.broadcast_to(factor, (1, CHANNELS, HEIGHT, WIDTH))
                              / meta["constant_scale"]), -128, 127).astype(np.int8)
    operand = operand[0].transpose(1, 2, 0)
    compact = np.zeros((HEIGHT, 16), np.int8)
    for row in range(HEIGHT):
        compact[row, :CHANNELS] = operand[row, 0, :]
    linear = np.zeros((HEIGHT, WIDTH, CHANNELS), np.int8)
    for pixel in range(HEIGHT * WIDTH):
        row, column = divmod(pixel, WIDTH)
        if pixel < HEIGHT:
            linear[row, column] = compact[pixel, :CHANNELS]
    cases = np.fromfile(SOURCE / "input002.u8", np.uint8).reshape(-1, HEIGHT, WIDTH, CHANNELS)
    return np.stack([mul_reference(reference(case, quantization), linear) for case in cases])


class BroadcastNotchTests(unittest.TestCase):
    def results(self):
        return json.loads((SUITE / "notch_results.json").read_text())

    def outputs(self, name):
        index = self.results()[name]["index"]
        return np.frombuffer((SUITE / f"output{index:03d}.i8").read_bytes(),
                             np.int8).reshape(-1, HEIGHT, WIDTH, CHANNELS)

    def test_baseline_is_exact_and_the_notch_removes_the_hang(self):
        results = self.results()
        self.assertTrue(results["baseline_materialized"]["passed"])
        self.assertTrue(results["baseline_materialized"]["exact"])
        for name in COMPACT:
            with self.subTest(variant=name):
                record = results[name]
                self.assertTrue(record["passed"])
                self.assertFalse(record["exact"])
                self.assertEqual(record["differing_bytes"], 3025)
        hung = results["compact_mode2_notch_w-1"]
        self.assertFalse(hung["passed"])
        self.assertIn("runs=0 rc=1", hung["output"])
        self.assertIn("job timeout", hung["dmesg"])
        self.assertIn("soft reset", hung["dmesg"])

    def test_compact_operands_are_read_linearly_not_broadcast(self):
        prediction = linear_read_prediction()
        expected = np.frombuffer((SOURCE / "expected002.i8").read_bytes(),
                                 np.int8).reshape(prediction.shape)
        self.assertFalse(np.array_equal(prediction, expected))
        for name in COMPACT:
            with self.subTest(variant=name):
                self.assertTrue(np.array_equal(self.outputs(name), prediction))

    def test_variants_carry_the_decoded_registers(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        by_name = {entry["variant"]: entry for entry in manifest}
        self.assertEqual(by_name["baseline_materialized"]["0x506c"], "0x0")
        self.assertEqual(by_name["compact_stride_plane_notch_w-1"]["0x506c"], "0x40")
        self.assertEqual(by_name["compact_stride_1_notch_w-1"]["0x5040"], "0x10")
        self.assertEqual(by_name["compact_mode2_notch_w-1"]["0x5034"], "0x80000004")
        self.assertEqual(by_name["compact_mode1_surf1_notch_w-1"]["0x5034"], "0x60000004")
        self.assertTrue((SUITE / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
