"""MIT. Retained evidence for the LUT index probe (P6 blocker).

The probe in `research/lut_index_probe/` measured the hardware table index directly
with a sign-aware output-code ramp. These tests re-derive the headline measurements
from the retained board outputs, so the finding cannot drift, and pin the blocker
that keeps non-power-of-two LUT bands rejected.
"""
from pathlib import Path
import json
import math
import sys
import unittest

import numpy as np
from onnx import helper as h, numpy_helper as nh

from open_rknpu.lut import BASE_WEIGHT_SCALE, compile_lut, negative_half_gain

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "lut_index_probe"
sys.path.insert(0, str(ROOT / "research"))
import analyze_lut_index  # noqa: E402

MANIFEST = {m["index"]: m for m in json.loads((PROBE / "manifest.json").read_text())}
ANALYSIS = json.loads((PROBE / "analysis.json").read_text())
CODES = [code for code, _ in analyze_lut_index.decode_context()]
NON_POWER_OF_TWO = ["npot_0_02", "npot_0_03", "npot_1_48", "npot_0_024", "npot_0_012"]


def lut_stem(gain):
    weight = (np.eye(3, dtype=np.float32) * np.float32(gain)).reshape(3, 3, 1, 1)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node("Sigmoid", ["conv"], ["output"])], "reject",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(weight, "w"), nh.from_array(np.zeros(3, np.float32), "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class LutIndexProbe(unittest.TestCase):
    def test_probe_is_retained(self):
        self.assertTrue((PROBE / "README.md").exists())
        self.assertTrue((PROBE / "analysis.json").exists())
        self.assertEqual(len(MANIFEST), 30)
        for index in MANIFEST:
            for name in (f"model{index:03}.bin", f"input{index:03}.u8",
                         f"expected{index:03}.i8", f"out{index:03}.i8"):
                self.assertGreater((PROBE / name).stat().st_size, 0, name)

    def test_measured_index_reproduces_the_retained_analysis(self):
        for stem in range(6):
            meta = MANIFEST[stem * 10]
            recorded = ANALYSIS[meta["name"]]
            measured = analyze_lut_index.measured_index(stem, MANIFEST)
            with self.subTest(stem=meta["name"]):
                self.assertEqual(len(measured), 768)
                self.assertEqual(sum(1 for value in measured if value is not None),
                                 recorded["readable"])
                signed = {}
                for index, code in zip(measured, CODES):
                    if index is None:
                        continue
                    gain = meta["gain"] * (meta["hardware_gain"] if code < 0 else 1.0)
                    delta = index - (512 + round(64 * gain * code))
                    signed[delta] = signed.get(delta, 0) + 1
                # Every deviation from the emitted reference is one index step.
                self.assertLessEqual(set(signed), {-1, 0, 1}, signed)
                self.assertEqual(signed.get(0, 0), recorded["index_exact"])
                self.assertEqual({str(k): v for k, v in signed.items()},
                                 recorded["index_signed"])

    def test_control_stem_index_is_exact_and_reference_is_byte_exact(self):
        recorded = ANALYSIS["ctrl_1_32"]
        self.assertEqual(recorded["readable"], 768)
        self.assertEqual(recorded["index_exact"], 768)
        self.assertEqual(recorded["bytes_mismatch"], 0)
        self.assertGreater(recorded["pos_affine"], 0)
        self.assertGreater(recorded["neg_affine"], 0)

    def test_non_power_of_two_bands_leave_one_index_step(self):
        for name in NON_POWER_OF_TWO:
            recorded = ANALYSIS[name]
            with self.subTest(name=name):
                self.assertGreaterEqual(recorded["bytes_mismatch"], 60)
                self.assertLessEqual(recorded["bytes_mismatch"], 110)
                self.assertEqual(recorded["bytes_worst"], 1)
                self.assertLessEqual(set(recorded["index_signed"]), {"-1", "0", "1"})
                ratio = recorded["index_exact"] / recorded["readable"]
                self.assertGreaterEqual(ratio, 0.75)
                self.assertLessEqual(ratio, 0.9)
                # No integer (A, B, J), J <= 14, reproduces the measured index,
                # so the hardware argument is not an affinely rounded code.
                self.assertEqual(recorded["pos_affine"], 0)
                self.assertEqual(recorded["neg_affine"], 0)

    def test_emitter_still_rejects_non_power_of_two_bands(self):
        probed = {m["ratio"]: m for m in MANIFEST.values()}
        for gain in (0.02, 0.03, 1 / 48, 0.024, 0.012):
            with self.subTest(gain=gain):
                with self.assertRaisesRegex(ValueError, "power of two"):
                    compile_lut(lut_stem(gain), 1, 128)
                scale = abs(gain) / 255
                ratio = BASE_WEIGHT_SCALE / scale
                self.assertGreater(abs(ratio - 2 ** round(math.log2(ratio))),
                                   1e-6 * ratio)
                # The same band was probed on the board (float32 weight scale).
                match = min(probed, key=lambda r: abs(r - ratio))
                self.assertLess(abs(match - ratio), 1e-3 * ratio)
                self.assertEqual(probed[match]["hardware_gain"],
                                 negative_half_gain(scale))


if __name__ == "__main__":
    unittest.main()
