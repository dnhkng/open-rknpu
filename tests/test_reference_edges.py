"""SPDX-License-Identifier: MIT

Edge coverage for the calibration paths the profile table advertises but the main parity
test does not reach: a join chain whose final stage is a Conv tail (rather than a Mul
join), and the explicit rejection of calibration on a height-strip tiled chain. Both are
new in the F3 pass, and both are user-visible promises, so they are pinned here rather
than left to the sampled replay.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_calibration_parity as parity  # noqa: E402  (the deterministic model builders)

from open_rknpu.scheduler import compile_sequence  # noqa: E402
from open_rknpu.sequence import decode_sequence  # noqa: E402


def compile_model(model, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path, **kwargs)


class JoinChainTailCalibrationTests(unittest.TestCase):
    def test_a_conv_tail_carries_its_measured_band(self):
        model = parity._join_chain(tail=3)
        ranges, names = parity._requested_ranges(model)
        binary, meta = compile_model(model, calibration_ranges=ranges)
        info = decode_sequence(binary)
        # The tail Conv is the graph output, so its measured range must be the band.
        self.assertAlmostEqual(info["output_scale"],
                               float(ranges[model.graph.output[0].name]["scale"]), places=6)
        self.assertEqual(info["output_zero_point"],
                         int(ranges[model.graph.output[0].name]["zero_point"]))
        self.assertTrue(any(name.endswith("_act") or name.startswith("h")
                            for name in names), names)

    def test_a_tiled_native_chain_rejects_calibration_ranges(self):
        model = parity._native_chain()
        with self.assertRaisesRegex(ValueError,
                                    "calibration is unsupported for the height-strip tiled chain"):
            compile_model(model, tiles=2, calibration_ranges={})


if __name__ == "__main__":
    unittest.main()
