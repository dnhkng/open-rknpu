"""SPDX-License-Identifier: MIT

Replay the board-verified Clip[0,6]/ReLU6 suite through the integer reference.

`research/conv_clip_suite/` recorded four fused `Conv -> Clip[0,6]` containers with their
inputs, expected outputs, output band and the *upper clamp code* they used. The reference
(`open_rknpu.native.native_input_reference`) used to model the CNA accumulator but not the
clamp, so a user comparing against it saw a mismatch and could not tell whether the hardware
or the reference was wrong. It now takes `upper_code`, and `compile_native_input` publishes
the same value as `meta["clip_upper_code"]`. These tests replay the recorded evidence
byte-for-byte.
"""
from pathlib import Path
import json
import unittest

import numpy as np
import onnx

from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "conv_clip_suite"


def quantization_from(meta):
    return Quantization(**{key: np.array(value) if isinstance(value, list) else value
                           for key, value in meta["quantization"].items()})


def input_batches(entry):
    """Recorded inputs as (count, height, width, channels) UINT8, derived from the model."""
    model = onnx.load(SUITE / f"model{entry['index']:03}.onnx")
    shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    _, channels, height, width = shape
    data = np.fromfile(SUITE / f"input{entry['index']:03}.u8", dtype=np.uint8)
    return data.reshape(-1, height, width, channels)


class ClipReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((SUITE / "manifest.json").read_text())

    def compile(self, entry):
        return compile_sequence(str(SUITE / f"model{entry['index']:03}.onnx"),
                                entry["input_scale"], entry["input_zero_point"],
                                output_range=entry["compile_output_range"])

    def test_recorded_board_evidence_replays_exactly(self):
        """The recorded expected bytes are reproduced when `upper_code` is supplied."""
        for entry in self.manifest:
            with self.subTest(model=entry["index"]):
                _, meta = self.compile(entry)
                quantization = quantization_from(meta)
                expected = np.fromfile(SUITE / f"expected{entry['index']:03}.i8", dtype=np.int8)
                produced = np.stack([
                    native_input_reference(sample, quantization, entry["input_zero_point"],
                                           entry["pads"], entry["strides"],
                                           upper_code=entry["upper_code"])
                    for sample in input_batches(entry)])
                self.assertEqual(produced.astype(np.int8).tobytes(), expected.tobytes())

    def test_meta_exposes_the_same_clamp_code(self):
        for entry in self.manifest:
            with self.subTest(model=entry["index"]):
                _, meta = self.compile(entry)
                self.assertEqual(meta["fused_activation"], "Clip[0,6]")
                self.assertEqual(meta["clip_upper_code"], entry["upper_code"])

    def test_the_clamp_actually_changes_the_result(self):
        """Without the clamp the reference is strictly higher somewhere."""
        entry = self.manifest[0]
        _, meta = self.compile(entry)
        quantization = quantization_from(meta)
        sample = input_batches(entry)[0]
        unclamped = native_input_reference(sample, quantization, entry["input_zero_point"],
                                           entry["pads"], entry["strides"])
        clamped = native_input_reference(sample, quantization, entry["input_zero_point"],
                                         entry["pads"], entry["strides"],
                                         upper_code=entry["upper_code"])
        self.assertTrue((unclamped.astype(np.int16) > clamped.astype(np.int16)).any())
        self.assertTrue((clamped.astype(np.int16) <= entry["upper_code"]).all())

    def test_default_upper_code_is_backwards_compatible(self):
        """`upper_code=None` (the default) equals `min(reference, upper)` applied outside."""
        for entry in self.manifest:
            with self.subTest(model=entry["index"]):
                _, meta = self.compile(entry)
                quantization = quantization_from(meta)
                for sample in input_batches(entry):
                    default = native_input_reference(sample, quantization,
                                                     entry["input_zero_point"], entry["pads"],
                                                     entry["strides"])
                    external = np.minimum(default, entry["upper_code"]).astype(np.int8)
                    internal = native_input_reference(sample, quantization,
                                                      entry["input_zero_point"], entry["pads"],
                                                      entry["strides"],
                                                      upper_code=entry["upper_code"])
                    self.assertEqual(external.tobytes(), internal.tobytes())


if __name__ == "__main__":
    unittest.main()
