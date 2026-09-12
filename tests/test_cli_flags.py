"""SPDX-License-Identifier: MIT

The CLI's target/quantization contract and the legacy container round-trip.

Two published interfaces used to be silently wrong and are pinned here:

* `open-rknpu compile --target/--quantize` accepted their values and ignored them, so a
  future target could have been mis-compiled without any error. They are now validated
  against the supported sets and echoed in the summary line.
* `open_rknpu.model.decode` did not return the `register_count` that `encode` requires, so
  `encode(decode(blob))` was not callable. It now round-trips byte-identically.
"""
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu import cli
from open_rknpu.compiler import compile_model
from open_rknpu.model import decode, encode


def conv_model(path):
    """One 8x8/C3 3x3 Conv, the smallest legacy-profile graph."""
    rng = np.random.default_rng(7)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "cli_flags",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                         [nh.from_array(rng.uniform(-.3, .3, (3, 3, 3, 3)).astype(np.float32), "w"),
                          nh.from_array(rng.uniform(-.5, .5, (3,)).astype(np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def run_cli(*argv):
    """Drive cli.main() with argv, returning (exit status, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    previous = sys.argv
    sys.argv = ["open-rknpu", *[str(value) for value in argv]]
    status = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main()
    except SystemExit as exit_status:
        status = int(exit_status.code or 0)
    finally:
        sys.argv = previous
    return status, stdout.getvalue(), stderr.getvalue()


class CliTargetTests(unittest.TestCase):
    def test_target_and_quantization_are_echoed(self):
        with tempfile.TemporaryDirectory() as folder:
            model = conv_model(Path(folder) / "model.onnx")
            output = Path(folder) / "model.bin"
            status, stdout, stderr = run_cli("compile", model, "-o", output,
                                             "--target", "rv1103", "--quantize", "int8")
            self.assertEqual(status, 0, stderr)
            self.assertIn("rv1103/int8", stdout)
            self.assertTrue(output.is_file())
            self.assertEqual(decode(output.read_bytes())["target"], "rv1103")

    def test_unsupported_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = conv_model(Path(folder) / "model.onnx")
            status, _, stderr = run_cli("compile", model, "-o", Path(folder) / "m.bin",
                                        "--target", "rv1106")
            self.assertEqual(status, 2, stderr)
            self.assertIn("invalid choice", stderr)

    def test_unsupported_quantization_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = conv_model(Path(folder) / "model.onnx")
            status, _, stderr = run_cli("compile", model, "-o", Path(folder) / "m.bin",
                                        "--quantize", "int16")
            self.assertEqual(status, 2, stderr)
            self.assertIn("invalid choice", stderr)

    def test_supported_sets_are_declared_once(self):
        self.assertEqual(cli.TARGETS, ("rv1103",))
        self.assertEqual(cli.QUANTIZATIONS, ("int8",))


class LegacyRoundTripTests(unittest.TestCase):
    def test_encode_decode_encode_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            model = conv_model(Path(folder) / "model.onnx")
            payload, metadata = compile_model(model)
            blob = encode(payload, metadata)
            self.assertEqual(blob, encode(payload, decode(blob)))

    def test_decode_exposes_the_register_count(self):
        with tempfile.TemporaryDirectory() as folder:
            model = conv_model(Path(folder) / "model.onnx")
            payload, metadata = compile_model(model)
            decoded = decode(encode(payload, metadata))
            self.assertEqual(decoded["register_count"], 126)
            self.assertEqual(decoded["payload_bytes"], len(payload))


if __name__ == "__main__":
    unittest.main()
