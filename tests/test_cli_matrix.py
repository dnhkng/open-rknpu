"""SPDX-License-Identifier: MIT

The `open-rknpu` CLI surface: one compile flag matrix plus the error paths.

`cli.main()` is driven in process with `sys.argv` patched, so every branch the installed
console script exercises is covered without a subprocess or a board. Each successful
`compile` must exit 0, write a container that `model.decode`/`decode_sequence` accepts,
and print the summary line `Compiled <model> -> <output> (<n> bytes, <mode> submission)`;
`--calibration` must also write the retained `.calibration.json` report, `inspect` must
print the decoded JSON, and `normalize` must rewrite the ONNX graph. The rejected
combinations (`--mutable-*`/Mul zero points without `--sequence`, calibration with an
output override, `--tiles 1`, one-sided output quantization) exit 1 with the compiler's
own reason, while unknown subcommands/arguments are argparse errors (exit 2).
"""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu import cli
from open_rknpu.compose import batched_layout
from open_rknpu.model import decode as decode_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def run(*argv):
    """Drive `cli.main()` with argv, returning (exit code, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    old = sys.argv
    sys.argv = ["open-rknpu", *argv]
    code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main()
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
    finally:
        sys.argv = old
    return code, stdout.getvalue(), stderr.getvalue()


class CliMatrixTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)

    def out(self, name="model.bin"):
        return self.tmp / name

    def assert_summary(self, text, model, output, size, mode="serial",
                       target="rv1103", quantize="int8"):
        """The summary keeps its established prefix and now names the target/quantization."""
        expected = "Compiled %s -> %s (%d bytes, %s submission" % (model, output, size, mode)
        self.assertTrue(text.strip().startswith(expected), text)
        self.assertIn("%s/%s)" % (target, quantize), text)

    def test_compile_default_legacy_path(self):
        model = RESEARCH / "generated/k3relu_heldout.onnx"
        output = self.out("legacy.bin")
        code, text, err = run("compile", str(model), "-o", str(output))
        self.assertEqual((code, err), (0, ""))
        data = output.read_bytes()
        info = decode_model(data)
        self.assertIn(info["format_version"], (1, 2))
        self.assert_summary(text, model, output, len(data))
        self.assertEqual(data[:8], b"ORNPUBIN")

    def test_compile_sequence_and_expose_intermediates(self):
        model = RESEARCH / "chain_multi_suite/model000.onnx"
        output = self.out("chain.bin")
        code, text, err = run("compile", str(model), "--sequence", "--expose-intermediates",
                              "-o", str(output))
        self.assertEqual((code, err), (0, ""))
        data = output.read_bytes()
        info = decode_sequence(data)
        self.assertEqual((info["format_version"], info["output_tensor_count"]), (5, 3))
        self.assertEqual(data, compile_sequence(model, expose_intermediates=True)[0])
        self.assertIn("(4768 bytes, serial submission,", text)

    def test_input_quantization_overrides(self):
        cases = (
            ("legacy", RESEARCH / "mnist_first_suite/model000.onnx",
             ["--input-scale", ".25", "--input-zero-point", "135"], 0.25, 135),
            ("sequence", RESEARCH / "native_input_suite/model000.onnx",
             ["--sequence", "--input-scale", ".5", "--input-zero-point", "255"], 0.5, 255),
        )
        for label, model, flags, scale, zero_point in cases:
            with self.subTest(path=label):
                output = self.out(label + ".bin")
                code, text, err = run("compile", str(model), *flags, "-o", str(output))
                self.assertEqual((code, err), (0, ""))
                info = decode_model(output.read_bytes())
                self.assertEqual((info["input_scale"], info["input_zero_point"]), (scale, zero_point))
                self.assertIn("submission,", text)

    def test_output_quantization_overrides(self):
        cases = (
            ("legacy", RESEARCH / "wide_suite/model028.onnx", []),
            ("sequence", RESEARCH / "chain_multi_suite/model000.onnx", ["--sequence"]),
        )
        for label, model, flags in cases:
            with self.subTest(path=label):
                output = self.out(label + ".bin")
                code, text, err = run("compile", str(model), *flags,
                                      "--output-scale", "0.5", "--output-zero-point", "7",
                                      "-o", str(output))
                self.assertEqual((code, err), (0, ""))
                info = decode_model(output.read_bytes())
                self.assertEqual((info["output_scale"], info["output_zero_point"]), (0.5, 7))
                self.assertIn("submission,", text)

    def test_submission_serial_and_batched(self):
        model = RESEARCH / "walk_join_suite/model000.onnx"
        for mode, expected in (("serial", True), ("batched", False)):
            with self.subTest(submission=mode):
                output = self.out(mode + ".bin")
                code, text, err = run("compile", str(model), "--sequence",
                                      "--submission", mode, "-o", str(output))
                self.assertEqual((code, err), (0, ""))
                data = output.read_bytes()
                info = decode_sequence(data)
                self.assertEqual(info["serial"], expected)
                self.assert_summary(text, model, output, len(data), mode)
                if mode == "batched":
                    self.assertIsNone(batched_layout(data, info))

    def test_tiles_height_strips(self):
        model = RESEARCH / "deep_chain_suite/model002.onnx"
        output = self.out("tiled.bin")
        code, text, err = run("compile", str(model), "--sequence", "--tiles", "2", "-o", str(output))
        self.assertEqual((code, err), (0, ""))
        data = output.read_bytes()
        untiled = compile_sequence(model)[0]
        self.assertEqual(data, compile_sequence(model, tiles=2)[0])
        self.assertGreater(decode_sequence(data)["task_count"],
                           decode_sequence(untiled)["task_count"])
        self.assertIn("submission,", text)

    def test_mutable_weight_and_constant_descriptors(self):
        cases = (
            ("weights", RESEARCH / "native_input_suite/model000.onnx", "--mutable-weights",
             "conv.parameters", 1),
            ("constants", RESEARCH / "mul_broadcast_suite/model001.onnx", "--mutable-constants",
             "mul.factor", 3),
        )
        for label, model, flag, name, kind in cases:
            with self.subTest(descriptor=label):
                output = self.out(label + ".bin")
                code, text, err = run("compile", str(model), "--sequence", flag, "-o", str(output))
                self.assertEqual((code, err), (0, ""))
                info = decode_sequence(output.read_bytes())
                self.assertEqual((info["format_version"], info["constant_count"]), (4, 1))
                self.assertEqual((info["constants"][0]["name"], info["constants"][0]["kind"]),
                                 (name, kind))
                self.assertIn("submission,", text)

    def test_per_channel_mul_and_reuse_intermediates(self):
        cases = (
            ("per-channel", RESEARCH / "per_channel_mul_suite/model000.onnx", "--per-channel-mul",
             dict(per_channel_mul=True)),
            ("reuse", RESEARCH / "chain_reuse_suite/model000.onnx", "--reuse-intermediates",
             dict(reuse_intermediates=True)),
        )
        for label, model, flag, kwargs in cases:
            with self.subTest(flag=flag):
                output = self.out(label + ".bin")
                code, text, err = run("compile", str(model), "--sequence", flag, "-o", str(output))
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(output.read_bytes(), compile_sequence(model, **kwargs)[0])
                decode_sequence(output.read_bytes())
                self.assertIn("submission,", text)

    def test_calibration_writes_report(self):
        model = RESEARCH / "walk_chain_suite/model000.onnx"
        folder = self.tmp / "calibration"
        folder.mkdir()
        np.save(folder / "batch.npy", np.random.default_rng(7).integers(0, 256, (8, 3, 8, 8),
                                                                      dtype=np.uint8))
        output = self.out("calibrated.bin")
        code, text, err = run("compile", str(model), "--sequence", "--calibration", str(folder),
                              "-o", str(output))
        self.assertEqual((code, err), (0, ""))
        report = json.loads(output.with_suffix(".calibration.json").read_text())
        self.assertEqual((report["samples"], sorted(report["ranges"])), (8, ["conv2", "relu0"]))
        data = output.read_bytes()
        self.assertEqual(data, compile_sequence(model, calibration_ranges=report["ranges"])[0])
        # The measured range really is what the container carries: it beats the
        # analytic bound the default compile would have used.
        analytic = compile_sequence(model)[0]
        self.assertNotEqual(decode_sequence(data)["output_scale"],
                            decode_sequence(analytic)["output_scale"])
        self.assertIn("submission,", text)

    def test_inspect_prints_decoded_json(self):
        model = RESEARCH / "native_input_suite/model000.onnx"
        output = self.out("inspect.bin")
        self.assertEqual(run("compile", str(model), "--sequence", "-o", str(output))[0], 0)
        code, text, err = run("inspect", str(output))
        self.assertEqual((code, err), (0, ""))
        info = json.loads(text)
        self.assertEqual(info["format_version"], 3)
        self.assertEqual(info["shape_nhwc"], [1, 5, 5, 2])

    def test_normalize_writes_model(self):
        source = self.tmp / "normalize.onnx"
        onnx.save(h.make_model(h.make_graph(
            [h.make_node("Reshape", ["data", "shape"], ["y"])], "constant", [],
            [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 12])],
            [nh.from_array(np.arange(24, dtype=np.float32).reshape(2, 3, 4), "data"),
             nh.from_array(np.array([0, -1], dtype=np.int64), "shape")]),
            opset_imports=[h.make_opsetid("", 14)]), source)
        output = self.tmp / "nested" / "normalized.onnx"
        code, text, err = run("normalize", str(source), "-o", str(output))
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(output.is_file())
        self.assertEqual(len(onnx.load(output).graph.node), 0)
        self.assertIn("Normalized", text)

    def test_rejected_flag_combinations(self):
        cases = (
            ("mutable-without-sequence",
             ["compile", str(RESEARCH / "native_input_suite/model000.onnx"), "--mutable-weights"],
             "mutable parameters require --sequence"),
            ("mutable-constants-without-sequence",
             ["compile", str(RESEARCH / "mul_broadcast_suite/model001.onnx"), "--mutable-constants"],
             "mutable parameters require --sequence"),
            ("mul-zero-points-without-sequence",
             ["compile", str(RESEARCH / "generated/k3relu_heldout.onnx"),
              "--mul-a-zero-point", "5"],
             "Mul operand zero points require --sequence"),
            ("tiles-one",
             ["compile", str(RESEARCH / "deep_chain_suite/model002.onnx"), "--sequence",
              "--tiles", "1"],
             "tiles must be an integer of at least 2"),
            ("one-sided-output-quantization",
             ["compile", str(RESEARCH / "wide_suite/model028.onnx"), "--output-scale", "0.5"],
             "specify output scale and zero point together"),
        )
        for label, argv, message in cases:
            with self.subTest(case=label):
                output = self.out(label + ".bin")
                code, text, err = run(*argv, "-o", str(output))
                self.assertEqual(code, 1, err)
                self.assertIn(message, err)
                self.assertEqual(text, "")
                self.assertFalse(output.exists())

    def test_calibration_conflicts_with_output_override(self):
        folder = self.tmp / "calibration"
        folder.mkdir()
        np.save(folder / "batch.npy", np.zeros((4, 3, 8, 8), np.uint8))
        output = self.out("conflict.bin")
        code, text, err = run("compile", str(RESEARCH / "chain_multi_suite/model000.onnx"),
                              "--sequence", "--calibration", str(folder),
                              "--output-scale", "1.0", "--output-zero-point", "0", "-o", str(output))
        self.assertEqual(code, 1)
        self.assertIn("calibration and output quantization overrides cannot be combined", err)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".calibration.json").exists())

    def test_argparse_and_missing_file_errors(self):
        output = self.out("missing.bin")
        model = str(RESEARCH / "generated/k3relu_heldout.onnx")
        cases = (
            ("unknown-subcommand", ["frobnicate"], 2, "invalid choice"),
            ("unknown-argument", ["compile", model, "--frobnicate", "-o", str(output)], 2,
             "unrecognized arguments"),
            ("missing-model", ["compile", str(self.tmp / "absent.onnx"), "-o", str(output)], 1,
             "No such file or directory"),
        )
        for label, argv, code, message in cases:
            with self.subTest(case=label):
                status, text, err = run(*argv)
                self.assertEqual(status, code, err)
                self.assertIn(message, err)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
