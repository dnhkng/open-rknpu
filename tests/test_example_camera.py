"""SPDX-License-Identifier: MIT

E5: `examples/camera/` builds deterministically, the C harness reproduces the Python
conversion byte for byte, and a device that is not V4L2 fails cleanly.

Three things are checked, all host-only, no board and no network:

1. `build.py` runs twice into two temporary directories (with the process working directory
   also temporary) and the whole artifact set - model, three capture formats, packed input,
   integer reference, report - is byte-identical between the runs; `model.bin` decodes to the
   documented `[1,32,32,3] -> [1,16,16,4]` chain-walk container, and `expected.i8` equals
   `open_rknpu.walk.chain_walk_reference` recomputed here on the written `input.u8`.
2. `camera_npu.c` is host-compiled against `runtime/open_rknpu.c` and its `--emit-input` mode
   is run on the synthetic YUYV, NV12 and RGB frames; each output must equal
   `examples/camera/convert.py`'s bytes exactly. That byte equality is the conversion
   contract - the C and Python implementations of the YUV->RGB -> nearest downsample ->
   packed NHWC rule have to agree on every byte.
3. Pointing `--device` at a regular file (not a V4L2 node) exits non-zero with the errno
   named, instead of crashing or silently retrying. The capture node is opened before the
   NPU, which is what makes the `rkipc`-held ISP's `EBUSY` the reported error on the board.
"""
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "camera"
BUILD = EXAMPLE / "build.py"
CONVERT = EXAMPLE / "convert.py"
HARNESS = EXAMPLE / "camera_npu.c"
RUNTIME = ROOT / "runtime" / "open_rknpu.c"
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
ARTIFACTS = ("model.bin", "frame.yuyv", "frame.nv12", "frame.rgb", "input.u8",
             "expected.i8", "report.json")
FRAME_WIDTH, FRAME_HEIGHT = 64, 48

COMPILER = os.environ.get("OPEN_RKNPU_HOST_CC") or shutil.which("cc") or shutil.which("gcc")
DEFAULT_FLAGS = ["-O2", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-Iruntime"]
BUILD_FLAGS = shlex.split(os.environ.get("OPEN_RKNPU_HOST_CFLAGS", "")) or DEFAULT_FLAGS


def load_module(name, path):
    """Import an example module by path, under a private name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMERA_BUILD = load_module("camera_build", BUILD)
CAMERA_CONVERT = load_module("camera_convert", CONVERT)


def run_build(output, cwd):
    """Run the example's build.py with a temporary working directory and output."""
    return subprocess.run([sys.executable, str(BUILD), "--out", str(output)], cwd=str(cwd),
                          env=ENV, capture_output=True, text=True, timeout=120)


class CameraBuildTests(unittest.TestCase):
    """The deterministic artifact set and the integer reference it publishes."""

    @classmethod
    def setUpClass(cls):
        cls.first = tempfile.TemporaryDirectory(prefix="camera-build-a-")
        cls.cwd = tempfile.TemporaryDirectory(prefix="camera-cwd-")
        cls.process = run_build(cls.first.name, cls.cwd.name)
        cls.model = CAMERA_BUILD.build_model()
        cls.binary, cls.meta = CAMERA_BUILD.compile_model(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.first.cleanup()
        cls.cwd.cleanup()

    def test_build_succeeds(self):
        self.assertEqual(self.process.returncode, 0, self.process.stdout + self.process.stderr)

    def test_build_runs_twice_and_is_byte_repeatable(self):
        with tempfile.TemporaryDirectory(prefix="camera-build-b-") as second, \
                tempfile.TemporaryDirectory(prefix="camera-cwd-") as cwd:
            process = run_build(second, cwd)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            for name in ARTIFACTS:
                with self.subTest(artifact=name):
                    self.assertEqual((Path(self.first.name) / name).read_bytes(),
                                     (Path(second) / name).read_bytes(),
                                     "%s changed between builds" % name)

    def test_container_decodes_to_the_documented_chain(self):
        written = (Path(self.first.name) / "model.bin").read_bytes()
        self.assertEqual(written, self.binary, "the published container is not the compiled one")
        info = decode_sequence(written)
        self.assertEqual(info["format_version"], 5)
        self.assertEqual(info["task_count"], 3)
        self.assertEqual(list(info["shape_nhwc"]), [1, 32, 32, 3])
        self.assertEqual(list(info["output_shape_nhwc"]), [1, 16, 16, 4])
        self.assertEqual(info["input_bytes"], 32 * 32 * 3)
        self.assertEqual(info["output_bytes"], 16 * 16 * 4)
        self.assertEqual(self.meta["profile"], "chain-walk")
        self.assertEqual(list(self.meta["walk_ops"]), ["conv", "MaxPool", "conv"])

    def test_expected_is_the_profile_reference_on_the_written_input(self):
        out = Path(self.first.name)
        packed = (out / "input.u8").read_bytes()
        self.assertEqual(len(packed), 32 * 32 * 3)
        reference = CAMERA_BUILD.integer_reference(self.model, self.meta, packed)
        written = np.fromfile(out / "expected.i8", dtype=np.int8).reshape(reference.shape)
        np.testing.assert_array_equal(written, reference)
        self.assertGreater(len(np.unique(reference)), 1, "the reference output is degenerate")

    def test_report_records_the_geometry_bands_and_conversion(self):
        report = json.loads((Path(self.first.name) / "report.json").read_text())
        self.assertEqual(report["profile"], "chain-walk")
        self.assertEqual(report["task_count"], 3)
        self.assertEqual(report["model"]["input_shape_nhwc"], [1, 32, 32, 3])
        self.assertEqual(report["model"]["output_shape_nhwc"], [1, 16, 16, 4])
        self.assertEqual(report["frame"]["width"], FRAME_WIDTH)
        self.assertEqual(report["frame"]["height"], FRAME_HEIGHT)
        self.assertEqual(sorted(report["frame"]["formats"]),
                         ["nv12", "rgb", "yuyv"])
        self.assertEqual(report["frame"]["primary"], "frame.nv12")
        self.assertEqual(len(report["bands"]), 3)
        self.assertEqual([band["op"] for band in report["bands"]], ["conv", "MaxPool", "conv"])
        for key in ("decode", "formats", "downsample", "layout"):
            self.assertIn(key, report["conversion"])
        # Every artifact hash in the report is the hash of the file on disk.
        for name, entry in sorted(report["artifacts"].items()):
            self.assertIn(name, ARTIFACTS)
            data = (Path(self.first.name) / name).read_bytes()
            self.assertEqual(entry["bytes"], len(data))
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())


@unittest.skipUnless(COMPILER, "no host C compiler (cc/gcc) found; camera harness tests skipped")
class CameraConversionContractTests(unittest.TestCase):
    """The C conversion must reproduce `convert.py` byte for byte on every capture format."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="camera-c-")
        cls.datadir = Path(cls.tmp.name) / "artifacts"
        CAMERA_BUILD.write_artifacts(cls.datadir)
        cls.binary = Path(cls.tmp.name) / "camera_npu"
        cls.build = subprocess.run(
            [COMPILER, *BUILD_FLAGS, str(HARNESS.relative_to(ROOT)),
             str(RUNTIME.relative_to(ROOT)), "-o", str(cls.binary)],
            cwd=ROOT, capture_output=True, text=True, timeout=180)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def emit(self, fmt):
        """Run the harness's convert-only mode on one synthetic frame."""
        target = Path(self.tmp.name) / ("c-%s.u8" % fmt)
        process = subprocess.run(
            [str(self.binary), "--model", str(self.datadir / "model.bin"),
             "--frame", str(self.datadir / ("frame.%s" % fmt)), "--format", fmt,
             "--width", str(FRAME_WIDTH), "--height", str(FRAME_HEIGHT),
             "--emit-input", str(target)],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        return process, target

    def test_the_harness_compiles_and_links_against_the_runtime(self):
        self.assertEqual(self.build.returncode, 0,
                         "camera_npu.c failed to build:\n%s\n%s"
                         % (self.build.stdout, self.build.stderr))

    def test_emit_input_matches_the_python_conversion(self):
        for fmt in ("yuyv", "nv12", "rgb"):
            with self.subTest(format=fmt):
                process, target = self.emit(fmt)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertIn("EMIT", process.stdout)
                frame = (self.datadir / ("frame.%s" % fmt)).read_bytes()
                reference = CAMERA_CONVERT.convert_frame(frame, fmt, FRAME_WIDTH, FRAME_HEIGHT,
                                                         32, 32)
                written = target.read_bytes()
                self.assertEqual(len(written), len(reference))
                self.assertEqual(written, reference,
                                 "the C conversion of %s differs from convert.py" % fmt)
                if fmt == "nv12":
                    self.assertEqual(written, (self.datadir / "input.u8").read_bytes(),
                                     "the NV12 frame is the primary frame behind input.u8")

    def test_a_device_that_is_not_v4l2_fails_cleanly(self):
        not_a_device = Path(self.tmp.name) / "not-a-video-device"
        not_a_device.write_bytes(b"this is not a V4L2 node\n")
        process = subprocess.run(
            [str(self.binary), "--model", str(self.datadir / "model.bin"),
             "--device", str(not_a_device), "--format", "nv12",
             "--width", str(FRAME_WIDTH), "--height", str(FRAME_HEIGHT)],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertIn(str(not_a_device), process.stderr)
        self.assertIn("VIDIOC_QUERYCAP", process.stderr)
        self.assertIn("ENOTTY", process.stderr)
        self.assertNotIn("SUMMARY", process.stdout)

    def test_an_absent_node_names_the_errno_before_touching_the_npu(self):
        # The board's rkisp_mainpath reports EBUSY from VIDIOC_REQBUFS while rkipc holds it.
        # A capture node that cannot even be opened fails at open(); both paths name the
        # errno, and the node is opened before the NPU, so rkipc's EBUSY is what is reported.
        missing = Path(self.tmp.name) / "absent-video-node"
        process = subprocess.run(
            [str(self.binary), "--model", str(self.datadir / "model.bin"),
             "--device", str(missing), "--format", "yuyv",
             "--width", str(FRAME_WIDTH), "--height", str(FRAME_HEIGHT)],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertIn("ENOENT", process.stderr)

    def test_usage_and_bad_options_are_rejected(self):
        process = subprocess.run([str(self.binary)], cwd=ROOT, capture_output=True, text=True,
                                 timeout=60)
        self.assertEqual(process.returncode, 2)
        self.assertIn("usage:", process.stderr)
        process = subprocess.run([str(self.binary), "--model", "x", "--format", "bogus"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 2)
        self.assertIn("unknown format", process.stderr)


if __name__ == "__main__":
    unittest.main()
