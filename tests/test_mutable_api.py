"""SPDX-License-Identifier: MIT

The mutable-parameter API (`open_rknpu.mutable`, checklist F11).

A v4 container is the only one that carries named packed constant regions, and the runtime's
`ornpu_set_constant` replaces a region *whole*: the length has to match the descriptor and
the task program (which holds the band's multiplier/shift/zero-point registers) has to stay
the same. Those three rules are exactly what a host-side helper must enforce before a user
sends a container to the board, so this module pins:

* `compile_mutable` returns a v4 container with a `conv.parameters` (kind 1) or `mul.factor`
  (kind 3) region, and refuses to return a container with no region at all;
* `constant_regions` decodes the descriptors with their absolute file offsets, and raises
  the exact message for the formats that have no table (legacy and v5);
* `constant_payload` and `replace_constant` round-trip, reject a wrong-length replacement,
  an unknown name and an out-of-range index, and recompute the checksum so the result is a
  valid container;
* a region grafted from a same-band container reproduces that container byte-for-byte, and
  `program_bytes` is the comparison that tells a user whether two containers share a band;
* the retained `research/mutable_weights_suite` container still behaves as recorded.
"""
from pathlib import Path
import os
import struct
import subprocess
import sys
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from open_rknpu.mutable import (KIND_NAMES, compile_mutable, constant_payload,
                                constant_regions, container_info, graft_region,
                                program_bytes, replace_constant)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "mutable_weights_suite"
SIZE = 8
KERNEL = 3
IN_CHANNELS = 4
OUT_CHANNELS = 6


def conv_model(weights, bias):
    """A 3x3 C4->C6 Conv on 8x8 that routes to `native16-input`, the mutable profile.

    C3 input is deliberately not used: that shape is matched by a v3 profile before the
    native16 path, so it has no constant table. `test_a_c3_conv_has_no_constant_table`
    pins that boundary.
    """
    node = helper.make_node("Conv", ["input", "w", "b"], ["output"],
                            kernel_shape=[KERNEL, KERNEL], pads=[KERNEL // 2] * 4)
    graph = helper.make_graph(
        [node], "mutable_conv",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT,
                                       [1, IN_CHANNELS, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT,
                                       [1, OUT_CHANNELS, SIZE, SIZE])],
        [numpy_helper.from_array(weights.astype(np.float32), "w"),
         numpy_helper.from_array(bias.astype(np.float32), "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def conv_weights(seed, mirror=False):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.4, 0.4, (OUT_CHANNELS, IN_CHANNELS, KERNEL, KERNEL))
    if mirror:
        weights = weights[:, :, ::-1, ::-1].copy()
    return weights, rng.uniform(-0.3, 0.3, OUT_CHANNELS)


def mutable_mul_model(factor):
    node = helper.make_node("Mul", ["input", "factor"], ["output"])
    graph = helper.make_graph(
        [node], "mutable_mul",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [numpy_helper.from_array(np.asarray(factor, np.float32).reshape(1, 3, 1, 1), "factor")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class CompileMutableTests(unittest.TestCase):
    def test_a_native_conv_emits_one_conv_parameters_region(self):
        weights, bias = conv_weights(11)
        binary, meta, regions = compile_mutable(conv_model(weights, bias), mutable_weights=True)
        self.assertEqual(container_info(binary)["format_version"], 4)
        self.assertEqual([region.name for region in regions], ["conv.parameters"])
        region = regions[0]
        self.assertEqual(region.kind, 1)
        self.assertEqual(region.kind_name, KIND_NAMES[1])
        self.assertEqual(len(constant_payload(binary, name=region.name)), region.size)
        self.assertEqual(region.byte_offset % 8, 0, "regions are 8-byte aligned in the payload")
        # The descriptor repr is the debugging surface a user pastes into an issue.
        self.assertIn("conv.parameters", str(region))
        self.assertIn("kind=1", str(region))
        self.assertIn("packed Conv parameters", str(region))

    def test_the_constant_mul_profile_emits_a_factor_region(self):
        factor = np.array([0.5, -1.25, 2.0], np.float32)
        binary, _meta, regions = compile_mutable(mutable_mul_model(factor),
                                                mutable_constants=True)
        self.assertEqual([region.name for region in regions], ["mul.factor"])
        self.assertEqual(regions[0].kind, 3)

    def test_compile_mutable_needs_a_mutable_flag(self):
        weights, bias = conv_weights(12)
        with self.assertRaisesRegex(ValueError, "needs mutable_weights or mutable_constants"):
            compile_mutable(conv_model(weights, bias))

    def test_a_profile_without_a_mutable_form_is_rejected_by_the_scheduler(self):
        node = helper.make_node("Conv", ["input", "w", "b"], ["c"],
                                kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        pool = helper.make_node("MaxPool", ["c"], ["output"], kernel_shape=[2, 2], strides=[2, 2])
        weights, bias = conv_weights(13)
        graph = helper.make_graph(
            [node, pool], "no_mutable_form",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT,
                                           [1, IN_CHANNELS, SIZE, SIZE])],
            [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT,
                                           [1, OUT_CHANNELS, 4, 4])],
            [numpy_helper.from_array(weights.astype(np.float32), "w"),
             numpy_helper.from_array(bias.astype(np.float32), "b")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        with self.assertRaisesRegex(ValueError, "mutable weights currently require one native Conv"):
            compile_mutable(model, mutable_weights=True)


class RegionAccessTests(unittest.TestCase):
    def test_replace_constant_round_trips_its_own_payload(self):
        weights, bias = conv_weights(21)
        binary, _meta, regions = compile_mutable(conv_model(weights, bias), mutable_weights=True)
        payload = constant_payload(binary, name="conv.parameters")
        self.assertEqual(replace_constant(binary, payload, name="conv.parameters"), binary)

    def test_program_bytes_are_stable_for_the_same_model(self):
        weights, bias = conv_weights(31)
        first, _m1, _r1 = compile_mutable(conv_model(weights, bias), mutable_weights=True)
        second, _m2, _r2 = compile_mutable(conv_model(weights, bias), mutable_weights=True)
        self.assertEqual(first, second, "the compiler must be deterministic")
        self.assertEqual(program_bytes(first), program_bytes(second))

    def test_program_bytes_catch_a_one_ulp_band_change(self):
        # Mirroring the kernel taps preserves the per-channel weight statistics but changes
        # the summation order of the analytic output band by one float32 ulp. `program_bytes`
        # is the check a caller makes before grafting a region, so it has to catch even that.
        first, _m1, _r1 = compile_mutable(conv_model(*conv_weights(31)), mutable_weights=True)
        second, _m2, _r2 = compile_mutable(conv_model(*conv_weights(31, mirror=True)),
                                           mutable_weights=True)
        self.assertNotEqual(first[72:76], second[72:76], "the output band differs by one ulp")
        self.assertNotEqual(program_bytes(first), program_bytes(second))
        self.assertLess(abs(struct.unpack("<f", first[72:76])[0]
                            - struct.unpack("<f", second[72:76])[0]), 1e-4)

    def test_a_graft_between_different_band_containers_is_left_to_the_caller(self):
        # The API does not silently refuse a cross-band graft: it produces a valid container
        # and exposes `program_bytes` so the caller can decide. This test documents that
        # contract explicitly rather than implying a band check that does not exist.
        first, _m1, _r1 = compile_mutable(conv_model(*conv_weights(31)), mutable_weights=True)
        second, _m2, _r2 = compile_mutable(conv_model(*conv_weights(31, mirror=True)),
                                           mutable_weights=True)
        grafted = replace_constant(first, constant_payload(second, index=0), index=0)
        self.assertEqual(len(grafted), len(first))
        self.assertNotEqual(grafted, first)
        self.assertNotEqual(program_bytes(grafted), program_bytes(second),
                            "the graft keeps the host program, so its band is still the host's")
        from open_rknpu.model import checksum
        masked = bytearray(grafted)
        stored = int.from_bytes(masked[80:84], "little")
        masked[80:84] = b"\0" * 4
        self.assertEqual(stored, checksum(bytes(masked)))

    def test_graft_region_reproduces_a_same_band_donor(self):
        band = {"scale": 0.125, "zero_point": -7}
        host, _mh, _rh = compile_mutable(conv_model(*conv_weights(81)), mutable_weights=True,
                                        output_range=band)
        donor, _md, _rd = compile_mutable(conv_model(*conv_weights(81, mirror=True)),
                                         mutable_weights=True, output_range=band)
        self.assertEqual(program_bytes(host), program_bytes(donor))
        self.assertEqual(graft_region(host, donor), donor)

    def test_graft_region_refuses_a_different_band_donor(self):
        host, _mh, _rh = compile_mutable(conv_model(*conv_weights(82)), mutable_weights=True,
                                        output_range={"scale": 0.125, "zero_point": -7})
        donor, _md, _rd = compile_mutable(conv_model(*conv_weights(82, mirror=True)),
                                         mutable_weights=True,
                                         output_range={"scale": 0.25, "zero_point": -3})
        with self.assertRaisesRegex(ValueError, "do not share a task program"):
            graft_region(host, donor)

    def test_graft_region_reports_a_donor_without_the_region(self):
        band = {"scale": 0.125, "zero_point": -7}
        host, _mh, _rh = compile_mutable(conv_model(*conv_weights(83)), mutable_weights=True,
                                        output_range=band)
        factor = np.array([0.5, -1.25, 2.0], np.float32)
        donor, _md, _rd = compile_mutable(mutable_mul_model(factor), mutable_constants=True,
                                         output_range=band)
        with self.assertRaisesRegex(ValueError, "the donor has no constant region named 'conv.parameters'"):
            graft_region(host, donor)

    def test_replace_constant_validates_length_name_and_index(self):
        binary, _meta, region = compile_mutable(conv_model(*conv_weights(51)),
                                                mutable_weights=True)
        region = region[0]
        with self.assertRaisesRegex(ValueError, "replacement has 1 bytes, region 'conv.parameters'"):
            replace_constant(binary, b"\x00", name="conv.parameters")
        with self.assertRaisesRegex(ValueError, "no constant region named 'mul.factor'"):
            replace_constant(binary, b"\x00" * region.size, name="mul.factor")
        with self.assertRaisesRegex(ValueError, "constant region index 3 out of range"):
            replace_constant(binary, b"\x00" * region.size, index=3)

    def test_constant_payload_by_index_matches_by_name(self):
        binary, _meta, regions = compile_mutable(conv_model(*conv_weights(61)),
                                                 mutable_weights=True)
        self.assertEqual(constant_payload(binary, index=0),
                         constant_payload(binary, name=regions[0].name))

    def test_a_replaced_container_has_a_valid_checksum(self):
        binary, _meta, region = compile_mutable(conv_model(*conv_weights(71)),
                                                mutable_weights=True)
        region = region[0]
        payload = bytearray(constant_payload(binary, name=region.name))
        payload[0] ^= 0xFF
        updated = replace_constant(binary, bytes(payload), name=region.name)
        from open_rknpu.model import checksum
        masked = bytearray(updated)
        stored = int.from_bytes(masked[80:84], "little")
        masked[80:84] = b"\0" * 4
        self.assertEqual(stored, checksum(bytes(masked)), "the spliced container must re-hash")


class ProfileBoundaryTests(unittest.TestCase):
    """The mutable region is a property of the profile, not of the API."""

    def test_a_c3_conv_has_no_constant_table(self):
        rng = np.random.default_rng(91)
        weights = rng.uniform(-0.4, 0.4, (3, 3, KERNEL, KERNEL))
        bias = rng.uniform(-0.3, 0.3, 3)
        node = helper.make_node("Conv", ["input", "w", "b"], ["output"],
                                kernel_shape=[KERNEL, KERNEL], pads=[KERNEL // 2] * 4)
        graph = helper.make_graph(
            [node], "c3_conv",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
            [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
            [numpy_helper.from_array(weights.astype(np.float32), "w"),
             numpy_helper.from_array(bias.astype(np.float32), "b")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        with self.assertRaisesRegex(ValueError, "no mutable constant region"):
            compile_mutable(model, mutable_weights=True)


class FormatRejectionTests(unittest.TestCase):
    def test_legacy_and_v5_containers_have_no_constant_table(self):
        legacy = (ROOT / "research" / "mnist_first_suite" / "model000.bin").read_bytes()
        v5 = (ROOT / "research" / "walk_chain_suite" / "model000.bin").read_bytes()
        self.assertEqual(legacy[:8], b"ORNPUBIN")
        self.assertEqual(v5[:8], b"ORNPUSEQ")
        with self.assertRaisesRegex(ValueError, "legacy containers have no constant descriptor table"):
            constant_regions(legacy)
        with self.assertRaisesRegex(ValueError, "v5 containers have no constant descriptor table"):
            constant_regions(v5)

    def test_a_v3_container_has_no_constant_table(self):
        v3 = (ROOT / "research" / "chain_suite" / "model000.bin").read_bytes()
        info = container_info(v3)
        if info["format_version"] != 3:
            self.skipTest(f"chain_suite/model000 is v{info['format_version']}, not v3")
        with self.assertRaisesRegex(ValueError, "legacy containers have no constant descriptor table"):
            constant_regions(v3)


class RetainedContainerTests(unittest.TestCase):
    """The published mutable-weights suite must keep behaving as recorded."""

    def test_the_retained_container_exposes_its_recorded_region(self):
        binary = (SUITE / "model000.bin").read_bytes()
        info = container_info(binary)
        self.assertEqual(info["format_version"], 4)
        regions = constant_regions(binary)
        self.assertEqual([region.name for region in regions], ["conv.parameters"])
        self.assertEqual(len(constant_payload(binary, name="conv.parameters")), regions[0].size)
        self.assertEqual(replace_constant(binary, constant_payload(binary, index=0), index=0),
                         binary)

    def test_the_retained_replacement_region_has_the_descriptor_size(self):
        parameters = (SUITE / "parameters.bin").read_bytes()
        binary = (SUITE / "model000.bin").read_bytes()
        region = constant_regions(binary)[0]
        self.assertEqual(len(parameters), region.size,
                         "the suite's donor region no longer matches the descriptor")
        self.assertNotEqual(parameters, constant_payload(binary, index=0))


class CookbookExampleTests(unittest.TestCase):
    """The shipped example is the user-facing documentation, so it must keep running."""

    def test_the_mutable_cookbook_example_runs_and_makes_its_claims(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "examples/cookbook/08_mutable_api.py"],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("programs identical, packed regions differ", result.stdout)
        self.assertIn("graft_region(host, donor) == donor container", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
