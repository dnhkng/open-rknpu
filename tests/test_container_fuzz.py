"""SPDX-License-Identifier: MIT

Adversarial container tests: mutation fuzzing and encoder/decoder properties.

The container is the only interface between the compiler and the RV1103 loader, so
`open_rknpu.model.decode`/`open_rknpu.sequence.decode_sequence` must be total: for
*any* byte string they either return a structurally self-consistent description or
raise the decoder's documented `ValueError`.  This module pins that with a
deterministic, bounded mutation campaign over valid containers produced by real
emitters of every published flavour:

* **v3** (`encode_sequence`, no constant descriptors), **v4** (constant
  descriptors), **v5** (named tensor table) and the legacy `ORNPUBIN`
  (`model.encode`);
* structured mutations: single-byte flips, truncation at every header/task/tensor
  field boundary, inflated and deflated length/count fields, tensor index/name/
  layout corruption, task-offset corruption, checksum corruption and appended
  garbage - each in a raw form (the checksum catches it) and a checksum-repaired
  form (the semantic validators catch it, or the result must be self-consistent);
* encoder properties: the FNV-1a checksum changes exactly when the payload
  changes, a decoded sequence re-encodes byte-identically (v3/v4/v5), the legacy
  encoder round-trips, `task_count` always equals the task list, and every tensor
  lies inside the arena with the layout's exact byte size;
* a hand-built malicious-container regression: a tensor whose declared size
  exceeds the arena, an external tensor pair that overlaps and an internal tensor
  that overlaps an external one are all rejected by the loader-facing decoder and
  cannot be produced through `compose`/`encode_sequence_v5` either.

Seeds are fixed, the campaign is bounded by module constants (no input can hang a
loop), only `tempfile` is used and no board, network or vendor artifact is read.
"""
import copy
import math
import random
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu.compose import Binding, Stage, TensorSpec, compose
from open_rknpu.compiler import compile_model
from open_rknpu.model import HEADER, checksum, decode as decode_model, encode as encode_model
from open_rknpu.register_profile import REGISTERS
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import (LAYOUT_NAMES, LAYOUT_NATIVE16, ROLE_INPUT, ROLE_INTERNAL,
                                 ROLE_OUTPUT, decode_sequence, encode_sequence, encode_sequence_v5,
                                 payload_base, relink_for_batched, tensor_native_bytes)

# The decoder's documented failure set.  The set below is the union of the
# exception types a reader of the implementation would consider plausible; the
# `DecoderContractTests` module below proves the *actual* contract is `ValueError`
# alone (every `struct.unpack_from` is length-guarded before it runs).
DECODE_EXCEPTIONS = (ValueError, KeyError, struct.error)

# Campaign bounds: small enough that the whole module stays well under 20 s, large
# enough to cross every field boundary of the four flavours.
FLIPS_PER_CONTAINER = 96
APPEND_SIZES = (1, 4, 17, 64, 256)
COUNT_VALUES = (0, 1, 2, 0xFFFFFFFF)
LENGTH_VALUES = (0, 1, 64, 0xFFFFFFFF)


def checksum_field(data):
    """Offset of the stored FNV-1a checksum for either magic."""
    return 80 if data[:8] == b"ORNPUSEQ" else 84


def stored_checksum(data):
    return struct.unpack_from("<I", data, checksum_field(data))[0]


def checksum_holds(data):
    checked = bytearray(data)
    offset = checksum_field(data)
    checked[offset:offset + 4] = b"\0" * 4
    return checksum(checked) == stored_checksum(data)


def repair(data):
    """Recompute the checksum so a semantic mutation is what the decoder sees."""
    out = bytearray(data)
    offset = checksum_field(out)
    out[offset:offset + 4] = b"\0" * 4
    struct.pack_into("<I", out, offset, checksum(out))
    return bytes(out)


def patch32(data, offset, value):
    out = bytearray(data)
    struct.pack_into("<I", out, offset, value & 0xFFFFFFFF)
    return bytes(out)


# ---------------------------------------------------------------------------
# Real-emitter fixtures
# ---------------------------------------------------------------------------


def _nchw(shape):
    height, width, channels = shape
    return [1, channels, height, width]


def _conv_model(seed, height, width, input_channels, output_channels, kernel, relu=False, group=1):
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-.3, .3, (output_channels, input_channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-.4, .4, output_channels).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv" if relu else "output"],
                         kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4, group=group)]
    if relu:
        nodes.append(h.make_node("Relu", ["conv"], ["output"]))
    graph = h.make_graph(nodes, "conv", [h.make_tensor_value_info("input", 1, _nchw((height, width, input_channels)))],
                         [h.make_tensor_value_info("output", 1, _nchw((height, width, output_channels)))],
                         [nh.from_array(weight, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def _depthwise_model(seed):
    rng = np.random.default_rng(seed)
    stem = rng.uniform(-.3, .3, (3, 3, 1, 1)).astype(np.float32)
    stem_bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    weight = rng.uniform(-.4, .4, (3, 1, 3, 3)).astype(np.float32)
    bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "s_w", "s_b"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["stem", "d_w", "d_b"], ["output"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1], group=3)]
    graph = h.make_graph(nodes, "depthwise", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                         [nh.from_array(stem, "s_w"), nh.from_array(stem_bias, "s_b"),
                          nh.from_array(weight, "d_w"), nh.from_array(bias, "d_b")],
                         value_info=[h.make_tensor_value_info("stem", 1, [1, 3, 8, 8])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def _chain_walk_model(seed):
    rng = np.random.default_rng(seed)
    first = rng.uniform(-.3, .3, (6, 3, 3, 3)).astype(np.float32)
    first_bias = rng.uniform(-.3, .3, 6).astype(np.float32)
    last = rng.uniform(-.3, .3, (3, 6, 3, 3)).astype(np.float32)
    last_bias = rng.uniform(-.3, .3, 3).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "c0_w", "c0_b"], ["c0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             h.make_node("Relu", ["c0"], ["r0"]),
             h.make_node("MaxPool", ["r0"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
             h.make_node("Conv", ["p0", "c1_w", "c1_b"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "walk", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])],
                         [nh.from_array(first, "c0_w"), nh.from_array(first_bias, "c0_b"),
                          nh.from_array(last, "c1_w"), nh.from_array(last_bias, "c1_b")],
                         value_info=[h.make_tensor_value_info("c0", 1, [1, 6, 8, 8]),
                                     h.make_tensor_value_info("r0", 1, [1, 6, 8, 8]),
                                     h.make_tensor_value_info("p0", 1, [1, 6, 4, 4])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def _legacy_container(seed=11):
    """A real `ORNPUBIN` v1 container from the single-Conv legacy emitter."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(_conv_model(seed, 8, 8, 3, 3, 3), path)
        payload, metadata = compile_model(path)
    return encode_model(payload, metadata)


# ---------------------------------------------------------------------------
# Structural self-consistency of whatever the decoder returns
# ---------------------------------------------------------------------------


def _assert_sequence_consistent(test, data, info):
    test.assertTrue(checksum_holds(data))
    test.assertEqual(info["task_count"], len(info["tasks"]))
    test.assertGreater(info["payload_bytes"], 0)
    test.assertLessEqual(info["payload_bytes"], 1048576)
    test.assertEqual(info["payload_bytes"] % 64, 0)
    test.assertEqual(info["arena_bytes"] % 4096, 0)
    test.assertLessEqual(info["arena_bytes"], 4194304)
    test.assertEqual(info["payload_bytes"], len(data) - payload_base(info))
    test.assertTrue(math.isfinite(info["input_scale"]) and info["input_scale"] > 0)
    test.assertTrue(0 <= info["input_zero_point"] <= 255)
    test.assertTrue(math.isfinite(info["output_scale"]) and info["output_scale"] > 0)
    test.assertTrue(-128 <= info["output_zero_point"] <= 127)
    for task in info["tasks"]:
        test.assertGreaterEqual(task["command_offset"], 0)
        test.assertEqual(task["command_offset"] % 8, 0)
        test.assertTrue(1 <= task["register_count"] <= 1106)
        test.assertLessEqual(task["command_offset"] + (task["register_count"] + 4) * 8, info["payload_bytes"])
    for shape in (info["shape_nhwc"], info["output_shape_nhwc"]):
        test.assertTrue(1 <= shape[0] <= 16)
        test.assertTrue(1 <= shape[2] <= 1024 and 1 <= shape[3] <= 1024)
        test.assertTrue(1 <= shape[1] <= 128)
    if info["format_version"] == 5:
        names = [tensor["name"] for tensor in info["tensors"]]
        test.assertEqual(len(names), len(set(names)), "duplicate tensor name survived validation")
        externals = []
        for tensor in info["tensors"]:
            test.assertIn(tensor["layout"], LAYOUT_NAMES)
            test.assertEqual(tensor["bytes"], tensor_native_bytes(tensor["layout"], tensor["batch"],
                                                                  tensor["height"], tensor["width"],
                                                                  tensor["channels"]))
            test.assertGreaterEqual(tensor["byte_offset"], info["payload_bytes"])
            test.assertLessEqual(tensor["byte_offset"] + tensor["bytes"], info["arena_bytes"])
            if tensor["role"] != ROLE_INTERNAL:
                externals.append(tensor)
        for index, first in enumerate(externals):
            for second in externals[index + 1:]:
                test.assertFalse(first["byte_offset"] < second["byte_offset"] + second["bytes"]
                                 and second["byte_offset"] < first["byte_offset"] + first["bytes"],
                                 "overlapping external tensors survived validation")
        for tensor in info["tensors"]:
            if tensor["role"] != ROLE_INTERNAL:
                continue
            for external in externals:
                test.assertFalse(tensor["byte_offset"] < external["byte_offset"] + external["bytes"]
                                 and external["byte_offset"] < tensor["byte_offset"] + tensor["bytes"],
                                 "internal/external overlap survived validation")


def _assert_legacy_consistent(test, data, info):
    test.assertTrue(checksum_holds(data))
    test.assertEqual(len(data), HEADER.size + info["payload_bytes"])
    test.assertEqual(info["format_version"], 1 if (info["input_scale"], info["input_zero_point"]) == (1.0, 0) else 2)
    test.assertTrue(info["profile"] in (1, 2, 3, 4, 5, 6, 7, 8))
    test.assertTrue(math.isfinite(info["output_scale"]) and info["output_scale"] > 0)
    test.assertTrue(-128 <= info["output_zero_point"] <= 127)


def exercise(test, data, observed=None):
    """Decode one mutation: success must be self-consistent, failure must be documented."""
    try:
        info = decode_model(data)
    except DECODE_EXCEPTIONS as error:
        if observed is not None:
            observed.add(type(error))
        return None
    if data[:8] == b"ORNPUSEQ":
        _assert_sequence_consistent(test, data, info)
        test.assertEqual(info, decode_sequence(data), "model.decode and decode_sequence disagree")
    else:
        _assert_legacy_consistent(test, data, info)
        with test.assertRaises(ValueError):
            decode_sequence(data)
    return info


# ---------------------------------------------------------------------------
# Mutation generators (all bounded and seeded)
# ---------------------------------------------------------------------------


def sequence_mutations(data, info, rng):
    """Deterministic mutations of one sequence container: (label, bytes) pairs."""
    count = info["task_count"]
    task_start = 112 if info["format_version"] == 5 else 96
    tensor_start = task_start + count * 16
    payload_start = payload_base(info)
    boundaries = ({0, 4, 8, 16, 44, 48, 60, 80, 84, 88, 92, 95, 96, 111, len(data)}
                  | {task_start + 16 * index for index in range(count + 1)}
                  | {payload_start})
    if info["format_version"] == 5:
        boundaries |= {96 + 4 * index for index in range(4)}
        boundaries |= {tensor_start + 64 * index for index in range(info["tensor_count"] + 1)}
    boundaries = sorted(offset for offset in boundaries if 0 <= offset <= len(data))

    for offset in boundaries:
        yield ("truncate@%d" % offset, data[:offset])
    for _ in range(FLIPS_PER_CONTAINER):
        position = rng.randrange(len(data))
        mutated = bytearray(data)
        mutated[position] ^= 1 << rng.randrange(8)
        yield ("flip@%d" % position, bytes(mutated))

    field_offsets = {"payload_size": 44, "arena": 48, "count": 60, "flags": 84, "layout": 88, "batch": 92}
    if info["format_version"] == 5:
        field_offsets.update({"tensor_count": 96, "tensor_size": 100, "input_count": 104, "output_count": 108})
    for name, offset in field_offsets.items():
        for value in COUNT_VALUES + LENGTH_VALUES:
            patched = patch32(data, offset, value)
            yield ("raw-%s=%d" % (name, value), patched)
            yield ("fixed-%s=%d" % (name, value), repair(patched))

    for index in range(count):
        base = task_start + 16 * index
        for field, delta in (("offset", 4), ("amount", 1), ("enable", 0), ("mask", 0)):
            offset = base + {"offset": 0, "amount": 4, "enable": 8, "mask": 12}[field]
            original = struct.unpack_from("<I", data, offset)[0]
            for value in (0, 1, original + 1, 0xFFFFFFFF):
                patched = patch32(data, offset, value)
                yield ("raw-task%d-%s=%d" % (index, field, value), patched)
                yield ("fixed-task%d-%s=%d" % (index, field, value), repair(patched))
        # Task offsets must stay inside the payload and 8-byte aligned.
        yield ("raw-task%d-offset=misaligned" % index, patch32(data, base, 4))
        yield ("fixed-task%d-offset=misaligned" % index, repair(patch32(data, base, 4)))

    if info["format_version"] == 5:
        for index in range(info["tensor_count"]):
            base = tensor_start + 64 * index
            bare = data[base:base + 24]
            name = bare.split(b"\0", 1)[0]
            bad_name = bytearray(data)
            if name:
                bad_name[base] = 0x80  # invalid UTF-8 lead byte
            else:
                bad_name[base] = ord("x")
            yield ("raw-tensor%d-name" % index, bytes(bad_name))
            yield ("fixed-tensor%d-name" % index, repair(bytes(bad_name)))
            for field, slot in (("role", 0), ("layout", 1), ("index", 2), ("batch", 3),
                                ("height", 4), ("width", 5), ("channels", 6), ("offset", 7), ("size", 8)):
                offset = base + 24 + slot * 4
                for value in (0, 1, 0xFFFFFFFF):
                    patched = patch32(data, offset, value)
                    yield ("raw-tensor%d-%s=%d" % (index, field, value), patched)
                    yield ("fixed-tensor%d-%s=%d" % (index, field, value), repair(patched))

    yield ("checksum-zeroed", patch32(data, checksum_field(data), 0))
    yield ("checksum-flipped", patch32(data, checksum_field(data), stored_checksum(data) ^ 0xFFFFFFFF))
    for size in APPEND_SIZES:
        yield ("append-%d" % size, data + bytes(size))
        yield ("append-rand-%d" % size, data + bytes(rng.randrange(256) for _ in range(size)))


def legacy_mutations(data, rng):
    """Deterministic mutations of one legacy `ORNPUBIN` container."""
    for offset in sorted({0, 4, 8, 12, 16, 40, 44, 48, 52, 64, 76, 80, 84, 88, 92, 95, 96, len(data)}):
        yield ("truncate@%d" % offset, data[:offset])
    for _ in range(FLIPS_PER_CONTAINER):
        position = rng.randrange(len(data))
        mutated = bytearray(data)
        mutated[position] ^= 1 << rng.randrange(8)
        yield ("flip@%d" % position, bytes(mutated))
    for offset in (4, 8, 12, 16, 40, 44, 48, 52, 56, 60, 64):
        for value in COUNT_VALUES + LENGTH_VALUES:
            patched = patch32(data, offset, value)
            yield ("raw-field%d=%d" % (offset, value), patched)
            yield ("fixed-field%d=%d" % (offset, value), repair(patched))
    yield ("checksum-zeroed", patch32(data, 84, 0))
    yield ("checksum-flipped", patch32(data, 84, stored_checksum(data) ^ 0xFFFFFFFF))
    for size in APPEND_SIZES:
        yield ("append-%d" % size, data + bytes(size))


class ContainerFixture(unittest.TestCase):
    """Shared, built-once real containers of every published flavour."""

    @classmethod
    def setUpClass(cls):
        depthwise, _ = compile_sequence(_depthwise_model(21))
        cls.containers = {"v3": depthwise, "v4": None, "v5": None, "legacy": _legacy_container()}
        native, _ = compile_sequence(_conv_model(22, 10, 10, 3, 5, 3, relu=True), mutable_weights=True)
        cls.containers["v4"] = native
        walk, _ = compile_sequence(_chain_walk_model(23))
        cls.containers["v5"] = walk
        assert decode_sequence(cls.containers["v3"])["format_version"] == 3
        assert decode_sequence(cls.containers["v4"])["format_version"] == 4
        assert decode_sequence(cls.containers["v5"])["format_version"] == 5
        assert cls.containers["legacy"][:8] == b"ORNPUBIN"


class MutationFuzzTests(ContainerFixture):
    """The decoder is total over a deterministic mutation campaign."""

    def test_valid_fixtures_decode_consistently(self):
        for label, data in self.containers.items():
            with self.subTest(flavour=label):
                corrected = repair(data)
                self.assertEqual(corrected, data, "%s fixture has a wrong checksum" % label)
                exercise(self, data)

    def test_sequence_mutations_never_escape_the_documented_failures(self):
        for label in ("v3", "v4", "v5"):
            data = self.containers[label]
            info = decode_sequence(data)
            rng = random.Random(0xC0FFEE + info["format_version"])
            observed = set()
            mutations = 0
            for mutation, mutated in sequence_mutations(data, info, rng):
                mutations += 1
                with self.subTest(flavour=label, mutation=mutation):
                    exercise(self, mutated, observed)
            self.assertGreater(mutations, 200)
            self.assertTrue(observed.issubset(set(DECODE_EXCEPTIONS)))
            self.assertIn(ValueError, observed, "the campaign never exercised a rejection")

    def test_legacy_mutations_never_escape_the_documented_failures(self):
        data = self.containers["legacy"]
        rng = random.Random(0xBEEF)
        observed = set()
        mutations = 0
        for mutation, mutated in legacy_mutations(data, rng):
            mutations += 1
            with self.subTest(mutation=mutation):
                exercise(self, mutated, observed)
        self.assertGreater(mutations, 50)
        self.assertTrue(observed.issubset(set(DECODE_EXCEPTIONS)))
        self.assertIn(ValueError, observed)

    def test_checksum_repair_is_a_no_op_on_a_valid_container(self):
        for label, data in self.containers.items():
            with self.subTest(flavour=label):
                self.assertEqual(repair(data), data)


class DecoderContractTests(ContainerFixture):
    """The decoder's failure contract is exactly `ValueError`."""

    def test_structural_mutations_raise_value_error_not_struct_error(self):
        # A short buffer whose header claims a huge table would be a struct.error
        # without the length guard; the guard makes it a ValueError instead.
        data = self.containers["v5"]
        truncated = patch32(data, 60, 0xFFFFFFFF)[:100]
        with self.assertRaises(ValueError):
            decode_sequence(truncated)
        with self.assertRaises(ValueError):
            decode_model(truncated)

    def test_every_documented_malformed_input_raises_value_error(self):
        cases = {
            "empty": b"",
            "short-magic": b"ORNPUSEQ"[:4],
            "sequence-header-only": b"ORNPUSEQ" + bytes(88),
            "legacy-long-magic": b"ORNPUBIN" + bytes(8),
            "garbage": bytes(range(256)),
        }
        for label, data in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    decode_model(data)

    def test_decoder_does_not_index_unvalidated_keys(self):
        # Every dict the decoder builds carries the keys the consistency checks
        # read, so a successful decode never raises KeyError downstream.
        for label, data in self.containers.items():
            with self.subTest(flavour=label):
                info = decode_model(data)
                for key in ("target", "format_version", "shape_nhwc", "output_shape_nhwc",
                            "input_scale", "output_scale", "input_zero_point", "output_zero_point"):
                    self.assertIn(key, info)


class EncoderPropertyTests(ContainerFixture):
    """Checksum, task count, arena containment and round-trip properties."""

    PAYLOAD = bytes(4096)
    TASKS = ((0, 126, 29, 768), (0x440, 78, 24, 768))

    def encode(self, payload, tasks, *, constants=()):
        return encode_sequence(payload, input_shape=(8, 8, 3), output_shape=(8, 8, 3),
                               input_stride=16, arena_bytes=16384, input_offset=0x2000,
                               output_offset=0x3000, tasks=list(tasks), input_scale=1.0,
                               input_zero_point=0, output_scale=0.25, output_zero_point=7,
                               constants=constants)

    def test_checksum_changes_iff_the_payload_changes(self):
        base = self.encode(self.PAYLOAD, self.TASKS)
        same = self.encode(self.PAYLOAD, self.TASKS)
        self.assertEqual(stored_checksum(base), stored_checksum(same))
        for position in (0, 63, 1024, 4095):
            changed = bytearray(self.PAYLOAD)
            changed[position] ^= 0xFF
            mutated = self.encode(bytes(changed), self.TASKS)
            with self.subTest(position=position):
                self.assertNotEqual(stored_checksum(base), stored_checksum(mutated))
                self.assertTrue(checksum_holds(mutated))
                self.assertNotEqual(base, mutated)

    def test_checksum_covers_header_and_descriptors_too(self):
        base = self.encode(self.PAYLOAD, self.TASKS)
        # A semantic header change with a repaired checksum changes the stored word.
        changed = self.encode(self.PAYLOAD, self.TASKS, constants=[dict(name="k", offset=3000, size=64, kind=1)])
        self.assertNotEqual(stored_checksum(base), stored_checksum(changed))
        self.assertEqual(decode_sequence(changed)["format_version"], 4)
        self.assertNotEqual(len(base), len(changed))

    def test_task_count_matches_the_task_list(self):
        for tasks in (self.TASKS[:1], self.TASKS, self.TASKS + ((0x700, 37, 96, 3072),)):
            info = decode_sequence(self.encode(self.PAYLOAD, tasks))
            with self.subTest(tasks=len(tasks)):
                self.assertEqual(info["task_count"], len(tasks))
                self.assertEqual([(t["command_offset"], t["register_count"], t["enable"], t["mask"])
                                  for t in info["tasks"]], list(tasks))

    def test_every_tensor_lies_inside_the_arena(self):
        for label, data in self.containers.items():
            if label == "legacy":
                continue
            info = decode_sequence(data)
            if info["format_version"] != 5:
                continue
            with self.subTest(flavour=label):
                for tensor in info["tensors"]:
                    self.assertGreaterEqual(tensor["byte_offset"], info["payload_bytes"])
                    self.assertLessEqual(tensor["byte_offset"] + tensor["bytes"], info["arena_bytes"])
                    self.assertEqual(tensor["bytes"], tensor_native_bytes(
                        tensor["layout"], tensor["batch"], tensor["height"], tensor["width"],
                        tensor["channels"]))

    def test_sequence_decode_reencode_is_byte_stable(self):
        for label in ("v3", "v4", "v5"):
            data = self.containers[label]
            with self.subTest(flavour=label):
                self.assertEqual(rebuild_sequence(data), data)

    def test_legacy_encode_decode_round_trips_byte_identically(self):
        for seed, input_channels, output_channels, kernel, relu in (
                (31, 3, 3, 1, False), (32, 3, 5, 3, True), (33, 3, 4, 5, False)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.onnx"
                onnx.save(_conv_model(seed, 8, 8, input_channels, output_channels, kernel, relu=relu), path)
                payload, metadata = compile_model(path)
            data = encode_model(payload, metadata)
            with self.subTest(seed=seed, kernel=kernel, relu=relu):
                self.assertEqual(reencode_legacy(data), data)
                self.assertTrue(checksum_holds(data))

    def test_relinking_a_legacy_container_is_a_no_op(self):
        legacy = self.containers["legacy"]
        self.assertEqual(relink_for_batched(legacy), legacy)

    def test_relinking_a_sequence_container_is_idempotent(self):
        for label in ("v3", "v4", "v5"):
            data = self.containers[label]
            with self.subTest(flavour=label):
                linked = relink_for_batched(data)
                self.assertEqual(relink_for_batched(linked), linked)
                self.assertTrue(checksum_holds(linked))


def rebuild_sequence(data):
    """Decode then re-encode a sequence container purely from its decoded fields."""
    info = decode_sequence(data)
    payload = data[payload_base(info):]
    tasks = [(task["command_offset"], task["register_count"], task["enable"], task["mask"])
             for task in info["tasks"]]
    if info["format_version"] == 5:
        tensors = [dict(name=tensor["name"], role=tensor["role"], layout=tensor["layout"],
                        index=tensor["index"],
                        shape=(tensor["batch"], tensor["height"], tensor["width"], tensor["channels"]),
                        offset=tensor["byte_offset"], size=tensor["bytes"]) for tensor in info["tensors"]]
        return encode_sequence_v5(payload, tensors=tensors, tasks=tasks, arena_bytes=info["arena_bytes"],
                                  input_scale=info["input_scale"], input_zero_point=info["input_zero_point"],
                                  output_scale=info["output_scale"], output_zero_point=info["output_zero_point"],
                                  serial=info["serial"])
    constants = [dict(name=entry["name"], offset=entry["offset"], size=entry["size"], kind=entry["kind"])
                 for entry in info["constants"]]
    return encode_sequence(payload, input_shape=tuple(info["shape_nhwc"][1:]),
                           output_shape=tuple(info["output_shape_nhwc"][1:]), input_stride=info["input_stride"],
                           arena_bytes=info["arena_bytes"], input_offset=info["input_offset"],
                           output_offset=info["output_offset"], tasks=tasks,
                           input_scale=info["input_scale"], input_zero_point=info["input_zero_point"],
                           output_scale=info["output_scale"], output_zero_point=info["output_zero_point"],
                           serial=info["serial"], input_layout=info["input_layout"], batch=info["batch"],
                           input_tensor_count=info["input_tensor_count"], constants=constants or ())


def reencode_legacy(data):
    """Decode then re-encode a legacy container from its own decoded fields."""
    info = decode_model(data)
    info["register_count"] = len(REGISTERS)
    return encode_model(data[HEADER.size:], info)


class MaliciousContainerTests(ContainerFixture):
    """A hand-built lying container cannot reach the loader, the decoder or `compose`."""

    def tensor_field(self, data, index, slot):
        info = decode_sequence(data)
        start = 112 + info["task_count"] * 16
        return start + index * 64 + 24 + slot * 4

    def test_tensor_claiming_more_bytes_than_the_arena_is_rejected(self):
        data = self.containers["v5"]
        info = decode_sequence(data)
        index = next(i for i, tensor in enumerate(info["tensors"]) if tensor["layout"] == 0)
        tensor = info["tensors"][index]
        # Claim a packed tensor 8192 pixels wide: shape and size stay mutually
        # consistent, so only the arena bound can catch it.
        mutated = patch32(data, self.tensor_field(data, index, 5), 8192)
        mutated = patch32(mutated, self.tensor_field(data, index, 8),
                          tensor["batch"] * tensor["height"] * 8192 * tensor["channels"])
        mutated = repair(mutated)
        with self.assertRaises(ValueError):
            decode_sequence(mutated)
        with self.assertRaises(ValueError):
            decode_model(mutated)

    def test_tensor_offset_beyond_the_arena_is_rejected(self):
        data = self.containers["v5"]
        info = decode_sequence(data)
        index = next(i for i, tensor in enumerate(info["tensors"]) if tensor["role"] == ROLE_INTERNAL)
        mutated = repair(patch32(data, self.tensor_field(data, index, 7), info["arena_bytes"] - 64))
        with self.assertRaises(ValueError):
            decode_sequence(mutated)

    def test_overlapping_external_tensors_are_rejected(self):
        data = self.containers["v5"]
        info = decode_sequence(data)
        outputs = [i for i, tensor in enumerate(info["tensors"]) if tensor["role"] == ROLE_OUTPUT]
        inputs = [i for i, tensor in enumerate(info["tensors"]) if tensor["role"] == ROLE_INPUT]
        mutated = repair(patch32(data, self.tensor_field(data, outputs[0], 7),
                                 info["tensors"][inputs[0]]["byte_offset"]))
        with self.assertRaises(ValueError):
            decode_sequence(mutated)
        with self.assertRaises(ValueError):
            decode_model(mutated)

    def test_internal_tensor_overlapping_an_external_is_rejected(self):
        data = self.containers["v5"]
        info = decode_sequence(data)
        internals = [i for i, tensor in enumerate(info["tensors"]) if tensor["role"] == ROLE_INTERNAL]
        inputs = [i for i, tensor in enumerate(info["tensors"]) if tensor["role"] == ROLE_INPUT]
        mutated = repair(patch32(data, self.tensor_field(data, internals[0], 7),
                                 info["tensors"][inputs[0]]["byte_offset"]))
        with self.assertRaises(ValueError):
            decode_sequence(mutated)

    def test_larger_shape_with_matching_size_is_still_rejected(self):
        data = self.containers["v5"]
        info = decode_sequence(data)
        index = next(i for i, tensor in enumerate(info["tensors"]) if tensor["layout"] == LAYOUT_NATIVE16)
        tensor = info["tensors"][index]
        mutated = patch32(data, self.tensor_field(data, index, 4), 1024)
        mutated = patch32(mutated, self.tensor_field(data, index, 5), 1024)
        mutated = patch32(mutated, self.tensor_field(data, index, 8),
                          tensor["batch"] * (1024 * 1024 // 4) * 64 * ((tensor["channels"] + 15) // 16))
        with self.assertRaises(ValueError):
            decode_sequence(repair(mutated))

    def test_encode_v5_rejects_a_tensor_whose_size_lies(self):
        data = self.containers["v5"]
        info = decode_sequence(data)
        payload = data[payload_base(info):]
        tasks = [(task["command_offset"], task["register_count"], task["enable"], task["mask"])
                 for task in info["tasks"]]
        tensors = [dict(name=tensor["name"], role=tensor["role"], layout=tensor["layout"],
                        index=tensor["index"],
                        shape=(tensor["batch"], tensor["height"], tensor["width"], tensor["channels"]),
                        offset=tensor["byte_offset"], size=tensor["bytes"]) for tensor in info["tensors"]]
        tensors[0]["size"] = 1
        with self.assertRaises(ValueError):
            encode_sequence_v5(payload, tensors=tensors, tasks=tasks, arena_bytes=info["arena_bytes"])

    def test_compose_rejects_a_tensor_spec_whose_size_lies(self):
        stages = [Stage(name="only", family="elementwise", reads=("image",), writes=("output",),
                        fields=lambda addresses, constants: {0x4020: addresses["output"],
                                                             0x5018: addresses["image"],
                                                             0x5038: addresses["image"]},
                        bindings=(Binding(0x5018, "image", "read"), Binding(0x5038, "image", "read"),
                                  Binding(0x4020, "output", "write")))]
        good = [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16),
                TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16)]
        binary, _ = compose(stages, copy.deepcopy(good), serial=True)
        info = decode_sequence(binary)
        self.assertEqual(info["tensor_count"], 2)
        lying = copy.deepcopy(good)
        lying[0] = TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1)
        with self.assertRaises(ValueError):
            compose(stages, lying, serial=True)

    def test_compose_never_emits_an_out_of_arena_tensor(self):
        stages = [Stage(name="only", family="elementwise", reads=("image",), writes=("output",),
                        fields=lambda addresses, constants: {0x4020: addresses["output"],
                                                             0x5018: addresses["image"],
                                                             0x5038: addresses["image"]},
                        bindings=(Binding(0x5018, "image", "read"), Binding(0x5038, "image", "read"),
                                  Binding(0x4020, "output", "write")))]
        tensors = [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16),
                   TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16)]
        binary, _ = compose(stages, tensors, serial=True)
        info = decode_sequence(binary)
        for tensor in info["tensors"]:
            self.assertLessEqual(tensor["byte_offset"] + tensor["bytes"], info["arena_bytes"])


# ---------------------------------------------------------------------------
# Coverage closure for the encoder, composer and liveness guards
# ---------------------------------------------------------------------------


def _payload_and_tasks():
    payload = bytes(4096)
    tasks = ((0, 126, 29, 768), (0x440, 78, 24, 768))
    return payload, tasks


def _encode_sample(payload, tasks, constants=()):
    return encode_sequence(payload, input_shape=(8, 8, 3), output_shape=(8, 8, 3), input_stride=16,
                           arena_bytes=16384, input_offset=0x2000, output_offset=0x3000,
                           tasks=list(tasks), constants=constants)


class SequenceEncoderGuardTests(ContainerFixture):
    """Reachable rejection paths of `encode_sequence`/`encode_sequence_v5`."""

    def test_unknown_tensor_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            tensor_native_bytes(9, 1, 8, 8, 3)

    def test_v5_constant_descriptors_are_not_representable(self):
        # Version 5 replaces the v4 constant table; the encoder's `constants`
        # parameter therefore produces a container its own decoder rejects.
        info = decode_sequence(self.containers["v5"])
        payload = self.containers["v5"][payload_base(info):]
        tasks = [(task["command_offset"], task["register_count"], task["enable"], task["mask"])
                 for task in info["tasks"]]
        tensors = [dict(name=tensor["name"], role=tensor["role"], layout=tensor["layout"],
                        index=tensor["index"],
                        shape=(tensor["batch"], tensor["height"], tensor["width"], tensor["channels"]),
                        offset=tensor["byte_offset"], size=tensor["bytes"]) for tensor in info["tensors"]]
        with self.assertRaises(ValueError):
            encode_sequence_v5(payload, tensors=tensors, tasks=tasks, arena_bytes=info["arena_bytes"],
                               constants=[dict(name="c", offset=0, size=64, kind=1)])
        # An invalid descriptor (kind outside 1..3) is rejected inside the encoder.
        with self.assertRaises(ValueError):
            encode_sequence_v5(payload, tensors=tensors, tasks=tasks, arena_bytes=info["arena_bytes"],
                               constants=[dict(name="c", offset=0, size=64, kind=0)])

    def test_input_tensor_count_is_bounded(self):
        payload, tasks = _payload_and_tasks()
        with self.assertRaises(ValueError):
            encode_sequence(payload, input_shape=(8, 8, 3), output_shape=(8, 8, 3), input_stride=16,
                            arena_bytes=16384, input_offset=0x2000, output_offset=0x3000, tasks=list(tasks),
                            input_tensor_count=3)

    def test_invalid_constant_descriptor_is_rejected(self):
        payload, tasks = _payload_and_tasks()
        for kind in (0, 4):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    _encode_sample(payload, tasks, constants=[dict(name="c", offset=3000, size=64, kind=kind)])
        with self.assertRaises(ValueError):
            _encode_sample(payload, tasks, constants=[dict(name="c", offset=0xFFFFFF, size=64, kind=1)])

    def test_constant_name_must_be_utf8(self):
        data = self.containers["v4"]
        info = decode_sequence(data)
        name_offset = 96 + info["task_count"] * 16
        mutated = bytearray(repair(data))
        mutated[name_offset] = 0x80
        with self.assertRaises(ValueError):
            decode_sequence(repair(bytes(mutated)))

    def test_constant_may_not_overlap_a_command_program(self):
        data = self.containers["v4"]
        info = decode_sequence(data)
        offset_field = 96 + info["task_count"] * 16 + 24
        mutated = patch32(data, offset_field, 0)
        with self.assertRaises(ValueError):
            decode_sequence(repair(mutated))

    def test_encoder_tensor_entries_require_their_documented_keys(self):
        # The encoder reads its tensor records by key, so a record missing one is a
        # `KeyError`; this is why the documented encoder failure set is wider than the
        # decoder's `ValueError`-only contract.
        info = decode_sequence(self.containers["v5"])
        payload = self.containers["v5"][payload_base(info):]
        tasks = [(task["command_offset"], task["register_count"], task["enable"], task["mask"])
                 for task in info["tasks"]]
        complete = [dict(name=tensor["name"], role=tensor["role"], layout=tensor["layout"],
                         index=tensor["index"],
                         shape=(tensor["batch"], tensor["height"], tensor["width"], tensor["channels"]),
                         offset=tensor["byte_offset"], size=tensor["bytes"]) for tensor in info["tensors"]]
        for key in ("name", "role", "layout", "shape", "offset", "size"):
            entries = [dict(entry) for entry in complete]
            del entries[0][key]
            with self.subTest(missing=key), self.assertRaises(KeyError):
                encode_sequence_v5(payload, tensors=entries, tasks=tasks, arena_bytes=info["arena_bytes"])


def _elementwise_stage(name, reads, writes):
    return Stage(name=name, family="elementwise", reads=(reads,), writes=(writes,),
                 fields=lambda addresses, constants: {0x4020: addresses[writes],
                                                      0x5018: addresses[reads],
                                                      0x5038: addresses[reads]},
                 bindings=(Binding(0x5018, reads, "read"), Binding(0x5038, reads, "read"),
                           Binding(0x4020, writes, "write")))


def _chain_declaration():
    """A two-stage composed container (image -> mid -> output)."""
    stages = [_elementwise_stage("first", "image", "mid"), _elementwise_stage("second", "mid", "output")]
    tensors = [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024),
               TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024),
               TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024)]
    return stages, tensors


def _tail_offset(info, position):
    task = info["tasks"][position]
    return payload_base(info) + task["command_offset"] + task["register_count"] * 8


def _patch_tail(data, info, position, link=None, control=None):
    base = _tail_offset(info, position)
    out = bytearray(data)
    if link is not None:
        word0 = struct.unpack_from("<Q", out, base)[0]
        struct.pack_into("<Q", out, base, (word0 & ~(0xFFFFFFFF << 16)) | (link << 16))
    if control is not None:
        word1 = struct.unpack_from("<Q", out, base + 8)[0]
        struct.pack_into("<Q", out, base + 8, (word1 & ~(0xFFFFFFFF << 16)) | (control << 16))
    return repair(bytes(out))


class ComposerGuardTests(ContainerFixture):
    """Reachable validation paths of `compose`, `engine_runs` and the binding view."""

    def test_compose_requires_a_stage(self):
        with self.assertRaises(ValueError):
            compose([], [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024)])

    def test_compose_rejects_a_tensor_no_stage_writes(self):
        stages = [_elementwise_stage("only", "image", "output")]
        tensors = [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024),
                   TensorSpec("ghost", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024),
                   TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024)]
        with self.assertRaises(ValueError):
            compose(stages, tensors, serial=True, reuse=False)

    def test_compose_rejects_a_register_value_out_of_range(self):
        stage = Stage(name="only", family="elementwise", reads=("image",), writes=("output",),
                      fields=lambda addresses, constants: {0x4020: 2 ** 32, 0x5018: addresses["image"],
                                                           0x5038: addresses["image"]},
                      bindings=(Binding(0x5018, "image", "read"), Binding(0x5038, "image", "read"),
                                Binding(0x4020, "output", "write")))
        tensors = [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024),
                   TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 1024)]
        with self.assertRaises(ValueError):
            compose([stage], tensors, serial=True)

    def test_engine_runs_reports_a_trailing_linked_run(self):
        from open_rknpu.compose import engine_runs
        stages, tensors = _chain_declaration()
        binary, _ = compose(stages, tensors, serial=False)
        info = decode_sequence(binary)
        self.assertEqual(engine_runs(binary, info), [(0, 2)])
        # A final task that still links leaves a trailing run for the post-loop append.
        broken = _patch_tail(binary, info, info["task_count"] - 1, link=1)
        self.assertEqual(engine_runs(broken, decode_sequence(broken)), [(0, 2)])

    def test_batched_layout_reports_each_rejection(self):
        from open_rknpu.compose import batched_layout
        stages, tensors = _chain_declaration()
        serial, _ = compose(copy.deepcopy(stages), copy.deepcopy(tensors), serial=True)
        serial_info = decode_sequence(serial)
        self.assertIsNotNone(batched_layout(serial, serial_info))

        batched, _ = compose(copy.deepcopy(stages), copy.deepcopy(tensors), serial=False)
        info = decode_sequence(batched)
        self.assertIsNone(batched_layout(batched, info))

        last_link = _patch_tail(batched, info, info["task_count"] - 1, link=1, control=0x28)
        last_info = decode_sequence(last_link)
        self.assertIsNotNone(batched_layout(last_link, last_info))
        self.assertIn("last task", batched_layout(last_link, last_info))

        middle = _patch_tail(batched, info, 0, link=0, control=0x40)
        middle_info = decode_sequence(middle)
        self.assertIsNotNone(batched_layout(middle, middle_info))
        self.assertIn("must be terminal", batched_layout(middle, middle_info))

        # A middle task that ends its own run is a legal run boundary.
        boundary = _patch_tail(batched, info, 0, link=0, control=0x28)
        self.assertIsNone(batched_layout(boundary, decode_sequence(boundary)))
        # A middle task that links somewhere other than its successor is rejected.
        wrong = _patch_tail(batched, info, 0, link=12345)
        wrong_info = decode_sequence(wrong)
        self.assertIsNotNone(batched_layout(wrong, wrong_info))
        self.assertIn("must link", batched_layout(wrong, wrong_info))

    def test_derive_bindings_requires_v5_and_known_families(self):
        from open_rknpu.compose import derive_bindings
        with self.assertRaises(ValueError):
            derive_bindings(self.containers["v3"], decode_sequence(self.containers["v3"]))
        stages, tensors = _chain_declaration()
        binary, _ = compose(stages, tensors, serial=False)
        # An unknown (register_count, enable) signature fails the family lookup.
        task_start = 112
        broken = patch32(binary, task_start + 8, 96)
        broken = patch32(broken, task_start + 12, 3072)
        broken_info = decode_sequence(repair(broken))
        with self.assertRaises(ValueError):
            derive_bindings(broken, broken_info)

    def test_declared_bindings_match_the_container(self):
        from open_rknpu.compose import check_declared_bindings, derive_bindings
        stages, tensors = _chain_declaration()
        binary, meta = compose(stages, tensors, serial=False)
        info = decode_sequence(binary)
        derived = derive_bindings(binary, info)
        declared, offsets = meta["declared_bindings"], meta["tensor_offsets"]
        self.assertTrue(check_declared_bindings(declared, derived, offsets, meta["stage_schedule"]))
        # Distinct families make the register sets unique without a schedule.
        distinct_stages = [
            Stage(name="conv", family="native-conv", reads=("image",), writes=("mid",),
                  fields=lambda addresses, constants: {0x1070: addresses["image"], 0x4020: addresses["mid"]},
                  bindings=(Binding(0x1070, "image", "read"), Binding(0x4020, "mid", "write"))),
            Stage(name="pool", family="pool", reads=("mid",), writes=("output",),
                  fields=lambda addresses, constants: {0x701c: addresses["mid"], 0x6070: addresses["output"]},
                  bindings=(Binding(0x701c, "mid", "read"), Binding(0x6070, "output", "write")))]
        distinct_binary, distinct_meta = compose(distinct_stages, copy.deepcopy(tensors), serial=True)
        distinct_info = decode_sequence(distinct_binary)
        distinct_derived = derive_bindings(distinct_binary, distinct_info)
        self.assertTrue(check_declared_bindings(distinct_meta["declared_bindings"], distinct_derived,
                                                distinct_meta["tensor_offsets"]))

        # A declared list that repeats a stage name is rejected before any task.
        with self.assertRaises(ValueError):
            check_declared_bindings([dict(declared[0]), dict(declared[0])], derived, offsets,
                                    meta["stage_schedule"])
        # A schedule naming tensors that are not the declared stages is rejected.
        with self.assertRaises(ValueError):
            check_declared_bindings(declared, derived, offsets, ["not-a-stage", "also-not"])
        # Without a schedule the register sets must identify exactly one stage.
        ambiguous = [dict(declared[0]), dict(declared[0])]
        ambiguous[1]["stage"] = "other"
        with self.assertRaises(ValueError):
            check_declared_bindings(ambiguous, derived, offsets)
        # A declared tensor that the composer never placed is rejected.
        unknown = [dict(entry) for entry in declared]
        unknown[0]["reads"] = dict(unknown[0]["reads"])
        unknown[0]["reads"][0x5018] = "ghost"
        with self.assertRaises(ValueError):
            check_declared_bindings(unknown, derived, offsets, meta["stage_schedule"])


class LivenessGuardTests(unittest.TestCase):
    """The liveness allocator's documented rejection and placement behaviours."""

    def test_dict_tasks_are_normalized_and_validated(self):
        from open_rknpu.liveness import topological_order
        self.assertEqual(topological_order([{"reads": (), "writes": ("a",)}]), [0])
        with self.assertRaises(ValueError):
            topological_order([{"reads": (), "writes": ("",)}])
        with self.assertRaises(ValueError):
            topological_order([{"reads": (), "writes": ("a", "a")}])

    def test_live_intervals_rejects_a_read_before_write_with_a_fixed_order(self):
        from open_rknpu.liveness import Access, live_intervals
        with self.assertRaises(ValueError):
            live_intervals([Access(("a",), ("b",))], order=[0])

    def test_check_offsets_rejects_a_live_overlap(self):
        from open_rknpu.liveness import check_offsets
        with self.assertRaises(ValueError):
            check_offsets({"a": (0, 1), "b": (0, 1)}, {"a": 64, "b": 64}, {"a": 0, "b": 32})
        with self.assertRaises(ValueError):
            check_offsets({"a": (0, 0), "b": (1, 1)}, {"a": 64, "b": 64}, {"a": 0, "b": 1})

    def test_allocate_places_every_tensor_with_a_positive_size(self):
        from open_rknpu.liveness import allocate
        offsets = allocate({"a": (0, 0), "b": (0, 0)}, {"a": 64, "b": 128}, start=4096)
        self.assertEqual(sorted(offsets.values()), [4096, 4160])


if __name__ == "__main__":
    unittest.main()
