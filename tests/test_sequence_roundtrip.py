"""SPDX-License-Identifier: MIT

Container/loader parity for the two published executable formats.

The C loader (`runtime/open_rknpu.c`) is the only consumer of `model*.bin`, so the
compiler's own reader must agree with it byte for byte and field for field:

* every emitter's `meta` must describe the container it produced - shapes, input/output
  quantization band, task count with each task's enable/mask family, the v5 tensor
  table's roles/layouts/geometry and arena size, and the stored checksum;
* a container that is decoded and re-encoded from its own fields must come back
  byte-identical (v3, v4 constants and v5's serial/batched/multi-output forms), which is
  what makes `decode -> edit -> encode` tooling safe;
* the public encoders and `decode_sequence` reject malformed containers (bad magic or
  version, payload/tensor/task table that lies about its size, duplicated tensor names,
  non-contiguous external indices, wrong checksum);
* the legacy `ORNPUBIN` encoder/decoder round-trips and rejects a corrupted checksum or
  a truncated file;
* every published `research/*_suite/model*.bin` decodes the way the C loader would -
  sequence magic through `decode_sequence`, legacy magic through `model.decode` - and
  its declared geometry matches the sibling `manifest.json` where one exists.

This is the container half of the C-loader mirror; `tests/test_container_bindings.py`
adds the task read/write binding proof on top.
"""
from pathlib import Path
import json
import math
import re
import struct
import unittest

from open_rknpu.compose import FAMILY_BY_SIGNATURE
from open_rknpu.compiler import compile_model
from open_rknpu.model import checksum, decode as decode_model, encode as encode_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import (LAYOUT_NAMES, LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT,
                                 ROLE_INTERNAL, ROLE_OUTPUT, decode_sequence, encode_sequence,
                                 encode_sequence_v5, payload_base, tensor_native_bytes)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"

# (label, model, compile kwargs, expected per-task (register_count, enable, mask))
EMITTERS = (
    ("image-conv", "native_input_suite/model000.onnx", {}, [(126, 29, 768)]),
    ("chain-walk", "walk_chain_suite/model000.onnx", {},
     [(126, 29, 768), (37, 96, 3072), (126, 29, 768)]),
    ("diamond", "diamond_suite/model000.onnx", {},
     [(126, 29, 768), (126, 29, 768), (126, 29, 768), (78, 24, 768)]),
    ("join-dag", "join_dag_suite/model000.onnx", {},
     [(126, 29, 768), (126, 29, 768), (126, 29, 768), (126, 29, 768),
      (78, 24, 768), (78, 24, 768)]),
    ("depthwise", "depthwise_suite/model000.onnx", {}, [(126, 29, 768), (126, 29, 768)]),
    ("depthwise-join", "depthwise_join_suite/model000.onnx", {},
     [(126, 29, 768), (126, 29, 768), (126, 29, 768), (78, 24, 768)]),
)


def _checksum_offset(data):
    return 84 if data[:8] == b"ORNPUBIN" else 80


def _stored_checksum(data):
    return struct.unpack_from("<I", data, _checksum_offset(data))[0]


def _checksum_holds(data):
    """The stored checksum covers the whole file with its own field zeroed."""
    checked = bytearray(data)
    offset = _checksum_offset(data)
    checked[offset:offset + 4] = b"\0" * 4
    return checksum(checked) == _stored_checksum(data)


def _declared_shapes(meta):
    """Emitter shapes as NHWC; the walk profiles declare NCHW input/output_shape."""
    if "shape_nhwc" in meta:
        return list(meta["shape_nhwc"]), list(meta["output_shape_nhwc"])
    batch, channels, height, width = meta["input_shape"]
    out_batch, out_channels, out_height, out_width = meta["output_shape"]
    return [batch, height, width, channels], [out_batch, out_height, out_width, out_channels]


def _tasks_of(info):
    return [(task["command_offset"], task["register_count"], task["enable"], task["mask"])
            for task in info["tasks"]]


def _rebuild(data):
    """Decode a container and re-encode it purely from the decoded fields."""
    info = decode_sequence(data)
    payload = data[payload_base(info):]
    tasks = _tasks_of(info)
    if info["format_version"] == 5:
        tensors = [dict(name=t["name"], role=t["role"], layout=t["layout"], index=t["index"],
                        shape=(t["batch"], t["height"], t["width"], t["channels"]),
                        offset=t["byte_offset"], size=t["bytes"]) for t in info["tensors"]]
        return encode_sequence_v5(payload, tensors=tensors, tasks=tasks,
                                  arena_bytes=info["arena_bytes"], input_scale=info["input_scale"],
                                  input_zero_point=info["input_zero_point"],
                                  output_scale=info["output_scale"],
                                  output_zero_point=info["output_zero_point"], serial=info["serial"])
    constants = tuple(dict(name=c["name"], offset=c["offset"], size=c["size"], kind=c["kind"])
                      for c in info["constants"])
    return encode_sequence(payload, input_shape=tuple(info["shape_nhwc"][1:]),
                           output_shape=tuple(info["output_shape_nhwc"][1:]),
                           input_stride=info["input_stride"], arena_bytes=info["arena_bytes"],
                           input_offset=info["input_offset"], output_offset=info["output_offset"],
                           tasks=tasks, input_scale=info["input_scale"],
                           input_zero_point=info["input_zero_point"],
                           output_scale=info["output_scale"],
                           output_zero_point=info["output_zero_point"], serial=info["serial"],
                           input_layout=info["input_layout"], batch=info["batch"],
                           input_tensor_count=info["input_tensor_count"], constants=constants)


def _recompute(data, offset, value, fmt="<I"):
    """Mutate one header/tensor field and re-seal the container checksum."""
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, value)
    out[80:84] = b"\0" * 4
    struct.pack_into("<I", out, 80, checksum(out))
    return bytes(out)


class EmitterMetaParityTests(unittest.TestCase):
    """Every emitter's declared meta must describe the container it wrote."""

    def assert_emitter_parity(self, label, data, meta, expected_tasks):
        info = decode_sequence(data)
        where = label
        in_shape, out_shape = _declared_shapes(meta)
        self.assertEqual(info["shape_nhwc"], in_shape, where)
        self.assertEqual(info["output_shape_nhwc"], out_shape, where)
        self.assertEqual(info["batch"], in_shape[0], where)
        self.assertAlmostEqual(info["input_scale"], meta.get("input_scale", 1.0), places=6, msg=where)
        self.assertEqual(info["input_zero_point"], meta.get("input_zero_point", 0), where)
        self.assertAlmostEqual(info["output_scale"], meta["output_scale"], places=6, msg=where)
        self.assertEqual(info["output_zero_point"], meta["output_zero_point"], where)
        if "arena_bytes" in meta:
            self.assertEqual(info["arena_bytes"], meta["arena_bytes"], where)
        self.assertTrue(_checksum_holds(data), where)
        self.assertEqual([(t["register_count"], t["enable"], t["mask"]) for t in info["tasks"]],
                         [(count, enable, mask) for count, enable, mask in expected_tasks], where)
        self.assertEqual(info["task_count"], len(expected_tasks), where)
        # A declared binding list names the same task families, in schedule order.
        if "declared_bindings" in meta:
            families = [FAMILY_BY_SIGNATURE[(count, enable)].name for count, enable, _ in expected_tasks]
            self.assertEqual([entry["family"] for entry in meta["declared_bindings"]], families, where)
            declared_names = {name for entry in meta["declared_bindings"]
                              for name in list(entry["reads"].values()) + list(entry["writes"].values())}
            self.assertEqual(declared_names, set(meta["tensor_offsets"]), where)
        self.check_tensor_table(info, meta, where)

    def check_tensor_table(self, info, meta, where):
        """v5 tensor roles, layouts, geometry and offsets match the emitter meta."""
        if info["format_version"] != 5:
            self.assertEqual(info.get("tensors", []), [], where)
            return
        tensors = info["tensors"]
        self.assertTrue(tensors, where)
        inputs = [t for t in tensors if t["role"] == ROLE_INPUT]
        outputs = [t for t in tensors if t["role"] == ROLE_OUTPUT]
        self.assertEqual([t["index"] for t in inputs], [0], where)
        self.assertEqual([t["index"] for t in outputs], [0], where)
        names = {t["name"] for t in tensors}
        self.assertEqual(names, set(meta["tensor_offsets"]), where)
        self.assertGreaterEqual(len(names), 2, where)
        for tensor in tensors:
            self.assertIn(tensor["layout_name"], LAYOUT_NAMES.values(), where)
            self.assertEqual(tensor["bytes"],
                             tensor_native_bytes(tensor["layout"], tensor["batch"], tensor["height"],
                                                 tensor["width"], tensor["channels"]), where)
            self.assertEqual(tensor["byte_offset"], meta["tensor_offsets"][tensor["name"]], where)
            self.assertLessEqual(tensor["byte_offset"] + tensor["bytes"], info["arena_bytes"], where)
            self.assertEqual(tensor["role_name"],
                             {ROLE_INPUT: "input", ROLE_OUTPUT: "output",
                              ROLE_INTERNAL: "internal"}[tensor["role"]], where)
        for tensor in inputs:
            self.assertEqual(tensor["layout"], LAYOUT_PACKED_U8, where)
        for tensor in tensors:
            if tensor["role"] != ROLE_INPUT:
                self.assertEqual(tensor["layout"], LAYOUT_NATIVE16, where)
        if "output_tensors" in meta:
            self.assertEqual([t["name"] for t in outputs], list(meta["output_tensors"]), where)

    def test_emitters_declare_the_container_they_wrote(self):
        for label, relative, kwargs, tasks in EMITTERS:
            with self.subTest(emitter=label):
                data, meta = compile_sequence(RESEARCH / relative, **kwargs)
                self.assert_emitter_parity(label, data, meta, tasks)

    def test_emitter_containers_match_their_published_artifacts(self):
        # The published artifact is the same bytes the board ran; a recompile must
        # reproduce it and the emitter meta must still describe it.
        for label, relative, kwargs, tasks in EMITTERS:
            with self.subTest(emitter=label):
                data, meta = compile_sequence(RESEARCH / relative, **kwargs)
                artifact = (RESEARCH / relative).with_suffix(".bin").read_bytes()
                self.assertEqual(data, artifact)
                self.assert_emitter_parity(label, artifact, meta, tasks)


class SequenceReencodeTests(unittest.TestCase):
    """encode -> decode -> re-encode is byte-identical for the same inputs."""

    def test_synthetic_v3_v4_v5_roundtrip(self):
        payload = bytes(4096)
        tasks = [(0, 126, 29, 768)]
        v3 = encode_sequence(payload, input_shape=(8, 8, 3), output_shape=(8, 8, 3), input_stride=16,
                             arena_bytes=16384, input_offset=8192, output_offset=12288, tasks=tasks,
                             output_scale=0.5, output_zero_point=3)
        v4 = encode_sequence(payload, input_shape=(8, 8, 3), output_shape=(8, 8, 3), input_stride=16,
                             arena_bytes=16384, input_offset=8192, output_offset=12288, tasks=tasks,
                             output_scale=0.5, output_zero_point=3,
                             constants=[dict(name="conv.parameters", offset=2048, size=1024, kind=1)])
        v5 = encode_sequence_v5(
            payload, tasks=tasks, arena_bytes=16384, output_scale=0.5, output_zero_point=3,
            tensors=[dict(name="input", role=ROLE_INPUT, layout=LAYOUT_PACKED_U8, index=0,
                          shape=(1, 8, 8, 3), offset=8192, size=384),
                     dict(name="output", role=ROLE_OUTPUT, layout=LAYOUT_NATIVE16, index=0,
                          shape=(1, 8, 8, 3), offset=12288, size=1024)])
        for label, data, version, constants in (("v3", v3, 3, 0), ("v4", v4, 4, 1), ("v5", v5, 5, 0)):
            with self.subTest(form=label):
                info = decode_sequence(data)
                self.assertEqual((info["format_version"], info.get("constant_count", 0)),
                                 (version, constants))
                self.assertEqual(_rebuild(data), data)

    def test_real_containers_reencode_byte_identically(self):
        # v3 published, v4 mutable descriptors, v5 serial/batched/multi-output.
        cases = (
            ("v3-depthwise", RESEARCH / "depthwise_suite/model000.bin"),
            ("v3-native", RESEARCH / "native_input_suite/model000.bin"),
            ("v3-two-input", RESEARCH / "standalone_mul_suite/model000.bin"),
            ("v5-walk", RESEARCH / "walk_chain_suite/model000.bin"),
            ("v5-diamond", RESEARCH / "diamond_suite/model000.bin"),
        )
        for label, path in cases:
            with self.subTest(form=label):
                data = path.read_bytes()
                self.assertEqual(_rebuild(data), data)
                self.assertTrue(_checksum_holds(data))
        compiled = (
            ("v4-weights", RESEARCH / "native_input_suite/model000.onnx", dict(mutable_weights=True)),
            ("v4-constants", RESEARCH / "mul_broadcast_suite/model001.onnx", dict(mutable_constants=True)),
            ("v5-batched", RESEARCH / "walk_join_suite/model000.onnx", dict(submission="batched")),
            ("v5-multi-output", RESEARCH / "chain_multi_suite/model000.onnx",
             dict(expose_intermediates=True)),
        )
        for label, path, kwargs in compiled:
            with self.subTest(form=label):
                data, _ = compile_sequence(path, **kwargs)
                info = decode_sequence(data)
                self.assertEqual(_rebuild(data), data)
                self.assertEqual(info["serial"], kwargs.get("submission") != "batched")


class V5ValidatorTests(unittest.TestCase):
    """The v5 encoders and decoder reject malformed containers."""

    def setUp(self):
        raw = (RESEARCH / "walk_chain_suite/model000.bin").read_bytes()
        self.info = decode_sequence(raw)
        self.data = raw
        self.payload = raw[payload_base(self.info):]
        self.tensors = [dict(name=t["name"], role=t["role"], layout=t["layout"], index=t["index"],
                             shape=(t["batch"], t["height"], t["width"], t["channels"]),
                             offset=t["byte_offset"], size=t["bytes"]) for t in self.info["tensors"]]
        self.kwargs = dict(tensors=self.tensors, tasks=_tasks_of(self.info),
                           arena_bytes=self.info["arena_bytes"], input_scale=self.info["input_scale"],
                           input_zero_point=self.info["input_zero_point"],
                           output_scale=self.info["output_scale"],
                           output_zero_point=self.info["output_zero_point"], serial=self.info["serial"])

    def encode(self, **changes):
        return encode_sequence_v5(self.payload, **{**self.kwargs, **changes})

    def test_encoder_rejects_malformed_tensors_and_tasks(self):
        duplicate = [dict(t) for t in self.tensors]
        duplicate[1]["name"] = duplicate[0]["name"]
        gap = [dict(t) for t in self.tensors]
        gap[0]["index"] = 1
        oversized = [dict(t) for t in self.tensors]
        oversized[0]["size"] += 64
        degenerate = [dict(t) for t in self.tensors]
        degenerate[0]["shape"] = (1, 0, 8, 3)
        internal_only = [t for t in self.tensors if t["role"] != ROLE_OUTPUT]
        for label, changes in (("duplicated-name", dict(tensors=duplicate)),
                               ("index-gap", dict(tensors=gap)),
                               ("size-mismatch", dict(tensors=oversized)),
                               ("bad-shape", dict(tensors=degenerate)),
                               ("no-external-output", dict(tensors=internal_only)),
                               ("task-past-payload", dict(tasks=[(1 << 20, 126, 29, 768)]))):
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.encode(**changes)

    def test_decoder_rejects_magic_version_and_checksum(self):
        # The encoder always writes the right magic/version/checksum, so these are
        # decoder-level rejections of a file a loader may be handed.
        cases = (("magic", 0, 0x41, "<B"), ("version-6", 8, 6, "<I"),
                 ("version-4", 8, 4, "<I"), ("checksum", 80, 0x5A5A5A5A, "<I"))
        for label, offset, value, fmt in cases:
            with self.subTest(case=label):
                data = bytearray(self.data)
                struct.pack_into(fmt, data, offset, value)
                with self.assertRaises(ValueError):
                    decode_sequence(bytes(data))

    def test_decoder_rejects_lying_lengths(self):
        task_start = 112
        tensor_start = 112 + 16 * self.info["task_count"]
        cases = {
            "payload-longer": _recompute(self.data, 44, self.info["payload_bytes"] + 64),
            "payload-shorter": _recompute(self.data, 44, self.info["payload_bytes"] - 64),
            "stride-wrong": _recompute(self.data, 40, 15),
            "task-count-zero": _recompute(self.data, 60, 0),
            "task-offset-huge": _recompute(self.data, task_start, 1 << 20),
            "arena-zero": _recompute(self.data, 48, 0),
        }
        for label, data in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                decode_sequence(data)
        # A duplicated tensor name and a non-contiguous external index are semantic
        # table errors, so they must survive the checksum being recomputed.
        duplicate = bytearray(self.data)
        duplicate[tensor_start + 64:tensor_start + 88] = duplicate[tensor_start:tensor_start + 24]
        duplicate[80:84] = b"\0" * 4
        struct.pack_into("<I", duplicate, 80, checksum(duplicate))
        with self.assertRaises(ValueError):
            decode_sequence(bytes(duplicate))
        with self.assertRaises(ValueError):
            decode_sequence(_recompute(self.data, tensor_start + 32, 5))

    def test_decoder_rejects_truncation_and_trailing_bytes(self):
        for data in (self.data[:-1], self.data[:-64], self.data + b"x"):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                decode_sequence(data)


class LegacyOrnpubinTests(unittest.TestCase):
    """The legacy ORNPUBIN encoder/decoder round-trips and rejects corruption."""

    @classmethod
    def setUpClass(cls):
        payload, metadata = compile_model(RESEARCH / "generated/k3relu_heldout.onnx")
        cls.metadata = metadata
        cls.data = encode_model(payload, metadata)

    def test_roundtrip_and_fields(self):
        info = decode_model(self.data)
        self.assertEqual((info["target"], info["format_version"]), ("rv1103", 1))
        self.assertEqual(info["shape_nhwc"], list(self.metadata["shape_nhwc"]))
        self.assertEqual(info["output_shape_nhwc"], list(self.metadata["output_shape_nhwc"]))
        self.assertEqual(info["payload_bytes"], len(self.data) - 96)
        self.assertTrue(_checksum_holds(self.data))

    def test_decoder_rejects_corrupted_checksum(self):
        data = bytearray(self.data)
        data[84] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            decode_model(bytes(data))
        data = bytearray(self.data)
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            decode_model(bytes(data))

    def test_decoder_rejects_truncated_file(self):
        for data in (self.data[:96], self.data[:95], self.data[:-1]):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                decode_model(data)


class PublishedSuiteSweepTests(unittest.TestCase):
    """Mirror the C loader over every published board artifact, fast (decode only)."""

    @classmethod
    def setUpClass(cls):
        cls.models = sorted(RESEARCH.glob("*_suite/model*.bin"))
        cls.manifests = {}
        for path in cls.models:
            suite = path.parent.name
            if suite not in cls.manifests:
                manifest = path.parent / "manifest.json"
                entries = json.loads(manifest.read_text()) if manifest.exists() else []
                cls.manifests[suite] = {entry["index"]: entry for entry in entries
                                        if isinstance(entry, dict) and "index" in entry}

    def test_every_suite_container_decodes(self):
        self.assertGreater(len(self.models), 2000)
        sequence = legacy = 0
        for path in self.models:
            data = path.read_bytes()
            where = str(path.relative_to(ROOT))
            with self.subTest(model=where):
                if data[:8] == b"ORNPUSEQ":
                    info = decode_sequence(data)
                    self.assertEqual(decode_model(data), info, where)
                    sequence += 1
                else:
                    # The loader dispatches on magic: legacy containers are read by
                    # `model.decode` and the sequence reader must not accept them.
                    self.assertEqual(data[:8], b"ORNPUBIN", where)
                    self.assertIn(decode_model(data)["format_version"], (1, 2), where)
                    with self.assertRaisesRegex(ValueError, "magic"):
                        decode_sequence(data)
                    legacy += 1
        self.assertEqual(sequence + legacy, len(self.models))
        self.assertGreater(sequence, 1900)
        self.assertGreater(legacy, 300)

    def test_manifest_geometry_matches_where_declared(self):
        checked = 0
        for path in self.models:
            match = re.fullmatch(r"model(\d+)", path.stem)
            if match is None:
                continue
            entry = self.manifests[path.parent.name].get(int(match.group(1)))
            if not entry:
                continue
            info = decode_model(path.read_bytes())
            where = str(path.relative_to(ROOT))
            with self.subTest(model=where):
                if "shape_nhwc" in entry:
                    self.assertEqual(list(entry["shape_nhwc"]), info["shape_nhwc"], where)
                if "output_shape_nhwc" in entry:
                    self.assertEqual(list(entry["output_shape_nhwc"]), info["output_shape_nhwc"], where)
                if "input_scale" in entry:
                    self.assertTrue(math.isclose(entry["input_scale"], info["input_scale"],
                                                 rel_tol=1e-6), where)
                if "output_scale" in entry:
                    self.assertTrue(math.isclose(entry["output_scale"], info["output_scale"],
                                                 rel_tol=1e-6), where)
                if "output_zero_point" in entry:
                    self.assertEqual(entry["output_zero_point"], info["output_zero_point"], where)
                checked += 1
        # The manifests that declare NHWC geometry are a large minority; a silent
        # convention change would otherwise erase the check.
        self.assertGreater(checked, 300)


if __name__ == "__main__":
    unittest.main()
