# SPDX-License-Identifier: MIT
"""Pins the legacy ``ORNPUBIN`` container (v1/v2) and the single-Conv emitter.

`compiler.compile_model` emits the original one-program container; `model.encode`
wraps the 8192-byte payload with a fixed 96-byte header and `model.decode` parses
and validates it.  The tests here pin:

* header geometry, the output band, the fixed 126-register program, the weight and
  bias/lane blocks and the FNV checksum of freshly compiled Conv and Conv+Relu
  graphs;
* that decoding and re-encoding the container is byte identical;
* the documented rejections for a corrupted checksum, a truncated file, a wrong
  magic and several mismatched header geometries (checksum recomputed so the
  semantic check, not the checksum, is what fires);
* that the analytic (auto) output band covers the float range of a spread of
  inputs computed with ONNX's `ReferenceEvaluator`, and is not absurdly wide
  (ratio below 4x);
* that the sequence path (`compile_sequence`) and the legacy path
  (`compile_model` + `encode`) agree on the same single-Conv graph: the integer
  outputs are compared with an exact (0 LSB) tolerance and the bands must match.

No board is used.
"""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.compiler import compile_model
from open_rknpu.model import HEADER, checksum, decode, encode
from open_rknpu.quantization import Quantization, reference
from open_rknpu.register_profile import REGISTERS
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

REGISTER_COUNT = len(REGISTERS)


def conv_model(kernel=1, out_channels=3, input_channels=3, height=8, width=8, seed=0, relu=False):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.4, 0.4, (out_channels, input_channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-1, 1, out_channels).astype(np.float32)
    conv_output = "conv" if relu else "output"
    nodes = [h.make_node("Conv", ["input", "w", "b"], [conv_output], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    if relu:
        nodes.append(h.make_node("Relu", ["conv"], ["output"]))
    graph = h.make_graph(nodes, "conv", [h.make_tensor_value_info("input", 1, [1, input_channels, height, width])],
                         [h.make_tensor_value_info("output", 1, [1, out_channels, height, width])],
                         [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def compile_legacy(model, *args, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_model(path, *args, **kwargs)


def compile_sequence_path(model, *args, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path, *args, **kwargs)


def quantization_of(meta):
    return Quantization(**{key: np.array(value) if isinstance(value, list) else value
                           for key, value in meta["quantization"].items()})


def reencode(data):
    """Decode then re-encode the container with the constant ORNPUBIN register count."""
    info = decode(data)
    info["register_count"] = REGISTER_COUNT
    return encode(data[HEADER.size:], info)


def rewrite(data, offset, value):
    """Patch one little-endian uint32 and repair the checksum so semantics are tested."""
    patched = bytearray(data)
    struct.pack_into("<I", patched, offset, value)
    patched[84:88] = b"\0" * 4
    struct.pack_into("<I", patched, 84, checksum(patched))
    return bytes(patched)


def observed_float_range(model, weights):
    """Reference float outputs over random and corner inputs in the UINT8 domain."""
    out_channels, input_channels = weights.shape[0], weights.shape[1]
    height = width = 8
    inputs = []
    for channel in range(out_channels):
        for sign in (1, -1):
            sample = np.zeros((height, width, input_channels), np.float32)
            for index in range(input_channels):
                total = float(weights[channel, index].sum())
                sample[:, :, index] = 255.0 if (total > 0) == (sign > 0) else 0.0
            inputs.append(sample)
    rng = np.random.default_rng(0)
    inputs.extend(rng.integers(0, 256, (height, width, input_channels)).astype(np.float32) for _ in range(64))
    inputs.append(np.zeros((height, width, input_channels), np.float32))
    inputs.append(np.full((height, width, input_channels), 255.0, np.float32))
    evaluator = ReferenceEvaluator(model)
    return np.stack([evaluator.run(None, {"input": sample[None].transpose(0, 3, 1, 2)})[0] for sample in inputs])


class LegacyContainerTests(unittest.TestCase):
    def test_header_geometry_and_blocks(self):
        for kernel in (1, 3, 5):
            for relu in (False, True):
                with self.subTest(kernel=kernel, relu=relu):
                    model = conv_model(kernel=kernel, seed=kernel, relu=relu)
                    payload, meta = compile_legacy(model)
                    data = encode(payload, meta)
                    info = decode(data)
                    q = quantization_of(meta)
                    channels = meta["output_shape_nhwc"][3]
                    input_channels = meta["shape_nhwc"][3]
                    # Header geometry: the raw fixed struct plus the decoded view.
                    (magic, version, header_size, target, profile, height, width, in_channels,
                     out_channels, in_stride, out_stride, payload_size, register_count, in_offset,
                     out_offset, arena, header_kernel, header_relu, header_zero_point, header_scale,
                     _stored, _reserved0, _reserved1) = HEADER.unpack_from(data)
                    self.assertEqual(magic, b"ORNPUBIN")
                    self.assertEqual((version, header_size, target, profile), (1, 96, 1103, 1))
                    self.assertEqual((height, width, in_channels, out_channels), (8, 8, input_channels, channels))
                    self.assertEqual((in_stride, out_stride), (16, 16))
                    self.assertEqual(payload_size, 8192)
                    self.assertEqual(register_count, REGISTER_COUNT)
                    self.assertEqual((in_offset, out_offset, arena), (8192, 12288, 16384))
                    self.assertEqual((header_kernel, header_relu), (kernel, int(relu)))
                    self.assertEqual((header_zero_point, header_scale),
                                     (meta["output_zero_point"], meta["output_scale"]))
                    self.assertEqual(info["target"], "rv1103")
                    self.assertEqual(info["format_version"], 1)
                    self.assertEqual(info["profile"], 1)
                    self.assertEqual(info["task_count"], 1)
                    self.assertEqual(info["shape_nhwc"], [1, 8, 8, input_channels])
                    self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, channels])
                    self.assertEqual((info["kernel_size"], info["relu"]), (kernel, relu))
                    self.assertEqual(info["operators"], ["Conv", "Relu"] if relu else ["Conv"])
                    self.assertEqual(info["arena_bytes"], 16384)
                    self.assertEqual(info["payload_bytes"], 8192)
                    self.assertEqual(info["input_bytes"], 8 * 8 * input_channels)
                    self.assertEqual(info["output_bytes"], 8 * 8 * channels)
                    # Output band mirrors the compiled quantization.
                    self.assertEqual(info["output_scale"], meta["output_scale"])
                    self.assertEqual(info["output_zero_point"], meta["output_zero_point"])
                    # One fixed 126-register program followed by the terminal sequence.
                    self.assertEqual(REGISTER_COUNT, 126)
                    first_word = struct.unpack_from("<Q", payload, 0)[0]
                    self.assertEqual(first_word & 0xFFFF, REGISTERS[0][0])
                    terminal = struct.unpack_from("<4Q", payload, REGISTER_COUNT * 8)
                    self.assertEqual(list(terminal), [0x0101000000000010, 0x0101000000280014,
                                                       0x0041000000000000, 0x00810000001D0008])
                    # Weight, weight-zero-point, bias and channel-multiplier blocks.
                    aligned = (channels + 3) // 4 * 4
                    weight_offset = 0x440
                    bias_offset = weight_offset + ((kernel * kernel * aligned * 4 + 63) // 64) * 64
                    for channel in range(channels):
                        channel_weights = q.weights[channel].reshape(input_channels, kernel, kernel)
                        for kh in range(kernel):
                            for kw in range(kernel):
                                row = list(channel_weights[:, kh, kw])
                                row += [int(q.weight_zero_points[channel])] * (4 - input_channels)
                                offset = weight_offset + (kh * kernel + kw) * aligned * 4 + channel * 4
                                self.assertEqual(list(struct.unpack_from("<4b", payload, offset)), row)
                        block = bias_offset + (channel // 4) * 32
                        lane = channel % 4
                        self.assertEqual(struct.unpack_from("<i", payload, block + lane * 4)[0],
                                         int(q.biases[channel]))
                        self.assertEqual(struct.unpack_from("<h", payload, block + 16 + lane * 2)[0],
                                         -int(q.weight_zero_points[channel]))
                        self.assertEqual(struct.unpack_from("<H", payload, block + 24 + lane * 2)[0],
                                         int(q.channel_multipliers[channel]))
                    # FNV checksum over the container with the stored field zeroed.
                    checked = bytearray(data)
                    checked[84:88] = b"\0" * 4
                    self.assertEqual(struct.unpack_from("<I", data, 84)[0], checksum(checked))

    def test_affine_input_band_round_trip(self):
        model = conv_model(kernel=3, seed=4)
        payload, meta = compile_legacy(model, input_scale=0.5, input_zero_point=128)
        data = encode(payload, meta)
        info = decode(data)
        self.assertEqual(info["format_version"], 2)
        self.assertEqual((info["input_scale"], info["input_zero_point"]), (0.5, 128))
        self.assertEqual(reencode(data), data)

    def test_encode_decode_is_byte_identical(self):
        for kernel in (1, 3, 5):
            for relu in (False, True):
                with self.subTest(kernel=kernel, relu=relu):
                    data = encode(*compile_legacy(conv_model(kernel=kernel, seed=kernel, relu=relu)))
                    self.assertEqual(reencode(data), data)

    def test_rejects_corrupted_checksum_and_truncation(self):
        data = encode(*compile_legacy(conv_model(kernel=3, seed=6)))
        corrupted = bytearray(data)
        corrupted[200] ^= 1
        with self.assertRaisesRegex(ValueError, "model checksum mismatch"):
            decode(bytes(corrupted))
        with self.assertRaisesRegex(ValueError, "truncated model header"):
            decode(data[: HEADER.size - 1])
        for truncated in (data[:-1], data + b"x"):
            with self.subTest(length=len(truncated)):
                with self.assertRaisesRegex(ValueError, "incorrect model length"):
                    decode(truncated)

    def test_rejects_wrong_magic(self):
        data = bytearray(encode(*compile_legacy(conv_model(seed=7))))
        data[0:8] = b"BADMAGIC"
        with self.assertRaisesRegex(ValueError, "unsupported model format"):
            decode(bytes(data))

    def test_sequence_container_dispatch(self):
        sequence, _meta = compile_sequence_path(conv_model(kernel=1, seed=9))
        self.assertTrue(sequence[:8] == b"ORNPUSEQ")
        self.assertEqual(decode(sequence), decode_sequence(sequence))

    def test_rejects_invalid_v2_input_quantization(self):
        data = encode(*compile_legacy(conv_model(seed=10), input_scale=0.5, input_zero_point=128))
        self.assertEqual(decode(data)["format_version"], 2)
        # reserved0 carries the float32 input scale, reserved1 the UINT8 zero point.
        for offset, value in ((88, 0), (88, 0x7FC00000), (88, 0xBF800000), (92, 256)):
            with self.subTest(offset=offset, value=value):
                with self.assertRaisesRegex(ValueError, "invalid input quantization"):
                    decode(rewrite(data, offset, value))

    def test_rejects_mismatched_geometry(self):
        data = encode(*compile_legacy(conv_model(kernel=3, seed=8)))
        cases = [
            (8, 3, "unsupported model format"),          # version must be 1 or 2
            (16, 1104, "unsupported NPU target/profile"),  # target must be rv1103
            (20, 99, "unsupported NPU target/profile"),  # profile outside 1..8
            (32, 4, "unsupported tensor shape"),          # input channels must be 1 or 3
            (40, 32, "unsupported memory layout"),        # input stride disagrees with the width
            (48, 4096, "unsupported memory layout"),      # payload size is fixed at 8192
            (52, 64, "unsupported memory layout"),        # register count is fixed at 126
            (56, 0, "unsupported memory layout"),         # input offset is fixed
            (68, 4, "unsupported model options"),         # kernel must be 1, 3 or 5
            (72, 2, "unsupported model options"),         # relu flag must be 0 or 1
            (76, 128, "invalid output quantization"),     # zero point outside INT8
            (80, 0, "invalid output quantization"),       # zero output scale
        ]
        for offset, value, message in cases:
            with self.subTest(offset=offset, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    decode(rewrite(data, offset, value))

    def test_analytic_band_covers_observed_range(self):
        for kernel in (1, 3, 5):
            for relu in (False, True):
                with self.subTest(kernel=kernel, relu=relu):
                    model = conv_model(kernel=kernel, seed=kernel, relu=relu)
                    _payload, meta = compile_legacy(model)
                    weights = np.asarray(meta["float_weights"], np.float64)
                    outputs = observed_float_range(model, weights)
                    scale = meta["output_scale"]
                    zero_point = meta["output_zero_point"]
                    codes = np.rint(outputs / scale) + zero_point
                    self.assertGreaterEqual(float(codes.min()), -128.0)
                    self.assertLessEqual(float(codes.max()), 127.0)
                    observed = float(outputs.max() - outputs.min())
                    ratio = 255 * scale / observed if observed else float("inf")
                    self.assertLess(ratio, 4.0)

    def test_sequence_and_legacy_paths_agree(self):
        # Exact (0 LSB): both paths call the same `compile_model` core, so the sequence
        # wrapper must not perturb the quantization metadata or the integer reference.
        rng = np.random.default_rng(13)
        samples = rng.integers(0, 256, (16, 8, 8, 3), dtype=np.uint8)
        for kernel in (1, 3, 5):
            for relu in (False, True):
                with self.subTest(kernel=kernel, relu=relu):
                    model = conv_model(kernel=kernel, seed=kernel, relu=relu)
                    payload, legacy_meta = compile_legacy(model)
                    sequence, sequence_meta = compile_sequence_path(model)
                    legacy = decode(encode(payload, legacy_meta))
                    wrapped = decode_sequence(sequence)
                    self.assertEqual((wrapped["output_scale"], wrapped["output_zero_point"]),
                                     (legacy["output_scale"], legacy["output_zero_point"]))
                    self.assertEqual((sequence_meta["output_scale"], sequence_meta["output_zero_point"]),
                                     (legacy_meta["output_scale"], legacy_meta["output_zero_point"]))
                    legacy_q = quantization_of(legacy_meta)
                    sequence_q = quantization_of(sequence_meta)
                    for sample in samples:
                        np.testing.assert_array_equal(reference(sample, legacy_q), reference(sample, sequence_q))


if __name__ == "__main__":
    unittest.main()
