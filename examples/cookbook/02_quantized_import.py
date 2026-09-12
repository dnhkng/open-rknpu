"""SPDX-License-Identifier: MIT

Cookbook 2/7: importing already-quantized models (`QLinearConv` and `DQ->Conv->Q`).

A quantized model arrives with its INT8 weights already chosen. The scheduler has
two import paths for that (`open_rknpu/quantized_import.py`):

* `QLinearConv` - one node whose scale/zero-point tensors are initializers;
* `DequantizeLinear -> DequantizeLinear -> Conv -> QuantizeLinear` - the ONNX
  Q/DQ pair, with a float bias that is rounded once into accumulator units.

Both are compiled onto the native16 Conv profile with the supplied weight codes
*preserved bit for bit* - the compiler does not re-quantize them. This script
asserts that by reading the packed weights back out of the container and by
checking the integer output against an independent implementation written below:
an explicit integer accumulator, the documented two-stage requantization
(per-channel multiplier with round-half-to-even, then the multiplier/shift pair)
and saturation to INT8.

The script prints one report block per import path and a final summary line.

Run:
    PYTHONPATH=src python examples/cookbook/02_quantized_import.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from cookbook_common import build_dir, deterministic_cases, qfrom, registers
from open_rknpu.native import native_input_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence, payload_base

FOLDER = build_dir("02_quantized_import")
SEED = 20202
# Bounded single-task geometry: ic <= 32 (one 32-lane part) and oc <= 16 (one
# output block) keep the independent packing reader below small and exact.
KERNEL, IN_CHANNELS, OUT_CHANNELS, HEIGHT, WIDTH = 3, 4, 5, 8, 8
X_SCALE, X_ZP, Y_SCALE, Y_ZP = 1.0, 128, 0.02, -5


# --------------------------------------------------------------------------- #
# The two ONNX models
# --------------------------------------------------------------------------- #
def weights_and_parameters(seed, per_channel=True):
    """Source INT8 codes and their per-channel (or scalar) quantization."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(-120, 121, (OUT_CHANNELS, IN_CHANNELS, KERNEL, KERNEL)).astype(np.int8)
    if per_channel:
        scales = rng.uniform(0.02, 0.05, OUT_CHANNELS).astype(np.float32)
        zero_points = rng.integers(-3, 4, OUT_CHANNELS).astype(np.int8)
    else:
        scales = np.array([0.03], np.float32)
        zero_points = np.array([1], np.int8)
    return codes, scales, zero_points


def build_qlinear(seed):
    """One `QLinearConv`; activation scale/zero-point are scalar initializers."""
    codes, scales, zero_points = weights_and_parameters(seed)
    bias = np.random.default_rng(seed + 1).integers(-500, 500, OUT_CHANNELS).astype(np.int32)
    inputs = ["input", "xscale", "xzp", "weights", "wscale", "wzp", "yscale", "yzp", "bias"]
    initializers = [nh.from_array(codes, "weights"), nh.from_array(scales, "wscale"),
                    nh.from_array(zero_points, "wzp"), nh.from_array(np.array(X_SCALE, np.float32), "xscale"),
                    nh.from_array(np.array(X_ZP, np.uint8), "xzp"),
                    nh.from_array(np.array(Y_SCALE, np.float32), "yscale"),
                    nh.from_array(np.array(Y_ZP, np.int8), "yzp"), nh.from_array(bias, "bias")]
    node = h.make_node("QLinearConv", inputs, ["output"], kernel_shape=[KERNEL, KERNEL],
                       pads=[KERNEL // 2] * 4, strides=[1, 1])
    graph = h.make_graph([node], "qlinear_import",
                         [h.make_tensor_value_info("input", onnx.TensorProto.UINT8,
                                                   [1, IN_CHANNELS, HEIGHT, WIDTH])],
                         [h.make_tensor_value_info("output", onnx.TensorProto.INT8,
                                                   [1, OUT_CHANNELS, HEIGHT, WIDTH])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model, codes, scales, zero_points


def build_qdq(seed):
    """The ONNX Q/DQ pair: input DQ, weight DQ (axis 0), Conv, output Q."""
    codes, scales, zero_points = weights_and_parameters(seed)
    bias = np.random.default_rng(seed + 1).uniform(-1.0, 1.0, OUT_CHANNELS).astype(np.float32)
    initializers = [nh.from_array(codes, "weights"), nh.from_array(scales, "wscale"),
                    nh.from_array(zero_points, "wzp"), nh.from_array(np.array(X_SCALE, np.float32), "xscale"),
                    nh.from_array(np.array(X_ZP, np.uint8), "xzp"),
                    nh.from_array(np.array(Y_SCALE, np.float32), "yscale"),
                    nh.from_array(np.array(Y_ZP, np.int8), "yzp"), nh.from_array(bias, "bias")]
    nodes = [h.make_node("DequantizeLinear", ["input", "xscale", "xzp"], ["x_float"]),
             h.make_node("DequantizeLinear", ["weights", "wscale", "wzp"], ["w_float"], axis=0),
             h.make_node("Conv", ["x_float", "w_float", "bias"], ["y_float"],
                         kernel_shape=[KERNEL, KERNEL], pads=[KERNEL // 2] * 4, strides=[1, 1]),
             h.make_node("QuantizeLinear", ["y_float", "yscale", "yzp"], ["output"])]
    graph = h.make_graph(nodes, "qdq_import",
                         [h.make_tensor_value_info("input", onnx.TensorProto.UINT8,
                                                   [1, IN_CHANNELS, HEIGHT, WIDTH])],
                         [h.make_tensor_value_info("output", onnx.TensorProto.INT8,
                                                   [1, OUT_CHANNELS, HEIGHT, WIDTH])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model, codes, scales, zero_points


# --------------------------------------------------------------------------- #
# Independent readers / references (no open_rknpu arithmetic below this line)
# --------------------------------------------------------------------------- #
def unpack_source_weights(payload, offset, codes, zero_points):
    """Rebuild the source INT8 tensor from the packed region.

    `runtime/sequence_format.md` and `native.py` document the packing as
    `[tap][lane][member]` with `lane` a 16-wide output lane and `member` an input
    channel inside a 32-lane part, the planes grouped in output blocks of 16.
    For `ic <= 32` and `oc <= 16` (the geometry this script compiles) the region
    is exactly `k*k*lanes*oc` bytes: `tap*lanes*oc + output*lanes + member`.
    Unused members are filled with that output channel's weight zero point.
    """
    oc, ic, k, _ = codes.shape
    lanes = -(-ic // 16) * 16
    region = np.frombuffer(payload, dtype=np.int8, count=k * k * lanes * oc, offset=offset)
    rebuilt = np.zeros((oc, ic, k, k), np.int8)
    padding_exact = True
    for tap in range(k * k):
        y, x = divmod(tap, k)
        for output in range(oc):
            base = tap * lanes * oc + output * lanes
            for channel in range(ic):
                rebuilt[output, channel, y, x] = region[base + channel]
            for channel in range(ic, lanes):
                padding_exact = padding_exact and int(region[base + channel]) == int(zero_points[output])
    return rebuilt, lanes, padding_exact


def independent_quantized_conv(case, codes, zero_points, biases, channel_multipliers,
                               multiplier, shift, output_zero_point, input_zero_point,
                               pads, strides=(1, 1)):
    """Integer accumulator + requantization + saturation, written from scratch.

    `acc = sum((code - input_zp) * (weight - weight_zp)) + bias` over the receptive
    field (zero-point padding contributes nothing). The accumulator is requantized
    the way `docs/quantization.md` documents the hardware path: per-channel
    multiply then round-half-to-even at 2^14, then the multiplier/shift pair with
    its own tie correction, then saturate to [-128, 127].
    """
    height, width, ic = case.shape
    oc, _, kernel, _ = codes.shape
    top, left, bottom, right = pads
    padded = np.pad(case.astype(np.int64) - input_zero_point, ((top, bottom), (left, right), (0, 0)),
                    constant_values=0)
    centered = codes.astype(np.int64) - zero_points.astype(np.int64)[:, None, None, None]
    accumulator = np.zeros((height, width, oc), np.int64)
    for oy in range(height):
        for ox in range(width):
            patch = padded[oy:oy + kernel, ox:ox + kernel, :]
            for output in range(oc):
                accumulator[oy, ox, output] = int((patch * centered[output].transpose(1, 2, 0)).sum()) \
                    + int(biases[output])
    product = accumulator * channel_multipliers.astype(np.int64)
    channel_scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    requantized = channel_scaled * int(multiplier)
    if shift:
        requantized = (requantized + (1 << (shift - 1)) - 1 + ((requantized >> shift) & 1)) >> shift
    return np.clip(requantized + int(output_zero_point), -128, 127).astype(np.int8)


def check(name, model, codes, path):
    """Compile one import graph and assert weight survival and integer arithmetic."""
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    info = decode_sequence(binary)
    (FOLDER / ("%s.bin" % name)).write_bytes(binary)
    quantization = qfrom(meta["quantization"])

    assert "preserved bit-for-bit" in meta["quantized_import"], meta["quantized_import"]
    assert quantization.weights.shape == (OUT_CHANNELS, IN_CHANNELS * KERNEL * KERNEL)
    assert np.array_equal(np.asarray(quantization.weights, np.int64).reshape(codes.shape), codes), \
        "emitter metadata does not carry the source codes"

    # 1. The source codes survive into the container payload, bit for bit.
    task_registers = registers(binary, info)
    weight_offset, weight_bytes = task_registers[0x1110], task_registers[0x1030]
    rebuilt, lanes, padding_exact = unpack_source_weights(binary[payload_base(info):],
                                                          weight_offset, codes, quantization.weight_zero_points)
    assert weight_bytes == KERNEL * KERNEL * lanes * OUT_CHANNELS, (weight_bytes, lanes)
    assert np.array_equal(rebuilt, codes), "supplied INT8 weight bytes do not survive into the container"
    assert padding_exact, "unused lanes are not filled with the channel's weight zero point"

    # 2. The integer output matches the independent accumulator/requantizer.
    rng = np.random.default_rng(SEED + 100 + len(name))
    cases = deterministic_cases(rng, 4, HEIGHT, WIDTH, IN_CHANNELS)
    checked = 0
    for case in cases:
        profile = native_input_reference(case, quantization, int(meta["input_zero_point"]),
                                         pads=meta["conv_pads"], strides=tuple(meta["conv_strides"]),
                                         dilations=tuple(meta["conv_dilations"]))
        independent = independent_quantized_conv(
            case, quantization.weights.reshape(codes.shape),
            np.asarray(quantization.weight_zero_points), np.asarray(quantization.biases),
            np.asarray(quantization.channel_multipliers), quantization.multiplier, quantization.shift,
            int(quantization.output_zero_point), int(meta["input_zero_point"]), meta["conv_pads"])
        assert np.array_equal(profile, independent), \
            "%s: profile and independent quantized conv disagree" % name
        checked += int(profile.size)
    print("  %-16s tasks=%d container=%d B  weights %d/%d source INT8 bytes exact (offset %d)" %
          (name, info["task_count"], len(binary), int(rebuilt.size), int(codes.size), weight_offset))
    print("  %-16s padding lanes exact, integer output exact on %d cases (%d codes) vs independent conv" %
          ("", len(cases), checked))
    return dict(name=name, profile=meta["profile"], tasks=info["task_count"], bytes=len(binary),
                codes=int(codes.size), cases=len(cases), checks=checked)


def main():
    rows = []
    model, codes, _, _ = build_qlinear(SEED)
    rows.append(check("qlinear", model, codes, FOLDER / "qlinear.onnx"))
    model, codes, _, _ = build_qdq(SEED + 7)
    rows.append(check("dq_conv_q", model, codes, FOLDER / "dq_conv_q.onnx"))
    parts = ["%s(%d B, %d codes, %d exact)" % (r["name"], r["bytes"], r["codes"], r["checks"])
             for r in rows]
    print("02_quantized_import: " + " | ".join(parts) + " -> %s" % FOLDER)


if __name__ == "__main__":
    main()
