"""SPDX-License-Identifier: MIT

Cookbook 3/7: mutable packed parameters (`ornpu_set_constant`).

`docs/plans/pipelining-plan.md` and `docs/container-format.md` describe the v4
container: alongside the task table it carries *named packed constant regions*.
`native.py` emits `conv.parameters` (kind 1, the whole packed weight + bias +
per-channel conversion region) when compiled with `mutable_weights=True`, and the
constant-Mul profile emits `mul.factor` (kind 3) with `mutable_constants=True`.

On the board the runtime replaces one complete region, never a sub-field:

    ornpu_model *model;
    ornpu_constant_info constant;
    ornpu_open("model.bin", &model);
    ornpu_get_constant(model, 0, &constant);            /* byte_offset, bytes, kind */
    ornpu_set_constant(model, 0, packed_region, constant.bytes);
    ornpu_run(model, input, info.input_bytes, output, info.output_bytes);
    ornpu_close(model);

The region must be exactly `constant.bytes` long and must have been packed for the
*container's* band: the program's multiplier/shift/zero-point registers are part
of the task program, not of the region, so a replacement has to share the band.

This script demonstrates the semantics on the host by rebuilding the payload:
it compiles a Conv twice with the same quantization band but different packed
weights (the kernel taps are mirrored, which preserves every per-channel
min/max, so both containers have byte-identical task programs), copies the second
region into the first container with `ornpu_set_constant`'s copy + re-checksum,
and asserts the result is byte-identical to the second container and that the
integer output changed as expected. It repeats the container-level check for the
`mul.factor` variant.

Run:
    PYTHONPATH=src python examples/cookbook/03_mutable_parameters.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from onnx import helper as h, numpy_helper as nh

from cookbook_common import (build_dir, checksum_masked, constant_region, conv_node, initializer,
                             model_graph, qfrom, splice_constant, tensor_info)
from open_rknpu.native import native_input_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence, payload_base

FOLDER = build_dir("03_mutable_parameters")
SEED = 30303
IN_CHANNELS, OUT_CHANNELS, KERNEL, SIZE = 4, 6, 3, 8


# --------------------------------------------------------------------------- #
# Conv: mutable_weights
# --------------------------------------------------------------------------- #
def build_conv(seed, mirror):
    """A Conv that routes to native16; `mirror` flips the kernel taps in y and x."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.4, 0.4, (OUT_CHANNELS, IN_CHANNELS, KERNEL, KERNEL)).astype(np.float32)
    if mirror:
        weights = weights[:, :, ::-1, ::-1].copy()
    bias = rng.uniform(-0.3, 0.3, OUT_CHANNELS).astype(np.float32)
    node = conv_node("input", "output", "w", "b", KERNEL, pads=[KERNEL // 2] * 4, strides=(1, 1))
    return model_graph([node], "mutable_conv", [tensor_info("input", [1, IN_CHANNELS, SIZE, SIZE])],
                       [tensor_info("output", [1, OUT_CHANNELS, SIZE, SIZE])],
                       [initializer("w", weights), initializer("b", bias)])


def conv_demo():
    """Two same-band containers; the second region replaces the first."""
    first, first_meta = compile_sequence(build_conv(SEED, False), mutable_weights=True)
    second, second_meta = compile_sequence(build_conv(SEED, True), mutable_weights=True)
    first_info, second_info = decode_sequence(first), decode_sequence(second)
    assert first_info["format_version"] == 4 and second_info["format_version"] == 4
    descriptor, region = constant_region(first, first_info, "conv.parameters")
    _, replacement = constant_region(second, second_info, "conv.parameters")
    assert descriptor["kind"] == 1 and region != replacement

    # Same band => identical task programs, so only the named region differs.
    end = payload_base(first_info) + descriptor["offset"]
    assert checksum_masked(first[:end]) == checksum_masked(second[:end]), \
        "mirrored taps changed the task program; the band must be identical"
    spliced = splice_constant(first, first_info, "conv.parameters", replacement)
    assert spliced == second, "splice_constant did not reproduce the second container"

    first_q, second_q = qfrom(first_meta["quantization"]), qfrom(second_meta["quantization"])
    assert first_q.output_scale == second_q.output_scale
    assert first_q.output_zero_point == second_q.output_zero_point
    case = np.random.default_rng(SEED + 1).integers(0, 256, (SIZE, SIZE, IN_CHANNELS), dtype=np.uint8)
    before = native_input_reference(case, first_q, int(first_meta["input_zero_point"]))
    after = native_input_reference(case, second_q, int(second_meta["input_zero_point"]))
    assert not np.array_equal(before, after), "replacing the weights did not change the output"
    delta = int(np.abs(before.astype(int) - after.astype(int)).max())
    (FOLDER / "conv_before.bin").write_bytes(first)
    (FOLDER / "conv_after.bin").write_bytes(spliced)
    print("  conv.parameters  kind=%d offset=%d bytes=%d; program bytes identical, splice == rebuilt container" %
          (descriptor["kind"], descriptor["offset"], descriptor["size"]))
    print("  conv weights replace : output changed on %d/%d codes, max |delta| = %d LSB" %
          (int((before != after).sum()), before.size, delta))
    return dict(name="conv.parameters", offset=descriptor["offset"], size=descriptor["size"],
                changed=int((before != after).sum()))


# --------------------------------------------------------------------------- #
# Mul: mutable_constants
# --------------------------------------------------------------------------- #
def build_mul(factor):
    """One `Mul` by a per-channel constant (the profile requires RGB 5..8)."""
    node = h.make_node("Mul", ["input", "factor"], ["output"])
    graph = h.make_graph([node], "mutable_mul", [tensor_info("input", [1, 3, SIZE, SIZE])],
                         [tensor_info("output", [1, 3, SIZE, SIZE])],
                         [nh.from_array(np.asarray(factor, np.float32).reshape(1, 3, 1, 1), "factor")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def mul_demo():
    """The v4 `mul.factor` descriptor and its packed INT8 codes."""
    factor_a = np.array([0.5, -1.25, 2.0], np.float32)
    factor_b = np.array([2.0, 0.5, -1.25], np.float32)
    first, first_meta = compile_sequence(build_mul(factor_a), mutable_constants=True)
    second, second_meta = compile_sequence(build_mul(factor_b), mutable_constants=True)
    first_info, second_info = decode_sequence(first), decode_sequence(second)
    assert first_info["format_version"] == 4
    descriptor, region = constant_region(first, first_info, "mul.factor")
    _, replacement = constant_region(second, second_info, "mul.factor")
    assert descriptor["kind"] == 3 and descriptor["size"] == 16 and region != replacement

    end = payload_base(first_info) + descriptor["offset"]
    assert checksum_masked(first[:end]) == checksum_masked(second[:end]), \
        "the two Mul containers share one program apart from the factor region"
    spliced = splice_constant(first, first_info, "mul.factor", replacement)
    assert spliced == second, "splice_constant did not reproduce the second Mul container"

    # Independent check of the packed codes: the profile stores [round(factor/scale)]
    # for the three channels in the first 16-byte lane block.
    scale = float(first_meta["constant_scale"])
    assert scale == second_meta["constant_scale"], "the two factors must share one scale"
    expected_a = np.clip(np.rint(factor_a / scale), -128, 127).astype(np.int8)
    codes_a = np.frombuffer(region, np.int8)[:3]
    assert np.array_equal(codes_a, expected_a), (codes_a.tolist(), expected_a.tolist())
    codes_b = np.frombuffer(replacement, np.int8)[:3]
    assert np.array_equal(codes_b, np.roll(codes_a, 1)), "replacement codes are not the rotated factor"
    assert not np.array_equal(codes_a, codes_b)
    (FOLDER / "mul_before.bin").write_bytes(first)
    (FOLDER / "mul_after.bin").write_bytes(spliced)
    print("  mul.factor       kind=%d offset=%d bytes=%d; program bytes identical, splice == rebuilt container" %
          (descriptor["kind"], descriptor["offset"], descriptor["size"]))
    print("  mul factor codes : %s -> %s (real factor %s -> %s)" %
          (codes_a.tolist(), codes_b.tolist(), factor_a.tolist(), factor_b.tolist()))
    return dict(name="mul.factor", offset=descriptor["offset"], size=descriptor["size"],
                changed=3)


def main():
    rows = [conv_demo(), mul_demo()]
    print("03_mutable_parameters: " + " | ".join(
        "%s@%d+%d" % (row["name"], row["offset"], row["size"]) for row in rows) +
        " -> %s" % FOLDER)


if __name__ == "__main__":
    main()
