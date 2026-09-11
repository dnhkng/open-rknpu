"""Fit the K5 phase arithmetic for the retained depthwise ConvTranspose probe.

Reconstructs the K5 quantization meta (the public emitter currently rejects K5),
then scores candidate accumulator/bias/rounding models against the retained board
output `/tmp/m5k_out.i8`. Development-only: the public emitter never reads this.

    PYTHONPATH=src python research/analyze_k5_phase.py
"""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.chain import native_quantize
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "transpose_k5_suite"
ACTUAL = Path("/tmp/m5k_out.i8")


def reconstruct(index=0):
    model = onnx.load(SUITE / f"model{index:03}.onnx")
    graph = model.graph
    stem, trans = graph.node[0], graph.node[1]
    attrs = {a.name: h.get_attribute_value(a) for a in trans.attribute}
    import tempfile
    sub = h.make_model(h.make_graph([stem], "stem", list(graph.input),
        [h.make_tensor_value_info(stem.output[0], 1, [1, 3, 8, 8])],
        [t for t in graph.initializer if t.name in stem.input[1:]]),
        opset_imports=list(model.opset_import))
    sub.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stem.onnx"
        onnx.save(sub, path)
        _, first = compile_sequence(path)
    q1 = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                         for k, v in first["quantization"].items()})
    constants = {t.name: nh.to_array(t) for t in graph.initializer}
    weights = constants[trans.input[1]]
    bias = constants[trans.input[2]]
    q = native_quantize(weights, bias, first["output_scale"], first["output_zero_point"], None, symmetric=True)
    return q1, q, attrs, weights, bias


def requantize(acc, q, stage1, stage2):
    product = acc * q.channel_multipliers
    if stage1 == "ties_even":
        scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    elif stage1 == "half_up":
        scaled = (product + 8192) >> 14
    elif stage1 == "floor":
        scaled = (product + 8191) >> 14
    elif stage1 == "ties_away":
        scaled = np.where(product >= 0, (product + 8192) >> 14, (product + 8191) >> 14)
    else:
        raise ValueError(stage1)
    product = scaled * q.multiplier
    if q.shift:
        if stage2 == "ties_even":
            product = product + (1 << (q.shift - 1)) - 1 + ((product >> q.shift) & 1)
        elif stage2 == "half_up":
            product = product + (1 << (q.shift - 1))
        elif stage2 == "floor":
            product = product + (1 << (q.shift - 1)) - 1
        else:
            raise ValueError(stage2)
        result = (product >> q.shift) + q.output_zero_point
    else:
        result = product + q.output_zero_point
    return np.clip(result, -128, 127).astype(np.int8)


def candidate(x, q1, q, weights, attrs, correction, stage1, stage2, flip):
    strides = attrs["strides"]
    pads = attrs["pads"]
    oh = 7 * strides[0] + 5 - pads[0] - pads[2]
    ow = 7 * strides[1] + 5 - pads[1] - pads[3]
    kernel = q.weights.reshape(3, 5, 5)
    if flip:
        kernel = kernel[:, ::-1, ::-1]
    a = reference(x, q1).astype(np.int64) - q1.output_zero_point
    if correction == "all_taps":
        base = q.biases + q1.output_zero_point * q.weights.sum(axis=1)
    elif correction == "none":
        base = q.biases.copy()
    elif correction == "in_range":
        base = q.biases + q1.output_zero_point * q.weights.sum(axis=1)
    else:
        raise ValueError(correction)
    acc = np.broadcast_to(base, (oh, ow, 3)).copy()
    for iy in range(8):
        for ix in range(8):
            for ky in range(5):
                for kx in range(5):
                    oy = iy * strides[0] + ky - pads[0]
                    ox = ix * strides[1] + kx - pads[1]
                    if 0 <= oy < oh and 0 <= ox < ow:
                        acc[oy, ox] += a[iy, ix] * kernel[:, ky, kx]
    return requantize(acc, q, stage1, stage2)


def main():
    q1, q, attrs, weights, bias = reconstruct()
    samples = np.fromfile(SUITE / "input000.u8", np.uint8).reshape(-1, 8, 8, 3)
    actual = np.frombuffer(ACTUAL.read_bytes(), np.int8).reshape(-1, 15, 15, 3)
    print("samples", len(samples), "actual shape", actual.shape,
          "q1 zp", q1.output_zero_point, "q shift", q.shift)
    results = []
    for correction in ("all_taps", "none"):
        for stage1 in ("ties_even", "half_up", "floor", "ties_away"):
            for stage2 in ("ties_even", "half_up", "floor"):
                for flip in (False, True):
                    bad = 0
                    maxdelta = 0
                    for sample, truth in zip(samples, actual):
                        got = candidate(sample, q1, q, weights, attrs, correction, stage1, stage2, flip)
                        delta = got.astype(int) - truth.astype(int)
                        bad += int(np.count_nonzero(delta))
                        maxdelta = max(maxdelta, int(np.abs(delta).max()))
                    results.append((bad, maxdelta, correction, stage1, stage2, flip))
    results.sort()
    for row in results[:10]:
        print("mismatch=%5d maxdelta=%d correction=%s stage1=%s stage2=%s flip=%s" % row)


if __name__ == "__main__":
    main()
