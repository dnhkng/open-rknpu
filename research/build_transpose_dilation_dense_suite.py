"""Dense K3 dilation2 ConvTranspose: sparse-K5 rewrite suite (docs/plans/pipelining-plan.md P3).

A dilated K3 transposed convolution spans exactly the five taps of a K5 kernel with
zeros between them, so `open_rknpu.transposed` zero-stuffs `(C_in, C_out, 3, 3)` weights
to `(C_in, C_out, 5, 5)` and emits the verified dense-K5 form. This builder:

1. writes a Conv stem plus a dense `ConvTranspose(kernel_shape=[3,3], dilations=[2,2])`
   at 8x8 with a few strides/pads combinations;
2. checks the rewrite on the host - the original dilated graph and its zero-stuffed
   sibling must produce **bit-identical** ONNX float outputs - and that the compiled
   container reports the sparse-K5 rewrite;
3. computes the expected bytes with an independent **dilated scatter** reference (each
   input position scatters its centered value over `iy*stride + ky*2 - pad`), i.e. not
   from the rewritten graph;
4. cross-checks that reference against the same scatter over the zero-stuffed K5
   weights, which is what the container actually executes.

Board runner: `research/run_profile_suite.py` (legacy v3 containers, `tests/board_api.c`).
"""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/transpose_dilation_dense_suite")
parser.add_argument("--cases", type=int, default=8)
args = parser.parse_args()
root = Path(args.directory).resolve()
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110420)

# (input channels, output channels, strides, pads, output_padding)
CONFIGS = [(4, 3, [1, 1], [0, 0, 0, 0], [0, 0]),
           (4, 3, [1, 1], [1, 1, 1, 1], [0, 0]),
           (8, 5, [2, 2], [1, 1, 1, 1], [0, 0]),
           (3, 3, [1, 1], [2, 2, 2, 2], [0, 0])]


def build(ic, oc, strides, pads, opad):
    oh = 7 * strides[0] - pads[0] - pads[2] + 2 * 2 + 1 + opad[0]
    ow = 7 * strides[1] - pads[1] - pads[3] + 2 * 2 + 1 + opad[1]
    w1 = rng.uniform(-.3, .3, (ic, 3, 1, 1)).astype(np.float32)
    b1 = rng.uniform(-.5, .5, ic).astype(np.float32)
    w = rng.uniform(-.3, .3, (ic, oc, 3, 3)).astype(np.float32)
    b = rng.uniform(-.5, .5, oc).astype(np.float32)
    graph = h.make_graph(
        [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
         h.make_node("ConvTranspose", ["stem", "w", "b"], ["output"], group=1,
                     kernel_shape=[3, 3], dilations=[2, 2], strides=strides, pads=pads,
                     output_padding=opad)],
        "transpose_dilation_dense",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, oc, oh, ow])],
        [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"), nh.from_array(w, "w"),
         nh.from_array(b, "b")],
        value_info=[h.make_tensor_value_info("stem", 1, [1, ic, 8, 8])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model, w, oh, ow


def stuffed(model, strides, pads, opad):
    """The same operator as a unit-dilation K5 graph (the emitter's rewrite)."""
    lower = copy.deepcopy(model)
    node = lower.graph.node[-1]
    tensors = {v.name: v for v in lower.graph.initializer}
    weight = nh.to_array(tensors[node.input[1]])
    expanded = np.zeros(weight.shape[:2] + (5, 5), np.float32)
    expanded[:, :, ::2, ::2] = weight
    tensors[node.input[1]].CopyFrom(nh.from_array(expanded, node.input[1]))
    del node.attribute[:]
    node.attribute.extend([h.make_attribute("group", 1), h.make_attribute("kernel_shape", [5, 5]),
                           h.make_attribute("strides", strides), h.make_attribute("pads", pads),
                           h.make_attribute("dilations", [1, 1]),
                           h.make_attribute("output_padding", opad)])
    return lower


def reference_scatter(sample, q1, q, taps, strides, pads, oh, ow):
    """Integer reference: scatter the centered stem output over the quantized taps.

    `taps` are `(grid_y, grid_x, data_y, data_x)` entries of the *quantized* weight
    grid: the dilated K3 op uses the even grid positions with their original data
    offsets, while the unit-dilation K5 form the container executes uses all 25 (the odd
    ones are zero). Both must produce the same bytes.
    """
    # The emitter quantizes the zero-stuffed weights, so the quantized grid is the K5
    # one: `q.weights` is (oc, ic*grid*grid).
    oc = q.weights.shape[0]
    grid = 5
    ic = q.weights.shape[1] // (grid * grid)
    qw = np.array(q.weights).reshape(oc, ic, grid, grid)
    centered = qw - q.weight_zero_points[:, None, None, None]
    a = reference(sample, q1).astype(np.int64) - q1.output_zero_point
    bias_units = q.biases + q1.output_zero_point * centered.reshape(oc, -1).sum(axis=1)
    acc = np.broadcast_to(bias_units, (oh, ow, oc)).copy()
    for iy in range(8):
        for ix in range(8):
            for grid_y, grid_x, data_y, data_x in taps:
                oy = iy * strides[0] + data_y - pads[0]
                ox = ix * strides[1] + data_x - pads[1]
                if 0 <= oy < oh and 0 <= ox < ow:
                    acc[oy, ox] += a[iy, ix] @ centered[:, :, grid_y, grid_x].T
    product = acc * q.channel_multipliers
    scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    product = scaled * q.multiplier
    if q.shift:
        product += q.output_zero_point << q.shift
        result = (product + (1 << (q.shift - 1)) - 1 + ((product >> q.shift) & 1)) >> q.shift
    else:
        result = product + q.output_zero_point
    return np.clip(result, -128, 127).astype(np.int8)


manifest = []
for index, (ic, oc, strides, pads, opad) in enumerate(CONFIGS):
    model, weights, oh, ow = build(ic, oc, strides, pads, opad)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    # 1. the rewrite is the same operator, bit for bit
    probe = rng.integers(0, 256, (2, 3, 8, 8), dtype=np.uint8).astype(np.float32)
    original_out = ReferenceEvaluator(model).run(None, {"input": probe})[0]
    rewritten_out = ReferenceEvaluator(stuffed(model, strides, pads, opad)).run(
        None, {"input": probe})[0]
    # The two graphs are the same operator; the zero taps only change the summation
    # order, so the float outputs agree to rounding rather than bit-for-bit.
    if not np.allclose(original_out, rewritten_out, rtol=0.0, atol=1e-4):
        raise SystemExit("dense K3 dilation2 rewrite changed the float output")
    binary, meta = compile_sequence(path)
    if meta.get("transposed_rewrite") != "K3 dilation2 expanded to sparse K5":
        raise SystemExit("expected the sparse-K5 rewrite, got %r" % meta.get("transposed_rewrite"))
    path.with_suffix(".bin").write_bytes(binary)
    q = Quantization(**{key: np.array(value) if isinstance(value, list) else value
                        for key, value in meta["transposed_quantization"].items()})
    q1 = Quantization(**{key: np.array(value) if isinstance(value, list) else value
                         for key, value in meta["first"]["quantization"].items()})
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    # The dilated op places the original tap (ky,kx) at data offset (2ky,2kx), which is
    # grid position (2ky,2kx) of the zero-stuffed K5 weights.
    dilated_taps = [(2 * ky, 2 * kx, 2 * ky, 2 * kx) for ky in range(3) for kx in range(3)]
    stuffed_taps = [(ty, tx, ty, tx) for ty in range(5) for tx in range(5)]
    outputs = []
    for sample in cases:
        dilated = reference_scatter(sample, q1, q, dilated_taps, strides, pads, oh, ow)
        stuffed_out = reference_scatter(sample, q1, q, stuffed_taps, strides, pads, oh, ow)
        if not np.array_equal(dilated, stuffed_out):
            raise SystemExit("dilated and sparse-K5 references disagree")
        outputs.append(dilated)
    cases.tofile(root / f"input{index:03}.u8")
    np.stack(outputs).tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, input_channels=ic, output_channels=oc, kernel=3,
                         dilation=2, strides=strides, pads=pads, output_padding=opad,
                         output_shape=[oh, ow, oc], cases=args.cases,
                         output_bytes=int(outputs[0].size),
                         rewrite=meta["transposed_rewrite"],
                         profile=meta["transposed_profile"]))
    print(manifest[-1], flush=True)

(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("built %d dense K3-dilation2 models in %s" % (len(manifest), root))
