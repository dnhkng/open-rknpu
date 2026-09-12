"""SPDX-License-Identifier: MIT

E7: the depthwise-separable classifier block, the most common mobile block.

The graph is the MobileNetV1 first separable layer on an 8x8 RGB image, a 2x2
stride-2 pool and a 1x1 classifier head:

    input [1,3,8,8] -> Conv 3x3, group=3 (depthwise, pad 1, stride 1)
                    -> Relu
                    -> Conv 1x1, 3->8 (pointwise)
                    -> Relu
                    -> MaxPool 2x2 / stride 2
                    -> Conv 1x1, 8->4 (classifier head)

Why not the classic `stem -> depthwise -> pointwise -> pool`? The dedicated
native depthwise profile (`open_rknpu.depthwise`) matches a whole graph shape:
`Conv[/Relu] -> depthwise Conv`, optionally `-> pointwise Conv`. It has no pool
task, and the terminal-pool profile accepts a single legacy Conv before the
pool, so any depthwise graph that ends in a pool is rejected with

    ValueError: depthwise sequence requires Conv[/Relu] -> depthwise Conv

(recorded in `build/report.json` under `rejected_attempts`). A depthwise Conv
that reads the graph input is instead lowered by the front end's verified
image-input depthwise rewrite (`normalize.py` turns group=C into a zero-filled
dense kernel), which lets the graph enter the board-verified op-level chain walk
(`open_rknpu.walk`, profile `chain-walk`). The classifier head after the pool is
what keeps the pool interior to the chain, and it is a real 1x1 Conv head.

The integer reference is the profile's own reference for the container that
compiled: `open_rknpu.walk.chain_walk_reference` over the quantizations the
emitter recorded and the op list `parse_chain` derives from the *normalized*
graph, which is exactly what `compile_chain_walk` consumes. Every published
`expected.i8` byte is that reference; the board compares it byte for byte.

Deterministic, host-only, no board and no network. Run from the repository root:

    PYTHONPATH=src python examples/depthwise_separable/build.py
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper as nh

from open_rknpu.normalize import normalize_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain

DEFAULT_OUT = Path(__file__).resolve().parent / "build"

FLOAT = TensorProto.FLOAT
HEIGHT = 8
WIDTH = 8
IMAGE_CHANNELS = 3      # RGB input: the depthwise group count
HIDDEN = 8              # pointwise output channels
CLASSES = 4             # classifier-head output channels
CASES = 8               # deterministic input cases (3 fixed corners + 5 random)
SEED = 110701
INPUT_SCALE = 1.0
INPUT_ZERO_POINT = 0

GRAPH = [
    "input [1,3,8,8] float32",
    "Conv 3x3 group=3 depthwise, pads 1, stride 1",
    "Relu",
    "Conv 1x1 pointwise 3->8",
    "Relu",
    "MaxPool 2x2 stride 2",
    "Conv 1x1 classifier head 8->4",
]


# --------------------------------------------------------------------------- #
# ONNX builders
# --------------------------------------------------------------------------- #
def tensor_info(name, shape):
    return h.make_tensor_value_info(name, FLOAT, list(shape))


def initializer(name, array):
    return nh.from_array(np.asarray(array, np.float32), name)


def conv_node(source, output, weights, bias, kernel, pads=None, strides=(1, 1), group=1):
    """A `Conv` with explicit symmetric padding, stride and group."""
    if pads is None:
        pads = [kernel // 2] * 4
    return h.make_node("Conv", [source, weights, bias], [output], kernel_shape=[kernel, kernel],
                       pads=list(pads), strides=list(strides), group=group)


def relu_node(source, output):
    return h.make_node("Relu", [source], [output])


def pool_node(source, output):
    """The only verified pool shape: 2x2 stride 2, no padding, no ceil mode."""
    return h.make_node("MaxPool", [source], [output], kernel_shape=[2, 2], strides=[2, 2],
                       pads=[0, 0, 0, 0])


def _model(nodes, inputs, outputs, constants, name):
    graph = h.make_graph(list(nodes), name, list(inputs), list(outputs), list(constants))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _block_constants(seed, stem):
    """Deterministic float32 weights for the stem, depthwise, pointwise and head."""
    rng = np.random.default_rng(seed)
    constants = []
    if stem:
        constants += [initializer("stem_w", rng.uniform(-0.5, 0.5, (IMAGE_CHANNELS, IMAGE_CHANNELS, 3, 3))),
                      initializer("stem_b", rng.uniform(-1.0, 1.0, IMAGE_CHANNELS))]
    constants += [initializer("dw_w", rng.uniform(-0.5, 0.5, (IMAGE_CHANNELS, 1, 3, 3))),
                  initializer("dw_b", rng.uniform(-1.0, 1.0, IMAGE_CHANNELS)),
                  initializer("pw_w", rng.uniform(-0.5, 0.5, (HIDDEN, IMAGE_CHANNELS, 1, 1))),
                  initializer("pw_b", rng.uniform(-1.0, 1.0, HIDDEN)),
                  initializer("head_w", rng.uniform(-0.5, 0.5, (CLASSES, HIDDEN, 1, 1))),
                  initializer("head_b", rng.uniform(-1.0, 1.0, CLASSES))]
    return constants


def build_model(seed=SEED):
    """The delivered graph: image depthwise -> pointwise -> 2x2 pool -> head."""
    constants = _block_constants(seed, stem=False)
    nodes = [conv_node("input", "depthwise", "dw_w", "dw_b", 3, group=IMAGE_CHANNELS),
             relu_node("depthwise", "depthwise_relu"),
             conv_node("depthwise_relu", "pointwise", "pw_w", "pw_b", 1),
             relu_node("pointwise", "pointwise_relu"),
             pool_node("pointwise_relu", "pool"),
             conv_node("pool", "output", "head_w", "head_b", 1)]
    return _model(nodes, [tensor_info("input", [1, IMAGE_CHANNELS, HEIGHT, WIDTH])],
                  [tensor_info("output", [1, CLASSES, HEIGHT // 2, WIDTH // 2])], constants,
                  "depthwise_separable")


def build_rejected_native_block(seed=SEED):
    """The classic `stem -> depthwise -> pointwise -> pool` the native profile rejects."""
    constants = _block_constants(seed + 1, stem=True)
    nodes = [conv_node("input", "stem", "stem_w", "stem_b", 3),
             relu_node("stem", "stem_relu"),
             conv_node("stem_relu", "depthwise", "dw_w", "dw_b", 3, group=IMAGE_CHANNELS),
             relu_node("depthwise", "depthwise_relu"),
             conv_node("depthwise_relu", "pointwise", "pw_w", "pw_b", 1),
             pool_node("pointwise", "output")]
    return _model(nodes, [tensor_info("input", [1, IMAGE_CHANNELS, HEIGHT, WIDTH])],
                  [tensor_info("output", [1, HIDDEN, HEIGHT // 2, WIDTH // 2])], constants,
                  "depthwise_separable_native")


def build_rejected_image_block(seed=SEED):
    """The literal `depthwise -> pointwise -> pool` graph with a terminal pool."""
    constants = _block_constants(seed + 2, stem=False)
    nodes = [conv_node("input", "depthwise", "dw_w", "dw_b", 3, group=IMAGE_CHANNELS),
             relu_node("depthwise", "depthwise_relu"),
             conv_node("depthwise_relu", "pointwise", "pw_w", "pw_b", 1),
             pool_node("pointwise", "output")]
    return _model(nodes, [tensor_info("input", [1, IMAGE_CHANNELS, HEIGHT, WIDTH])],
                  [tensor_info("output", [1, HIDDEN, HEIGHT // 2, WIDTH // 2])], constants,
                  "depthwise_separable_image")


# --------------------------------------------------------------------------- #
# Compilation and the profile's integer reference
# --------------------------------------------------------------------------- #
def compile_block(model):
    """`compile_sequence` on a saved copy; returns `(binary, meta)` verbatim."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "depthwise_separable.onnx"
        onnx.save(model, path)
        return compile_sequence(path, INPUT_SCALE, INPUT_ZERO_POINT)


def deterministic_cases(seed=SEED + 1, count=CASES):
    """`count` packed NHWC UINT8 cases with the fixed all-0, all-255 and 128 corners."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, (count, HEIGHT, WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
    values[0] = 0
    values[1] = 255
    values[2] = 128
    return values


def integer_reference(model, meta, cases):
    """The `chain-walk` reference for the container `compile_block` produced.

    `compile_chain_walk` parses the graph *after* `normalize_model` folded the
    group Conv into a dense kernel, so the reference parses the same normalized
    graph and replays the emitter's own quantizations op by op.
    """
    spec = parse_chain(normalize_model(model).graph)
    if spec is None:
        raise ValueError("the normalized graph is not a supported chain walk")
    quantizations = load_quantizations(meta)
    grids = [chain_walk_reference(case, quantizations, spec["ops"], INPUT_ZERO_POINT)
             for case in cases]
    return np.stack(grids).astype(np.int8)


def rejection_of(model):
    """The exact `ValueError` text this graph raises, or `None` if it compiles."""
    try:
        compile_block(model)
    except ValueError as error:
        return str(error)
    return None


def rejected_attempts(seed=SEED):
    """The two out-of-envelope graphs that motivate the delivered one."""
    return [dict(name="stem-depthwise-pointwise-pool", error=rejection_of(build_rejected_native_block(seed))),
            dict(name="image-depthwise-pointwise-pool", error=rejection_of(build_rejected_image_block(seed)))]


def report_payload(model, binary, meta, cases, expected):
    """The JSON-safe summary written to `build/report.json`."""
    info = decode_sequence(binary)
    return dict(
        example="depthwise_separable",
        profile=meta.get("profile"),
        graph=GRAPH,
        container_bytes=len(binary),
        format_version=info["format_version"],
        tasks=info["task_count"],
        input_shape_nhwc=[int(v) for v in info["shape_nhwc"]],
        output_shape_nhwc=[int(v) for v in info["output_shape_nhwc"]],
        input_bytes=int(info["input_bytes"]),
        output_bytes=int(info["output_bytes"]),
        input_scale=float(info["input_scale"]),
        input_zero_point=int(info["input_zero_point"]),
        output_scale=float(info["output_scale"]),
        output_zero_point=int(info["output_zero_point"]),
        cases=int(cases.shape[0]),
        expected_bytes=int(expected.size),
        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        expected_case0=expected[0].ravel().tolist(),
        expected_case1=expected[1].ravel().tolist(),
        reference="open_rknpu.walk.chain_walk_reference (profile chain-walk)",
        reference_note=("the depthwise Conv reads the graph input, so the front-end image-input "
                        "depthwise rewrite lowers it to a dense kernel and the chain walk emits it; "
                        "the native depthwise profile accepts no pool"),
        rejected_attempts=rejected_attempts(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the depthwise-separable classifier example.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: examples/depthwise_separable/build)")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    model = build_model()
    binary, meta = compile_block(model)
    cases = deterministic_cases()
    expected = integer_reference(model, meta, cases)

    (out / "prefix.bin").write_bytes(binary)
    cases.tofile(out / "inputs.u8")
    expected.tofile(out / "expected.i8")
    report = report_payload(model, binary, meta, cases, expected)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
