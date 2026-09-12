# SPDX-License-Identifier: MIT
"""`Concat(axis=1)` of sibling Conv branches stacked into one wide Conv.

Independently generated graphs with two or three Conv branches that read the same graph
input with identical geometry. `open_rknpu.normalize` stacks their weights on the
output-channel axis and their biases on the channel axis, deletes the branches and the
`Concat`, and leaves one wide Conv; the generator asserts the emitted container is
byte-identical to the equivalent hand-written wide Conv. Expected bytes come from the
native input profile's own documented integer reference
(`open_rknpu.native.native_input_reference`), because every merged graph has more than 16
output channels and therefore routes to `native16-input`. Board runner: `tests/board_api.c`.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/wide_concat_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)

# (label, branch output channels, input channels, kernel, branches carry a bias)
GRAPHS = [
    ("concat2-c3-o10-10-k1", (10, 10), 3, 1, False),
    ("concat2-c3-o10-10-k3", (10, 10), 3, 3, True),
    ("concat3-c3-o8-8-8-k1", (8, 8, 8), 3, 1, True),
    ("concat3-c1-o8-8-8-k3", (8, 8, 8), 1, 3, True),
    ("concat2-c1-o12-12-k3", (12, 12), 1, 3, False),
    ("concat2-c3-o10-10-k3-bias-free", (10, 10), 3, 3, False),
    ("concat3-c3-o6-8-10-k1", (6, 8, 10), 3, 1, True),
    ("concat2-c3-o9-12-k3", (9, 12), 3, 3, False),
    ("concat3-c1-o10-10-4-k3", (10, 10, 4), 1, 3, True),
    ("concat2-c3-o11-11-k3", (11, 11), 3, 3, True),
    ("concat2-c1-o10-10-k1", (10, 10), 1, 1, True),
    ("concat3-c3-o7-7-7-k3", (7, 7, 7), 3, 3, True),
]


def quant_from(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


def branch_parts(splits, in_channels, kernel, seed):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-.8, .8, (oc, in_channels, kernel, kernel)).astype(np.float32),
             rng.uniform(-1, 1, oc).astype(np.float32)) for oc in splits]


def build(spec, seed):
    label, splits, in_channels, kernel, biases = spec
    parts = branch_parts(splits, in_channels, kernel, seed)
    nodes, initializers, concat_inputs = [], [], []
    for index, (oc, (weights, bias)) in enumerate(zip(splits, parts)):
        inputs = ["input", "w%d" % index] + (["b%d" % index] if biases else [])
        nodes.append(h.make_node("Conv", inputs, ["c%d" % index],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        initializers.append(nh.from_array(weights, "w%d" % index))
        if biases:
            initializers.append(nh.from_array(bias, "b%d" % index))
        concat_inputs.append("c%d" % index)
    nodes.append(h.make_node("Concat", concat_inputs, ["output"], axis=1))
    graph = h.make_graph(
        nodes, label.replace("-", "_"),
        [h.make_tensor_value_info("input", 1, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, sum(splits), 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def wide(spec, seed):
    label, splits, in_channels, kernel, biases = spec
    parts = branch_parts(splits, in_channels, kernel, seed)
    weights = np.concatenate([value for value, _ in parts], axis=0)
    initializers = [nh.from_array(weights, "w")]
    inputs = ["input", "w"]
    if biases:
        initializers.append(nh.from_array(np.concatenate([b for _, b in parts]), "b"))
        inputs.append("b")
    nodes = [h.make_node("Conv", inputs, ["output"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    graph = h.make_graph(
        nodes, "wide",
        [h.make_tensor_value_info("input", 1, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, sum(splits), 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


manifest = []
for index, spec in enumerate(GRAPHS):
    label, splits, in_channels, kernel, biases = spec
    seed = 110550 + index
    model = build(spec, seed)
    onnx.save(model, root / f"model{index:03}.onnx")
    binary, meta = compile_sequence(root / f"model{index:03}.onnx")
    if meta.get("profile") != "native16-input":
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}, "
                         f"not native16-input")
    expected_container, _ = compile_sequence(wide(spec, seed))
    if binary != expected_container:
        raise SystemExit(f"model{index:03} ({label}) is not byte-identical to the wide Conv")
    (root / f"model{index:03}.bin").write_bytes(binary)
    rng = np.random.default_rng(seed)
    cases = rng.integers(0, 256, (args.cases, 8, 8, in_channels), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    quantization = quant_from(meta["quantization"])
    outputs = np.stack([native_input_reference(case, quantization, meta["input_zero_point"],
                                               pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta.get("conv_dilations", (1, 1))))
                        for case in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label, branches=list(splits),
                         input_channels=in_channels, output_channels=sum(splits),
                         kernel=kernel, biases=biases, input_shape=[1, 8, 8, in_channels],
                         output_shape=[1, 8, 8, sum(splits)], profile="native16-input",
                         cases=args.cases, output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} wide-Concat models in {root}")
