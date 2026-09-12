"""Walk-elementwise suite: a constant elementwise stage inside a Conv chain.

Independently generated graphs whose elementwise op (`Add`/`Sub`/`Max`/`Mul` against a
broadcast constant) sits *inside* the chain, so no profile above the op-level walk
matches them. Every model is lowered by `open_rknpu.walk.compile_chain_walk`, which
reuses the standalone elementwise emitter's 78-word DPU program with its second operand
read from a payload constant grid. Expected bytes come from the walk's own documented
integer reference (`open_rknpu.walk.chain_walk_reference`). Board runner:
`research/run_v5_suite.py` (v5 named-tensor containers).

    PYTHONPATH=src python research/build_walk_elementwise_suite.py
    PYTHONPATH=src python research/run_v5_suite.py walk_elementwise_suite
    PYTHONPATH=src python research/build_suite_readmes.py
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.accuracy import error_metrics
from open_rknpu.scheduler import compile_sequence
from open_rknpu.walk import (chain_walk_reference, compile_chain_walk, load_quantizations,
                             parse_chain)

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/walk_elementwise_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110880)


def constant_of(mode, channels, seed):
    generator = np.random.default_rng(seed)
    if mode == "scalar":
        return np.float32(0.5)
    if mode == "scalar-rank3":
        # The walk accepts a scalar as `()`, `(1,)` or `(1,1,1)`. `normalize.py` folds
        # a `()`/`(1,)` Mul into the feeding Conv's weights before dispatch, but not the
        # `(1,1,1)` form, so a scalar Mul reaches the walk by default as this shape only.
        return np.full((1, 1, 1), 0.5, np.float32)
    if mode == "per-channel":
        return generator.uniform(.2, 1.2, (1, channels, 1, 1)).astype(np.float32)
    return generator.uniform(.2, 1.2, (1, channels, 8, 8)).astype(np.float32)


def build(op, mode, pooled, seed):
    """`Conv -> op(constant) -> [MaxPool] -> Conv` on the 8x8 RGB boundary."""
    generator = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["c0"], kernel_shape=[1, 1])]
    inits = [nh.from_array(generator.uniform(-.6, .6, (8, 3, 1, 1)).astype(np.float32), "w0"),
             nh.from_array(generator.uniform(-1, 1, (8,)).astype(np.float32), "b0"),
             nh.from_array(constant_of(mode, 8, seed + 1), "k")]
    nodes.append(h.make_node(op, ["c0", "k"], ["e0"]))
    source, height, width = "e0", 8, 8
    if pooled:
        nodes.append(h.make_node("MaxPool", [source], ["p0"],
                                 kernel_shape=[2, 2], strides=[2, 2]))
        source, height, width = "p0", 4, 4
    nodes.append(h.make_node("Conv", [source, "w1", "b1"], ["output"], kernel_shape=[1, 1]))
    inits += [nh.from_array(generator.uniform(-.6, .6, (3, 8, 1, 1)).astype(np.float32), "w1"),
              nh.from_array(generator.uniform(-1, 1, (3,)).astype(np.float32), "b1")]
    graph = h.make_graph(nodes, "walk_elementwise",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, height, width])],
                         inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


# (label, op, constant mode, interior 2x2 pool). Every op is covered once per constant
# mode; four models carry the pool and the rest are bare chains. Every model either
# dispatches to the walk by default or is one of the two pinned default rejections below.
GRAPHS = [
    ("add-scalar", "Add", "scalar", False),
    ("add-per-channel", "Add", "per-channel", False),
    ("add-spatial", "Add", "spatial", False),
    ("sub-scalar-pool", "Sub", "scalar", True),
    ("sub-per-channel", "Sub", "per-channel", False),
    ("sub-spatial-pool", "Sub", "spatial", True),
    ("max-scalar", "Max", "scalar", False),
    ("max-per-channel-pool", "Max", "per-channel", True),
    ("max-spatial", "Max", "spatial", False),
    ("mul-scalar", "Mul", "scalar-rank3", False),
    ("mul-spatial-pool", "Mul", "spatial", True),
    ("mul-per-channel", "Mul", "per-channel", False),
]

# Models whose *default* `compile_sequence` outcome is the pinned depthwise rejection:
# `normalize.py` folds a per-channel `Mul` into the feeding Conv's weights, so that
# graph becomes `Conv -> Conv` and no profile accepts it. The retained container below
# is still the walk's direct lowering (the board runs the container), so the outcome is
# stable and recorded in the manifest/README rather than left to drift.
PINNED_DEFAULT_REJECTIONS = {"mul-per-channel"}


def float_targets(model, cases):
    """The float32 ONNX result for each uint8 HWC case, through `ReferenceEvaluator`."""
    evaluator = ReferenceEvaluator(model)
    name = model.graph.input[0].name
    return np.stack([evaluator.run(None, {name: case.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
                     .transpose(1, 2, 0) for case in cases])


manifest = []
report_records = []
for index, (label, op, mode, pooled) in enumerate(GRAPHS):
    model = build(op, mode, pooled, 7100 + index)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    # Every model is lowered by the walk itself; where the default dispatch reaches the
    # same walk container it must reproduce it byte for byte, and where `normalize.py`
    # pre-empts the constant the default outcome is a pinned rejection.
    binary, meta = compile_chain_walk(model)
    if meta.get("profile") != "chain-walk" or meta.get("walk_elementwise") != [op]:
        raise SystemExit(f"model{index:03} ({label}) did not lower through the walk")
    try:
        sequence_binary, routed = compile_sequence(path)
    except ValueError as error:
        sequence_binary, routed = None, str(error)
    if sequence_binary is not None and routed.get("walk_elementwise") == [op]:
        dispatch = "chain-walk"
        if sequence_binary != binary:
            raise SystemExit(f"model{index:03} ({label}) dispatch container differs from the walk")
    elif sequence_binary is None and label in PINNED_DEFAULT_REJECTIONS:
        dispatch = "pinned default rejection: " + routed
    else:
        raise SystemExit(f"model{index:03} ({label}) has no stable default outcome: {routed}")
    (root / f"model{index:03}.bin").write_bytes(binary)
    spec = parse_chain(model.graph)
    quantizations = load_quantizations(meta)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([chain_walk_reference(case, quantizations, spec["ops"])
                        for case in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    metrics = error_metrics(outputs, float_targets(model, cases),
                            meta["output_scale"], meta["output_zero_point"])
    report_records.append(dict(index=index, label=label, op=op, constant_mode=mode,
                               pooled=pooled, cases=args.cases,
                               output_scale=meta["output_scale"],
                               output_zero_point=meta["output_zero_point"],
                               float_mae=metrics["mae"], float_max_error=metrics["max_error"]))
    manifest.append(dict(index=index, label=label, op=op, constant_mode=mode,
                         constant_shape=list(np.asarray(spec["ops"][1]["constant"]).shape),
                         pooled=pooled,
                         ops=[item["kind"] for item in spec["ops"]],
                         activations=[item["activation"] for item in spec["ops"]
                                      if item["kind"] == "conv"],
                         input_shape=list(spec["input_shape"]),
                         output_shape=list(spec["output_shape"]),
                         profile=meta["profile"], dispatch=dispatch, cases=args.cases,
                         output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "accuracy_report.json").write_text(json.dumps(report_records, indent=2) + "\n")
print(f"Built {len(manifest)} walk-elementwise models in {root}")
