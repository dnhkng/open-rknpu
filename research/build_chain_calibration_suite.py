"""Chain-calibration suite: analytic versus measured bands on the native chain.

Each `Conv/Relu` chain is compiled twice through `open_rknpu.scheduler` semantics:
once with analytic bands and once with bands measured from a separate deterministic
calibration batch (`open_rknpu.calibration.measure`). The expected bytes come from the
documented integer reference (`open_rknpu.chain_n.chain_n_reference`), so the board
check compares exact INT8 execution, not accuracy. Board runner: tests/board_api.c.

    PYTHONPATH=src python research/build_chain_calibration_suite.py
    PYTHONPATH=src python research/build_suite_readmes.py
"""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.accuracy import error_metrics, layerwise_quantization_error
from open_rknpu.calibration import measure, measured_tensor_names
from open_rknpu.chain_n import chain_n_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/chain_calibration_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20240912)

# Each configuration is `[(channels, kernel), ...]`: hidden channels 3..16, the last
# layer producing the three external output channels, K1 or padded K3.
GRAPHS = {
    "chain3": [(5, 1), (5, 3), (3, 1)],
    "chain4": [(8, 3), (6, 1), (4, 3), (3, 1)],
    "chain5": [(12, 3), (10, 3), (7, 1), (4, 3), (3, 1)],
}


def chain_graph(name, layers):
    nodes = []
    tensors = []
    previous = "input"
    for index, (channels, kernel) in enumerate(layers):
        inputs = 3 if index == 0 else layers[index - 1][0]
        weight = rng.uniform(-.7, .8, (channels, inputs, kernel, kernel)).astype(np.float32)
        bias = rng.uniform(-2, 2, (channels,)).astype(np.float32)
        tensors += [nh.from_array(weight, f"w{index}"), nh.from_array(bias, f"b{index}")]
        nodes.append(h.make_node("Conv", [previous, f"w{index}", f"b{index}"], [f"c{index}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        previous = f"c{index}"
        if index < len(layers) - 1:
            nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
            previous = f"r{index}"
    graph = h.make_graph(nodes, name,
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info(previous, 1, [1, layers[-1][0], 8, 8])],
                         tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def analytic_ranges(model, meta):
    """The per-tensor scales the analytic compile chose, keyed by measured name."""
    names = measured_tensor_names(model)
    return {name: {"scale": meta["quantizations"][index]["output_scale"],
                   "zero_point": meta["quantizations"][index]["output_zero_point"]}
            for index, name in enumerate(names)}


manifest = []
report_records = []
for name, layers in GRAPHS.items():
    model = chain_graph(name, layers)
    path = root / f"{name}.onnx"
    onnx.save(model, path)
    calibration_dir = root / f"{name}_calibration"
    calibration_dir.mkdir(exist_ok=True)
    calibration = rng.integers(0, 256, (24, 3, 8, 8), dtype=np.uint8)
    np.save(calibration_dir / "cal.npy", calibration)
    # Relative paths in the published report (no maintainer-specific absolute path).
    report = measure(path, os.path.relpath(calibration_dir))
    (calibration_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Evaluation inputs are drawn after the ranges are fixed.
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    evaluator = ReferenceEvaluator(model)
    input_name = model.graph.input[0].name
    targets = np.stack([evaluator.run(None, {input_name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
                        .transpose(1, 2, 0) for x in cases])
    variants = {}
    for calibrated in (False, True):
        index = len(manifest)
        binary, meta = compile_sequence(path, calibration_ranges=report["ranges"] if calibrated else None)
        (root / f"model{index:03}.bin").write_bytes(binary)
        cases.tofile(root / f"input{index:03}.u8")
        outputs = np.stack([chain_n_reference(x, meta["quantizations"]) for x in cases])
        outputs.tofile(root / f"expected{index:03}.i8")
        metrics = error_metrics(outputs, targets, meta["output_scale"], meta["output_zero_point"])
        record = {"index": index, "graph": name, "layers": len(layers), "calibrated": calibrated,
                  "cases": args.cases,
                  "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                  "float_mae": metrics["mae"], "float_max_error": metrics["max_error"]}
        manifest.append(record)
        variants[calibrated] = (meta, outputs)
        print(record, flush=True)
    layer_analytic = layerwise_quantization_error(model, analytic_ranges(model, variants[False][0]), cases)
    layer_calibrated = layerwise_quantization_error(model, report["ranges"], cases)
    report_records.append({"graph": name, "layers": len(layers),
        "analytic": {"metrics": error_metrics(variants[False][1], targets,
                                              variants[False][0]["output_scale"],
                                              variants[False][0]["output_zero_point"]),
                     "layers": layer_analytic["layers"]},
        "calibrated": {"metrics": error_metrics(variants[True][1], targets,
                                                variants[True][0]["output_scale"],
                                                variants[True][0]["output_zero_point"]),
                       "layers": layer_calibrated["layers"]}})
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "accuracy_report.json").write_text(json.dumps(report_records, indent=2) + "\n")
print(f"Built {len(manifest)} chain-calibration models in {root}")
