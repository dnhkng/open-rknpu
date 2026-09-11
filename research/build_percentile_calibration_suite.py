"""Percentile and KL activation calibration suite (docs/plans/pipelining-plan.md P7 residual).

One `Conv 1x1 -> Relu -> Conv 1x1` graph at 8x8/C3 whose calibration activations are
heavy-tailed: the calibration inputs are mostly small with a small fraction of saturated
pixels. Four variants of the same graph are compiled through `compile_sequence(path,
calibration_ranges=...)`:

* `analytic` - no calibration, the compiler's conservative analytic range;
* `minmax` - the observed minimum and maximum, which the outliers stretch;
* `percentile` - the observed minimum with the upper 0.1% tail clipped;
* `kl` - TensorRT-style saturation: the threshold minimizing the KL divergence between
  the activation histogram and its 128-level quantization.

The accuracy table separates the two things a measured range trades against each other:
the error on the *bulk* distribution (evaluation inputs drawn like the calibration set)
and the error on *extremes* (saturated inputs). Percentile/KL give the bulk more
resolution and pay for it by clipping the rarest activations; min/max does the opposite.
Exactness (the board check) is unaffected by the choice: every variant's INT8 output is
compared against the integer reference of its own quantization.

Board runner: `research/run_profile_suite.py` (legacy v3 containers, `tests/board_api.c`).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.accuracy import error_metrics
from open_rknpu.calibration import measure
from open_rknpu.chain import native_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/percentile_calibration_suite")
parser.add_argument("--cases", type=int, default=16)
parser.add_argument("--percentile", type=float, default=99.9)
args = parser.parse_args()
root = Path(args.directory).resolve()
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110410)

hidden = 8
w1 = rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32)
w1[0] *= 20.0                                    # one long-tailed hidden channel
b1 = rng.uniform(-1, 1, (hidden,)).astype(np.float32)
w2 = rng.uniform(.05, .25, (3, hidden, 1, 1)).astype(np.float32)
b2 = rng.uniform(-1, 1, (3,)).astype(np.float32)
nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
         h.make_node("Relu", ["c1"], ["r1"]),
         h.make_node("Conv", ["r1", "w2", "b2"], ["output"], kernel_shape=[1, 1])]
graph = h.make_graph(nodes, "percentile_calibration",
    [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
    [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
    [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"),
     nh.from_array(w2, "w2"), nh.from_array(b2, "b2")])
graph.value_info.append(h.make_tensor_value_info("r1", 1, [1, hidden, 8, 8]))
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
model.ir_version = 8
path = root / "calibrated.onnx"
onnx.save(model, path)


def background(count, seed):
    return np.random.default_rng(seed).integers(0, 24, (count, 8, 8, 3), dtype=np.uint8)


def tailed(count, seed, fraction=0.002):
    """Background plus a small fraction of saturated pixels."""
    local = np.random.default_rng(seed)
    cases = local.integers(0, 24, (count, 8, 8, 3), dtype=np.uint8)
    cases[local.random(cases.shape) < fraction] = 255
    return cases


calibration_dir = root / "calibration"
calibration_dir.mkdir(exist_ok=True)
# NCHW arrays for the ONNX host reference; the runtime fixtures below are NHWC.
np.save(calibration_dir / "cal.npy", tailed(32, 110411).transpose(0, 3, 1, 2))

bulk = background(args.cases, 110412)
extreme = tailed(3, 110413, fraction=1.0)
extreme[0] = 0
extreme[1] = 255
extreme[2] = 128
cases = np.concatenate([bulk, extreme])

evaluator = ReferenceEvaluator(model)
input_name = model.graph.input[0].name


def float_reference(images):
    return np.stack([evaluator.run(None, {input_name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
                     .transpose(1, 2, 0) for x in images])


def qfrom(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


manifest, accuracy = [], []
for index, method in enumerate((None, "minmax", "percentile", "kl")):
    report = None
    if method is None:
        binary, meta = compile_sequence(path)
        label = "analytic"
    else:
        report = measure(path, calibration_dir, method=method, percentile=args.percentile)
        (root / f"calibration_report_{method}.json").write_text(json.dumps(report, indent=2) + "\n")
        binary, meta = compile_sequence(path, calibration_ranges=report["ranges"])
        label = method
    (root / f"model{index:03}.bin").write_bytes(binary)
    cases.tofile(root / f"input{index:03}.u8")
    integer = np.stack([native_reference(reference(x, qfrom(meta["first"]["quantization"])),
                                         qfrom(meta["second"])) for x in cases])
    integer.tofile(root / f"expected{index:03}.i8")
    bulk_metrics = error_metrics(integer[:len(bulk)], float_reference(bulk),
                                 meta["output_scale"], meta["output_zero_point"])
    extreme_metrics = error_metrics(integer[len(bulk):], float_reference(extreme),
                                    meta["output_scale"], meta["output_zero_point"])
    record = dict(index=index, method=label, cases=len(cases), output_bytes=int(integer[0].size),
                  output_scale=meta["output_scale"], output_zero_point=meta["output_zero_point"],
                  bulk_mae=bulk_metrics["mae"], bulk_max_error=bulk_metrics["max_error"],
                  extreme_max_error=extreme_metrics["max_error"],
                  extreme_mae=extreme_metrics["mae"])
    if report is not None:
        entry = report["ranges"][list(report["ranges"])[0]]
        record.update(measured_min=entry["min"], measured_max=entry["max"])
    if method == "percentile":
        record["percentile"] = report["percentile"]
    if method == "kl":
        record["kl_bins"] = report["bins"]
    manifest.append(record)
    accuracy.append(record)
    print(record, flush=True)

(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(root / "accuracy_report.json").write_text(json.dumps(accuracy, indent=2) + "\n")
print("built %d percentile-calibration models in %s" % (len(manifest), root))
