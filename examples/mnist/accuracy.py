"""Held-out accuracy for the MNIST variants from the integer references.

This separates integer exactness (verify.py / the board) from float-model
accuracy. Predictions come from the documented quantized integer computation
followed by the ONNX suffix; the calibrated variant uses the measured Conv2
output range recorded by `build.py --native --calibrate`.

Run from the repository root after `sanity.py` preparation and the three builds:

    PYTHONPATH=src python examples/mnist/accuracy.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization, reference
from open_rknpu.chain import native_quantize, native_reference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

SOURCE = ROOT / "research/pretrained/mnist"
model = onnx.load(SOURCE / "normalized.onnx")
g = model.graph
weights = {t.name: nh.to_array(t) for t in g.initializer}
inputs = np.fromfile(HERE / "sanity-results/inputs.u8", np.uint8).reshape(100, 28, 28, 1)
reference_report = json.loads((HERE / "sanity-results/reference.json").read_text())
labels = np.array(reference_report["labels"])


def subgraph(nodes, info):
    m = h.make_model(h.make_graph(list(nodes), "suffix", [info], list(g.output), list(g.initializer)),
                     opset_imports=list(model.opset_import))
    m.ir_version = model.ir_version
    return ReferenceEvaluator(m)


pool_info = next(v for v in g.value_info if v.name == g.node[2].output[0])
conv2_info = next(v for v in g.value_info if v.name == g.node[3].output[0])
suffix_hybrid = subgraph(g.node[3:], pool_info)
suffix_native = subgraph(g.node[4:], conv2_info)

binary, meta = compile_sequence(HERE / "build/prefix.onnx",
                                reference_report["input_scale"], reference_report["input_zero_point"])
q1 = Quantization(**{k: np.array(v) if isinstance(v, list) else v for k, v in meta["quantization"].items()})
stem_scale, stem_zp = meta["output_scale"], meta["output_zero_point"]
pooled = np.stack([reference(image, q1).reshape(14, 2, 14, 2, 8).max(axis=(1, 3)) for image in inputs])
pooled_float = (pooled.astype(np.float32) - stem_zp) * np.float32(stem_scale)


def logits_from(second_integer, q2):
    result = []
    for activation in second_integer:
        values = (activation.astype(np.float32) - q2.output_zero_point) * np.float32(q2.output_scale)
        result.append(suffix_native.run(None, {g.node[3].output[0]: values.transpose(2, 0, 1)[None]})[0].ravel())
    return np.array(result)


hybrid = np.array([suffix_hybrid.run(None, {g.node[2].output[0]: a.transpose(2, 0, 1)[None]})[0].ravel()
                   for a in pooled_float])
q2_analytic = native_quantize(weights[g.node[3].input[1]], weights[g.node[3].input[2]], stem_scale, stem_zp)
native_analytic = logits_from([native_reference(a, q2_analytic, stem_zp) for a in pooled], q2_analytic)
calibrated_range = json.loads((HERE / "build-native-calibrated/reference.json").read_text())["native_conv2_output_range"]
q2_calibrated = native_quantize(weights[g.node[3].input[1]], weights[g.node[3].input[2]], stem_scale, stem_zp,
                                calibrated_range)
native_calibrated = logits_from([native_reference(a, q2_calibrated, stem_zp) for a in pooled], q2_calibrated)

variants = {"float": np.array(reference_report["float_predictions"]),
            "float_quantized_input": np.array(reference_report["float_with_quantized_input_predictions"]),
            "hybrid": hybrid, "native": native_analytic, "native_calibrated": native_calibrated}
float_reference = np.array(reference_report["float_predictions"])
results = {}
for name, logits in variants.items():
    predictions = logits.argmax(1) if logits.ndim == 2 and logits.shape[1] == 10 else logits
    results[name] = {"correct": int((predictions == labels).sum()), "accuracy": float((predictions == labels).mean()),
                     "agreement_with_float": int((predictions == float_reference).sum()),
                     "prediction_histogram": np.bincount(predictions, minlength=10).tolist()}
report = {"count": len(labels), "seed": 110309, "calibration_samples": 256,
          "conv2_output_scales": {"analytic": q2_analytic.output_scale, "calibrated": q2_calibrated.output_scale},
          "results": results}
(HERE / "sanity-results/accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
