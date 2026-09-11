"""Check upstream trained MNIST baseline and open frontend normalization.

Run with PYTHONPATH=src python research/verify_pretrained_mnist.py.
This is host evidence only; no claim of NPU support or dataset accuracy.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from open_rknpu.normalize import normalize_model

ROOT = Path(__file__).resolve().parent / "pretrained/mnist"


def main():
    source = json.loads((ROOT / "source.json").read_text())
    digest = hashlib.sha256((ROOT / "mnist-12.tar.gz").read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("upstream archive checksum mismatch")
    model = onnx.load(ROOT / "mnist-12/mnist-12.onnx")
    normalized = normalize_model(model)
    original_eval = ReferenceEvaluator(model)
    normalized_eval = ReferenceEvaluator(normalized)
    fixture = ROOT / "mnist-12/test_data_set_0"
    x = onnx.numpy_helper.to_array(onnx.load_tensor(fixture / "input_0.pb"))
    expected = onnx.numpy_helper.to_array(onnx.load_tensor(fixture / "output_0.pb"))
    actual = original_eval.run(None, {model.graph.input[0].name: x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
    rng = np.random.default_rng(110318)
    max_error = 0.0
    for data in [x, *[rng.uniform(x.min(), x.max(), x.shape).astype(np.float32) for _ in range(32)]]:
        a = original_eval.run(None, {model.graph.input[0].name: data})[0]
        b = normalized_eval.run(None, {model.graph.input[0].name: data})[0]
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
        max_error = max(max_error, float(np.max(np.abs(a - b))))
    onnx.save(normalized, ROOT / "normalized.onnx")
    report = dict(source, original_nodes=len(model.graph.node),
                  normalized_nodes=len(normalized.graph.node), normalization_cases=33,
                  normalization_max_abs_error=max_error,
                  upstream_max_abs_error=float(np.max(abs(actual - expected))),
                  fixture_prediction=int(actual.argmax()), fixture_expected=int(expected.argmax()),
                  input_min=float(x.min()), input_max=float(x.max()),
                  hardware_verified=False)
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
