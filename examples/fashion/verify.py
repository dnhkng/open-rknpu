"""SPDX-License-Identifier: MIT — Verify pulled board outputs against ONNX reference."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--native", action="store_true")
parser.add_argument("--calibrated", action="store_true")
args = parser.parse_args()
name = "build-native-calibrated" if args.calibrated else ("build-native" if args.native else "build")
p = Path(__file__).resolve().parent / name
a = np.fromfile(p / "actual.f32", "<f4").reshape(-1, 10)
b = np.fromfile(p / "expected.f32", "<f4").reshape(-1, 10)
x = np.fromfile(p / "actual-prefix.i8", np.int8)
y = np.fromfile(p / "prefix_expected.i8", np.int8)
np.testing.assert_array_equal(x, y)
np.testing.assert_allclose(a, b, atol=0.0002, rtol=0.00002)
np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
reference = json.loads((p / "reference.json").read_text())
assert int(a[0].argmax()) == reference["original_prediction"]
report = dict(cases=len(a), exact_npu_bytes=len(x), logits=a.size,
              suffix_max_abs_error=float(abs(a-b).max()),
              fixture_prediction=int(a[0].argmax()),
              fixture_float_model_max_abs_error=float(abs(a[0]-np.array(reference["original_logits"])).max()),
              dataset_accuracy_measured=False)
(p / "verified.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
