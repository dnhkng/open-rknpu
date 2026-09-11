"""MIT licensed. Fixed 100-image test-set sanity check; calibration is separate.

Prepare: PYTHONPATH=src python examples/mnist/sanity.py
Report after board runs: same command with --report.
The calibrated variant is scored only when sanity-results/native_calibrated.f32
exists (produced by running the build-native-calibrated prefix on the board).
"""
import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "sanity-results"
SOURCE = ROOT / "research/pretrained/mnist"
parser = argparse.ArgumentParser()
parser.add_argument("--report", action="store_true")
args = parser.parse_args()
OUT.mkdir(exist_ok=True)
if not args.report:
    def load(name, digest):
        data = (SOURCE / "test-data" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("dataset checksum mismatch")
        return gzip.decompress(data)
    images = load("t10k-images-idx3-ubyte.gz", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6")
    labels = load("t10k-labels-idx1-ubyte.gz", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6")
    assert struct.unpack_from(">4I", images) == (2051, 10000, 28, 28)
    assert struct.unpack_from(">2I", labels) == (2049, 10000)
    indices = np.random.default_rng(110309).choice(10000, 100, replace=False)
    x = np.frombuffer(images, np.uint8, offset=16).reshape(10000, 1, 28, 28)[indices].astype(np.float32) / 255
    y = np.frombuffer(labels, np.uint8, offset=8)[indices]
    meta = json.loads((HERE / "build/reference.json").read_text())
    q = np.clip(np.rint(x/meta["input_scale"])+meta["input_zero_point"], 0, 255).astype(np.uint8)
    q.tofile(OUT / "inputs.u8")
    dequant = (q.astype(np.float32)-meta["input_zero_point"])*np.float32(meta["input_scale"])
    model = onnx.load(SOURCE / "normalized.onnx")
    evaluator = ReferenceEvaluator(model)
    preds = lambda batch: [int(evaluator.run(None, {model.graph.input[0].name:a[None]})[0].argmax()) for a in batch]
    folders = [d for d in ("build", "build-native", "build-native-calibrated") if (HERE/d/"prefix.bin").exists()]
    report = dict(count=100, seed=110309, indices=indices.tolist(), labels=y.tolist(),
                  preprocessing="uint8 / 255, NCHW, per upstream README",
                  calibration_or_tuning=False, float_predictions=preds(x),
                  float_with_quantized_input_predictions=preds(dequant),
                  input_scale=meta["input_scale"], input_zero_point=meta["input_zero_point"],
                  encoded_input_unique_values=np.unique(q).tolist(),
                  model_hashes={d:hashlib.sha256((HERE/d/"prefix.bin").read_bytes()).hexdigest() for d in folders})
    (OUT / "reference.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Prepared 100 fixed test images; input codes:", report["encoded_input_unique_values"])
else:
    report = json.loads((OUT / "reference.json").read_text())
    truth = np.array(report["labels"])
    predictions = {"float":np.array(report["float_predictions"]),
                   "float_quantized_input":np.array(report["float_with_quantized_input_predictions"])}
    modes = [m for m in ("hybrid", "native", "native_calibrated") if (OUT/(m+".f32")).exists()]
    for mode in modes:
        logits = np.fromfile(OUT / (mode+".f32"), "<f4")
        assert logits.size == 1000 and np.isfinite(logits).all()
        predictions[mode] = logits.reshape(100,10).argmax(1)
    # Reproduce each offloaded variant from the independent integer model, so a
    # board mismatch is attributed to quantization/hardware, not to a missing step.
    from open_rknpu.scheduler import compile_sequence
    from open_rknpu.quantization import Quantization, reference
    from open_rknpu.chain import native_reference
    from native import extend
    binary, meta = compile_sequence(HERE / "build/prefix.onnx", report["input_scale"], report["input_zero_point"])
    q1 = Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta["quantization"].items()})
    model = onnx.load(SOURCE / "normalized.onnx")
    g = model.graph
    weights = {t.name:onnx.numpy_helper.to_array(t) for t in g.initializer}
    w2, b2 = weights[g.node[3].input[1]], weights[g.node[3].input[2]]
    variants = {}
    emitted, q2 = extend(binary, w2, b2)
    assert hashlib.sha256(emitted).hexdigest() == report["model_hashes"]["build-native"]
    variants["native"] = q2
    calibrated_path = HERE / "build-native-calibrated/reference.json"
    if "build-native-calibrated" in report["model_hashes"]:
        calibrated_range = json.loads(calibrated_path.read_text())["native_conv2_output_range"]
        emitted_cal, q2_cal = extend(binary, w2, b2, calibrated_range)
        assert hashlib.sha256(emitted_cal).hexdigest() == report["model_hashes"]["build-native-calibrated"]
        variants["native_calibrated"] = q2_cal
    split = g.node[3].output[0]
    info = next(v for v in g.value_info if v.name==split)
    suffix = onnx.helper.make_model(onnx.helper.make_graph(list(g.node[4:]), "suffix", [info], list(g.output), list(g.initializer)), opset_imports=list(model.opset_import))
    suffix.ir_version = model.ir_version
    evaluator = ReferenceEvaluator(suffix)
    images_in = np.fromfile(OUT / "inputs.u8",np.uint8).reshape(100,28,28,1)
    for name, q2v in variants.items():
        expected = []
        for image in images_in:
            first = reference(image,q1).reshape(14,2,14,2,8).max(axis=(1,3))
            second = native_reference(first,q2v,meta["output_zero_point"])
            activation = (second.astype(np.float32)-q2v.output_zero_point)*np.float32(q2v.output_scale)
            expected.append(evaluator.run(None,{split:activation.transpose(2,0,1)[None]})[0].ravel())
        actual = np.fromfile(OUT / (name+".f32"), "<f4").reshape(100,10)
        np.testing.assert_allclose(actual,np.array(expected),atol=0.0002,rtol=0.00002)
        report[f"{name}_integer_reference_max_logit_error"] = float(abs(actual-np.array(expected)).max())
        report[f"{name}_conv2_output_scale"] = q2v.output_scale
    results = {k:dict(correct=int((p==truth).sum()), agreement_with_float=int((p==predictions["float"]).sum()),
                     prediction_histogram=np.bincount(p, minlength=10).tolist()) for k,p in predictions.items()}
    report["results"] = results
    report["board_predictions"] = {k:predictions[k].tolist() for k in modes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(results, indent=2))
