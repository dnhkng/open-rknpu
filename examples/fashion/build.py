"""SPDX-License-Identifier: MIT
Build the trained Fashion-MNIST hybrid example using only open host tools.
Run from repository root with PYTHONPATH=src python examples/mnist/build.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization, reference

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research" / "pretrained" / "fashion-mnist"
OUT = Path(__file__).resolve().parent / "build"


def main():
    path = SOURCE / "normalized.onnx"
    # Pin the trained companion model by hash and by the pinned op sequence.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pinned = (SOURCE / "model.sha256").read_text().split()[0]
    if digest != pinned:
        raise ValueError("This example requires the trained Fashion-MNIST model recorded in model.sha256")
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true", help="offload Conv2 using the native14 profile")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate the native Conv2 output range from a held-out calibration split")
    args = parser.parse_args()
    if args.calibrate and not args.native:
        raise ValueError("--calibrate requires --native")
    if args.native:
        OUT = OUT.with_name("build-native-calibrated" if args.calibrate else "build-native")
    OUT.mkdir(exist_ok=True)
    model = onnx.load(path)
    g = model.graph
    split = g.node[2].output[0]
    split_info = next(v for v in g.value_info if v.name == split)
    def subgraph(nodes, inputs, outputs):
        m = h.make_model(h.make_graph(nodes, "mnist", inputs, outputs, list(g.initializer)),
                         opset_imports=list(model.opset_import))
        m.ir_version = model.ir_version
        return m
    prefix = subgraph(list(g.node[:3]), list(g.input), [split_info])
    suffix = subgraph(list(g.node[3:]), [split_info], list(g.output))
    onnx.save(prefix, OUT / "prefix.onnx")
    # No ONNX fixture ships with the locally trained model: take the input range from the
    # first test image, which is in [0, 1] like the training data. The dataset is fetched,
    # not vendored, so fall back to a documented analytic range when it is absent - the
    # compiled prefix is identical either way, only the chosen input band changes.
    import gzip
    dataset = SOURCE / "test-data" / "t10k-images-idx3-ubyte.gz"
    if dataset.is_file():
        raw = gzip.decompress(dataset.read_bytes())
        first = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28)[0].astype(np.float32) / 255
        x = first[None, None]
        scale = float(np.float32((x.max() - x.min()) / 255)) or 1 / 255
        zp = int(np.clip(np.rint(-x.min() / scale), 0, 255))
    else:
        print("Fashion-MNIST test data not found at %s; using the documented analytic\n"
              "scale 1/255, zero point 0 input band. Run\n"
              "`python examples/fetch_idx.py --dataset fashion` to derive it from the data."
              % dataset.relative_to(ROOT))
        x = np.zeros((1, 1, 28, 28), np.float32)
        scale, zp = 1 / 255, 0
    binary, meta = compile_sequence(OUT / "prefix.onnx", scale, zp)
    (OUT / "prefix.bin").write_bytes(binary)
    q = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                        for k, v in meta["quantization"].items()})
    rng = np.random.default_rng(110308)
    inputs = rng.integers(0, 256, (8, 28, 28, 1), dtype=np.uint8)
    inputs[0] = np.clip(np.rint(x[0].transpose(1, 2, 0) / scale) + zp, 0, 255).astype(np.uint8)
    inputs[1] = zp
    inputs[2] = 0
    inputs[3] = 255
    activations = np.stack([reference(v, q).reshape(14, 2, 14, 2, 8).max(axis=(1, 3)) for v in inputs])
    dequant = (activations.astype(np.float32) - meta["output_zero_point"]) * np.float32(meta["output_scale"])
    weights = {t.name: nh.to_array(t) for t in g.initializer}
    output_range = None
    if args.calibrate:
        import gzip
        import struct
        def load(name, digest):
            data = (SOURCE / "test-data" / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("dataset checksum mismatch")
            return gzip.decompress(data)
        images = load("t10k-images-idx3-ubyte.gz",
                      "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073")
        assert struct.unpack_from(">4I", images) == (2051, 10000, 28, 28)
        held = np.random.default_rng(110309).choice(10000, 100, replace=False)
        candidates = np.setdiff1d(np.arange(10000), held)
        indices = np.random.default_rng(110340).choice(candidates, 256, replace=False)
        calibration = (np.frombuffer(images, np.uint8, offset=16)
                       .reshape(10000, 1, 28, 28)[indices].astype(np.float32) / 255)
        encoded = np.clip(np.rint(calibration / scale) + zp, 0, 255).astype(np.uint8)
        pooled = np.stack([reference(v.transpose(1, 2, 0), q).reshape(14, 2, 14, 2, 8).max(axis=(1, 3))
                           for v in encoded])
        activations_float = ((pooled.astype(np.float32) - meta["output_zero_point"])
                             * np.float32(meta["output_scale"]))
        node = g.node[3]
        conv2 = h.make_model(h.make_graph([node], "conv2",
            [h.make_tensor_value_info(node.input[0], 1, [1, 8, 14, 14])],
            [h.make_tensor_value_info(node.output[0], 1, [1, 16, 14, 14])],
            [t for t in g.initializer if t.name in node.input[1:]]), opset_imports=list(model.opset_import))
        conv2.ir_version = model.ir_version
        evaluate = ReferenceEvaluator(conv2)
        low = high = 0.0
        for activation in activations_float:
            values = evaluate.run(None, {node.input[0]: activation.transpose(2, 0, 1)[None]})[0]
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
        cal_scale = float(np.float32((high - low) / 255)) or 1.0
        cal_zp = int(np.clip(np.rint(-128 - low / cal_scale), -128, 127))
        output_range = {"scale": cal_scale, "zero_point": cal_zp}
        print(json.dumps({"conv2_calibration_samples": len(indices), "low": low, "high": high,
                          "output_range": output_range}, indent=2))
    if args.native:
        from native import extend
        from open_rknpu.chain import native_reference
        binary, q2 = extend(binary, weights[g.node[3].input[1]], weights[g.node[3].input[2]], output_range)
        activations = np.stack([native_reference(a, q2, meta["output_zero_point"]) for a in activations])
        meta["output_scale"], meta["output_zero_point"] = q2.output_scale, q2.output_zero_point
        dequant = (activations.astype(np.float32) - q2.output_zero_point) * np.float32(q2.output_scale)
        split = g.node[3].output[0]
        split_info = next(v for v in g.value_info if v.name == split)
        suffix = subgraph(list(g.node[4:]), [split_info], list(g.output))
        (OUT / "prefix.bin").write_bytes(binary)
    evaluator = ReferenceEvaluator(suffix)
    expected = np.concatenate([evaluator.run(None, {split: a.transpose(2, 0, 1)[None]})[0] for a in dequant])
    inputs.tofile(OUT / "inputs.u8")
    activations.tofile(OUT / "prefix_expected.i8")
    expected.astype("<f4").tofile(OUT / "expected.f32")
    weights = {t.name: nh.to_array(t) for t in g.initializer}
    header = ["/* Generated trained model weights; upstream Apache-2.0 license applies. */"]
    for name, array in [("conv_w", weights[g.node[3].input[1]]),
                        ("conv_b", weights[g.node[3].input[2]]),
                        ("dense_w", weights[g.node[7].input[1]]),
                        ("dense_b", weights[g.node[8].input[1]])]:
        values = ",".join(float(v).hex() + "f" for v in array.ravel())
        header.append(f"static const float {name}[{array.size}] = {{{values}}};")
    (OUT / "weights.h").write_text("\n".join(header) + "\n")
    original = ReferenceEvaluator(model).run(None, {g.input[0].name: x})[0].ravel()
    report = dict(cases=len(inputs), placement={"NPU": ["Conv1", "Relu1", "MaxPool1"],
                  "CPU": ["Conv2", "Relu2", "MaxPool2", "Reshape", "MatMul", "Add"]},
                  original_logits=original.tolist(), hybrid_reference_logits=expected[0].tolist(),
                  original_prediction=int(original.argmax()), hybrid_prediction=int(expected[0].argmax()),
                  fixture_max_logit_error=float(abs(original - expected[0]).max()),
                  input_scale=scale, input_zero_point=zp,
                  output_scale=meta["output_scale"], output_zero_point=meta["output_zero_point"],
                  dataset_accuracy_measured=False)
    if args.native:
        report["placement"]["NPU"].append("Conv2")
        report["placement"]["CPU"].remove("Conv2")
    if output_range is not None:
        report["native_conv2_calibrated"] = True
        report["native_conv2_output_range"] = output_range
    (OUT / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
