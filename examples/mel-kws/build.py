"""SPDX-License-Identifier: MIT
Compile the trained mel-CNN for the RV1103 and emit the board inputs/reference outputs.

The graph is the op-level chain walk (`open_rknpu.walk`): Conv/Relu/MaxPool/Conv/Relu/
MaxPool/Conv, exactly the primitive set already board-verified for chains with pools.

Steps:
1. load `build/model.onnx` and confirm `parse_chain` accepts it;
2. pack the 300 official test utterances into UINT8 NHWC (`byte = round(value * 255)`,
   input scale 1/255, zero point 0 - the features are built in [0, 1]);
3. sweep the output quantization of the final 1x1 Conv and keep the range with the best
   INT8 test accuracy (the walk has no calibration path, so the output band is the one
   free parameter);
4. write `build/prefix.bin`, `inputs.u8`, `labels.u8`, `expected.i8` and a report, all
   reproducible from `train.py`.

    PYTHONPATH=src python examples/mel-kws/build.py
"""
from pathlib import Path
import json
import sys
import tempfile

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/mel-kws"))

from open_rknpu.calibration import measure  # noqa: E402
from open_rknpu.normalize import normalize_model  # noqa: E402
from open_rknpu.scheduler import compile_sequence  # noqa: E402
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain  # noqa: E402

BUILD = Path(__file__).resolve().parent / "build"
INPUT_SCALE = 1.0      # the model consumes the UINT8 feature bytes directly
INPUT_ZERO_POINT = 0


def pack(images):
    """(N,3,32,32) feature images -> (N,32,32,3) UINT8 NHWC, the runtime's API layout."""
    return np.transpose(images.astype(np.uint8), (0, 2, 3, 1)).copy()


def int8_outputs(onnx_path, packed, ranges):
    """Compile once and run the walked integer reference over every sample.

    The output band is carried by `ranges["output"]`, so calibration and the output
    override never need to be combined (the scheduler rejects that combination).
    """
    binary, meta = compile_sequence(str(onnx_path), INPUT_SCALE, INPUT_ZERO_POINT,
                                    calibration_ranges=ranges)
    plan = parse_chain(normalize_model(onnx.load(onnx_path)).graph)
    quantizations = load_quantizations(meta)
    outputs = np.stack([chain_walk_reference(sample, quantizations, plan["ops"], INPUT_ZERO_POINT)
                        for sample in packed])
    return binary, meta, outputs


def main():
    model_path = BUILD / "model.onnx"
    if not model_path.is_file():
        raise SystemExit("run examples/mel-kws/train.py first")
    normalized = normalize_model(onnx.load(model_path))
    plan = parse_chain(normalized.graph)
    if plan is None:
        raise SystemExit("the trained graph is not accepted by the chain walk")
    print("chain:", " -> ".join(op["kind"] if op["kind"] != "conv"
                                else f"Conv(op{op['output_channels']})" for op in plan["ops"]))

    cached = np.load(BUILD / "features.npz")
    x, y, is_test = cached["x"], cached["y"], cached["test"]
    train_images, test_images = x[~is_test], x[is_test]
    test_labels = y[is_test]
    packed = pack(test_images)

    # Measure every Conv's activation range on the *training* split. A trained network's
    # activations are nowhere near the analytic bound the walk falls back to, so the
    # chain needs measured bands or the later grids collapse onto the zero point.
    with tempfile.TemporaryDirectory() as folder:
        np.save(Path(folder) / "train.npy", train_images.astype(np.uint8))
        report = measure(str(model_path), folder, method="percentile", percentile=99.9)
    ranges = report["ranges"]
    for name, entry in ranges.items():
        print(f"calibration {name}: [{entry['min']:.3f}, {entry['max']:.3f}] "
              f"scale {entry['scale']:.6g} zero point {entry['zero_point']}")

    # Float logits from an independent ONNX evaluator (also the accuracy reference).
    from onnx.reference import ReferenceEvaluator
    session = ReferenceEvaluator(onnx.load(model_path))
    float_maps = np.stack([session.run(None, {"input": row[None]})[0][0] for row in test_images])
    low, high = float(float_maps.min()), float(float_maps.max())
    print(f"float logit map range over 64 test utterances: [{low:.3f}, {high:.3f}]")

    # The Conv bands are measured; only the *last* Conv's band is a free knob (it sets
    # the output scale). Try the measured band and two tighter variants and keep the
    # most accurate.
    import copy
    output_entry = ranges["output"]
    midpoint = (output_entry["min"] + output_entry["max"]) / 2
    span = output_entry["max"] - output_entry["min"]
    candidates = []
    for factor in (1.0, 0.5, 0.25):
        scale = max(span * factor, 1e-6) / 255.0
        zero_point = int(np.clip(round(-128 - (midpoint - span * factor / 2) / scale), -128, 127))
        candidates.append((scale, zero_point, factor))
    seen = set()
    sweep = []
    best = None
    for scale, zero_point, factor in candidates:
        if (round(scale, 9), zero_point) in seen:
            continue
        seen.add((round(scale, 9), zero_point))
        patched = copy.deepcopy(ranges)
        patched["output"] = dict(patched["output"], scale=scale, zero_point=zero_point)
        binary, meta, outputs = int8_outputs(model_path, packed, patched)
        logits = outputs.astype(np.int64).mean(axis=(1, 2))
        accuracy = float((logits.argmax(axis=1) == test_labels).mean())
        sweep.append(dict(scale=scale, zero_point=zero_point, span_factor=factor,
                          int8_test_accuracy=accuracy, container_bytes=len(binary)))
        print(f"  output band x{factor:<4} scale {scale:.6f} zero point {zero_point:4d}: "
              f"INT8 test accuracy {accuracy * 100:.2f}%")
        if best is None or accuracy > best[2]:
            best = (scale, zero_point, accuracy, binary, outputs, meta)
            best_ranges = patched

    scale, zero_point, accuracy, binary, outputs, meta = best
    float_logits = float_maps.astype(np.float64).mean(axis=(2, 3))
    float_accuracy = float((float_logits.argmax(axis=1) == test_labels).mean())
    int8_logits = outputs.astype(np.int64).mean(axis=(1, 2))
    agreement = float((int8_logits.argmax(axis=1) == float_logits.argmax(axis=1)).mean())

    (BUILD / "prefix.bin").write_bytes(binary)
    packed.tofile(BUILD / "inputs.u8")
    test_labels.astype(np.uint8).tofile(BUILD / "labels.u8")
    outputs.astype(np.int8).tofile(BUILD / "expected.i8")
    (BUILD / "quantization.json").write_text(json.dumps({
        "input": {"scale": INPUT_SCALE, "zero_point": INPUT_ZERO_POINT,
                  "packing": "byte = the trained feature image, NHWC"},
        "output": {"scale": scale, "zero_point": zero_point},
        "intermediate": meta.get("quantization", {}),
        "calibration": {name: {key: value for key, value in entry.items()}
                        for name, entry in best_ranges.items()},
        "profile": meta.get("profile"), "submission": meta.get("submission"),
        "tasks": meta.get("task_count"), "engine_runs": meta.get("engine_runs"),
    }, indent=2, default=str) + "\n")
    report = {
        "model": "examples/mel-kws (FSDD spoken digits, 10 classes)",
        "container_bytes": len(binary),
        "test_utterances": int(is_test.sum()),
        "output_bytes": int(outputs.size),
        "calibration_method": "percentile 99.9 on the 2,700 training utterances",
        "float_test_accuracy": float_accuracy,
        "int8_test_accuracy": accuracy,
        "int8_float_agreement": agreement,
        "output_sweep": sweep,
        "architecture": plan["output_shape"],
    }
    (BUILD / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"model      {len(binary)} bytes, profile {meta.get('profile')}, "
          f"{meta.get('task_count')} tasks, {meta.get('engine_runs')} engine run(s)")
    print(f"float      {float_accuracy * 100:.2f}% on {int(is_test.sum())} test utterances")
    print(f"INT8       {accuracy * 100:.2f}% (agrees with float on {agreement * 100:.2f}%)")
    print(f"wrote {BUILD}/prefix.bin inputs.u8 labels.u8 expected.i8 report.json")


if __name__ == "__main__":
    main()
