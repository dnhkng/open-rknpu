"""Full 10,000-image MNIST accuracy through the board, chunked to fit flash.

Streams test images to the board in chunks, runs the calibrated both-Conv NPU
prefix, pulls logits, and reports accuracy against the float model. Uses the
existing board deployment at /userdata/open-npu-research/mnist-calibrated.

    PYTHONPATH=src python examples/mnist/full_dataset.py
"""
import argparse
import gzip
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "research/pretrained/mnist"
OUT = HERE / "sanity-results"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
BOARD = "/userdata/open-npu-research/mnist-calibrated"

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, default=1000)
parser.add_argument("--prefix", default="calibrated_prefix.bin")
parser.add_argument("--local-only", action="store_true", help="skip board runs and score the integer reference")
parser.add_argument("--limit", type=int, default=10000)
args = parser.parse_args()

def adb(*arguments, timeout=180):
    result = subprocess.run([ADB, "-s", SERIAL, *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace"))
    return result.stdout

def load(name, digest):
    data = (SOURCE / "test-data" / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("dataset checksum mismatch")
    return gzip.decompress(data)

images = load("t10k-images-idx3-ubyte.gz", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6")
labels = load("t10k-labels-idx1-ubyte.gz", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6")
assert struct.unpack_from(">4I", images) == (2051, 10000, 28, 28)
assert struct.unpack_from(">2I", labels) == (2049, 10000)
x = np.frombuffer(images, np.uint8, offset=16).reshape(10000, 1, 28, 28).astype(np.float32) / 255
y = np.frombuffer(labels, np.uint8, offset=8)
meta = json.loads((HERE / "build/reference.json").read_text())
encoded = np.clip(np.rint(x / meta["input_scale"]) + meta["input_zero_point"], 0, 255).astype(np.uint8).reshape(10000, 28, 28, 1)
if args.prefix == "calibrated_prefix.bin":
    local_prefix = HERE / "build-native-calibrated/prefix.bin"
    expected = hashlib.sha256(local_prefix.read_bytes()).hexdigest()
    remote = adb("shell", f"sha256sum {BOARD}/{args.prefix}").decode().split()[0]
    if remote != expected:
        raise SystemExit(f"board prefix hash {remote} != local {expected}")

logits = np.zeros((10000, 10), np.float32)
LIMIT = min(args.limit, 10000)
if args.local_only:
    from open_rknpu.chain import native_quantize, native_reference
    from open_rknpu.quantization import Quantization, reference
    from open_rknpu.scheduler import compile_sequence
    binary, prefix_meta = compile_sequence(HERE / "build/prefix.onnx", meta["input_scale"], meta["input_zero_point"])
    q1 = Quantization(**{k: np.array(v) if isinstance(v, list) else v for k, v in prefix_meta["quantization"].items()})
    model = onnx.load(SOURCE / "normalized.onnx")
    g = model.graph
    weights = {t.name: onnx.numpy_helper.to_array(t) for t in g.initializer}
    calibrated_range = json.loads((HERE / "build-native-calibrated/reference.json").read_text())["native_conv2_output_range"]
    q2 = native_quantize(weights[g.node[3].input[1]], weights[g.node[3].input[2]],
                         prefix_meta["output_scale"], prefix_meta["output_zero_point"], calibrated_range)
    info = next(v for v in g.value_info if v.name == g.node[3].output[0])
    suffix = onnx.helper.make_model(onnx.helper.make_graph(list(g.node[4:]), "suffix", [info], list(g.output), list(g.initializer)),
                                    opset_imports=list(model.opset_import))
    suffix.ir_version = model.ir_version
    evaluator = ReferenceEvaluator(suffix)
    for start in range(0, LIMIT, args.chunk):
        stop = min(start + args.chunk, LIMIT)
        for offset, image in enumerate(encoded[start:stop]):
            first = reference(image, q1).reshape(14, 2, 14, 2, 8).max(axis=(1, 3))
            second = native_reference(first, q2, prefix_meta["output_zero_point"])
            activation = (second.astype(np.float32) - q2.output_zero_point) * np.float32(q2.output_scale)
            logits[start + offset] = evaluator.run(None, {g.node[3].output[0]: activation.transpose(2, 0, 1)[None]})[0].ravel()
        print(f"reference {stop}/{LIMIT}", flush=True)
else:
    chunk_input = OUT / "full_chunk.u8"
    chunk_logits = OUT / "full_chunk.f32"
    for start in range(0, LIMIT, args.chunk):
        stop = min(start + args.chunk, LIMIT)
        encoded[start:stop].tofile(chunk_input)
        adb("push", str(chunk_input), f"{BOARD}/chunk.u8")
        adb("shell", f"cd {BOARD} && ./mnist-run {args.prefix} chunk.u8 chunk.f32 /dev/null")
        adb("pull", f"{BOARD}/chunk.f32", str(chunk_logits))
        values = np.fromfile(chunk_logits, "<f4")
        logits[start:stop] = values.reshape(stop - start, 10)
        print(f"board {stop}/{LIMIT}", flush=True)
    adb("shell", f"rm -f {BOARD}/chunk.u8 {BOARD}/chunk.f32")

predictions = logits.argmax(1)
float_model = onnx.load(SOURCE / "normalized.onnx")
evaluator = ReferenceEvaluator(float_model)
float_predictions = np.zeros(10000, dtype=np.int64)
# The pinned model reshapes to a fixed [1,256], so the float reference runs one
# image at a time.
for start in range(LIMIT):
    outputs = evaluator.run(None, {float_model.graph.input[0].name: x[start:start + 1]})[0]
    float_predictions[start] = int(np.asarray(outputs).reshape(-1, 10).argmax(1)[0])
    if (start + 1) % 2000 == 0:
        print(f"float {start + 1}/{LIMIT}", flush=True)
report = {"count": int(LIMIT), "prefix": args.prefix, "chunk": args.chunk, "local_only": args.local_only,
          "board_accuracy": float((predictions[:LIMIT] == y[:LIMIT]).mean()),
          "float_accuracy": float((float_predictions[:LIMIT] == y[:LIMIT]).mean()),
          "agreement_with_float": int((predictions[:LIMIT] == float_predictions[:LIMIT]).sum()),
          "correct": int((predictions[:LIMIT] == y[:LIMIT]).sum()),
          "prediction_histogram": np.bincount(predictions[:LIMIT], minlength=10).tolist()}
(OUT / f"full_dataset_{args.prefix.replace('.bin', '')}_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
