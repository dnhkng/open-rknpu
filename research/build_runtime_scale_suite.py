"""Runtime per-channel scale Mul: two external inputs of different shapes.

`Mul(image[1,3,H,W], scale[1,3,1,1])` binds the per-channel operand as a named
external tensor: the runtime packs three bytes into the 16-byte operand row that
the verified elementwise per-channel mode already reads. This is the first profile
with unequal runtime input shapes.

Independently generated commands (no RKNN, no captures). Expected bytes come from
`open_rknpu.elementwise.runtime_scale_reference`. Board runner: tests/board_io.c,
which concatenates external inputs in tensor-index order: image bytes then the
three scale bytes (operand code + 128, because the native16 packer subtracts 128).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h
from open_rknpu.elementwise import runtime_scale_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/runtime_scale_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111100)

CONFIGS = []
for height, width in ((8, 8), (5, 5), (6, 7), (5, 8)):
    for codes in ([127, 127, 127], [64, 0, -64], [-128, 127, 32], [1, 2, 3]):
        CONFIGS.append((height, width, codes))


def build(height, width):
    graph = h.make_graph(
        [h.make_node("Mul", ["image", "scale"], ["output"])], "runtime_scale",
        [h.make_tensor_value_info("image", 1, [1, 3, height, width]),
         h.make_tensor_value_info("scale", 1, [1, 3, 1, 1])],
        [h.make_tensor_value_info("output", 1, [1, 3, height, width])], [])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


manifest = []
for index, (height, width, codes) in enumerate(CONFIGS):
    path = root / f"model{index:03}.onnx"
    onnx.save(build(height, width), path)
    binary, meta = compile_sequence(path)
    path.with_suffix(".bin").write_bytes(binary)
    code_array = np.asarray(codes, np.int32)
    cases = rng.integers(0, 256, (args.cases, height, width, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    packed = bytearray()
    expected = []
    for case in cases:
        packed.extend(case.tobytes())
        packed.extend((code_array + 128).astype(np.uint8).tobytes())
        expected.append(runtime_scale_reference(case, code_array, meta["branches"][0]["quantization"]))
    (root / f"input{index:03}.u8").write_bytes(bytes(packed))
    np.stack(expected).tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "height": height, "width": width, "codes": [int(v) for v in code_array],
                     "cases": args.cases, "profile": meta["profile"],
                     "operand_scale": meta["operand_scale"], "output_scale": meta["output_scale"],
                     "output_zero_point": meta["output_zero_point"],
                     "input_bytes": [height * width * 3, 3], "output_bytes": height * width * 3})
    print(index, f"{height}x{width}", codes, "out scale", round(meta["output_scale"], 4), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} runtime-scale models in {root}")
