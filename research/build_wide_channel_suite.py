"""Native16 input/output channels beyond the old C1..128 front-end cap (F10).

Independently generated graphs; expected bytes come from the documented native input
integer reference (`open_rknpu.native.native_input_reference`). Board runner:
`tests/board_api.c` via `research/run_profile_suite.py`.

The 2026-09-12 probe (`docs/investigation-log.md`, "channel-plane wall") measured both
structural walls, and this suite pins a model on each side of them:

* input: the CNA weight table groups 16-lane planes into 32-lane parts, and 511 parts is
  the largest table the driver completes. C16352 (511 parts) is byte-exact; C16368
  (512 parts) never completes (job timeout, driver soft reset), so the emitter refuses it.
* output: the surface block index `(oc-1)//16` is nine bits. C8192 (512 blocks) is
  byte-exact; C16384 writes the first 8192 channels correctly and then wrong bytes.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference
from open_rknpu.sequence import decode_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/wide_channel_suite")
parser.add_argument("--cases", type=int, default=None,
                    help="recorded inputs per model; defaults to the per-model plan")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)

# (label, ic, oc, kernel, height, width, stride, cases)
GRAPHS = [
    ("c129-in-k3", 129, 16, 3, 8, 8, 1, 8),
    ("c129-in-k3-s2", 129, 32, 3, 16, 16, 2, 4),
    ("c160-in-k1", 160, 8, 1, 10, 6, 1, 8),
    ("c192-in-k3", 192, 4, 3, 9, 9, 1, 4),
    ("c256-both-k1", 256, 256, 1, 8, 8, 1, 4),
    ("c1024-in-k1", 1024, 4, 1, 8, 8, 1, 4),
    ("c4096-in-k1", 4096, 4, 1, 8, 8, 1, 2),
    ("c8192-in-k1", 8192, 1, 1, 2, 2, 1, 2),
    ("c16352-in-k1", 16352, 1, 1, 2, 2, 1, 2),
    ("c1360-in-k3-tiled", 1360, 4, 3, 16, 16, 1, 2),
    ("c14560-in-k3", 14560, 1, 3, 1, 1, 1, 2),
    ("c144-in-k3-tall", 144, 8, 3, 40, 40, 1, 1),
    ("c8192-out-k1", 16, 8192, 1, 2, 2, 1, 2),
]


def quant_from(params):
    params = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        params[key] = np.array(params[key])
    return Quantization(**params)


manifest = []
for index, (label, ic, oc, kernel, height, width, stride, cases) in enumerate(GRAPHS):
    if args.cases is not None:
        cases = args.cases
    pad = kernel // 2
    oh = (height + 2 * pad - kernel) // stride + 1
    ow = (width + 2 * pad - kernel) // stride + 1
    # One generator per model: adding a model must not shift the others' weights.
    model_rng = np.random.default_rng(110912 + index * 7)
    w = model_rng.uniform(-.7, .8, (oc, ic, kernel, kernel)).astype(np.float32)
    b = model_rng.uniform(-2, 2, oc).astype(np.float32)
    graph = h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[kernel, kernel],
                     pads=[pad] * 4, strides=[stride, stride])],
        label.replace("-", "_"),
        [h.make_tensor_value_info("input", 1, [1, ic, height, width])],
        [h.make_tensor_value_info("output", 1, [1, oc, oh, ow])],
        [nh.from_array(w, "w"), nh.from_array(b, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    (root / f"model{index:03}.bin").write_bytes(binary)
    lanes = (ic + 15) // 16 * 16
    cases_rng = np.random.default_rng(210912 + index * 7)
    inputs = cases_rng.integers(0, 256, (cases, height, width, ic), dtype=np.uint8)
    inputs[0] = 0
    if cases > 1:
        inputs[1] = 255
    if cases > 2:
        inputs[2] = 128
    inputs.tofile(root / f"input{index:03}.u8")
    q = quant_from(meta["quantization"])
    outputs = np.stack([native_input_reference(x, q, meta["input_zero_point"],
                                               pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta["conv_dilations"]))
                        for x in inputs])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "label": label, "input_channels": ic, "output_channels": oc,
                     "kernel": kernel, "stride": stride, "height": height, "width": width,
                     "lanes": lanes, "planes": lanes // 16, "weight_parts": (lanes + 31) // 32,
                     "output_blocks": (oc + 15) // 16, "cases": cases,
                     "tasks": decode_sequence(binary)["task_count"],
                     "container_bytes": len(binary), "output_bytes": int(outputs[0].size)})
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} wide-channel models in {root}")
