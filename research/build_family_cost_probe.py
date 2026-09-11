"""Per-family task cost probe (S5): matched CNA, pool and elementwise containers.

The plan's S5 residual asked for a per-family cost table but the verified tree has no
comparable single-family containers, so this probe builds them at one geometry:

* `conv`      - one 1x1 Conv C3->C3 at 8x8, one CNA task;
* `conv_pool` - the *same* Conv weights and bias followed by a 2x2 stride-2 MaxPool,
                two tasks (CNA + DPU pool); the difference against `conv` isolates the
                pool task;
* `ew2`       - a copy of `runtime_scale_suite/model000` (the verified per-channel
                runtime-scale profile), which is two DPU elementwise tasks, so its cost
                per task is the elementwise family cost directly.

Expected bytes come from the library references (conv requantisation, a 2x2 max pool,
and `runtime_scale_reference` for the copied model).
"""
from pathlib import Path
import json
import shutil

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "family_cost_probe"
OUT.mkdir(exist_ok=True)
CASES = 16
rng = np.random.default_rng(20260910)

WEIGHT = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32)
BIAS = rng.uniform(-2, 2, (3,)).astype(np.float32)


def conv_model(pool):
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1])]
    output = "conv"
    if pool:
        nodes.append(h.make_node("MaxPool", ["conv"], ["output"], kernel_shape=[2, 2],
                                 strides=[2, 2]))
        output = "output"
    graph = h.make_graph(nodes, "conv_pool" if pool else "conv",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(output, 1, [1, 3, 4 if pool else 8, 4 if pool else 8])],
        [nh.from_array(WEIGHT, "w"), nh.from_array(BIAS, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def quantize(meta):
    values = dict(meta["quantization"])
    for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


def main():
    cases = rng.integers(0, 256, (CASES, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    manifest = []
    for index, (name, pool) in enumerate((("conv", False), ("conv_pool", True))):
        path = OUT / f"model{index:03}.onnx"
        onnx.save(conv_model(pool), path)
        binary, meta = compile_sequence(path)
        (OUT / f"model{index:03}.bin").write_bytes(binary)
        cases.tofile(OUT / f"input{index:03}.u8")
        grid = quantize(meta)
        expected = []
        for case in cases:
            codes = reference(case, grid)
            if pool:
                blocked = codes.astype(np.int64).reshape(4, 2, 4, 2, 3)
                codes = blocked.max(axis=(1, 3)).astype(np.int8)
            expected.append(codes)
        np.stack(expected).tofile(OUT / f"expected{index:03}.i8")
        manifest.append(dict(index=index, name=name, tasks=meta.get("register_count") or 1,
                             cases=CASES, output_bytes=int(np.stack(expected).nbytes // CASES)))
        print(manifest[-1], flush=True)
    # Elementwise family: the verified runtime-scale profile (two DPU tasks).
    source = ROOT / "runtime_scale_suite"
    for prefix, suffix in (("model", "bin"), ("input", "u8"), ("expected", "i8")):
        shutil.copyfile(source / f"{prefix}000.{suffix}", OUT / f"{prefix}002.{suffix}")
    manifest.append(dict(index=2, name="ew2", tasks=2, cases=CASES, output_bytes=192,
                         note="copy of runtime_scale_suite/model000 (two DPU tasks)"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built", len(manifest), "family cost models in", OUT)


if __name__ == "__main__":
    main()
