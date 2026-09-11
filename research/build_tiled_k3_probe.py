"""K > 1 height-strip tiling suite (docs/plans/pipelining-plan.md S7).

Two `[Conv, Relu]*(N-1) + [Conv]` chains whose strips need halo rows:

* `k3`    16 layers of padded 3x3 (halo 1 on every layer);
* `mixed` 12 layers alternating 3x3 and 1x1 with varying hidden channels, so both
  the shared-surface and the no-halo geometry run inside one container.

For each chain the builder writes the untiled serial container (the control and the
source of the expected bytes) plus tiled twins: `tiles=2` and `tiles=4`, each in serial
and in one linked batched job. Every variant is compiled from the same emitter
quantization chain, so all of them are checked against one integer reference.

The board runner is `run_tiled_k3_probe.py`; evidence lands in `board_results.txt` /
`board_results.json` next to the containers.
"""
from pathlib import Path
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.chain_n import chain_n_reference_layers
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.tiled_chain import compile_tiled_chain

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tiled_k3_probe"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260911)

CONFIGS = {
    "k3": [(6, 3)] * 15 + [(3, 3)],
    "mixed": [(4, 3), (6, 1), (8, 3), (10, 1), (10, 3), (8, 1),
              (6, 3), (4, 1), (3, 3), (3, 1), (3, 3), (3, 1)],
}
CASES = 4


def chain_model(layers):
    nodes, tensors, previous = [], [], "input"
    for index, (channels, kernel) in enumerate(layers):
        inputs = 3 if index == 0 else layers[index - 1][0]
        w = rng.uniform(-.7, .8, (channels, inputs, kernel, kernel)).astype(np.float32)
        b = rng.uniform(-2, 2, (channels,)).astype(np.float32)
        tensors += [nh.from_array(w, f"w{index}"), nh.from_array(b, f"b{index}")]
        nodes.append(h.make_node("Conv", [previous, f"w{index}", f"b{index}"], [f"c{index}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        previous = f"c{index}"
        if index < len(layers) - 1:
            nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
            previous = f"r{index}"
    graph = h.make_graph(nodes, "tiled_k3_chain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(previous, 1, [1, layers[-1][0], 8, 8])], tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def main():
    manifest = []
    inputs = rng.integers(0, 256, (CASES, 8, 8, 3), dtype=np.uint8)
    inputs[0] = 0
    inputs[1] = 255
    inputs[2] = 128
    for name, layers in CONFIGS.items():
        path = OUT / f"{name}.onnx"
        onnx.save(chain_model(layers), path)
        untitled, meta = compile_sequence(path)
        (OUT / f"{name}_untiled.bin").write_bytes(untitled)
        expected = [chain_n_reference_layers(x, meta["quantizations"])[-1].tobytes()
                    for x in inputs]
        for index, (case, reference) in enumerate(zip(inputs, expected)):
            case.tofile(OUT / f"{name}_input{index}.u8")
            (OUT / f"{name}_expected{index}.i8").write_bytes(reference)
        variants = {}
        for tiles in (1, 2, 4):
            for serial in (True, False):
                suffix = "" if serial else "_batched"
                data, tiled = compile_tiled_chain(onnx.load(path), tiles=tiles, serial=serial)
                (OUT / f"{name}_tiles{tiles}{suffix}.bin").write_bytes(data)
                variants[f"tiles{tiles}{suffix}"] = dict(
                    tasks=decode_sequence(data)["task_count"],
                    arena_bytes=decode_sequence(data)["arena_bytes"],
                    layout=tiled["layout"], submission=tiled["submission"],
                    bytes=len(data))
        entry = dict(name=name, layers=len(layers),
                     hidden_channels=[c for c, _ in layers],
                     kernels=[k for _, k in layers], cases=CASES,
                     untiled_tasks=decode_sequence(untitled)["task_count"],
                     untiled_arena_bytes=decode_sequence(untitled)["arena_bytes"],
                     untiled_bytes=len(untitled), output_bytes=len(expected[0]),
                     variants=variants)
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built", len(manifest), "K>1 chains in", OUT)


if __name__ == "__main__":
    main()
