"""Diamond tail suite: join output consumed by a Conv tail (name-bound composition).

The join task's output becomes an internal tensor that the first tail Conv reads by
name, so the elementwise join emitter and the native Conv emitter are composed
through the version-5 tensor table. The arena comes from `open_rknpu.liveness`,
which now schedules join and tail lifetimes.

Independently generated commands (no RKNN, no captures). Expected bytes come from
`open_rknpu.graph.diamond_reference`, which applies the tail quantizations after the
join. Board runner: tests/board_io.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.graph import diamond_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/diamond_tail_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(111000)

CONFIGS = []
for kind in ("Add", "Mul", "Sub", "Max"):
    for tails in ((1,), (3,), (1, 3)):
        CONFIGS.append((kind, tails))


def build(kind, tails, seed):
    rng = np.random.default_rng(seed)
    hidden = int(rng.integers(3, 17))
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["stem"], ["relu1"]),
             h.make_node("Conv", ["relu1", "wa", "ba"], ["head_a"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
             h.make_node("Conv", ["relu1", "wb", "bb"], ["head_b"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             h.make_node(kind, ["head_a", "head_b"], ["join_out"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-4, 4, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, 1, 1)).astype(np.float32), "wa"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "ba"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, 3, 3)).astype(np.float32), "wb"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "bb")]
    values = [h.make_tensor_value_info("relu1", 1, [1, hidden, 8, 8]),
              h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]),
              h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8]),
              h.make_tensor_value_info("join_out", 1, [1, 3, 8, 8])]
    source = "join_out"
    for index, kernel in enumerate(tails):
        last = index == len(tails) - 1
        weight, bias = f"tail_w{index}", f"tail_b{index}"
        output = "output" if last else f"tail_conv{index}"
        initializers.extend([nh.from_array(rng.uniform(-.5, .5, (3, 3, kernel, kernel)).astype(np.float32), weight),
                             nh.from_array(rng.uniform(-2, 2, (3,)).astype(np.float32), bias)])
        nodes.append(h.make_node("Conv", [source, weight, bias], [output], kernel_shape=[kernel, kernel],
                                 pads=[kernel // 2] * 4))
        if last:
            source = output
            break
        relu = output + "_relu"
        nodes.append(h.make_node("Relu", [output], [relu]))
        values.extend([h.make_tensor_value_info(output, 1, [1, 3, 8, 8]),
                       h.make_tensor_value_info(relu, 1, [1, 3, 8, 8])])
        source = relu
    graph = h.make_graph(nodes, "diamond_tail",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], initializers)
    for value in values:
        graph.value_info.append(value)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


manifest = []
for index, (kind, tails) in enumerate(CONFIGS):
    model = build(kind, tails, seed=111000 + index)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                           kind, meta["tail_quantization"], meta["join_zero_point"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "join": kind, "tail_kernels": list(tails),
                     "hidden_channels": meta["hidden_channels"], "cases": args.cases,
                     "outputs": 1, "output_bytes": 192, "profile": meta["profile"],
                     "schedule": meta["schedule"], "tensor_lifetimes": meta["tensor_lifetimes"],
                     "tensor_offsets": {name: int(value) for name, value in meta["tensor_offsets"].items()},
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                     "lifetime_bytes": meta["lifetime_bytes"], "allocated_bytes": meta["allocated_bytes"]})
    print(index, kind, tails, "tasks", len(meta["schedule"]), "schedule", meta["schedule"], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} diamond-tail models in {root}")
