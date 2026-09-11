"""Deep N-layer dense Conv chain suite: serial vs one batched job.

Purpose (docs/plans/pipelining-plan.md S2): a real, independently generated chain long enough
that submitting the whole same-engine run as one job beats one ioctl per layer. The
chain emitter writes the next-command link between tasks when `serial=False`, which
is what a batched list needs (measured rule: linked tasks, single engine, terminal
last task; depth is bounded only by the loader's 64-task table).

Layout: `modelNNN.bin` is the serial container (the shipping mode) and
`batched/modelNNN.bin` the batched twin of the same graph; `inputNNN.u8` /
`expectedNNN.i8` are shared by both (the math is identical).
"""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.chain_n import chain_n_reference_layers
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deep_chain_suite"
OUT.mkdir(parents=True, exist_ok=True)
BATCHED = OUT / "batched"
BATCHED.mkdir(exist_ok=True)
REUSE = OUT / "reuse"
REUSE.mkdir(exist_ok=True)
REUSE_BATCHED = OUT / "reuse_batched"
REUSE_BATCHED.mkdir(exist_ok=True)
rng = np.random.default_rng(20260910)

CONFIGS = [
    [(5, 1), (8, 3), (8, 1), (12, 3), (12, 1), (8, 3), (8, 1), (3, 1)],
    [(4, 3), (6, 1), (8, 3), (10, 1), (10, 3), (8, 1), (6, 3), (4, 1), (3, 3),
     (3, 1), (3, 3), (3, 1)],
    [(3, 1)] * 16,
]


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
    graph = h.make_graph(nodes, "deep_chain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(previous, 1, [1, layers[-1][0], 8, 8])], tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def main():
    manifest = []
    cases = rng.integers(0, 256, (16, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    for index, layers in enumerate(CONFIGS):
        path = OUT / f"model{index:03}.onnx"
        onnx.save(chain_model(layers), path)
        serial, meta = compile_sequence(path)
        (OUT / f"model{index:03}.bin").write_bytes(serial)
        batched, bmeta = compile_sequence(path, submission="batched")
        (BATCHED / f"model{index:03}.bin").write_bytes(batched)
        # Double buffering: the chain keeps two intermediate surfaces instead of one
        # per layer, so the arena stays flat as depth grows.
        reuse, rmeta = compile_sequence(path, reuse_intermediates=True)
        (REUSE / f"model{index:03}.bin").write_bytes(reuse)
        reuse_batched, rbmeta = compile_sequence(path, reuse_intermediates=True,
                                                 submission="batched")
        (REUSE_BATCHED / f"model{index:03}.bin").write_bytes(reuse_batched)
        cases.tofile(OUT / f"input{index:03}.u8")
        # The container exposes one output (the last layer), so expected bytes are
        # that layer's grid only; `expose_intermediates=False` keeps the suite in the
        # single-output profile the board runner checks.
        encoded = [chain_n_reference_layers(x, meta["quantizations"])[-1].tobytes()
                   for x in cases]
        (OUT / f"expected{index:03}.i8").write_bytes(b"".join(encoded))
        entry = dict(index=index, layers=len(layers), tasks=meta["layers"],
                     hidden_channels=[c for c, _ in layers],
                     kernels=[k for _, k in layers], cases=len(cases),
                     output_bytes=len(encoded[0]),
                     serial_bytes=len(serial), batched_bytes=len(batched),
                     serial_submission=meta["submission"], batched_submission=bmeta["submission"],
                     arena_bytes=decode_sequence(serial)["arena_bytes"],
                     reuse_arena_bytes=decode_sequence(reuse)["arena_bytes"],
                     reuse_double_buffered=rmeta.get("reused_intermediates"),
                     reuse_batched_submission=rbmeta.get("submission"))
        manifest.append(entry)
        print(entry, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built", len(manifest), "deep chains in", OUT)


if __name__ == "__main__":
    main()
