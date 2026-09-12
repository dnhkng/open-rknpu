"""SPDX-License-Identifier: MIT

E13: the multi-model lifecycle - two compiled containers, one process.

`ornpu_open` owns everything device-side for one model: it opens `/dev/rknpu`,
allocates a 4 KiB task buffer plus one dma arena of the container's
`arena_size`, copies the compiled payload (the packed weights) into that arena
and keeps the file descriptor and both mappings until `ornpu_close`. There is no
process-global device context and no sharing between two `ornpu_model` handles:
opening `model_a.bin` and `model_b.bin` reserves the sum of the two arenas plus
two task buffers, and each handle reads and writes only its own arena.

That is the practical lesson this example exists for:

* open each model **once** and reuse the handle across as many inferences as you
  need - the weights stay resident and no open/close work happens per call;
* a script that calls `ornpu_open` per inference re-creates the dma allocations
  and re-uploads the weights every time (thrash), and one that forgets
  `ornpu_close` leaks the arena and the task buffer for every call;
* the two models never share an arena, so the device budget of a two-model
  process is `arena(a) + arena(b) + 2 * 4096` bytes, not `max(...)`.

This script builds two small, differently-shaped prefixes and compiles each with
`open_rknpu.scheduler.compile_sequence`:

* `a` - a 3x3 Conv classifier prefix, `Conv(3x3, RGB->16) -> Relu` on a 16x16
  image (`native16-input`, one task, arena 12288 B);
* `b` - a depthwise-pointwise prefix, `Conv(1x1, RGB->C8) -> depthwise Conv(3x3,
  C8) -> pointwise Conv(1x1, C8)` on an 8x8 image (the legacy 8x8 depthwise
  profile, three tasks, arena 24576 B).

Both are checked byte-for-byte against the profile's Python integer reference
(`native.native_input_reference`; `quantization.reference` ->
`depthwise.depthwise_reference` -> `chain.native_reference`), which is this
project's correctness criterion; the ONNX float error printed next to it is only
a quantization-quality number. The published containers are re-framed as the v5
named-tensor executables `examples/multi_model/board.c` loads.

Artifacts (all deterministic, all under `--out`, default
`examples/multi_model/build/`):

    model_a.bin model_b.bin      compiled + v5-framed containers
    input_a.u8  input_b.u8       4 packed NHWC UINT8 cases per model
    expected_a.i8 expected_b.i8  the integer reference for those cases
    model_a.onnx model_b.onnx    the graphs the containers were compiled from
    report.json                  container/arena summary + the measured numbers

Run from the repository root:

    PYTHONPATH=src python examples/multi_model/build.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primitives"))

from common import (as_named_tensor_container, assert_container_band, conv_node,
                    deterministic_cases, initializer, model_graph, qfrom, relu_node,
                    report_checks, tensor_info)
from open_rknpu.chain import native_reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import reference
from open_rknpu.scheduler import compile_sequence

EXAMPLE = Path(__file__).resolve().parent
REPO = EXAMPLE.parents[1]
DEFAULT_OUT = EXAMPLE / "build"
SEED = 131401
CASES = 4
TASK_BUFFER_BYTES = 4096  # runtime/open_rknpu.c: buffers[0] is one 4 KiB task page


def display(path):
    """A repository-relative path when the artifact lives in the checkout."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def graph_a():
    """`Conv(3x3, 3->16) -> Relu` on a 16x16 RGB image (native16-input)."""
    rng = np.random.default_rng(SEED + 1)
    weights = rng.uniform(-0.25, 0.25, (16, 3, 3, 3)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, (16,)).astype(np.float32)
    nodes = [conv_node("input", "conv", "w", "b", 3), relu_node("conv", "output")]
    return model_graph(nodes, "multi_model_a", [tensor_info("input", [1, 3, 16, 16])],
                       [tensor_info("output", [1, 16, 16, 16])],
                       [initializer("w", weights), initializer("b", bias)]), dict(
                           label="conv3x3-classifier-prefix", shape=(16, 16, 3),
                           compile=dict(input_scale=0.5, input_zero_point=128),
                           reference="native.native_input_reference")


def graph_b():
    """`Conv(1x1) -> depthwise Conv(3x3) -> pointwise Conv(1x1)` on 8x8 RGB."""
    channels, pointwise = 8, 8
    rng = np.random.default_rng(SEED + 2)
    stem_weights = rng.uniform(-0.3, 0.3, (channels, 3, 1, 1)).astype(np.float32)
    stem_bias = rng.uniform(-0.5, 0.5, (channels,)).astype(np.float32)
    weights = rng.uniform(-0.3, 0.3, (channels, 1, 3, 3)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, (channels,)).astype(np.float32)
    head_weights = rng.uniform(-0.3, 0.3, (pointwise, channels, 1, 1)).astype(np.float32)
    head_bias = rng.uniform(-1.0, 1.0, (pointwise,)).astype(np.float32)
    nodes = [conv_node("input", "stem", "stem_w", "stem_b", 1),
             conv_node("stem", "depthwise", "dw_w", "dw_b", 3, group=channels),
             conv_node("depthwise", "output", "pw_w", "pw_b", 1)]
    constants = [initializer("stem_w", stem_weights), initializer("stem_b", stem_bias),
                 initializer("dw_w", weights), initializer("dw_b", bias),
                 initializer("pw_w", head_weights), initializer("pw_b", head_bias)]
    return model_graph(nodes, "multi_model_b", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, pointwise, 8, 8])], constants), dict(
                           label="depthwise-pointwise-prefix", shape=(8, 8, 3),
                           compile={},
                           reference="quantization.reference -> "
                                     "depthwise.depthwise_reference -> chain.native_reference")


def reference_grid(kind, meta, cases):
    """The profile's host integer reference for each packed `HWC` case of one model."""
    if kind == "a":
        quantization = qfrom(meta["quantization"])
        return np.stack([native_input_reference(case, quantization, int(meta["input_zero_point"]),
                                               pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta["conv_dilations"]))
                         for case in cases])
    stem = qfrom(meta["first"]["quantization"])
    layer = qfrom(meta["depthwise"])
    if not meta.get("depthwise_pointwise"):
        raise ValueError("model b lost its pointwise head; the reference would change")
    head = qfrom(meta["pointwise"])
    return np.stack([native_reference(depthwise_reference(reference(case, stem), layer,
                                                         stem.output_zero_point),
                                     head, layer.output_zero_point)
                     for case in cases])


def compile_graph(out, kind, model, **compile_kwargs):
    """Save the graph next to its container and compile it deterministically."""
    path = out / ("model_%s.onnx" % kind)
    onnx.save(model, path)
    binary, meta = compile_sequence(path, **compile_kwargs)
    return path, binary, meta


def publish(out, kind, binary, meta, inputs, expected):
    """Write `model_<kind>.bin` as a v5 container plus its `input`/`expected` fixtures."""
    framed, info = as_named_tensor_container(binary)
    assert_container_band(framed, meta, "multi_model %s" % kind)
    (out / ("model_%s.bin" % kind)).write_bytes(framed)
    inputs.tofile(out / ("input_%s.u8" % kind))
    expected.tofile(out / ("expected_%s.i8" % kind))
    return info


def model_row(kind, spec, onnx_path, binary, meta, info, suite_bytes, inputs, expected, check):
    """The JSON-safe per-model summary recorded in `report.json`."""
    return dict(kind=kind, label=spec["label"], onnx=onnx_path.name, reference=spec["reference"],
                compile=dict(spec["compile"]),
                profile=meta.get("sequence_profile") or meta.get("depthwise_profile")
                or meta.get("profile") or "unknown",
                container_bytes=len(binary), suite_container_bytes=suite_bytes,
                format_version=info["format_version"], tasks=info["task_count"],
                arena_bytes=info["arena_bytes"],
                input_shape=list(info["shape_nhwc"]), output_shape=list(info["output_shape_nhwc"]),
                cases=int(inputs.shape[0]), input_bytes_per_case=int(inputs[0].size),
                output_bytes_per_case=int(expected[0].size), exact_bytes_total=int(expected.size),
                input_scale=info["input_scale"], input_zero_point=info["input_zero_point"],
                output_scale=info["output_scale"], output_zero_point=info["output_zero_point"],
                check=check)


def print_board_recipe(out):
    """The exact cross-compile + adb commands for this example's two containers."""
    out = display(out)
    remote = "/userdata/open-npu-research/multi_model"
    print("  board recipe for %s:" % out)
    print("    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\")
    print("      --sysroot=\"$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot\" \\")
    print("      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime examples/multi_model/board.c \\")
    print("      runtime/open_rknpu.c -o %s/board_multi" % out)
    print("    adb shell mkdir -p %s" % remote)
    print("    adb push %s/board_multi %s/model_a.bin %s/model_b.bin %s/input_a.u8 "
          "%s/input_b.u8 %s/expected_a.i8 %s/expected_b.i8 %s/"
          % (out, out, out, out, out, out, out, remote))
    print("    adb shell 'cd %s && ./board_multi model_a.bin model_b.bin input_a.u8 input_b.u8 "
          "expected_a.i8 expected_b.i8 8'" % remote)


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--out", type=Path, default=DEFAULT_OUT,
                      help="artifact directory (default: examples/multi_model/build)")
    options = args.parse_args(argv)
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for kind, builder in (("a", graph_a), ("b", graph_b)):
        model, spec = builder()
        onnx_path, binary, meta = compile_graph(out, kind, model, **spec["compile"])
        height, width, channels = spec["shape"]
        rng = np.random.default_rng(SEED + 100 + (0 if kind == "a" else 1))
        cases = deterministic_cases(rng, CASES, height, width, channels)
        expected = reference_grid(kind, meta, cases)
        info = publish(out, kind, binary, meta, cases, expected)
        published = np.fromfile(out / ("expected_%s.i8" % kind), dtype=np.int8)
        check = report_checks(spec["label"], expected, published.reshape(expected.shape), model,
                              cases, meta, spec["reference"])
        suite_bytes = (out / ("model_%s.bin" % kind)).stat().st_size
        rows[kind] = model_row(kind, spec, onnx_path, binary, meta, info, suite_bytes, cases,
                               expected, check)
        print("  %-8s %-30s profile=%-20s tasks=%d arena=%d B container=%d B"
              % (kind, spec["label"], rows[kind]["profile"], rows[kind]["tasks"],
                 rows[kind]["arena_bytes"], rows[kind]["container_bytes"]))

    arena_a, arena_b = rows["a"]["arena_bytes"], rows["b"]["arena_bytes"]
    report = dict(
        example="multi_model", generated_by="examples/multi_model/build.py", target="rv1103",
        cases_per_model=CASES, models=rows,
        arena=dict(task_buffer_bytes_per_model=TASK_BUFFER_BYTES,
                   model_a_bytes=arena_a, model_b_bytes=arena_b,
                   combined_arena_bytes=arena_a + arena_b,
                   both_models_open_bytes=arena_a + arena_b + 2 * TASK_BUFFER_BYTES,
                   note="each ornpu_open mmaps one 4 KiB task buffer and one arena of the "
                        "container's arena_size; two ornpu_model handles own two independent "
                        "allocations and never share one"),
        lesson=[
            "ornpu_open reserves the arena and uploads the weights device-side; ornpu_close "
            "releases both",
            "two open models do not share an arena: the device budget is arena(a) + arena(b) "
            "+ 2 * 4096 bytes",
            "open each model once and alternate inferences; reopening per inference re-creates "
            "the dma allocations and re-uploads the weights, and a missing close leaks them",
        ])
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print_board_recipe(out)
    summary = " | ".join("%s %s %dB/%dtask arena=%dB ref %s"
                         % (kind, rows[kind]["profile"], rows[kind]["container_bytes"],
                            rows[kind]["tasks"], rows[kind]["arena_bytes"], rows[kind]["check"])
                         for kind in ("a", "b"))
    print("multi_model: %s -> %s" % (summary, display(out)))


if __name__ == "__main__":
    main()
