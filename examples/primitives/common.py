"""SPDX-License-Identifier: MIT

Shared helpers for the `examples/primitives/` RV1103 examples.

Every example in this folder is host-only and deterministic. It builds a small
ONNX graph with the node helpers below, compiles it with
`open_rknpu.scheduler.compile_sequence`, and publishes the container as a board
suite in the v5 named-tensor format that `tests/board_io.c` and
`research/run_v5_suite.py` load. No board, network, PyTorch or vendor compiler
is touched here; the board commands are only printed as text.

The correctness criterion is **byte equality with the profile's Python integer
reference for the same quantization parameters** - the discipline the
`research/build_*_suite.py` generators and the board runs use. `report_checks`
asserts it and raises on a mismatch, so a script never publishes a suite whose
`expected*.i8` disagrees with the reference. The ONNX float comparison is kept as
a clearly labelled secondary quantization-quality line and is never called exact.
`report_reference_unavailable` exists so a primitive without an independent
integer reference is reported instead of silently falling back to that metric.

Container framing. Several bounded profiles (a single native Conv, depthwise,
elementwise, the LUT stem, ConvTranspose, the legacy chain) emit the older v3/v4
container, which has no named-tensor table, so `tests/board_io.c` rejects it.
`publish_suite` therefore re-frames such a container as v5 with
`as_named_tensor_container`: the payload, task table, arena, shapes and
quantization are copied verbatim and only the descriptor table is added. The
manifest records both sizes.
"""
from dataclasses import fields as dataclass_fields
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT, ROLE_OUTPUT,
                                 decode_sequence, encode_sequence_v5, tensor_native_bytes)

FLOAT = TensorProto.FLOAT

# Profile keys are spelled differently by the emitters; this is the print order.
PROFILE_KEYS = ("sequence_profile", "profile", "lut_profile", "transposed_profile",
                "depthwise_profile", "elementwise_profile")


# --------------------------------------------------------------------------- #
# ONNX graph builders
# --------------------------------------------------------------------------- #
def tensor_info(name, shape, elem_type=FLOAT):
    """A `ValueInfoProto` for one external tensor."""
    return h.make_tensor_value_info(name, elem_type, list(shape))


def initializer(name, array):
    """A float32 (or int8) immutable initializer."""
    return nh.from_array(np.asarray(array), name)


def conv_node(source, output, weights, bias, kernel, pads=None, strides=(1, 1), dilations=(1, 1),
              group=1):
    """A `Conv` node with explicit kernel/stride/dilation/padding attributes."""
    if pads is None:
        pads = [kernel // 2] * 4
    return h.make_node("Conv", [source, weights, bias], [output], kernel_shape=[kernel, kernel],
                       pads=list(pads), strides=list(strides), dilations=list(dilations), group=group)


def relu_node(source, output):
    return h.make_node("Relu", [source], [output])


def pool_node(kind, source, output, kernel=2, strides=2):
    """A 2x2 stride-2 `MaxPool` or `AveragePool` (the only verified pool shape)."""
    return h.make_node(kind, [source], [output], kernel_shape=[kernel, kernel],
                       strides=[strides, strides], pads=[0, 0, 0, 0])


def join_node(kind, first, second, output):
    """An attribute-free `Add`/`Sub`/`Mul`/`Max` join of two same-shape tensors."""
    return h.make_node(kind, [first, second], [output])


def model_graph(nodes, name, inputs, outputs, constants, value_info=(), opset=13):
    """A checked `ModelProto` with one opset import and IR version 8."""
    graph = h.make_graph(list(nodes), name, list(inputs), list(outputs), list(constants),
                         value_info=list(value_info))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# --------------------------------------------------------------------------- #
# Compilation and container reporting
# --------------------------------------------------------------------------- #
def profile_of(meta):
    """A printable profile name for any emitter's metadata."""
    for key in PROFILE_KEYS:
        if meta.get(key):
            return str(meta[key])
    if meta.get("pool_stages"):
        return "conv-pool-terminal"
    if "limitations" in meta:
        return "legacy-conv"
    return str(meta.get("profile") or "unknown")


def container_info(binary, meta=None):
    """Decode a container and describe it (profile, bytes, tasks, shapes, band)."""
    info = decode_sequence(binary)
    report = dict(profile=profile_of(meta or {}), container_bytes=len(binary),
                  format_version=info["format_version"], tasks=info["task_count"],
                  input_shape=list(info["shape_nhwc"]), output_shape=list(info["output_shape_nhwc"]),
                  input_bytes=info["input_bytes"], output_bytes=info["output_bytes"],
                  input_scale=info["input_scale"], input_zero_point=info["input_zero_point"],
                  output_scale=info["output_scale"], output_zero_point=info["output_zero_point"],
                  input_layout=info["input_layout"])
    if meta:
        report["pool_stages"] = list(meta.get("pool_stages", ()))
    return report


def compile_and_report(model, path_or_name="model.onnx", label=None, **compile_kwargs):
    """Save `model` to a temporary ONNX file, compile it and print one report line.

    Returns `(binary, meta)` exactly as `compile_sequence` did; the printed line
    is the decoded container summary, so the caller can compare the compiled
    container with the reference it is about to check.
    """
    name = Path(str(path_or_name)).name
    if not name.endswith(".onnx"):
        name += ".onnx"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        onnx.save(model, path)
        binary, meta = compile_sequence(path, **compile_kwargs)
    report = container_info(binary, meta)
    print("  %-26s profile=%-24s tasks=%d container=%d B" %
          (label or name, report["profile"], report["tasks"], report["container_bytes"]))
    return binary, meta, report


# --------------------------------------------------------------------------- #
# Host integer reference versus the ONNX float model
# --------------------------------------------------------------------------- #
def qfrom(params):
    """A live `Quantization` from one emitter metadata entry.

    Some emitters nest the fields under `quantization`; others store them at the
    top level. Shape annotations and similar non-quantization keys are ignored.
    """
    if isinstance(params, Quantization):
        return params
    values = dict(params)
    if "quantization" in values:
        values = dict(values["quantization"])
    names = {field.name for field in dataclass_fields(Quantization)}
    values = {name: value for name, value in values.items() if name in names}
    for name in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        if name in values:
            values[name] = np.array(values[name])
    return Quantization(**values)


def float_reference(model, case, input_scale=1.0, input_zero_point=0):
    """ONNX float output for one packed `HWC` UINT8 case, as `HWC`.

    The graph is fed the dequantized input `(code - zero_point) * scale`, which is
    the real-valued tensor the compiled band claims to approximate.
    """
    values = (np.asarray(case, np.float64) - input_zero_point) * input_scale
    name = model.graph.input[0].name
    outputs = ReferenceEvaluator(model).run(None, {name: values.transpose(2, 0, 1)[None].astype(np.float32)})
    return np.asarray(outputs[0])[0].transpose(1, 2, 0)


def dequantization_error(integer, float_output, output_scale, output_zero_point):
    """Quantization error of an INT8 grid against the ONNX float output.

    This is a *quality* metric, not the correctness criterion. This project's
    correctness criterion is byte equality with the profile's integer reference
    (`assert_int8_reference`); the float number here only says how much the
    chosen per-tensor band costs against the trained model.
    """
    decoded = (np.asarray(integer, np.float64) - output_zero_point) * output_scale
    error = np.abs(decoded - np.asarray(float_output, np.float64))
    return dict(max_abs=float(error.max()), max_lsb=float(error.max() / output_scale),
                output_scale=float(output_scale), output_zero_point=int(output_zero_point))


def float_errors(integers, model, cases, meta, input_scale=None, input_zero_point=None):
    """One `dequantization_error` per case, sharing a single ONNX evaluator.

    The input band defaults to the container's, but profiles whose metadata omits
    it (the LUT profile) can pass it explicitly so the float model is fed the
    dequantized input it expects.
    """
    evaluator = ReferenceEvaluator(model)
    name = model.graph.input[0].name
    scale = float(meta.get("input_scale", 1.0) if input_scale is None else input_scale)
    zero_point = int(meta.get("input_zero_point", 0) if input_zero_point is None else input_zero_point)
    errors = []
    for integer, case in zip(integers, cases):
        values = (np.asarray(case, np.float64) - zero_point) * scale
        outputs = evaluator.run(None, {name: values.transpose(2, 0, 1)[None].astype(np.float32)})
        errors.append(dequantization_error(integer, np.asarray(outputs[0])[0].transpose(1, 2, 0),
                                           float(meta["output_scale"]),
                                           int(meta["output_zero_point"])))
    return errors


def worst_lsb(errors):
    """The largest float quantization error over the cases, in output-scale units."""
    return max(float(error["max_lsb"]) for error in errors)


def assert_int8_reference(integers, expected, label):
    """Assert byte equality between the computed reference and the profile reference.

    This is the whole point of the examples: the container that will run on the
    board is only trusted because an independently written integer reference
    produces the same bytes for the same quantization parameters - the same
    criterion the `research/build_*_suite.py` generators and the board runs use.
    A mismatch raises, so the script exits non-zero instead of publishing a
    suite whose `expected*.i8` is wrong.
    """
    integers = np.asarray(integers, np.int8)
    expected = np.asarray(expected, np.int8)
    if integers.shape != expected.shape:
        raise ValueError("%s: reference shape %s does not match the computed %s"
                         % (label, expected.shape, integers.shape))
    if not integers.size:
        raise ValueError("%s: empty reference" % label)
    differences = integers != expected
    if differences.any():
        flat = int(np.argmax(differences.reshape(-1)))
        index = np.unravel_index(flat, integers.shape)
        raise AssertionError("%s: integer reference mismatch: %d/%d bytes differ; "
                             "first at %s computed %d, expected %d" %
                             (label, int(differences.sum()), int(integers.size), index,
                              int(integers[index]), int(expected[index])))
    return dict(cases=int(integers.shape[0]), bytes=int(integers.size))


def report_checks(label, reference, published, model, cases, meta, reference_name,
                  input_scale=None, input_zero_point=None):
    """Assert byte-exact integer agreement, then print the two check lines.

    `reference` is the profile's Python integer reference computed from the
    compiled `meta`, and `published` is the `expected*.i8` this script is about to
    hand to the board. They must be identical byte for byte; anything else raises,
    so the script exits non-zero instead of publishing a suite whose expected
    bytes do not match the reference. The float number on the second line is a
    secondary quantization-quality metric and never the correctness claim.
    """
    result = assert_int8_reference(reference, published, label)
    errors = float_errors(reference, model, cases, meta, input_scale, input_zero_point)
    print("  %-26s int8 reference: exact (%d/%d cases, %d bytes) via %s" %
          (label, result["cases"], result["cases"], result["bytes"], reference_name))
    print("  %-26s float quantization error: %.2f LSB max (secondary, not correctness)" %
          (label, worst_lsb(errors)))
    return "int8 exact"


def read_expected(folder, index):
    """The published `expected*.i8` bytes, flat INT8 - what the board compares against."""
    return np.fromfile(Path(folder) / ("expected%03d.i8" % index), dtype=np.int8)


REPO_ROOT = Path(__file__).resolve().parents[2]


def cross_check_suite(suite, reference_for, compile_kwargs=(), limit=None):
    """Reproduce a board-verified `research/` suite with this script's pipeline.

    `reference_for(case, model, meta)` must return the INT8 grid for one packed
    case. Every recorded `expected*.i8` is asserted byte for byte, so the
    reference pipeline used by this example is shown to be the same one that
    produced board-exact evidence. Returns `(models, bytes)`.
    """
    root = REPO_ROOT / "research" / suite
    if not root.exists():
        raise FileNotFoundError("board suite research/%s is missing" % suite)
    manifest = root / "manifest.json"
    if manifest.exists():
        indexes = [entry["index"] for entry in json.loads(manifest.read_text())]
    else:
        indexes = sorted(int(path.stem[5:]) for path in root.glob("model*.onnx"))
    if limit is not None:
        indexes = indexes[:limit]
    models = values = 0
    for index in indexes:
        path = root / ("model%03d.onnx" % index)
        expected = np.fromfile(root / ("expected%03d.i8" % index), dtype=np.int8)
        binary, meta = compile_sequence(path, *compile_kwargs)
        info = decode_sequence(binary)
        batch, height, width, channels = info["shape_nhwc"]
        cases = np.fromfile(root / ("input%03d.u8" % index), dtype=np.uint8)
        cases = cases.reshape(-1, height, width, channels)
        model = onnx.load(path)
        got = np.stack([reference_for(case, model, meta) for case in cases])
        assert_int8_reference(got, expected.reshape(got.shape), "research/%s model%03d" % (suite, index))
        models += 1
        values += int(got.size)
    print("  %-26s board suite cross-check: exact (%d models, %d bytes) [research/%s]" %
          ("", models, values, suite))
    return models, values


def report_reference_unavailable(label, profile):
    """The honest report when a profile has no independent integer reference.

    No example in this folder needs it; keep it so a future primitive cannot
    quietly fall back to the float metric and call that "exact".
    """
    print("  %-26s int8 reference: not available for %s" % (label, profile))
    return "int8 not available"


def deterministic_cases(rng, count, height, width, channels):
    """`count` packed `HWC` UINT8 cases with fixed corners (all-zero, all-255, 128)."""
    if count < 3:
        raise ValueError("at least three cases are needed for the corner inputs")
    cases = rng.integers(0, 256, (count, height, width, channels), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    return cases



# --------------------------------------------------------------------------- #
# Board suite publishing
# --------------------------------------------------------------------------- #
def as_named_tensor_container(binary):
    """Return `binary` in the v5 named-tensor framing that `tests/board_io.c` loads.

    A v5 container is returned unchanged. For a v3 container the payload, task
    table, arena, shapes and quantization are copied verbatim and the two external
    tensor descriptors (`input0`, `output`) are added, which is exactly the
    framing the composed emitters already use. A v4 container carries named
    constants that v5 cannot express, so it is rejected instead of being silently
    stripped.
    """
    info = decode_sequence(binary)
    if info["format_version"] == 5:
        return binary, info
    if info.get("constant_count"):
        raise ValueError("a v4 container with named constants cannot be re-framed as v5")
    payload = binary[96 + info["task_count"] * 16:]
    layout = LAYOUT_NATIVE16 if info["input_layout"] == "native16" else LAYOUT_PACKED_U8
    batch, height, width, channels = info["shape_nhwc"]
    out_batch, out_height, out_width, out_channels = info["output_shape_nhwc"]
    tensors = [
        dict(name="input0", role=ROLE_INPUT, layout=layout, index=0,
             shape=(batch, height, width, channels), offset=info["input_offset"],
             size=tensor_native_bytes(layout, batch, height, width, channels)),
        dict(name="output", role=ROLE_OUTPUT, layout=LAYOUT_NATIVE16, index=0,
             shape=(out_batch, out_height, out_width, out_channels), offset=info["output_offset"],
             size=tensor_native_bytes(LAYOUT_NATIVE16, out_batch, out_height, out_width, out_channels)),
    ]
    tasks = [(task["command_offset"], task["register_count"], task["enable"], task["mask"])
             for task in info["tasks"]]
    framed = encode_sequence_v5(payload, tensors=tensors, tasks=tasks, arena_bytes=info["arena_bytes"],
                                input_scale=info["input_scale"],
                                input_zero_point=info["input_zero_point"],
                                output_scale=info["output_scale"],
                                output_zero_point=info["output_zero_point"],
                                serial=bool(info["serial"]))
    return framed, decode_sequence(framed)


def _json_extras(meta, limit=600):
    """Small JSON-safe metadata values worth recording in the manifest."""
    extras = {}
    for key, value in meta.items():
        if key in PROFILE_KEYS or key == "container":
            continue
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError):
            continue
        if len(encoded) <= limit:
            extras[key] = value
    return extras


def assert_container_band(binary, meta, label):
    """Assert the container's decoded shapes and quantization match the emitter meta.

    The published `expected*.i8` was produced from `meta`, so the container the
    board loads must carry exactly those shapes and that input/output band. A
    mismatch means the suite would be checked against the wrong grid.
    """
    info = decode_sequence(binary)
    pairs = [("input_shape", info["shape_nhwc"], meta.get("shape_nhwc")),
             ("output_shape", info["output_shape_nhwc"], meta.get("output_shape_nhwc")),
             ("output_scale", info["output_scale"], meta.get("output_scale")),
             ("output_zero_point", info["output_zero_point"], meta.get("output_zero_point")),
             ("input_scale", info["input_scale"], meta.get("input_scale")),
             ("input_zero_point", info["input_zero_point"], meta.get("input_zero_point"))]
    for name, decoded, declared in pairs:
        if declared is None:
            continue
        if name.endswith("scale"):
            # The header stores IEEE float32, so compare after the same round trip.
            same = float(np.float32(declared)) == float(decoded)
        else:
            same = list(decoded) == list(declared) if isinstance(decoded, list) else decoded == declared
        if not same:
            raise AssertionError("%s: container %s %r does not match emitter metadata %r"
                                 % (label, name, decoded, declared))
    return info


def publish_suite(folder, binary, inputs, expected, meta, index=0):
    """Write one board suite (`model/input/expected` + `manifest.json`).

    `inputs` is a `[cases,H,W,C]` UINT8 array and `expected` a `[cases,oh,ow,oc]`
    INT8 array, both in the packed order the runtime's flat API buffers use. The
    recorded container is the v5 framing `tests/board_io.c` loads; `container_bytes`
    keeps the size of the container as the emitter produced it.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    inputs = np.ascontiguousarray(inputs, dtype=np.uint8)
    expected = np.ascontiguousarray(expected, dtype=np.int8)
    if inputs.ndim != 4 or expected.ndim != 4 or inputs.shape[0] != expected.shape[0]:
        raise ValueError("inputs must be [cases,H,W,C] and expected [cases,oh,ow,oc]")
    framed, info = as_named_tensor_container(binary)
    assert_container_band(framed, meta, "suite model %03d" % index)
    (folder / ("model%03d.bin" % index)).write_bytes(framed)
    inputs.tofile(folder / ("input%03d.u8" % index))
    expected.tofile(folder / ("expected%03d.i8" % index))
    entry = dict(index=index, cases=int(inputs.shape[0]), output_bytes=int(expected[0].size),
                 input_shape=list(info["shape_nhwc"]), output_shape=list(info["output_shape_nhwc"]),
                 profile=profile_of(meta), format_version=info["format_version"],
                 tasks=info["task_count"], container_bytes=len(binary),
                 suite_container_bytes=len(framed), input_scale=info["input_scale"],
                 input_zero_point=info["input_zero_point"], output_scale=info["output_scale"],
                 output_zero_point=info["output_zero_point"])
    for key, value in _json_extras(meta).items():
        entry.setdefault(key, value)
    manifest_path = folder / "manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest = [row for row in manifest if row.get("index") != index]
    manifest.append(entry)
    manifest.sort(key=lambda row: row["index"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return entry


def print_board_recipe(folder):
    """Print the cross-compile and adb commands for one published suite.

    The compile line keeps the wording of `examples/mel-kws/README.md`; the adb
    block stages the same suite and runs `tests/board_io.c`, whose runner derives
    the per-model case count from each pushed input file.
    """
    folder = Path(folder)
    remote = "/userdata/open-npu-research/%s" % folder.name
    models = sorted(folder.glob("model*.bin"))
    print("  board recipe for %s:" % folder)
    print("    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\")
    print("      --sysroot=\"$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot\" \\")
    print("      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\")
    print("      runtime/open_rknpu.c -o %s/board_io" % folder)
    print("    adb shell mkdir -p %s" % remote)
    print("    adb push %s/board_io %s/*.bin %s/*.u8 %s/*.i8 %s/" %
          (folder, folder, folder, folder, remote))
    print("    adb shell 'cd %s && ./board_io . %d'" % (remote, len(models)))
    print("    # or the whole staged directory: "
          "PYTHONPATH=src python research/run_v5_suite.py %s --binary %s/board_io" % (folder, folder))


def print_summary(title, rows, folder=None):
    """The single final line of an example: one token per compiled model."""
    parts = ["%s %dB/%dtask ref %s" % (row["profile"], row["container_bytes"], row["tasks"],
                                       row["check"]) for row in rows]
    line = "%s: %s" % (title, " | ".join(parts))
    if folder is not None:
        line += " -> %s" % folder
    print(line)
