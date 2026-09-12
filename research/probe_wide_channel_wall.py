"""Boundary probe for the native16 channel-plane wall (F10 negative evidence).

`wide_channel_suite` pins the *accepted* sides of both walls (C16352 input, C8192 output).
This probe rebuilds the refused sides so the refusal can be re-measured on a board:

* C16368 input  - 512 32-lane weight parts; the emitter refuses it, and a container built
  with the bound lifted never completes: the job times out and the driver soft-resets the
  core (`RKNPU: job timeout ... soft reset`).
* C16384 output - 1024 surface blocks; a container built with the bound lifted runs but
  writes the first 8192 channels correctly and then wrong bytes.

The two refused containers cannot come out of the shipped emitter, so the probe lifts the
Python bound in memory and prints the cross-compile line for a runtime built with the same
lift (`-DORNPU_MAX_NATIVE_CHANNELS=... -DORNPU_MAX_OUTPUT_CHANNELS=...`, honoured by
`runtime/open_rknpu.h`). Nothing in the shipped path is modified.
"""
from pathlib import Path
import argparse
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

import open_rknpu.native as native
import open_rknpu.sequence as sequence
from open_rknpu.quantization import Quantization

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="/tmp/wide_channel_wall")
parser.add_argument("--cases", type=int, default=2)
args = parser.parse_args()
root = Path(args.directory)
root.mkdir(parents=True, exist_ok=True)

# (label, ic, oc, kernel, height, width, lift)
PLAN = [
    ("c16352-in-k1-accepted", 16352, 1, 1, 2, 2, False),
    ("c16368-in-k1-refused", 16368, 1, 1, 2, 2, True),
    ("c8192-out-k1-accepted", 16, 8192, 1, 2, 2, False),
    ("c16384-out-k1-refused", 16, 16384, 1, 2, 2, True),
]

manifest = []
for index, (label, ic, oc, kernel, height, width, lift) in enumerate(PLAN):
    saved = (native.MAX_NATIVE_CHANNELS, native.MAX_OUTPUT_CHANNELS, native.MAX_WEIGHT_PARTS,
             sequence.MAX_NATIVE_CHANNELS, sequence.MAX_OUTPUT_CHANNELS)
    if lift:
        native.MAX_NATIVE_CHANNELS = sequence.MAX_NATIVE_CHANNELS = 65536
        native.MAX_OUTPUT_CHANNELS = sequence.MAX_OUTPUT_CHANNELS = 65536
        native.MAX_WEIGHT_PARTS = 65536
    try:
        rng = np.random.default_rng(510912 + index)
        w = rng.uniform(-.7, .8, (oc, ic, kernel, kernel)).astype(np.float32)
        b = rng.uniform(-2, 2, oc).astype(np.float32)
        pad = kernel // 2
        graph = h.make_graph(
            [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[kernel, kernel],
                         pads=[pad] * 4, strides=[1, 1])],
            label.replace("-", "_"),
            [h.make_tensor_value_info("input", 1, [1, ic, height, width])],
            [h.make_tensor_value_info("output", 1, [1, oc, height, width])],
            [nh.from_array(w, "w"), nh.from_array(b, "b")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        model.ir_version = 8
        path = root / f"model{index:03}.onnx"
        onnx.save(model, path)
        binary, meta = native.compile_native_input(model)
    except ValueError as error:
        print(f"{label}: emitter refused: {error}", flush=True)
        manifest.append({"index": index, "label": label, "emitted": False, "error": str(error)})
        continue
    finally:
        (native.MAX_NATIVE_CHANNELS, native.MAX_OUTPUT_CHANNELS, native.MAX_WEIGHT_PARTS,
         sequence.MAX_NATIVE_CHANNELS, sequence.MAX_OUTPUT_CHANNELS) = saved
    (root / f"model{index:03}.bin").write_bytes(binary)
    inputs = np.random.default_rng(510912 + index).integers(
        0, 256, (args.cases, height, width, ic), dtype=np.uint8)
    inputs[0] = 0
    if args.cases > 1:
        inputs[1] = 255
    inputs.tofile(root / f"input{index:03}.u8")
    quant = Quantization(**{key: np.array(value) if isinstance(value, list) else value
                                   for key, value in meta["quantization"].items()})
    expected = np.stack([native.native_input_reference(
        image, quant, meta["input_zero_point"], pads=meta["conv_pads"],
        strides=tuple(meta["conv_strides"]), dilations=tuple(meta["conv_dilations"]))
        for image in inputs])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "label": label, "emitted": True, "input_channels": ic,
                     "output_channels": oc, "planes": (ic + 15) // 16,
                     "weight_parts": ((((ic + 15) // 16) * 16) + 31) // 32,
                     "output_blocks": (oc + 15) // 16, "container_bytes": len(binary),
                     "input_bytes": int(inputs[0].size),
                     "output_bytes": int(meta["output_shape_nhwc"][0] * height * width * oc)})
    print(manifest[-1], flush=True)

(root / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2) + "\n")
print(f"wrote {root}")
print("relaxed runtime:  research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc "
      "--sysroot=$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot "
      "-O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime -Itests "
      "-DORNPU_MAX_NATIVE_CHANNELS=65536u -DORNPU_MAX_OUTPUT_CHANNELS=65536u "
      "tests/board_api.c runtime/open_rknpu.c -o /tmp/board_api_wall")
