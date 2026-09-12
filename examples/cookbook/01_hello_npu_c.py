"""SPDX-License-Identifier: MIT

Cookbook 1/8: the smallest C program, and the container it runs.

This script compiles one tiny `Conv` to a container under
`examples/cookbook/build/01_hello_npu_c/`, writes one deterministic packed input,
asserts the container decodes and that recompiling it is byte-identical, and
prints the exact commands that build and run `hello_npu.c` (see
`01_hello_npu_c.md` for the annotated version).

The C program itself is host-compilable: the device calls are behind
`ORNPU_BOARD`, so `gcc ... -c hello_npu.c` proves it builds without the vendor
toolchain, and running it needs the RV1103 board (the script never touches one).

Run:
    PYTHONPATH=src python examples/cookbook/01_hello_npu_c.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx

from cookbook_common import (build_dir, container_info, conv_node, initializer, model_graph,
                             tensor_info)
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

FOLDER = build_dir("01_hello_npu_c")
SEED = 10101
# A single 3x3 RGB Conv at 8x8 is the one-task legacy image profile: one task,
# packed input, ready to run through the flat `ornpu_run` API.
CHANNELS, HEIGHT, WIDTH, KERNEL, OUT_CHANNELS = 3, 8, 8, 3, 4


def build_model():
    """One padded 3x3 Conv, float32 weights, deterministic."""
    rng = np.random.default_rng(SEED)
    weights = rng.uniform(-0.25, 0.25, (OUT_CHANNELS, CHANNELS, KERNEL, KERNEL)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, OUT_CHANNELS).astype(np.float32)
    node = conv_node("input", "output", "w", "b", KERNEL, pads=[1, 1, 1, 1], strides=(1, 1))
    return model_graph([node], "hello_npu", [tensor_info("input", [1, CHANNELS, HEIGHT, WIDTH])],
                       [tensor_info("output", [1, OUT_CHANNELS, HEIGHT, WIDTH])],
                       [initializer("w", weights), initializer("b", bias)])


def print_commands(container, input_file):
    """The exact host/board commands, as text - nothing here runs a board."""
    binary = container.name
    print("  build and run commands:")
    print("    # host compile check (no device, no vendor toolchain):")
    print("    gcc -O2 -Wall -Wextra -Werror -Iruntime -c examples/cookbook/hello_npu.c -o /tmp/hello_npu.o")
    print("    # board build (docs/board.md, 'Build the runtime'):")
    print("    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\")
    print("      --sysroot=\"$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot\" \\")
    print("      -O2 -std=gnu99 -Wall -Wextra -Werror -DORNPU_BOARD -Iruntime \\")
    print("      examples/cookbook/hello_npu.c runtime/open_rknpu.c -o %s/hello_npu" % FOLDER)
    print("    adb shell mkdir -p /userdata/open-npu-research/hello_npu")
    print("    adb push %s/hello_npu %s/%s %s/%s /userdata/open-npu-research/hello_npu/" %
          (FOLDER, FOLDER, binary, FOLDER, input_file.name))
    print("    adb shell 'cd /userdata/open-npu-research/hello_npu && ./hello_npu %s %s'" %
          (binary, input_file.name))


def main():
    model_path = FOLDER / "hello_npu.onnx"
    onnx.save(build_model(), model_path)
    binary, meta = compile_sequence(model_path)
    report = container_info(binary, meta)
    (FOLDER / "model.bin").write_bytes(binary)
    info = decode_sequence(binary)

    # The container the C program loads, asserted against the emitter metadata.
    assert info["task_count"] == 1, "hello world expects exactly one NPU task, got %d" % info["task_count"]
    assert info["input_layout"] == "packed", "the flat ornpu_run API needs the packed input layout"
    assert list(info["shape_nhwc"]) == [1, HEIGHT, WIDTH, CHANNELS], info["shape_nhwc"]
    assert list(info["output_shape_nhwc"]) == [1, HEIGHT, WIDTH, OUT_CHANNELS], info["output_shape_nhwc"]
    assert info["input_bytes"] == HEIGHT * WIDTH * CHANNELS, "flat input buffer size"

    rng = np.random.default_rng(SEED + 1)
    packed = rng.integers(0, 256, info["input_bytes"], dtype=np.uint8)
    (FOLDER / "input.u8").write_bytes(packed.tobytes())

    # Determinism: the same model compiles to the same bytes, so the reported
    # commands are reproducible and the checksum in the file is stable.
    repeat, _ = compile_sequence(model_path)
    assert repeat == binary, "recompiling the tiny model changed the container bytes"

    print("  01_hello_npu_c: model.bin %d B, %d task, input %d B, output %d B -> %s" %
          (len(binary), info["task_count"], info["input_bytes"], info["output_bytes"], FOLDER))
    print("  profile=%s input_scale=%.3g input_zero_point=%d" %
          (report["profile"], info["input_scale"], info["input_zero_point"]))
    print_commands(FOLDER / "model.bin", FOLDER / "input.u8")
    print("01_hello_npu_c: container decodes, recompiles byte-identically, host C build is a one-liner")


if __name__ == "__main__":
    main()
