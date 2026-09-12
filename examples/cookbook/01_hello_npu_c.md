<!-- SPDX-License-Identifier: MIT -->

# 01 - Hello, NPU: the smallest C program

`01_hello_npu_c.py` compiles a one-task container and prints the commands below;
`hello_npu.c` is the complete program that loads and runs it. The C API is three
calls plus a header-only inspect, exactly as `docs/board.md` states:

```c
ornpu_inspect("model.bin", &info);       /* header only, no device access */
ornpu_open("model.bin", &model);
ornpu_run(model, input_u8, info.input_bytes, output_i8, info.output_bytes);
ornpu_close(model);
```

`hello_npu.c` reads the packed NHWC UINT8 input (`info.input_bytes` bytes, or a
zero-filled buffer), runs one inference, prints the first output codes and exits.

## Generate the model

```sh
PYTHONPATH=src python examples/cookbook/01_hello_npu_c.py
```

That writes `examples/cookbook/build/01_hello_npu_c/` with `hello_npu.onnx`,
`model.bin` (the container) and `input.u8` (one deterministic 8x8 RGB case). The
script asserts the container decodes to exactly one task with the expected
packed input shape and recompiles byte-identically.

## Makefile snippet

```make
CC      ?= gcc
CROSS   ?= research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
SYSROOT ?= $(PWD)/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot
CFLAGS  ?= -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime
BUILD   := examples/cookbook/build/01_hello_npu_c

# Host build: proves hello_npu.c compiles; the device calls are compiled out.
$(BUILD)/hello_npu.host.o: examples/cookbook/hello_npu.c runtime/open_rknpu.h
	$(CC) $(CFLAGS) -c $< -o $@

# Board build: the real program, with the libc-only runtime linked in.
$(BUILD)/hello_npu: examples/cookbook/hello_npu.c runtime/open_rknpu.c runtime/open_rknpu.h
	$(CROSS) --sysroot="$(SYSROOT)" $(CFLAGS) -DORNPU_BOARD \
	  examples/cookbook/hello_npu.c runtime/open_rknpu.c -o $@
```

The host line is what the repository verifies:

```sh
gcc -O2 -Wall -Wextra -Werror -Iruntime -c examples/cookbook/hello_npu.c -o /tmp/hello_npu.o
```

The device calls are guarded by `ORNPU_BOARD`, so a host build compiles (and, if
run, explains how to build for the board) while the board build is the only one
that touches `/dev/rknpu`.

## Cross-compile and run on the board

From `docs/board.md` (build the runtime), with the cookbook paths:

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -DORNPU_BOARD -Iruntime \
  examples/cookbook/hello_npu.c runtime/open_rknpu.c \
  -o examples/cookbook/build/01_hello_npu_c/hello_npu

adb shell mkdir -p /userdata/open-npu-research/hello_npu
adb push examples/cookbook/build/01_hello_npu_c/hello_npu \
         examples/cookbook/build/01_hello_npu_c/model.bin \
         examples/cookbook/build/01_hello_npu_c/input.u8 \
         /userdata/open-npu-research/hello_npu/
adb shell 'cd /userdata/open-npu-research/hello_npu && ./hello_npu model.bin input.u8'
```

Expected shape of the output (values depend on the compiled weights):

```text
container: 8x8 input, 8x8 output, 192 input bytes, 256 output bytes
task_count=1 serial=1 profile tasks run as separate submissions
output codes [0..16): -12 7 33 -5 41 0 19 -27 8 12 -3 55 21 -9 4 17
```

The toolchain is not in the repository; `research/fetch_toolchain.py` fetches it.
Any ARM uClibc compiler with the board's sysroot works. Keep `rkipc` alive and
stage under `/userdata/open-npu-research/` (see `docs/board.md`).

## What is demonstrated here

* the whole C surface a first program needs, in one file;
* a host build that compiles without the vendor toolchain (the CI check);
* the container the script produced: one task, packed input, checksum valid, and
  reproducible byte for byte.

Deeper API material is in `docs/board.md`, `docs/api.md` and
`docs/container-format.md`; `examples/primitives/` compiles the operators and
`tests/board_io.c` is the reference runner for multi-model suites.
