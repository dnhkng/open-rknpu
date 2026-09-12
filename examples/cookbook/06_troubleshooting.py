"""SPDX-License-Identifier: MIT

Cookbook 6/7: what a rejection looks like, and what to do about it.

Five graphs that are deliberately outside the verified envelope are handed to
`compile_sequence`. Each must raise `ValueError` with a specific, stable message;
the script asserts the message and prints the fix/roadmap pointer from
`docs/roadmap.md` ("Deliberately out of scope today") and
`docs/plans/primitive-roadmap.md`.

The point is not that these graphs fail - it is that they fail with a message a
user can act on, and that the boundary is documented rather than silent.

Run:
    PYTHONPATH=src python examples/cookbook/06_troubleshooting.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper as nh

from cookbook_common import build_dir
from open_rknpu.scheduler import compile_sequence

FOLDER = build_dir("06_troubleshooting")
SEED = 60606


def make_model(nodes, inputs, outputs, initializers):
    graph = h.make_graph(nodes, "unsupported", inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def weights(shape, seed):
    return np.random.default_rng(seed).uniform(-0.25, 0.25, shape).astype(np.float32)


def case_conv_1d():
    """A rank-3 (1-D) Conv: the front end requires static NCHW rank 4."""
    kernel, channels, out_channels, length = 3, 3, 4, 8
    node = h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[kernel],
                       pads=[kernel // 2, kernel // 2], strides=[1])
    return make_model([node], [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, length])],
                      [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_channels, length])],
                      [nh.from_array(weights((out_channels, channels, kernel), SEED), "w"),
                       nh.from_array(np.zeros(out_channels, np.float32), "b")])


def case_kernel_33():
    """A 33x33 kernel: outside the verified register encoding (K <= 31)."""
    kernel, channels, out_channels, size = 33, 3, 4, 64
    node = h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4, strides=[1, 1])
    return make_model([node], [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, size, size])],
                      [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_channels, size, size])],
                      [nh.from_array(weights((out_channels, channels, kernel, kernel), SEED + 1), "w"),
                       nh.from_array(np.zeros(out_channels, np.float32), "b")])


def case_input_c129():
    """129 input channels: one past the compiler's C128 cap."""
    channels, out_channels, size = 129, 4, 8
    node = h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3],
                       pads=[1, 1, 1, 1], strides=[1, 1])
    return make_model([node], [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, size, size])],
                      [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_channels, size, size])],
                      [nh.from_array(weights((out_channels, channels, 3, 3), SEED + 2), "w"),
                       nh.from_array(np.zeros(out_channels, np.float32), "b")])


def case_matmul():
    """A fully connected layer as MatMul: no primitive, and not a Conv graph at all."""
    node = h.make_node("MatMul", ["input", "w"], ["output"])
    return make_model([node], [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 8])],
                      [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 8])],
                      [nh.from_array(np.eye(8, dtype=np.float32), "w")])


def case_dynamic_shape():
    """A symbolic H dimension: containers are immutable, so every dim must be static."""
    node = h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3],
                       pads=[1, 1, 1, 1], strides=[1, 1])
    return make_model([node], [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "H", 8])],
                      [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
                      [nh.from_array(weights((4, 3, 3, 3), SEED + 3), "w"),
                       nh.from_array(np.zeros(4, np.float32), "b")])


# (label, builder, expected substring, fix/roadmap pointer)
CASES = [
    ("1-D conv (rank-3 input)", case_conv_1d, "static NCHW input required",
     "reshape to [1,C,1,W] (docs/roadmap.md; docs/plans/primitive-roadmap.md, 1-D convolution)"),
    ("kernel 33x33", case_kernel_33, "odd K1..31",
     "split large kernels into shorter taps (docs/roadmap.md, 'Kernels > 31')"),
    ("input C129", case_input_c129, "input C1..128",
     "the C128 cap is a vendor register bound; split the channels (docs/primitives.md; primitive-roadmap.md)"),
    ("MatMul / fully connected", case_matmul, "sequence lowering requires one input, one output, and an initial Conv",
     "use a 1x1 Conv for a fully connected layer (docs/roadmap.md, MatMul/Gemm row)"),
    ("dynamic shape (dim_param)", case_dynamic_shape, "static batch1..16",
     "every dimension must be static; recompile per shape (docs/roadmap.md, 'Dynamic shapes')"),
]


def main():
    rows = []
    for index, (label, builder, expected, pointer) in enumerate(CASES):
        model_path = FOLDER / ("case%d.onnx" % index)
        onnx.save(builder(), model_path)
        try:
            compile_sequence(model_path)
        except ValueError as error:
            message = str(error)
            assert expected in message, "%s: expected %r in %r" % (label, expected, message)
            rows.append((label, message, pointer))
            print("  %-26s rejected: %s" % (label, message))
            print("  %-26s expected text present; fix: %s" % ("", pointer))
        else:
            raise AssertionError("%s compiled, but this cookbook asserts it is unsupported" % label)
    print("06_troubleshooting: %d/%d unsupported graphs rejected with ValueError and the documented text" %
          (len(rows), len(CASES)))
    print("06_troubleshooting: boundary is documented in docs/roadmap.md and docs/plans/primitive-roadmap.md")


if __name__ == "__main__":
    main()
