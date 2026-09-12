"""SPDX-License-Identifier: MIT

Independently generated N-layer dense Conv chain for RV1103.

Supported graph: `[Conv, Relu] * (N-1) + [Conv]` at fixed `[1,3,8,8]`,
N >= 2, hidden channels 3..16, kernels 1x1 or padded 3x3, constant float32
weights and bias. The first layer uses the established legacy program; every
later layer is a native16 program reading the previous signed intermediate.
Intermediate buffers share one arena, so tensor lifetime is explicit. No vendor
capture, RKNN model or library is read.
"""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .chain import NATIVE, native_quantize, native_reference
from .compiler import compile_model
from .register_profile import REGISTERS
from .sequence import (encode_sequence, encode_sequence_v5, LAYOUT_PACKED_U8, LAYOUT_NATIVE16,
                       ROLE_INPUT, ROLE_OUTPUT)

SURFACE = 8 * 8 * 16  # native16 bytes per 8x8 plane


def _align(n, a=64):
    return (n + a - 1) // a * a


def _validate(model):
    onnx.checker.check_model(model)
    graph = model.graph
    nodes = list(graph.node)
    if len(nodes) < 3 or len(nodes) % 2 == 0 or any(n.op_type not in ("Conv", "Relu") for n in nodes):
        raise ValueError("native chain requires [Conv, Relu]*(N-1) + [Conv]")
    for i, node in enumerate(nodes):
        expected = "Conv" if i % 2 == 0 else "Relu"
        if node.op_type != expected or node.domain not in ("", "ai.onnx"):
            raise ValueError("native chain requires alternating Conv and Relu")
    if not all(nodes[i].output[0] == nodes[i + 1].input[0] for i in range(len(nodes) - 1)):
        raise ValueError("native chain nodes must be connected in order")
    if len(graph.input) != 1 or len(graph.output) != 1 or nodes[0].input[0] != graph.input[0].name \
       or nodes[-1].output[0] != graph.output[0].name:
        raise ValueError("native chain requires one input and one output")
    for value in (graph.input[0], graph.output[0]):
        if value.type.tensor_type.elem_type != 1 or [d.dim_value for d in value.type.tensor_type.shape.dim] != [1, 3, 8, 8]:
            raise ValueError("native chain external tensors must be float32 [1,3,8,8]")
    constants = {t.name: nh.to_array(t) for t in graph.initializer}
    convs = nodes[::2]
    layers = []
    for index, conv in enumerate(convs):
        if len(conv.input) != 3 or any(name not in constants for name in conv.input[1:]):
            raise ValueError("native chain convolutions require constant weights and bias")
        w, b = constants[conv.input[1]], constants[conv.input[2]]
        if w.ndim != 4 or w.dtype != np.float32 or b.dtype != np.float32:
            raise ValueError("native chain weights and bias must be float32 rank-four")
        kernel = w.shape[2]
        attrs = {a.name: h.get_attribute_value(a) for a in conv.attribute}
        supported = {"kernel_shape": [kernel, kernel], "pads": [kernel // 2] * 4,
                     "strides": [1, 1], "dilations": [1, 1], "group": 1}
        if kernel not in (1, 3) or any(k not in supported or v != supported[k] for k, v in attrs.items()):
            raise ValueError("native chain supports 1x1 or padded 3x3, stride1, group1")
        layers.append((w, b, kernel))
    if layers[0][0].shape[1] != 3 or layers[-1][0].shape[0] != 3:
        raise ValueError("native chain requires three external input and output channels")
    for w, b, _ in layers:
        if not 3 <= w.shape[0] <= 16 or b.shape != (w.shape[0],):
            raise ValueError("native chain hidden channels must be 3..16 with matching bias")
    for index in range(1, len(layers)):
        if layers[index][0].shape[1] != layers[index - 1][0].shape[0]:
            raise ValueError("native chain hidden channel counts must match between layers")
    return graph, nodes, convs, layers, constants


def _layer_tensor_names(nodes, count):
    """The measured tensor of each layer, in layer order.

    `open_rknpu.calibration.measured_tensor_names` names a Conv output by the fused
    activation output when a `Relu` directly follows, so a hidden layer's tensor is its
    `Relu` output and the final Conv's is the Conv output.
    """
    return [nodes[2 * index + 1].output[0] if index < count - 1 else nodes[2 * index].output[0]
            for index in range(count)]


def compile_chain_n(model, output_range=None, expose_intermediates=False, reuse_intermediates=False,
                    serial=True, calibration_ranges=None):
    graph, nodes, convs, layers, constants = _validate(model)
    count = len(layers)
    if calibration_ranges is not None and output_range is not None:
        raise ValueError('calibration and output quantization overrides cannot be combined')
    # Every layer's band comes from the same measured-band helper the other profiles use;
    # a missing entry fails before any program is emitted.
    from .calibration import measured_range
    bands = [measured_range(calibration_ranges, name)
             for name in _layer_tensor_names(nodes, count)]
    # Layer 1 program and constants from the established single-layer emitter.
    first_activation = nodes[1]
    first_graph = h.make_graph([convs[0], first_activation], "layer1", list(graph.input),
        [h.make_tensor_value_info(first_activation.output[0], 1, [1, layers[0][0].shape[0], 8, 8])],
        [nh.from_array(constants[name], name) for name in convs[0].input[1:]])
    first_model = h.make_model(first_graph, opset_imports=list(model.opset_import))
    first_model.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layer1.onnx"
        onnx.save(first_model, path)
        first_data, first_meta = compile_model(path, calibration_ranges=calibration_ranges)
    quantizations = [None] * count
    quantizations[0] = first_meta
    scale, zero_point = first_meta["output_scale"], first_meta["output_zero_point"]
    for index in range(1, count):
        w, b, _ = layers[index]
        if calibration_ranges is not None:
            selected = bands[index]
        else:
            selected = output_range if index == count - 1 else None
        quantizations[index] = native_quantize(w, b, scale, zero_point, selected)
        # The graph is `[Conv, Relu]*(N-1) + [Conv]`, and a native layer carries its
        # activation in its own program (registers 0x4060/0x406c/0x40e0). Until S9 the
        # hidden layers left it off, so the Relu was silently dropped; the flag also
        # drives `chain_n_reference_layers`, so container and reference stay together.
        quantizations[index].relu = index < count - 1
        scale, zero_point = quantizations[index].output_scale, quantizations[index].output_zero_point
    # Layout: programs, layer constants, then arena tensors.
    programs = [index * 0x440 for index in range(count)]
    cursor = _align(programs[-1] + (126 + 4) * 8)
    first_kernel = layers[0][2]
    first_weight_size = _align(first_kernel * first_kernel * _align(layers[0][0].shape[0], 4) * 4)
    offsets = []
    offsets.append((cursor, cursor + first_weight_size))
    cursor = _align(cursor + first_weight_size + ((layers[0][0].shape[0] + 3) // 4) * 32)
    for index in range(1, count):
        w, _, kernel = layers[index]
        weight_size = _align(w.shape[0] * 16 * kernel * kernel)
        offsets.append((cursor, cursor + weight_size))
        cursor = _align(cursor + weight_size + ((w.shape[0] + 3) // 4) * 32)
    payload_size = _align(cursor)
    input_offset = _align(payload_size, 4096)
    if reuse_intermediates and not expose_intermediates and count > 3:
        # Lifetime reuse: layer L writes buffer (L-1)%2, which was last read by
        # layer L-1, so only two intermediate buffers are ever live.
        intermediates = [input_offset + SURFACE * (1 + (index % 2)) for index in range(count - 1)]
        output_offset = input_offset + SURFACE * 3
        arena = _align(output_offset + SURFACE, 4096)
    else:
        intermediates = [input_offset + SURFACE * (index + 1) for index in range(count - 1)]
        output_offset = input_offset + SURFACE * count
        arena = _align(output_offset + SURFACE, 4096)
    data = bytearray(payload_size)
    first_source = {w & 65535: (w >> 16) & 0xffffffff for w in struct.unpack_from("<126Q", first_data)}
    first_weights, first_bias = offsets[0]
    registers = dict(first_source)
    registers.update({0x1110: first_weights, 0x5020: first_bias, 0x4020: intermediates[0] if count > 1 else output_offset,
                      0x1070: input_offset})
    for i, (reg, default, tag) in enumerate(REGISTERS):
        struct.pack_into("<Q", data, programs[0] + i * 8, tag << 48 | registers.get(reg, default) << 16 | reg)
    _layer_tail(data, programs[0], programs[1] if count > 1 else None, serial)
    data[first_weights:first_weights + first_weight_size] = first_data[first_source[0x1110]:first_source[0x1110] + first_weight_size]
    first_bias_size = ((layers[0][0].shape[0] + 3) // 4) * 32
    data[first_bias:first_bias + first_bias_size] = first_data[first_source[0x5020]:first_source[0x5020] + first_bias_size]
    for index in range(1, count):
        w, b, kernel = layers[index]
        q = quantizations[index]
        hidden = w.shape[1]
        channels = w.shape[0]
        relu = bool(getattr(q, "relu", False))
        weights_offset, bias_offset = offsets[index]
        fields = dict(NATIVE)
        # 0x1184 is the input zero point the CNA declares *and* the value it pads borders
        # with (docs/registers.md). A hidden layer reads the previous layer's native16
        # grid, whose band is `quantizations[index-1]`, so the border must be that band's
        # zero point - leaving the register at its 0xff80 default pads with zero point 0
        # and corrupts deep chains (measured: 3.4e7 mean error on an 8-layer chain against
        # 43 with the propagated value).
        previous = quantizations[index - 1]
        previous_zero_point = (previous["output_zero_point"] if isinstance(previous, dict)
                               else previous.output_zero_point)
        fields.update({0x1184: int(previous_zero_point) & 0xffff,
                       0x1010: 0x108 if kernel == 3 else 0x104,
                       0x1030: channels * 16 * kernel * kernel, 0x1034: 16 * kernel * kernel,
                       0x1038: (kernel << 24) | (kernel << 16) | channels,
                       0x1068: 0x101 if kernel == 3 else 0, 0x1188: 8 * kernel * kernel,
                       0x1024: ((hidden - 1) << 16) | 16,
                       0x403c: ((channels - 1) << 16) | 0xf,
                       0x4060: 0x12 if relu else 0x13,
                       0x406c: 0 if relu else 0x80000000,
                       0x40e0: 0 if relu else 0x80000000,
                       0x1070: intermediates[index - 1], 0x1110: weights_offset, 0x5020: bias_offset,
                       0x4020: intermediates[index] if index < count - 1 else output_offset,
                       0x4080: q.output_zero_point & 0xffffffff, 0x4084: q.multiplier, 0x4088: q.shift})
        for i, (reg, default, tag) in enumerate(REGISTERS):
            struct.pack_into("<Q", data, programs[index] + i * 8, tag << 48 | fields.get(reg, default) << 16 | reg)
        _layer_tail(data, programs[index], programs[index + 1] if index + 1 < count else None, serial)
        for o in range(channels):
            rows = q.weights[o].reshape(hidden, kernel, kernel)
            for t in range(kernel * kernel):
                y, x = divmod(t, kernel)
                row = list(rows[:, y, x]) + [int(q.weight_zero_points[o])] * (16 - hidden)
                struct.pack_into("<16b", data, weights_offset + (t * channels + o) * 16, *row)
            block = bias_offset + (o // 4) * 32
            lane = o % 4
            struct.pack_into("<i", data, block + lane * 4, int(q.biases[o]))
            struct.pack_into("<h", data, block + 16 + lane * 2, -int(q.weight_zero_points[o]))
            struct.pack_into("<H", data, block + 24 + lane * 2, int(q.channel_multipliers[o]))
    final = quantizations[-1]
    final_scale = final["output_scale"] if isinstance(final, dict) else final.output_scale
    final_zero = final["output_zero_point"] if isinstance(final, dict) else final.output_zero_point
    tasks = [(program, 126, 29, 768) for program in programs]
    hidden_channels = [w.shape[0] for w, _, _ in layers]
    if expose_intermediates:
        # Version 5: every layer output is a named external output, so the
        # intermediate tensors are visible through ornpu_run_io().
        tensors = [dict(name="input0", role=ROLE_INPUT, layout=LAYOUT_PACKED_U8, index=0,
                        shape=(1, 8, 8, 3), offset=input_offset, size=8 * 16 * 3)]
        for index in range(count - 1):
            tensors.append(dict(name=f"layer{index + 1}", role=ROLE_OUTPUT, layout=LAYOUT_NATIVE16,
                                index=index, shape=(1, 8, 8, hidden_channels[index]),
                                offset=intermediates[index], size=SURFACE))
        tensors.append(dict(name="output", role=ROLE_OUTPUT, layout=LAYOUT_NATIVE16,
                            index=count - 1, shape=(1, 8, 8, 3), offset=output_offset, size=SURFACE))
        binary = encode_sequence_v5(data, tensors=tensors, tasks=tasks, arena_bytes=arena,
            input_scale=1.0, input_zero_point=0, output_scale=final_scale,
            output_zero_point=final_zero, serial=serial)
    else:
        binary = encode_sequence(data, input_shape=(8, 8, 3), output_shape=(8, 8, 3), input_stride=16,
            arena_bytes=arena, input_offset=input_offset, output_offset=output_offset,
            tasks=tasks, input_scale=1.0, input_zero_point=0,
            output_scale=final_scale, output_zero_point=final_zero, serial=serial)
    def as_metadata(value):
        if isinstance(value, dict):
            return value.get("quantization", value)
        return value.metadata()
    meta = dict(profile="native-chain", layers=count, hidden_channels=hidden_channels,
                kernels=[k for _, _, k in layers], output_scale=final_scale, output_zero_point=final_zero,
                shape_nhwc=[1, 8, 8, 3], output_shape_nhwc=[1, 8, 8, 3],
                quantizations=[as_metadata(m) for m in quantizations], first=first_meta,
                exposed_intermediates=bool(expose_intermediates),
                reused_intermediates=bool(reuse_intermediates and not expose_intermediates and count > 3),
                submission="batched" if not serial else "serial",
                output_tensors=(["layer%d" % (i + 1) for i in range(count - 1)] + ["output"]) if expose_intermediates else ["output"])
    return binary, meta


def _terminal(data, offset, link=0, control=0x28):
    """Task tail. A batched list links each task to the next program's payload offset
    (register 0x10, control 0x40) and ends terminal; measured rule, PIPELINING_PLAN S1."""
    for i, (reg, value, tag) in enumerate(((0x10, link, 0x101), (0x14, control, 0x101),
                                           (0, 0, 0x41), (8, 29, 0x81))):
        struct.pack_into("<Q", data, offset + (126 + i) * 8, tag << 48 | value << 16 | reg)


def _layer_tail(data, program, next_program, serial):
    if serial or next_program is None:
        _terminal(data, program)
    else:
        _terminal(data, program, link=next_program, control=0x40)


def chain_n_reference_layers(inputs, quantizations):
    """Per-layer integer outputs for one uint8 HWC input (layer 1 first)."""
    from .quantization import reference
    first = quantizations[0]
    if isinstance(first, dict) and "quantization" in first:
        first = first["quantization"]
    first = _as_quantization(first)
    outputs = [reference(inputs, first)]
    zero_point = first.output_zero_point
    for params in quantizations[1:]:
        quantization = _as_quantization(params)
        outputs.append(native_reference(outputs[-1], quantization, int(zero_point)))
        zero_point = quantization.output_zero_point
    return outputs


def chain_n_reference(inputs, quantizations):
    """Final-layer integer output for one uint8 HWC input."""
    return chain_n_reference_layers(inputs, quantizations)[-1]


def _as_quantization(params):
    from .quantization import Quantization
    if isinstance(params, Quantization):
        return params
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)
