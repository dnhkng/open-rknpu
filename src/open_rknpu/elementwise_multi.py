"""SPDX-License-Identifier: MIT

Three-input elementwise DAG for RV1103, emitted as a version-5 executable:

    a -> ConvA -+
                +-> Op1 -> C -+
    b -> ConvB -+              +-> Mul -> output
    c -> ConvC --------------- D

Two external inputs feed the first stage; a third external input is converted by
an identity Conv and multiplied with the first-stage result. The container names
all three inputs, so the runtime binds one buffer each through `ornpu_run_io`.
Independently generated commands; no vendor capture, RKNN model or library.
"""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .elementwise import mul_output_conversion, mul_requant_reference
from .elementwise_chain import (FIRST_STAGES, add_reference, sub_reference, max_reference)
from .quantization import Quantization, reference
from .register_profile import REGISTERS
from .sequence import encode_sequence_v5, decode_sequence, LAYOUT_PACKED_U8, LAYOUT_NATIVE16, ROLE_INPUT, ROLE_OUTPUT, ROLE_INTERNAL

SURFACE = 1024
PACKED_INPUT = 8 * 16 * 3  # HWC bytes for one 8x8 RGB input


def align(n, a=64):
    return (n + a - 1) // a * a


def _validate(model):
    onnx.checker.check_model(model)
    graph = model.graph
    nodes = list(graph.node)
    ops = [n.op_type for n in nodes]
    if (len(nodes) < 5 or len(nodes) % 2 == 0 or ops[0] != "Conv" or ops[1] != "Conv"
        or ops[2] not in FIRST_STAGES or ops[-1] != "Mul"):
        raise ValueError("multi-input DAG requires Conv, Conv, {Add|Mul|Sub|Max}, (Conv, Mul)*")
    conv_a, conv_b, first = nodes[0], nodes[1], nodes[2]
    extras = [(nodes[i], nodes[i + 1]) for i in range(3, len(nodes), 2)]
    if any(conv.op_type != "Conv" or mul.op_type != "Mul" for conv, mul in extras):
        raise ValueError("each extra input requires a Conv then a Mul")
    if any(n.domain not in ("", "ai.onnx") for n in nodes):
        raise ValueError("unsupported operator domain")
    if len(graph.input) != 2 + len(extras) or len(graph.output) != 1:
        raise ValueError("multi-input DAG requires one input per Conv and one output")
    if list(first.input) != [conv_a.output[0], conv_b.output[0]]:
        raise ValueError("the first stage must combine the two Conv branches")
    previous = first.output[0]
    for conv, mul in extras:
        if list(mul.input) != [previous, conv.output[0]]:
            raise ValueError("each extra stage must multiply the previous result by its Conv")
        previous = mul.output[0]
    if previous != graph.output[0].name:
        raise ValueError("the last stage must produce the graph output")
    convs = [conv_a, conv_b] + [conv for conv, _ in extras]
    if [node.input[0] for node in convs] != [v.name for v in graph.input]:
        raise ValueError("Conv branches must consume the graph inputs in order")
    if first.attribute or any(mul.attribute for _conv, mul in extras):
        raise ValueError("elementwise DAG operators take no attributes")
    for node in convs:
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        supported = {"kernel_shape": [1, 1], "pads": [0, 0, 0, 0], "strides": [1, 1],
                     "dilations": [1, 1], "group": 1}
        if any(k not in supported or v != supported[k] for k, v in attrs.items()):
            raise ValueError("elementwise DAG branches must be 1x1 Conv")
        if len(node.input) != 3:
            raise ValueError("elementwise DAG branches require constant weights and bias")
    for value in (*graph.input, *graph.output):
        if value.type.tensor_type.elem_type != 1 or [d.dim_value for d in value.type.tensor_type.shape.dim] != [1, 3, 8, 8]:
            raise ValueError("elementwise DAG external tensors must be float32 [1,3,8,8]")
    return graph, nodes, first_stage_of(nodes), extras


def first_stage_of(nodes):
    return nodes[2].op_type


def _identity_conv_model():
    w = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
    b = np.zeros(3, np.float32)
    graph = h.make_graph([h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[1, 1])],
        "identity", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(w, "w"), nh.from_array(b, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def compile_multi_input_dag(model):
    graph, nodes, first_stage, extras = _validate(model)
    count = len(extras)
    values = {v.name: v for v in [*graph.input, *graph.output, *graph.value_info]}
    first = nodes[2]
    if first.output[0] not in values:
        raise ValueError("missing static first-stage output shape")
    sub = h.make_model(h.make_graph(nodes[:3], "first_stage", list(graph.input[:2]),
        [values[first.output[0]]], list(graph.initializer)), opset_imports=list(model.opset_import))
    sub.ir_version = model.ir_version
    from .scheduler import compile_sequence
    from .compiler import compile_model
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "first.onnx"
        onnx.save(sub, path)
        binary, meta = compile_sequence(path)
    info = decode_sequence(binary)
    if info["task_count"] != 3 or info["output_shape_nhwc"] != [1, 8, 8, 3]:
        raise ValueError("unexpected first-stage container")
    branch_scale = float(meta["branches"][0]["output_scale"])
    first_scale = float(meta["output_scale"])
    if meta["output_zero_point"] != 0:
        raise ValueError("multi-input DAG requires zero-point-zero stages")
    if first_stage in ("Add", "Sub", "Max") and abs(first_scale - 2 * branch_scale) > 1e-6:
        raise ValueError("multi-input DAG requires the verified equal-scale " + first_stage)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "identity.onnx"
        onnx.save(_identity_conv_model(), path)
        third_payload, third_meta = compile_model(path, output_scale=branch_scale, output_zero_point=0)
    third_regs = {w & 65535: (w >> 16) & 0xffffffff for w in struct.unpack_from("<126Q", third_payload)}
    weight_size = align(third_regs[0x1030]); bias_size = ((3 + 3) // 4) * 32
    # Stage scales and conversions.
    stage_scales = [first_scale]
    conversions = []
    for _ in range(count):
        output_scale_stage, _zp, conversion, shift = mul_output_conversion(stage_scales[-1] * branch_scale, None)
        conversions.append((conversion, shift)); stage_scales.append(output_scale_stage)
    output_scale, output_zero_point = stage_scales[-1], 0
    # Payload: base, then per extra input a Conv program, a Mul program and constants.
    base_payload = bytearray(binary[96 + 16 * info["task_count"]:])
    payload = bytearray(base_payload)
    layout = []
    for k in range(count):
        payload.extend(bytes(align(len(payload)) - len(payload)))
        conv_program = len(payload)
        payload.extend(bytes(130 * 8))
        mul_program = align(len(payload))
        payload.extend(bytes(mul_program - len(payload)))
        payload.extend(bytes(82 * 8))
        weights = align(len(payload))
        payload.extend(bytes(weights - len(payload)))
        payload.extend(bytes(weight_size + bias_size))
        layout.append((conv_program, mul_program, weights, weights + weight_size))
    payload.extend(bytes(align(len(payload)) - len(payload)))
    # Place inputs and buffers after the payload so large graphs cannot collide.
    cursor = align(len(payload))
    addresses = {"a": cursor, "b": cursor + PACKED_INPUT}
    for k in range(count):
        addresses[f"c{k}"] = addresses["b"] + PACKED_INPUT * (k + 1)
    cursor = align(addresses[f"c{count - 1}"] + PACKED_INPUT)
    A = cursor; B = A + SURFACE; C = B + SURFACE
    D = [C + SURFACE * (1 + 2 * k) for k in range(count)]
    S = [C + SURFACE * (2 + 2 * k) for k in range(count)]
    arena = align(S[-1] + SURFACE, 4096)
    # Write the identity Conv programs.
    conv_program_bytes = bytearray(126 * 8)
    for k in range(count):
        conv_program, _mul_program, weights, bias = layout[k]
        fields = {0x1070: addresses[f"c{k}"], 0x4020: D[k], 0x1110: weights, 0x5020: bias}
        for i, (reg, default, tag) in enumerate(REGISTERS):
            value = fields.get(reg, third_regs.get(reg, default))
            struct.pack_into("<Q", conv_program_bytes, i * 8, tag << 48 | value << 16 | reg)
        payload[conv_program:conv_program + 126 * 8] = conv_program_bytes
        for j, (reg, value, tag) in enumerate(((0x10, 0, 0x101), (0x14, 0x28, 0x101), (0, 0, 0x41), (8, 29, 0x81))):
            struct.pack_into("<Q", payload, conv_program + (126 + j) * 8, tag << 48 | value << 16 | reg)
        payload[weights:weights + weight_size] = third_payload[third_regs[0x1110]:third_regs[0x1110] + weight_size]
        payload[bias:bias + bias_size] = third_payload[third_regs[0x5020]:third_regs[0x5020] + bias_size]
    # Write the terminal Mul programs.
    template = bytes(base_payload[info["tasks"][-1]["command_offset"]:info["tasks"][-1]["command_offset"] + 78 * 8])
    for k in range(count):
        _conv_program, mul_program, _weights, _bias = layout[k]
        previous = C if k == 0 else S[k - 1]
        conversion, shift = conversions[k]
        fields = {0x4048: 0, 0x4050: 0x30000002, 0x4070: 0x81004094, 0x4074: 0, 0x4078: 1,
                  0x4080: 0, 0x4084: conversion, 0x4088: shift,
                  0x5018: previous, 0x5038: D[k], 0x4020: S[k]}
        program = bytearray(template)
        for i, (reg, _default, tag) in enumerate([r for r in REGISTERS if r[0] >= 0x4000][:78]):
            word = struct.unpack_from("<Q", program, i * 8)[0]
            current = (word >> 16) & 0xffffffff
            value = fields.get(reg, current)
            struct.pack_into("<Q", program, i * 8, tag << 48 | value << 16 | reg)
        payload[mul_program:mul_program + 78 * 8] = program
        for j, (reg, value, tag) in enumerate(((0x10, 0, 0x101), (0x14, 0x28, 0x101), (0, 0, 0x41), (8, 24, 0x81))):
            struct.pack_into("<Q", payload, mul_program + (78 + j) * 8, tag << 48 | value << 16 | reg)
    relocation = {
        0x1070: {0x1000: addresses["a"], 0x1180: addresses["b"]},
        0x4020: {0x2000: A, 0x2400: B, 0x3000: C},
        0x5018: {0x2000: A},
        0x5038: {0x2400: B},
    }
    for task in info["tasks"]:
        for i in range(task["register_count"]):
            offset = task["command_offset"] + i * 8
            word = struct.unpack_from("<Q", payload, offset)[0]
            reg = word & 0xffff
            value = (word >> 16) & 0xffffffff
            if reg in relocation and value in relocation[reg]:
                struct.pack_into("<Q", payload, offset, (word >> 48) << 48 | relocation[reg][value] << 16 | reg)
    tasks = [(t["command_offset"], t["register_count"], t["enable"], t["mask"]) for t in info["tasks"]]
    for k in range(count):
        conv_program, mul_program, _weights, _bias = layout[k]
        tasks.append((conv_program, 126, 29, 768))
        tasks.append((mul_program, 78, 24, 768))
    tensors = [
        dict(name="a", role=ROLE_INPUT, layout=LAYOUT_PACKED_U8, index=0, shape=(1, 8, 8, 3), offset=addresses["a"], size=PACKED_INPUT),
        dict(name="b", role=ROLE_INPUT, layout=LAYOUT_PACKED_U8, index=1, shape=(1, 8, 8, 3), offset=addresses["b"], size=PACKED_INPUT),
    ]
    for k in range(count):
        tensors.append(dict(name=f"c{k}", role=ROLE_INPUT, layout=LAYOUT_PACKED_U8, index=2 + k,
                            shape=(1, 8, 8, 3), offset=addresses[f"c{k}"], size=PACKED_INPUT))
    for name, offset in (("A", A), ("B", B), ("C", C)):
        tensors.append(dict(name=name, role=ROLE_INTERNAL, layout=LAYOUT_NATIVE16, index=len(tensors),
                            shape=(1, 8, 8, 3), offset=offset, size=SURFACE))
    for k in range(count):
        tensors.append(dict(name=f"D{k + 1}", role=ROLE_INTERNAL, layout=LAYOUT_NATIVE16, index=len(tensors),
                            shape=(1, 8, 8, 3), offset=D[k], size=SURFACE))
        if k < count - 1:
            tensors.append(dict(name=f"S{k + 2}", role=ROLE_INTERNAL, layout=LAYOUT_NATIVE16, index=len(tensors),
                                shape=(1, 8, 8, 3), offset=S[k], size=SURFACE))
    tensors.append(dict(name="output", role=ROLE_OUTPUT, layout=LAYOUT_NATIVE16, index=0,
                        shape=(1, 8, 8, 3), offset=S[-1], size=SURFACE))
    result = encode_sequence_v5(payload, tensors=tensors, tasks=tasks, arena_bytes=arena,
        input_scale=1.0, input_zero_point=0, output_scale=output_scale, output_zero_point=output_zero_point, serial=True)
    chain = f"{first_stage}(a,b)"
    for _ in range(count):
        chain = f"Mul({chain},c)"
    return result, dict(meta, elementwise_chain=chain, first_stage=first_stage, extra_inputs=count,
                        third=third_meta["quantization"], output_scale=output_scale, output_zero_point=output_zero_point,
                        stage_scales=stage_scales, stage_conversions=[list(c) for c in conversions],
                        output_shape_nhwc=[1, 8, 8, 3])


def multi_input_reference(input_a, input_b, extra_inputs, meta):
    """Compose the first-stage, identity-Conv and Mul references."""
    quantizations = [Quantization(**{k: np.array(v) if isinstance(v, list) else v for k, v in branch["quantization"].items()})
                     for branch in meta["branches"]]
    third = Quantization(**{k: np.array(v) if isinstance(v, list) else v for k, v in meta["third"].items()})
    a = reference(input_a, quantizations[0])
    b = reference(input_b, quantizations[1])
    first = meta["first_stage"]
    if first == "Add":
        stage = add_reference(a, b)
    elif first == "Sub":
        stage = sub_reference(a, b)
    elif first == "Max":
        stage = max_reference(a, b)
    else:
        _, _, conversion, shift = mul_output_conversion(float(meta["branches"][0]["output_scale"])
                                                        * float(meta["branches"][1]["output_scale"]), None)
        stage = mul_requant_reference(a, b, conversion & 0xffff, shift, 0)
    for value, (conversion, shift) in zip(extra_inputs, meta["stage_conversions"]):
        stage = mul_requant_reference(stage, reference(value, third), conversion & 0xffff, shift, 0)
    return stage
