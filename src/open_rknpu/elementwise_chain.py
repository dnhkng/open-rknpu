"""SPDX-License-Identifier: MIT

N-stage elementwise DAG for RV1103: `Op1(a,b)` then `Mul(stage, a)` per stage.

The first stage is compiled with the verified two-branch elementwise emitter;
each later stage appends an independently generated 78-word Mul task that reads
the previous stage and the first branch output (fan-out). Two external inputs,
one output, no broadcasting. No vendor capture, RKNN model or library is read.
"""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h
from .elementwise import mul_output_conversion, mul_requant_reference
from .quantization import Quantization, reference
from .register_profile import REGISTERS
from .sequence import encode_sequence, decode_sequence

SURFACE = 1024


def align(n, a=64):
    return (n + a - 1) // a * a


def add_reference(a, b):
    return np.rint((a.astype(np.int32) + b.astype(np.int32)) / 2).astype(np.int8)


def sub_reference(a, b):
    return np.clip(np.rint((a.astype(np.int32) - b.astype(np.int32)) / 2), -128, 127).astype(np.int8)


def max_reference(a, b):
    return np.rint(np.maximum(a, b).astype(np.int32) / 2).astype(np.int8)


FIRST_STAGES = ("Add", "Mul", "Sub", "Max")


def _validate(model):
    onnx.checker.check_model(model)
    graph = model.graph
    nodes = list(graph.node)
    ops = [n.op_type for n in nodes]
    if (len(nodes) < 4 or ops[:2] != ["Conv", "Conv"] or ops[2] not in FIRST_STAGES
        or any(op != "Mul" for op in ops[3:])):
        raise ValueError("elementwise DAG requires Conv, Conv, {Add|Mul|Sub|Max}, Mul...")
    conv_a, conv_b, first = nodes[0], nodes[1], nodes[2]
    if any(n.domain not in ("", "ai.onnx") for n in nodes):
        raise ValueError("unsupported operator domain")
    if len(graph.input) != 2 or len(graph.output) != 1:
        raise ValueError("elementwise DAG requires two inputs and one output")
    if first.input[0] != conv_a.output[0] or first.input[1] != conv_b.output[0]:
        raise ValueError("the first stage must combine the two Conv branches")
    previous = first.output[0]
    for node in nodes[3:]:
        if list(node.input) != [previous, conv_a.output[0]]:
            raise ValueError("each later stage must multiply the previous stage by the first branch")
        previous = node.output[0]
    if previous != graph.output[0].name:
        raise ValueError("the last stage must produce the graph output")
    if conv_a.input[0] != graph.input[0].name or conv_b.input[0] != graph.input[1].name:
        raise ValueError("Conv branches must consume the two graph inputs in order")
    if any(n.attribute for n in nodes[2:]):
        raise ValueError("elementwise DAG operators take no attributes")
    for node in (conv_a, conv_b):
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        supported = {"kernel_shape": [1, 1], "pads": [0, 0, 0, 0], "strides": [1, 1],
                     "dilations": [1, 1], "group": 1}
        if any(k not in supported or v != supported[k] for k, v in attrs.items()):
            raise ValueError("elementwise DAG branches must be 1x1 Conv")
        if len(node.input) != 3:
            raise ValueError("elementwise DAG branches require constant weights and bias")
    for value in (graph.input[0], graph.input[1], graph.output[0]):
        if value.type.tensor_type.elem_type != 1 or [d.dim_value for d in value.type.tensor_type.shape.dim] != [1, 3, 8, 8]:
            raise ValueError("elementwise DAG external tensors must be float32 [1,3,8,8]")
    return graph, nodes


def compile_elementwise_dag(model):
    graph, nodes = _validate(model)
    first = nodes[2]
    first_stage = first.op_type
    extra_stages = len(nodes) - 3
    values = {v.name: v for v in [*graph.input, *graph.value_info, *graph.output]}
    if first.output[0] not in values:
        raise ValueError("missing static first-stage output shape")
    sub = h.make_model(h.make_graph(nodes[:3], "first_stage", list(graph.input),
        [values[first.output[0]]], list(graph.initializer)), opset_imports=list(model.opset_import))
    sub.ir_version = model.ir_version
    from .scheduler import compile_sequence
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
        raise ValueError("elementwise DAG requires zero-point-zero stages")
    if first_stage in ("Add", "Sub", "Max") and abs(first_scale - 2 * branch_scale) > 1e-6:
        raise ValueError("elementwise DAG requires the verified equal-scale " + first_stage)
    # Plan every appended Mul stage, then relocate the IO region past them.
    program_base = 96 + 16 * info["task_count"]
    payload = bytearray(binary[program_base:])
    base_payload = len(payload)
    template = bytes(payload[info["tasks"][-1]["command_offset"]:info["tasks"][-1]["command_offset"] + 78 * 8])
    programs_offset = base_payload
    appended_end = align(programs_offset + extra_stages * (78 + 4) * 8)
    new_input = align(appended_end, 4096)
    delta = new_input - info["input_offset"]
    stage_scales = [first_scale]
    stage_conversions = []
    addresses = [0x3000]
    for stage in range(extra_stages):
        product_scale = stage_scales[-1] * branch_scale
        output_scale_stage, _zp, conversion, shift = mul_output_conversion(product_scale, None)
        stage_conversions.append((conversion, shift))
        stage_scales.append(output_scale_stage)
        addresses.append(0x3400 + SURFACE * stage)
    # Write the appended programs with relocated addresses.
    program = bytearray(template)
    for stage in range(extra_stages):
        program_offset = programs_offset + stage * (78 + 4) * 8
        conversion, shift = stage_conversions[stage]
        fields = {0x4048: 0, 0x4050: 0x30000002, 0x4070: 0x81004094, 0x4074: 0, 0x4078: 1,
                  0x4080: 0, 0x4084: conversion, 0x4088: shift,
                  0x5018: addresses[stage] + delta, 0x5038: 0x2000 + delta,
                  0x4020: addresses[stage + 1] + delta}
        program[:] = template
        for i, (reg, _default, tag) in enumerate([r for r in REGISTERS if r[0] >= 0x4000][:78]):
            word = struct.unpack_from("<Q", program, i * 8)[0]
            current = (word >> 16) & 0xffffffff
            value = fields.get(reg, current)
            struct.pack_into("<Q", program, i * 8, tag << 48 | value << 16 | reg)
        payload[program_offset:program_offset + 78 * 8] = program
        payload.extend(bytes(program_offset + (78 + 4) * 8 - len(payload)))
        for j, (reg, value, tag) in enumerate(((0x10, 0, 0x101), (0x14, 0x28, 0x101), (0, 0, 0x41), (8, 24, 0x81))):
            struct.pack_into("<Q", payload, program_offset + (78 + j) * 8, tag << 48 | value << 16 | reg)
    if len(payload) < appended_end:
        payload.extend(bytes(appended_end - len(payload)))
    address_regs = {0x1070, 0x4020, 0x5018, 0x5038}
    for task in info["tasks"]:
        for i in range(task["register_count"]):
            offset = task["command_offset"] + i * 8
            word = struct.unpack_from("<Q", payload, offset)[0]
            reg = word & 0xffff
            value = (word >> 16) & 0xffffffff
            if reg in address_regs and info["input_offset"] <= value < info["arena_bytes"]:
                value += delta
                struct.pack_into("<Q", payload, offset, (word >> 48) << 48 | value << 16 | reg)
    tasks = [(t["command_offset"], t["register_count"], t["enable"], t["mask"]) for t in info["tasks"]]
    for stage in range(extra_stages):
        tasks.append((programs_offset + stage * (78 + 4) * 8, 78, 24, 768))
    output_offset = addresses[-1] + delta
    output_scale = stage_scales[-1]
    output_zero_point = 0
    chain = f"{first_stage}(a,b)"
    for _ in range(extra_stages):
        chain = f"Mul({chain},a)"
    result = encode_sequence(payload, input_shape=(info["shape_nhwc"][1], 8, 3), output_shape=(8, 8, 3),
        input_stride=16, arena_bytes=align(output_offset + SURFACE, 4096), input_offset=new_input,
        output_offset=output_offset, tasks=tasks, input_scale=info["input_scale"],
        input_zero_point=info["input_zero_point"], output_scale=output_scale,
        output_zero_point=output_zero_point, serial=True, input_tensor_count=2)
    return result, dict(meta, elementwise_chain=chain, first_stage=first_stage, stages=extra_stages + 1,
                        stage_scales=stage_scales, stage_conversions=[list(c) for c in stage_conversions],
                        output_scale=output_scale, output_zero_point=output_zero_point,
                        branch_scale=branch_scale, output_shape_nhwc=[1, 8, 8, 3])


def chain_reference(input_a, input_b, meta):
    """Compose the stage references for two uint8 HWC inputs."""
    quantizations = [Quantization(**{k: np.array(v) if isinstance(v, list) else v for k, v in branch["quantization"].items()})
                     for branch in meta["branches"]]
    a = reference(input_a, quantizations[0])
    b = reference(input_b, quantizations[1])
    first = meta.get("first_stage", "Add")
    if first == "Add":
        stage = add_reference(a, b)
    elif first == "Sub":
        stage = sub_reference(a, b)
    elif first == "Max":
        stage = max_reference(a, b)
    elif first == "Mul":
        _, _, conversion, shift = mul_output_conversion(float(meta["branches"][0]["output_scale"])
                                                        * float(meta["branches"][1]["output_scale"]), None)
        stage = mul_requant_reference(a, b, conversion & 0xffff, shift, 0)
    else:
        raise ValueError("unknown first stage: " + str(first))
    for conversion, shift in meta.get("stage_conversions", []):
        stage = mul_requant_reference(stage, a, conversion & 0xffff, shift, 0)
    return stage


compile_mul_after_add = compile_elementwise_dag
