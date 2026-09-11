"""SPDX-License-Identifier: MIT

Dense-plus-depthwise branch join on RV1103, composed through the v5 tensor table.

    input -> stem Conv 3->3 (1x1[, Relu]) -+-> dense Conv 3->3 (1x1/3x3) -+
                                           +-> depthwise Conv group3 (K1/3/5) -+-> join

Both branches read one shared stem grid and one elementwise join consumes them, so
this retires the plan's "multi-task pool/depthwise branches into Mul" blocker. The
depthwise task program is taken from the verified standalone depthwise emitter
(`open_rknpu.depthwise.compile_depthwise`), its four address registers are
relocated into this container's arena and its weight/bias blocks are copied
verbatim; the dense branch is emitted with the shared native field builder. The
arena is placed by `open_rknpu.liveness` from the task read/write sets. No vendor
capture or RKNN object is read.
"""
import struct
import tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .chain import native_quantize, native_reference
from .compiler import compile_model
from .depthwise import compile_depthwise, depthwise_reference
from .graph import (_align, _join_fields, _native_fields, _pack_native_head,
                    _terminal, _terminals, handoff_tails, join_reference, JOIN_TYPES)
from .liveness import Access, plan
from .quantization import Quantization
from .register_profile import REGISTERS
from .sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT, ROLE_INTERNAL,
                       ROLE_OUTPUT, decode_sequence, encode_sequence_v5)


def parse_depthwise_join(nodes):
    """True when the node list has the shared stem plus dense and depthwise branches.

    Structural only: `compile_depthwise_join` runs the full validation. A grouped
    Conv in the second branch position is what distinguishes this from the dense
    diamond, and the join must be the last node.
    """
    if not nodes or nodes[0].op_type != 'Conv':
        return False
    stem_end = 2 if len(nodes) > 1 and nodes[1].op_type == 'Relu' else 1
    if len(nodes) != stem_end + 3:
        return False
    dense, depthwise, join = nodes[stem_end], nodes[stem_end + 1], nodes[stem_end + 2]
    if dense.op_type != 'Conv' or depthwise.op_type != 'Conv' or join.op_type not in JOIN_TYPES:
        return False
    group = next((h.get_attribute_value(a) for a in depthwise.attribute if a.name == 'group'), 1)
    return int(group) > 1


def _validate(model):
    onnx.checker.check_model(model)
    g = model.graph
    nodes = list(g.node)
    if any(node.domain not in ('', 'ai.onnx') for node in nodes):
        raise ValueError('depthwise join requires default-domain nodes')
    stem_end = 2 if len(nodes) > 1 and nodes[1].op_type == 'Relu' else 1
    if len(nodes) != stem_end + 3 or nodes[0].op_type != 'Conv':
        raise ValueError('depthwise join requires stem[,Relu], dense Conv, depthwise Conv and one join')
    stem_nodes = nodes[:stem_end]
    dense, depthwise, join = nodes[stem_end], nodes[stem_end + 1], nodes[stem_end + 2]
    if dense.op_type != 'Conv' or depthwise.op_type != 'Conv' or join.op_type not in JOIN_TYPES:
        raise ValueError('depthwise join requires a dense Conv, a depthwise Conv and Add/Mul/Sub/Max')
    if join.attribute:
        raise ValueError('depthwise join must not carry attributes')
    stem_out = stem_nodes[-1].output[0]
    if dense.input[0] != stem_out or depthwise.input[0] != stem_out:
        raise ValueError('depthwise join branches must both consume the shared stem output')
    if list(join.input) != [dense.output[0], depthwise.output[0]]:
        raise ValueError('depthwise join must consume the dense branch then the depthwise branch')
    if len(g.input) != 1 or len(g.output) != 1 or join.output[0] != g.output[0].name:
        raise ValueError('depthwise join requires one input and one output')
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*g.input, *g.output, *g.value_info]}
    for value in (g.input[0], g.output[0], dense, depthwise):
        name = value.name if isinstance(value, onnx.ValueInfoProto) else value.output[0]
        if shapes.get(name) != [1, 3, 8, 8]:
            raise ValueError('depthwise join tensors must be float32 [1,3,8,8]')
    if (g.input[0].type.tensor_type.elem_type != 1 or g.output[0].type.tensor_type.elem_type != 1):
        raise ValueError('depthwise join external tensors must be float32')
    constants = {t.name: nh.to_array(t) for t in g.initializer}
    for node in (stem_nodes[0], dense, depthwise):
        if len(node.input) != 3 or any(name not in constants for name in node.input[1:]):
            raise ValueError('depthwise join convolutions require constant float32 weights and bias')
        for name in node.input[1:]:
            if constants[name].dtype != np.float32:
                raise ValueError('depthwise join constants must be float32')
    w1 = constants[stem_nodes[0].input[1]]
    if (w1.ndim != 4 or w1.shape != (3, 3, 1, 1)
            or constants[stem_nodes[0].input[2]].shape != (3,)):
        raise ValueError('depthwise join stem must be a 1x1 Conv 3->3')
    stem_attrs = {a.name: h.get_attribute_value(a) for a in stem_nodes[0].attribute}
    stem_supported = {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1],
                      'dilations': [1, 1], 'group': 1}
    if any(k not in stem_supported or v != stem_supported[k] for k, v in stem_attrs.items()):
        raise ValueError('unsupported stem attributes')
    if len(stem_nodes) == 2 and (stem_nodes[1].attribute or list(stem_nodes[1].input) != list(stem_nodes[0].output)):
        raise ValueError('depthwise join stem Relu must consume the stem output')
    wd = constants[dense.input[1]]
    kernel = wd.shape[2] if wd.ndim == 4 else 0
    dense_attrs = {a.name: h.get_attribute_value(a) for a in dense.attribute}
    dense_supported = {'kernel_shape': [kernel, kernel], 'pads': [kernel // 2] * 4,
                       'strides': [1, 1], 'dilations': [1, 1], 'group': 1}
    if (wd.ndim != 4 or kernel not in (1, 3) or wd.shape != (3, 3, kernel, kernel)
            or constants[dense.input[2]].shape != (3,)):
        raise ValueError('depthwise join dense branch must be a 3-output 1x1/3x3 Conv')
    if any(k not in dense_supported or v != dense_supported[k] for k, v in dense_attrs.items()):
        raise ValueError('unsupported dense branch attributes')
    if kernel == 3 and dense_attrs.get('pads') != [1, 1, 1, 1]:
        raise ValueError('depthwise join 3x3 dense branch requires symmetric pad1')
    ww = constants[depthwise.input[1]]
    dw_kernel = ww.shape[2] if ww.ndim == 4 else 0
    dw_attrs = {a.name: h.get_attribute_value(a) for a in depthwise.attribute}
    if (ww.ndim != 4 or dw_kernel not in (1, 3, 5) or ww.shape != (3, 1, dw_kernel, dw_kernel)
            or constants[depthwise.input[2]].shape != (3,)):
        raise ValueError('depthwise join branch must be a group3 depthwise 1x1/3x3/5x5 Conv')
    dw_supported = {'kernel_shape': [dw_kernel, dw_kernel], 'pads': [dw_kernel // 2] * 4,
                    'strides': [1, 1], 'dilations': [1, 1], 'group': 3}
    if any(k not in dw_supported or v != dw_supported[k] for k, v in dw_attrs.items()):
        raise ValueError('unsupported depthwise branch attributes')
    return (stem_nodes, dense, depthwise, join, constants, w1, wd, kernel, ww, dw_kernel)


def compile_depthwise_join(model, output_range=None, operand_zero_points=(0, 0),
                           asymmetric_depthwise=False, serial=True):
    """Dense and depthwise branches from one stem, folded by one elementwise join.

    Both branch grids are quantized with zero point zero: a Mul join folds the two
    free operand scales, while Add/Sub/Max need one shared scale. The depthwise
    branch keeps its verified weight pair layout, including the optional asymmetric
    per-channel weight zero point. Bounds: RGB 8x8 external tensors, 1x1 stem 3->3,
    dense 1x1/3x3 branch, depthwise 1x1/3x3/5x5 group3 branch, zero-point-zero
    operands, and an output override only for a Mul join.
    """
    if tuple(operand_zero_points) != (0, 0):
        raise ValueError('depthwise join requires zero-centered join operands')
    (stem_nodes, dense, depthwise, join, constants, w1, wd, kernel, ww,
     dw_kernel) = _validate(model)
    g = model.graph
    kind = join.op_type
    if output_range is not None and kind != 'Mul':
        raise ValueError('the depthwise join output override requires a Mul join')
    # Stem program from the established single-layer emitter.
    stem_out_name = stem_nodes[-1].output[0]
    first_graph = h.make_graph(stem_nodes, 'stem', list(g.input),
        [h.make_tensor_value_info(stem_out_name, 1, [1, 3, 8, 8])],
        [nh.from_array(constants[name], name) for name in stem_nodes[0].input[1:]])
    first = h.make_model(first_graph, opset_imports=list(model.opset_import))
    first.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        stem_path = Path(tmp) / 'stem.onnx'
        onnx.save(first, stem_path)
        stem_data, stem_meta = compile_model(stem_path)
    stem_scale = stem_meta['output_scale']
    stem_zp = stem_meta['output_zero_point']
    # A standalone depthwise compile supplies the natural branch range and the task
    # program that this container relocates.
    standalone_nodes = list(stem_nodes) + [depthwise]
    standalone = h.make_model(h.make_graph(standalone_nodes, 'depthwise_branch', list(g.input),
        [h.make_tensor_value_info(depthwise.output[0], 1, [1, 3, 8, 8])],
        [nh.from_array(constants[name], name)
         for node in (stem_nodes[0], depthwise) for name in node.input[1:]],
        value_info=[h.make_tensor_value_info(stem_out_name, 1, [1, 3, 8, 8])]),
        opset_imports=list(model.opset_import))
    standalone.ir_version = model.ir_version
    def adjusted_scale(scale, zero_point):
        return float(np.float32(scale * max(128 + zero_point, 127 - zero_point) / 127))
    _, natural_meta = compile_depthwise(standalone)
    natural_dense = native_quantize(wd, constants[dense.input[2]], stem_scale, stem_zp, None)
    adjusted_dense = adjusted_scale(natural_dense.output_scale, natural_dense.output_zero_point)
    adjusted_dw = adjusted_scale(natural_meta['output_scale'], natural_meta['output_zero_point'])
    if kind == 'Mul':
        scales = [adjusted_dense, adjusted_dw]
    else:
        scales = [max(adjusted_dense, adjusted_dw)] * 2
    q_dense = native_quantize(wd, constants[dense.input[2]], stem_scale, stem_zp,
                              dict(scale=scales[0], zero_point=0))
    branch, branch_meta = compile_depthwise(standalone, output_range=dict(scale=scales[1], zero_point=0),
                                            asymmetric_pair=asymmetric_depthwise)
    info = decode_sequence(branch)
    payload = branch[96 + 16 * info['task_count']:]
    dw_words = struct.unpack_from('<126Q', payload, 0x440)
    dw_regs = {word & 65535: (word >> 16) & 0xffffffff for word in dw_words}
    dw_weight_size = dw_kernel * dw_kernel * 32
    dw_weight_source = dw_regs[0x1110]
    dw_bias_source = dw_regs[0x5020]
    dw_bias_size = ((3 + 3) // 4) * 24
    # Payload layout: four programs, then the packed constants and the arena.
    stem_basis = 0
    dense_basis = 0x440
    dw_basis = 0x880
    join_basis = 0xcc0
    cursor = _align(join_basis + (78 + 4) * 8)
    stem_weight_size = _align(1 * 1 * _align(3, 4) * 4)
    stem_weights = cursor
    stem_bias = stem_weights + stem_weight_size
    stem_bias_size = ((3 + 3) // 4) * 32
    cursor = _align(stem_bias + stem_bias_size)
    dense_weight_size = _align(3 * 16 * kernel * kernel)
    dense_weights = cursor
    dense_bias = dense_weights + dense_weight_size
    cursor = _align(dense_bias + 32)
    dw_weights = cursor
    cursor = _align(dw_weights + dw_weight_size)
    dw_bias = cursor
    cursor = _align(dw_bias + dw_bias_size)
    payload_size = _align(cursor)
    surface = 8 * 8 * 16
    input_off = _align(payload_size, 4096)
    input_bytes = 8 * 16 * 3
    bindings = [Access(reads=('input0',), writes=('stem',)),
                Access(reads=('stem',), writes=('dense',)),
                Access(reads=('stem',), writes=('depthwise',)),
                Access(reads=('dense', 'depthwise'), writes=('output',))]
    sizes = {'stem': surface, 'dense': surface, 'depthwise': surface}
    order, intervals, offsets = plan(bindings, sizes, start=_align(input_off + input_bytes),
                                     defined=('input0',))
    output_off = _align(max(offsets[name] + sizes[name] for name in sizes))
    arena = _align(output_off + surface, 4096)
    layout = dict(offsets)
    layout['input0'] = input_off
    layout['output'] = output_off
    layout_sizes = dict(sizes)
    layout_sizes['input0'] = input_bytes
    layout_sizes['output'] = surface
    for external in ('input0', 'output'):
        offset = layout[external]
        size = layout_sizes[external]
        for name in layout:
            if name == external:
                continue
            other = layout[name]
            other_size = layout_sizes[name]
            if offset < other + other_size and other < offset + size:
                raise ValueError('depthwise join external tensor %s overlaps %s' % (external, name))
    data = bytearray(payload_size)
    # Stem program with relocated constants and native intermediate destination.
    stem_src = {w & 65535: (w >> 16) & 0xffffffff for w in struct.unpack_from('<126Q', stem_data)}
    registers = dict(stem_src)
    registers.update({0x1110: stem_weights, 0x5020: stem_bias, 0x4020: offsets['stem'], 0x1070: input_off})
    for i, (reg, default, tag) in enumerate(REGISTERS):
        struct.pack_into('<Q', data, stem_basis + i * 8, tag << 48 | registers.get(reg, default) << 16 | reg)
    _terminal(data, stem_basis, 29)
    data[stem_weights:stem_weights + stem_weight_size] = stem_data[stem_src[0x1110]:stem_src[0x1110] + stem_weight_size]
    data[stem_bias:stem_bias + stem_bias_size] = stem_data[stem_src[0x5020]:stem_src[0x5020] + stem_bias_size]
    # Dense branch.
    dense_fields = _native_fields(kernel, 3, dense_weights, dense_bias, offsets['dense'], offsets['stem'], q_dense)
    dense_fields[0x1184] = int(stem_zp) & 0xffff
    for i, (reg, default, tag) in enumerate(REGISTERS):
        struct.pack_into('<Q', data, dense_basis + i * 8, tag << 48 | dense_fields.get(reg, default) << 16 | reg)
    _terminal(data, dense_basis, 29)
    _pack_native_head(data, kernel, 3, dense_weights, dense_bias, q_dense)
    # Depthwise branch: the verified task program with its four addresses relocated.
    dw_regs.update({0x1070: offsets['stem'], 0x4020: offsets['depthwise'],
                    0x1110: dw_weights, 0x5020: dw_bias, 0x1184: int(stem_zp) & 0xffff})
    for i, (reg, default, tag) in enumerate(REGISTERS):
        struct.pack_into('<Q', data, dw_basis + i * 8, tag << 48 | dw_regs.get(reg, default) << 16 | reg)
    _terminal(data, dw_basis, 29)
    data[dw_weights:dw_weights + dw_weight_size] = payload[dw_weight_source:dw_weight_source + dw_weight_size]
    data[dw_bias:dw_bias + dw_bias_size] = payload[dw_bias_source:dw_bias_source + dw_bias_size]
    # Join.
    join_fields, output_scale, output_zero_point = _join_fields(
        kind, 8, 8, 3, offsets['dense'], offsets['depthwise'], output_off, surface,
        scales, (0, 0), output_range)
    for i, (reg, default, tag) in enumerate(r for r in REGISTERS if r[0] >= 0x4000):
        struct.pack_into('<Q', data, join_basis + i * 8, tag << 48 | join_fields.get(reg, default) << 16 | reg)
    _terminals(data, join_basis, 78, 24)
    tensors = [
        dict(name='input0', role=ROLE_INPUT, layout=LAYOUT_PACKED_U8, index=0, shape=(1, 8, 8, 3),
             offset=input_off, size=input_bytes),
        dict(name='stem', role=ROLE_INTERNAL, layout=LAYOUT_NATIVE16, index=0, shape=(1, 8, 8, 3),
             offset=offsets['stem'], size=surface),
        dict(name='dense', role=ROLE_INTERNAL, layout=LAYOUT_NATIVE16, index=0, shape=(1, 8, 8, 3),
             offset=offsets['dense'], size=surface),
        dict(name='depthwise', role=ROLE_INTERNAL, layout=LAYOUT_NATIVE16, index=1, shape=(1, 8, 8, 3),
             offset=offsets['depthwise'], size=surface),
        dict(name='output', role=ROLE_OUTPUT, layout=LAYOUT_NATIVE16, index=0, shape=(1, 8, 8, 3),
             offset=output_off, size=surface),
    ]
    tasks = [(stem_basis, 126, 29, 768), (dense_basis, 126, 29, 768),
             (dw_basis, 126, 29, 768), (join_basis, 78, 24, 768)]
    handoff_tails(data, tasks, serial)
    binary = encode_sequence_v5(data, tensors=tensors, tasks=tasks, arena_bytes=arena,
        input_scale=1.0, input_zero_point=0, output_scale=output_scale,
        output_zero_point=output_zero_point, serial=serial)
    live_bytes = max(offsets[name] + sizes[name] for name in sizes) - _align(input_off + input_bytes)
    meta = dict(profile='depthwise-join', engine_runs=([] if serial else [len(tasks)]),
        join=kind, dense_kernel=kernel, depthwise_kernel=dw_kernel,
        depthwise_asymmetric_pair=bool(asymmetric_depthwise),
        join_scales=[float(v) for v in scales], input_scale=1.0, input_zero_point=0,
        output_scale=output_scale, output_zero_point=output_zero_point,
        shape_nhwc=[1, 8, 8, 3], output_shape_nhwc=[1, 8, 8, 3], output_tensors=['output'],
        schedule=[bindings[index].writes[0] for index in order],
        tensor_lifetimes={name: list(intervals[name]) for name in sorted(intervals)},
        tensor_offsets={name: layout[name] for name in sorted(layout)},
        lifetime_bytes=live_bytes, allocated_bytes=sum(sizes.values()),
        stem_quantization=stem_meta['quantization'],
        head_quantization=[q_dense.metadata()],
        depthwise_quantization=branch_meta['depthwise'],
        weights=[wd.tolist()], bias=[constants[dense.input[2]].tolist()],
        depthwise_weights=[ww.tolist()], depthwise_bias=[constants[depthwise.input[2]].tolist()],
        stem_weights=w1.tolist(), stem_bias=constants[stem_nodes[0].input[2]].tolist())
    return binary, meta


def depthwise_join_reference(inputs, stem_quantization, dense_quantization,
                             depthwise_quantization, kind, output_range=None):
    """Composed integer output for one uint8 HWC input (stem, both branches, join)."""
    from .quantization import reference

    def quantize(params):
        if isinstance(params, Quantization):
            return params
        values = dict(params)
        for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
            values[key] = np.array(values[key])
        return Quantization(**values)

    stem = quantize(stem_quantization)
    grid = reference(inputs, stem)
    dense = native_reference(grid, quantize(dense_quantization), stem.output_zero_point)
    branch = depthwise_reference(grid, quantize(depthwise_quantization), stem.output_zero_point)
    return join_reference(kind, dense, branch)
