"""SPDX-License-Identifier: MIT

Two pooled branches from one stem, joined after the pool, on RV1103.

    input -> stem Conv 1x1[, Relu] -+-> Conv 1x1/3x3 -> MaxPool/AveragePool -+
                                    +-> Conv 1x1/3x3 -> MaxPool/AveragePool -+-> join

Each branch is the verified `Conv[/Relu] -> 2x2 stride-2 pool` profile, so the
pool tasks reuse the pool register builder from `open_rknpu.pooling`
(`pool_registers` / `pool_tag`) with relocated addresses; the join consumes the
two 4x4/C3 pooled grids. The arena is placed by `open_rknpu.liveness`. This
retires the plan's "multi-task pool branches into Mul" blocker. No vendor capture
or RKNN object is read.
"""
import struct
import tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .chain import native_quantize, native_reference
from .compose import Binding, ConstantSpec, Stage, TensorSpec, compose
from .compiler import compile_model
from .graph import (_align, _join_fields, _native_fields, _pack_native_head,
                    join_reference, JOIN_TYPES)
from .pooling import pool_registers
from .quantization import Quantization
from .sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT, ROLE_INTERNAL,
                       ROLE_OUTPUT)


def parse_pool_join(nodes):
    """Structural check for `stem[,Relu], head, pool, head, pool, join`."""
    if len(nodes) < 6 or nodes[0].op_type != 'Conv':
        return False
    stem_end = 2 if len(nodes) > 1 and nodes[1].op_type == 'Relu' else 1
    if len(nodes) != stem_end + 5:
        return False
    head_a, pool_a, head_b, pool_b, join = nodes[stem_end:]
    return (head_a.op_type == 'Conv' and pool_a.op_type in ('MaxPool', 'AveragePool')
            and head_b.op_type == 'Conv' and pool_b.op_type == pool_a.op_type
            and join.op_type in JOIN_TYPES)


def _validate(model):
    onnx.checker.check_model(model)
    g = model.graph
    nodes = list(g.node)
    if any(node.domain not in ('', 'ai.onnx') for node in nodes):
        raise ValueError('pool join requires default-domain nodes')
    stem_end = 2 if len(nodes) > 1 and nodes[1].op_type == 'Relu' else 1
    if len(nodes) != stem_end + 5 or nodes[0].op_type != 'Conv':
        raise ValueError('pool join requires stem[,Relu], Conv, pool, Conv, pool and one join')
    stem_nodes = nodes[:stem_end]
    head_a, pool_a, head_b, pool_b, join = nodes[stem_end:]
    if head_a.op_type != 'Conv' or head_b.op_type != 'Conv' or join.op_type not in JOIN_TYPES:
        raise ValueError('pool join requires two Conv branches and Add/Mul/Sub/Max')
    if pool_a.op_type not in ('MaxPool', 'AveragePool') or pool_b.op_type != pool_a.op_type:
        raise ValueError('pool join requires two matching MaxPool or AveragePool nodes')
    for pool in (pool_a, pool_b):
        attrs = {a.name: h.get_attribute_value(a) for a in pool.attribute}
        allowed = {'kernel_shape': [2, 2], 'strides': [2, 2], 'auto_pad': b'NOTSET',
                   'ceil_mode': 0, 'count_include_pad': 0, 'storage_order': 0,
                   'dilations': [1, 1]}
        if any(k not in allowed or v != allowed[k] for k, v in attrs.items()):
            raise ValueError('pool join supports 2x2 stride-2 pooling without extra attributes')
    if join.attribute:
        raise ValueError('pool join must not carry attributes')
    stem_out = stem_nodes[-1].output[0]
    if (head_a.input[0] != stem_out or head_b.input[0] != stem_out
            or list(pool_a.input) != [head_a.output[0]] or list(pool_b.input) != [head_b.output[0]]
            or list(join.input) != [pool_a.output[0], pool_b.output[0]]):
        raise ValueError('pool join branches must run stem -> Conv -> pool and join both pools in order')
    if len(g.input) != 1 or len(g.output) != 1 or join.output[0] != g.output[0].name:
        raise ValueError('pool join requires one input and one output')
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*g.input, *g.output, *g.value_info]}
    for value, shape in ((g.input[0], [1, 3, 8, 8]), (g.output[0], [1, 3, 4, 4]),
                         (head_a, [1, 3, 8, 8]), (head_b, [1, 3, 8, 8]),
                         (pool_a, [1, 3, 4, 4]), (pool_b, [1, 3, 4, 4])):
        name = value.name if isinstance(value, onnx.ValueInfoProto) else value.output[0]
        if shapes.get(name) != shape:
            raise ValueError('pool join tensors must be float32 [1,3,8,8] or [1,3,4,4]')
    if (g.input[0].type.tensor_type.elem_type != 1 or g.output[0].type.tensor_type.elem_type != 1):
        raise ValueError('pool join external tensors must be float32')
    constants = {t.name: nh.to_array(t) for t in g.initializer}
    for node in (stem_nodes[0], head_a, head_b):
        if len(node.input) != 3 or any(name not in constants for name in node.input[1:]):
            raise ValueError('pool join convolutions require constant float32 weights and bias')
        for name in node.input[1:]:
            if constants[name].dtype != np.float32:
                raise ValueError('pool join constants must be float32')
    w1 = constants[stem_nodes[0].input[1]]
    hidden = w1.shape[0] if w1.ndim == 4 else 0
    if (w1.ndim != 4 or w1.shape != (hidden, 3, 1, 1) or not 3 <= hidden <= 16
            or constants[stem_nodes[0].input[2]].shape != (hidden,)):
        raise ValueError('pool join stem must be a 1x1 Conv with hidden channels 3..16')
    stem_attrs = {a.name: h.get_attribute_value(a) for a in stem_nodes[0].attribute}
    stem_supported = {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1],
                      'dilations': [1, 1], 'group': 1}
    if any(k not in stem_supported or v != stem_supported[k] for k, v in stem_attrs.items()):
        raise ValueError('unsupported stem attributes')
    if len(stem_nodes) == 2 and (stem_nodes[1].attribute or list(stem_nodes[1].input) != list(stem_nodes[0].output)):
        raise ValueError('pool join stem Relu must consume the stem output')
    heads = []
    for head in (head_a, head_b):
        w = constants[head.input[1]]
        kernel = w.shape[2] if w.ndim == 4 else 0
        if (w.ndim != 4 or kernel not in (1, 3) or w.shape != (3, hidden, kernel, kernel)
                or constants[head.input[2]].shape != (3,)):
            raise ValueError('pool join branches must be dense 3-output 1x1/3x3 Conv with hidden inputs')
        attrs = {a.name: h.get_attribute_value(a) for a in head.attribute}
        supported = {'kernel_shape': [kernel, kernel], 'pads': [kernel // 2] * 4,
                     'strides': [1, 1], 'dilations': [1, 1], 'group': 1}
        if any(k not in supported or v != supported[k] for k, v in attrs.items()):
            raise ValueError('unsupported branch attributes')
        if kernel == 3 and attrs.get('pads') != [1, 1, 1, 1]:
            raise ValueError('pool join 3x3 branches require symmetric pad1')
        heads.append((w, constants[head.input[2]], kernel))
    return stem_nodes, pool_a.op_type, heads, join.op_type, constants, w1, hidden, join


def compile_pool_join(model, output_range=None, serial=True, reuse=True):
    """Two Conv+pool branches from one stem, folded by one elementwise join.

    Both pooled grids are quantized with zero point zero: a Mul join folds the two
    free branch scales, while Add/Sub/Max need one shared scale that both branches
    are re-quantized onto. Pooling preserves the grid scale, so the pooled operands
    carry the branch scales unchanged. Bounds: RGB 8x8 input, RGB 4x4 output, 1x1
    stem with 3..16 hidden channels, dense 1x1/3x3 branches, 2x2 stride-2
    MaxPool/AveragePool, and an output override only for a Mul join.
    """
    stem_nodes, pool_kind, heads, kind, constants, w1, hidden, _ = _validate(model)
    g = model.graph
    if output_range is not None and kind != 'Mul':
        raise ValueError('the pool join output override requires a Mul join')
    stem_out_name = stem_nodes[-1].output[0]
    first_graph = h.make_graph(stem_nodes, 'stem', list(g.input),
        [h.make_tensor_value_info(stem_out_name, 1, [1, hidden, 8, 8])],
        [nh.from_array(constants[name], name) for name in stem_nodes[0].input[1:]])
    first = h.make_model(first_graph, opset_imports=list(model.opset_import))
    first.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        stem_path = Path(tmp) / 'stem.onnx'
        onnx.save(first, stem_path)
        stem_data, stem_meta = compile_model(stem_path)
    stem_scale = stem_meta['output_scale']
    stem_zp = stem_meta['output_zero_point']
    natural = [native_quantize(w, b, stem_scale, stem_zp, None) for w, b, _ in heads]
    adjusted = [float(np.float32(q.output_scale * max(128 + q.output_zero_point,
                                                      127 - q.output_zero_point) / 127))
                for q in natural]
    scales = adjusted if kind == 'Mul' else [max(adjusted)] * 2
    q_heads = [native_quantize(w, b, stem_scale, stem_zp, dict(scale=s, zero_point=0))
               for (w, b, _), s in zip(heads, scales)]
    # Stages are declared once. `open_rknpu.compose` orders them topologically,
    # places the arena with `open_rknpu.liveness`, assigns program slots and constant
    # blocks, substitutes the planned addresses into each stage's register words, and
    # returns the declared binding view checked by `tests/test_container_bindings.py`.
    stem_weight_size = _align(1 * 1 * _align(hidden, 4) * 4)
    stem_bias_size = ((hidden + 3) // 4) * 32
    surface = 8 * 8 * 16
    pool_surface = 4 * 4 * 16
    stem_src = {w & 65535: (w >> 16) & 0xffffffff for w in struct.unpack_from('<126Q', stem_data)}

    def stem_fill(payload, offset):
        payload[offset:offset + stem_weight_size] = stem_data[
            stem_src[0x1110]:stem_src[0x1110] + stem_weight_size]
        payload[offset + stem_weight_size:offset + stem_weight_size + stem_bias_size] = stem_data[
            stem_src[0x5020]:stem_src[0x5020] + stem_bias_size]

    def stem_fields(addresses, constant_offsets):
        values = dict(stem_src)
        values.update({0x1110: constant_offsets['stem'],
                       0x5020: constant_offsets['stem'] + stem_weight_size,
                       0x4020: addresses['stem'], 0x1070: addresses['input0']})
        return values

    stages = [Stage(name='stem', family='native-conv', reads=('input0',), writes=('stem',),
                    fields=stem_fields,
                    constants=(ConstantSpec('stem', stem_weight_size + stem_bias_size, stem_fill),),
                    bindings=(Binding(0x1070, 'input0', 'read'),
                              Binding(0x4020, 'stem', 'write')))]

    def head_stage(name, kernel, quantization):
        weight_size = _align(3 * 16 * kernel * kernel)
        bias_size = 32

        def fill(payload, offset):
            _pack_native_head(payload, kernel, hidden, offset, offset + weight_size, quantization)

        def fields(addresses, constant_offsets):
            values = _native_fields(kernel, hidden, constant_offsets[name],
                                    constant_offsets[name] + weight_size,
                                    addresses[name], addresses['stem'], quantization)
            values[0x1184] = int(stem_zp) & 0xffff
            return values

        return Stage(name=name, family='native-conv', reads=('stem',), writes=(name,),
                     fields=fields,
                     constants=(ConstantSpec(name, weight_size + bias_size, fill),),
                     bindings=(Binding(0x1070, 'stem', 'read'),
                               Binding(0x4020, name, 'write')))

    for name, (_, _, kernel), quantization in zip(('head_a', 'head_b'), heads, q_heads):
        stages.append(head_stage(name, kernel, quantization))

    def pool_stage(name, source):
        def fields(addresses, constant_offsets):
            return pool_registers(pool_kind, 8, 8, 4, 4, addresses[source], addresses[name])

        return Stage(name=name, family='pool', reads=(source,), writes=(name,), fields=fields,
                     bindings=(Binding(0x701c, source, 'read'),
                               Binding(0x6070, name, 'write')))

    stages.append(pool_stage('pool_a', 'head_a'))
    stages.append(pool_stage('pool_b', 'head_b'))

    _, output_scale, output_zero_point = _join_fields(kind, 4, 4, 3, 0, 0, 0, pool_surface,
                                                      scales, (0, 0), output_range)

    def join_fields(addresses, constant_offsets):
        values, _, _ = _join_fields(kind, 4, 4, 3, addresses['pool_a'], addresses['pool_b'],
                                    addresses['output'], pool_surface, scales, (0, 0),
                                    output_range)
        return values

    stages.append(Stage(name='join', family='elementwise', reads=('pool_a', 'pool_b'),
                        writes=('output',), fields=join_fields,
                        bindings=(Binding(0x5018, 'pool_a', 'read'),
                                  Binding(0x5038, 'pool_b', 'read'),
                                  Binding(0x4020, 'output', 'write'))))

    tensors = [
        TensorSpec('input0', ROLE_INPUT, LAYOUT_PACKED_U8, (1, 8, 8, 3), 8 * 16 * 3, 0),
        TensorSpec('stem', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, hidden), surface, 0),
        TensorSpec('head_a', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), surface, 0),
        TensorSpec('head_b', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), surface, 1),
        TensorSpec('pool_a', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 4, 4, 3), pool_surface, 0),
        TensorSpec('pool_b', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 4, 4, 3), pool_surface, 1),
        TensorSpec('output', ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 4, 4, 3), pool_surface, 0),
    ]
    binary, composed = compose(stages, tensors, input_scale=1.0, input_zero_point=0,
                               output_scale=output_scale, output_zero_point=output_zero_point,
                               serial=serial, reuse=reuse)
    meta = dict(profile='pool-join', join=kind, pool=pool_kind, hidden_channels=hidden,
        head_kernels=[k for _, _, k in heads], join_scales=[float(v) for v in scales],
        input_scale=1.0, input_zero_point=0, output_scale=output_scale,
        output_zero_point=output_zero_point, shape_nhwc=[1, 8, 8, 3],
        output_shape_nhwc=[1, 4, 4, 3], output_tensors=['output'],
        lifetime_bytes=composed['live_bytes'],
        stem_quantization=stem_meta['quantization'],
        head_quantization=[q.metadata() for q in q_heads],
        weights=[w.tolist() for w, _, _ in heads], bias=[b.tolist() for _, b, _ in heads],
        stem_weights=w1.tolist(), stem_bias=constants[stem_nodes[0].input[2]].tolist())
    meta.update({key: value for key, value in composed.items() if key != 'profile'})
    return binary, meta


def pool_join_reference(inputs, stem_quantization, head_quantizations, kind, pool_kind):
    """Composed integer output for one uint8 HWC input (stem, both branches, join)."""
    from .quantization import reference

    def quantize(params):
        if isinstance(params, Quantization):
            return params
        values = dict(params)
        for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
            values[key] = np.array(values[key])
        return Quantization(**values)

    def pool(grid):
        height, width, channels = grid.shape
        blocked = grid.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
        if pool_kind == 'MaxPool':
            return blocked.max(axis=(1, 3)).astype(np.int8)
        return np.rint(blocked.sum(axis=(1, 3)) / 4).astype(np.int8)

    stem = quantize(stem_quantization)
    grid = reference(inputs, stem)
    first = pool(native_reference(grid, quantize(head_quantizations[0]), stem.output_zero_point))
    second = pool(native_reference(grid, quantize(head_quantizations[1]), stem.output_zero_point))
    return join_reference(kind, first, second)
