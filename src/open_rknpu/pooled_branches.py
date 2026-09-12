"""SPDX-License-Identifier: MIT

Pooled multi-layer branches on RV1103: every branch is a Conv chain that pools before
the joins.

    input -> stem Conv -+-> Conv -> Conv -> 2x2 pool -+
                        +-> Depthwise -> 2x2 pool -----+-> join -> [join ->] output
                        +-> Conv -> 2x2 pool ----------+

Each branch is one to three dense Conv layers (a depthwise layer is expanded to an
exact block-diagonal dense kernel, as in `open_rknpu.join_dag`), followed by a 2x2
stride-2 MaxPool or AveragePool, so every join operand is a 4x4/C3 grid and the joins
run at 4x4. Two or three branches are folded by one or two elementwise joins. The task
order and arena come from `open_rknpu.liveness`, with a fresh slot per internal. No
vendor capture or RKNN object is read.
"""
import struct
import tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .chain import native_reference
from .compiler import compile_model
from .compose import Binding, ConstantSpec, Stage, TensorSpec, compose
from .graph import _align, _join_fields, JOIN_TYPES, join_reference
from .join_dag import _band, _layer_fields, _pack_layer_head, _prepare
from .pooling import pool_registers
from .quantization import Quantization
from .sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT, ROLE_INTERNAL,
                       ROLE_OUTPUT)

POOLS = ('MaxPool', 'AveragePool')


def parse_pooled_branches(nodes):
    """Describe pooled multi-layer branches, or return None.

    Requires a 1x1 stem, two or three branches — each one to three Conv layers followed
    by a 2x2 stride-2 pool — and one or two joins over the pooled grids.
    """
    ops = [node.op_type for node in nodes]
    if len(ops) < 7 or ops[0] != 'Conv' or any(node.domain not in ('', 'ai.onnx') for node in nodes):
        return None
    stem_end = 2 if len(ops) > 1 and ops[1] == 'Relu' else 1
    for node in nodes[stem_end:]:
        if node.op_type not in ('Conv',) + POOLS + JOIN_TYPES:
            return None
    stem_out = nodes[stem_end - 1].output[0]
    branch_of = {}
    branches = []
    pools = []
    joins = []
    produced = set()
    for node in nodes[stem_end:]:
        if node.op_type == 'Conv':
            source = list(node.input)[:1]
            if source == [stem_out]:
                branches.append([node])
                branch_of[node.output[0]] = len(branches) - 1
            elif source and source[0] in branch_of:
                owner = branch_of[source[0]]
                if len(pools) > owner:
                    return None
                branches[owner].append(node)
                branch_of[node.output[0]] = owner
            else:
                return None
            produced.add(node.output[0])
        elif node.op_type in POOLS:
            owner = branch_of.get(list(node.input)[:1][0]) if node.input else None
            if (owner is None or len(pools) != owner
                    or list(node.input) != [branches[owner][-1].output[0]]):
                return None
            pools.append(node)
            produced.add(node.output[0])
        else:
            if node.attribute or len(node.input) != 2 or len(node.output) != 1:
                return None
            if node.input[0] == node.input[1] or any(name not in produced for name in node.input):
                return None
            joins.append(node)
            produced.add(node.output[0])
    if not 2 <= len(branches) <= 3 or len(pools) != len(branches):
        return None
    if len(joins) not in (1, 2) or len(joins) != len(branches) - 1:
        return None
    if any(len(branch) > 3 or not branch for branch in branches):
        return None
    # The pooled-branch emitter uses one pool kind for every branch; mixed kinds fall
    # through so another path (the op-level walk) can lower the graph instead.
    if len({node.op_type for node in pools}) != 1:
        return None
    names = [node.output[0] for branch in branches for node in branch]
    if len(set(names)) != len(names):
        return None
    return dict(stem=nodes[:stem_end], branches=branches, pools=pools, joins=joins,
                stem_out=stem_out, branch_of=branch_of)


def _validate(model):
    parts = parse_pooled_branches(list(model.graph.node))
    if parts is None:
        raise ValueError('pooled branches require a 1x1 stem, two or three Conv-chain branches with pools and one or two joins')
    g = model.graph
    stem_nodes = parts['stem']
    branches = parts['branches']
    pools = parts['pools']
    joins = parts['joins']
    if len(g.input) != 1 or len(g.output) != 1 or joins[-1].output[0] != g.output[0].name:
        raise ValueError('pooled branches require one input, one output and a final join')
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*g.input, *g.output, *g.value_info]}
    for value, wanted in ((g.input[0], [1, 3, 8, 8]), (g.output[0], [1, 3, 4, 4])):
        if value.type.tensor_type.elem_type != 1 or shapes.get(value.name) != wanted:
            raise ValueError('pooled branches external tensors must be float32 %s' % wanted)
    for branch in branches:
        for node in branch:
            declared = shapes.get(node.output[0])
            if declared is None or len(declared) != 4 or declared[0] != 1 or declared[2:] != [8, 8]:
                raise ValueError('pooled branches branch tensors must be float32 1xCx8x8')
    for pool in pools:
        attrs = {a.name: h.get_attribute_value(a) for a in pool.attribute}
        allowed = {'kernel_shape': [2, 2], 'strides': [2, 2], 'auto_pad': b'NOTSET',
                   'ceil_mode': 0, 'count_include_pad': 0, 'storage_order': 0,
                   'dilations': [1, 1]}
        if pool.domain not in ('', 'ai.onnx') or any(k not in allowed or v != allowed[k]
                                                     for k, v in attrs.items()):
            raise ValueError('pooled branches pools must be 2x2 stride-2 MaxPool/AveragePool')
    constants = {t.name: nh.to_array(t) for t in g.initializer}
    stem = stem_nodes[0]
    w1 = constants.get(stem.input[1]) if len(stem.input) > 1 else None
    hidden = w1.shape[0] if w1 is not None and w1.ndim == 4 else 0
    if (w1 is None or w1.dtype != np.float32 or w1.ndim != 4 or w1.shape != (hidden, 3, 1, 1)
            or not 3 <= hidden <= 16 or stem.input[2] not in constants
            or constants[stem.input[2]].shape != (hidden,)):
        raise ValueError('pooled branches stem must be a 1x1 Conv with hidden channels 3..16')
    stem_attrs = {a.name: h.get_attribute_value(a) for a in stem.attribute}
    stem_supported = {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1],
                      'dilations': [1, 1], 'group': 1}
    if any(k not in stem_supported or v != stem_supported[k] for k, v in stem_attrs.items()):
        raise ValueError('unsupported stem attributes')
    if len(stem_nodes) == 2 and (stem_nodes[1].attribute or list(stem_nodes[1].input) != list(stem.output)):
        raise ValueError('pooled branches stem Relu must consume the stem output')
    specs = []
    for branch in branches:
        layers = []
        input_channels = hidden
        for position, node in enumerate(branch):
            expected = parts['stem_out'] if position == 0 else branch[position - 1].output[0]
            if node.input[0] != expected:
                raise ValueError('pooled branches layers must chain off the previous layer')
            w = constants.get(node.input[1]) if len(node.input) > 1 else None
            b = constants.get(node.input[2]) if len(node.input) > 2 else None
            kernel = w.shape[2] if w is not None and w.ndim == 4 else 0
            if w is None or b is None or w.dtype != np.float32 or b.dtype != np.float32:
                raise ValueError('pooled branches layers require constant float32 weights and bias')
            attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
            group = int(attrs.get('group', 1))
            if group == 1 or (len(branch) > 1 and group == input_channels):
                if group > 1:
                    if (w.shape != (input_channels, 1, kernel, kernel) or b.shape != (input_channels,)
                            or kernel not in (1, 3) or attrs.get('pads') != [kernel // 2] * 4
                            or attrs.get('strides', [1, 1]) != [1, 1]
                            or attrs.get('dilations', [1, 1]) != [1, 1]):
                        raise ValueError('pooled branches chained depthwise layers need group C, 1x1/3x3 and symmetric pad')
                    expanded = np.zeros((input_channels, input_channels, kernel, kernel), np.float32)
                    for channel in range(input_channels):
                        expanded[channel, channel] = w[channel, 0]
                    output_channels = input_channels
                    w = expanded
                else:
                    output_channels = w.shape[0] if w.ndim == 4 else 0
                    if (kernel not in (1, 3) or w.shape != (output_channels, input_channels, kernel, kernel)
                            or b.shape != (output_channels,) or not 1 <= output_channels <= 16):
                        raise ValueError('pooled branches layers must be C1..16 1x1/3x3 Conv over the previous tensor')
                allowed = {'kernel_shape': [kernel, kernel], 'pads': [kernel // 2] * 4,
                           'strides': [1, 1], 'dilations': [1, 1], 'group': 1}
                if any(k not in allowed or v != allowed[k] for k, v in attrs.items() if k != 'group'):
                    raise ValueError('unsupported layer attributes')
                layers.append(dict(kind='dense', node=node, weights=w, bias=b, kernel=kernel,
                                   input_channels=input_channels, output_channels=output_channels,
                                   depthwise_expanded=bool(group > 1)))
                input_channels = output_channels
            else:
                raise ValueError('pooled branches layers must be dense or depthwise group C')
        if input_channels != 3:
            raise ValueError('pooled branches must end each chain with three channels')
        specs.append(layers)
    for position, join in enumerate(joins):
        first = pools[0].output[0] if position == 0 else joins[position - 1].output[0]
        if list(join.input) != [first, pools[position + 1].output[0]]:
            raise ValueError('pooled branches joins must fold the pooled branches in order')
    return dict(stem_nodes=stem_nodes, stem=stem, hidden=hidden, branches=specs,
                joins=joins, pools=pools, constants=constants)


def compile_pooled_branches(model, output_range=None, serial=True):
    """Compile pooled multi-layer branches folded by elementwise joins at 4x4."""
    onnx.checker.check_model(model)
    if any(n.domain not in ('', 'ai.onnx') for n in model.graph.node):
        raise ValueError('pooled branches require default-domain nodes')
    spec = _validate(model)
    pool_kind = spec['pools'][0].op_type
    if any(pool.op_type != pool_kind for pool in spec['pools']):
        # Defensive: the parser declines mixed pool kinds so the graph falls through to the
        # walk instead of reaching here (tests substitute the parser to exercise this).
        raise ValueError('pooled branches require matching pool kinds')
    g = model.graph
    stem_nodes = spec['stem_nodes']
    stem = spec['stem']
    hidden = spec['hidden']
    t = stem_nodes[-1].output[0]
    first_graph = h.make_graph(stem_nodes, 'stem', [g.input[0]],
        [h.make_tensor_value_info(t, 1, [1, hidden, 8, 8])],
        [nh.from_array(spec['constants'][name], name) for name in stem.input[1:]])
    first = h.make_model(first_graph, opset_imports=list(model.opset_import))
    first.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        stem_path = Path(tmp) / 'stem.onnx'
        onnx.save(first, stem_path)
        stem_data, stem_meta = compile_model(stem_path)
    stem_scale = stem_meta['output_scale']
    stem_zp = stem_meta['output_zero_point']
    # The pool preserves the grid band, so the join scale propagation can run over the
    # branch finals instead of the pooled tensors.
    alias_joins = []
    for position, join in enumerate(spec['joins']):
        first = spec['branches'][0][-1]['node'].output[0] if position == 0 else spec['joins'][position - 1].output[0]
        alias_joins.append(h.make_node(join.op_type, [first, spec['branches'][position + 1][-1]['node'].output[0]],
                                       [join.output[0]]))
    prepare_spec = dict(spec, joins=alias_joins)
    branches, by_name, scales = _prepare(model, prepare_spec, stem_scale, stem_zp)
    surface = 8 * 8 * 16
    pooled_surface = 4 * 4 * 16
    # Stages are declared once and assembled by `open_rknpu.compose`. This profile
    # gives every internal a fresh slot (`reuse=False`): the board returned stale data
    # when one slot was written by two different task families (investigation log).
    surface = 8 * 8 * 16
    pooled_surface = 4 * 4 * 16
    stem_weight_size = _align(1 * 1 * _align(hidden, 4) * 4)
    stem_bias_size = ((hidden + 3) // 4) * 32
    stem_src = {w & 65535: (w >> 16) & 0xffffffff
                for w in struct.unpack_from('<126Q', stem_data)}
    pooled_names = [pool.output[0] for pool in spec['pools']]

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
    tensors = [TensorSpec('input0', ROLE_INPUT, LAYOUT_PACKED_U8, (1, 8, 8, 3), 8 * 16 * 3),
               TensorSpec('stem', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, hidden), surface)]

    def layer_stage(name, source, previous_zero_point, entry):
        kernel = entry['layer']['kernel']
        input_channels = entry['layer']['input_channels']
        output_channels = entry['layer']['output_channels']
        quantization = entry['quantization']
        weight_size = _align(output_channels * 16 * kernel * kernel)
        bias_size = ((output_channels + 3) // 4) * 32

        def fill(payload, offset):
            _pack_layer_head(payload, kernel, input_channels, output_channels,
                             offset, offset + weight_size, quantization)

        def fields(addresses, constant_offsets):
            values = _layer_fields(kernel, input_channels, output_channels,
                                   constant_offsets[name], constant_offsets[name] + weight_size,
                                   addresses[name], addresses[source], quantization)
            # The CNA border path injects the zero point of the tensor actually read.
            values[0x1184] = int(previous_zero_point) & 0xffff
            return values

        return Stage(name=name, family='native-conv', reads=(source,), writes=(name,),
                     fields=fields,
                     constants=(ConstantSpec(name, weight_size + bias_size, fill),),
                     bindings=(Binding(0x1070, source, 'read'),
                               Binding(0x4020, name, 'write'))), output_channels

    layer_index = 0
    for entries in branches:
        source = 'stem'
        source_zp = stem_zp
        for entry in entries:
            name = entry['layer']['node'].output[0]
            stage, output_channels = layer_stage(name, source, source_zp, entry)
            stages.append(stage)
            tensors.append(TensorSpec(name, ROLE_INTERNAL, LAYOUT_NATIVE16,
                                      (1, 8, 8, output_channels), surface, layer_index))
            layer_index += 1
            source = name
            source_zp = _band(entry['quantization'])[1]

    def pool_stage(name, source, index):
        def fields(addresses, constant_offsets):
            return pool_registers(pool_kind, 8, 8, 4, 4, addresses[source], addresses[name])

        return Stage(name=name, family='pool', reads=(source,), writes=(name,), fields=fields,
                     bindings=(Binding(0x701c, source, 'read'),
                               Binding(0x6070, name, 'write')))

    for index, (entries, pooled) in enumerate(zip(branches, pooled_names)):
        stages.append(pool_stage(pooled, entries[-1]['layer']['node'].output[0], index))
        tensors.append(TensorSpec(pooled, ROLE_INTERNAL, LAYOUT_NATIVE16,
                                  (1, 4, 4, 3), pooled_surface, index))

    def join_stage(position, join, first, second, scale_first, scale_second):
        last_join = position == len(spec['joins']) - 1
        target = 'output' if last_join else join.output[0]
        override = output_range if last_join else None

        def fields(addresses, constant_offsets):
            values, _, _ = _join_fields(join.op_type, 4, 4, 3, addresses[first],
                                        addresses[second], addresses[target], pooled_surface,
                                        [scale_first, scale_second], (0, 0), override)
            return values

        stage = Stage(name='join%d' % position, family='elementwise', reads=(first, second),
                      writes=(target,), fields=fields,
                      bindings=(Binding(0x5018, first, 'read'),
                                Binding(0x5038, second, 'read'),
                                Binding(0x4020, target, 'write')))
        _, join_scale, join_zero = _join_fields(join.op_type, 4, 4, 3, 0, 0, 0, pooled_surface,
                                                [scale_first, scale_second], (0, 0), override)
        if not last_join:
            tensors.append(TensorSpec(join.output[0], ROLE_INTERNAL, LAYOUT_NATIVE16,
                                      (1, 4, 4, 3), pooled_surface, 0))
        return stage, join_scale, join_zero

    output_scale = output_zero_point = None
    for position, join in enumerate(spec['joins']):
        first, second = join.input
        scale_first = (scales[spec['branches'][0][-1]['node'].output[0]] if position == 0
                       else scales[spec['joins'][position - 1].output[0]])
        scale_second = scales[spec['branches'][position + 1][-1]['node'].output[0]]
        stage, join_scale, join_zero = join_stage(position, join, first, second,
                                                  scale_first, scale_second)
        stages.append(stage)
        if position == len(spec['joins']) - 1:
            output_scale, output_zero_point = join_scale, join_zero
    tensors.append(TensorSpec('output', ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 4, 4, 3), pooled_surface, 0))

    binary, composed = compose(stages, tensors, input_scale=1.0, input_zero_point=0,
                               output_scale=output_scale, output_zero_point=output_zero_point,
                               serial=serial, reuse=False)
    meta = dict(profile='pooled-branches', pool=pool_kind,
        join_expression=[dict(name=join.output[0], kind=join.op_type, inputs=list(join.input))
                         for join in spec['joins']],
        pooled_names=pooled_names,
        branch_names=[[entry['layer']['node'].output[0] for entry in entries] for entries in branches],
        branch_quantization=[[entry['quantization'].metadata() for entry in entries] for entries in branches],
        head_names=[entries[-1]['layer']['node'].output[0] for entries in branches],
        head_kinds=['dense'] * len(branches),
        head_quantization=[entries[-1]['quantization'].metadata() for entries in branches],
        depthwise_quantization=[None] * len(branches),
        join_scales=[float(scales[join.output[0]]) for join in spec['joins']],
        branch_final_scales={entries[-1]['layer']['node'].output[0]: float(scales[entries[-1]['layer']['node'].output[0]])
                             for entries in branches},
        input_scale=1.0, input_zero_point=0, output_scale=output_scale,
        output_zero_point=output_zero_point, shape_nhwc=[1, 8, 8, 3],
        output_shape_nhwc=[1, 4, 4, 3], output_tensors=['output'],
        lifetime_bytes=composed['live_bytes'],
        stem_quantization=stem_meta['quantization'])
    meta.update({key: value for key, value in composed.items() if key != 'profile'})
    return binary, meta


def pooled_branches_reference(inputs, stem_quantization, branch_names, branch_quantization,
                             expression, pool, pooled_names=None):
    """Fold each branch chain, pool it, then replay the join expression."""
    def quantize(params):
        if isinstance(params, Quantization):
            return params
        values = dict(params)
        for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
            values[key] = np.array(values[key])
        return Quantization(**values)

    from .quantization import reference
    stem = quantize(stem_quantization)
    stem_grid = reference(inputs, stem)
    values = {}
    for index, names in enumerate(branch_names):
        grid = stem_grid
        zero_point = stem.output_zero_point
        for position, name in enumerate(names):
            q = quantize(branch_quantization[index][position])
            grid = native_reference(grid, q, zero_point)
            zero_point = q.output_zero_point
            values[name] = grid
        height, width, channels = grid.shape
        blocked = grid.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
        key = pooled_names[index] if pooled_names else names[-1] + '_pooled'
        if pool == 'MaxPool':
            values[key] = blocked.max(axis=(1, 3)).astype(np.int8)
        else:
            values[key] = np.rint(blocked.sum(axis=(1, 3)) / 4).astype(np.int8)
    for step in expression:
        first = step['inputs'][0]
        second = step['inputs'][1]
        values[step['name']] = join_reference(step['kind'], values[first], values[second])
    return values[expression[-1]['name']]
