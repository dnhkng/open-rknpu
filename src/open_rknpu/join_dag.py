"""SPDX-License-Identifier: MIT

General join-expression DAG on RV1103: shared stem, multi-layer dense/depthwise
branches reused across several elementwise joins.

    input -> stem Conv -+-> Conv -> Conv -+
                        +-> Conv            +-> join0 = Op(b0, b1) -+
                        +-> Depthwise       +                       +-> join1 = Op(join0, b0)
                                                                             |
                                                                          output

Every join may read **any two previously produced tensors**, so a branch or an
earlier join result can feed more than one consumer, and a branch may itself be a
short chain (a residual-style block). The task order and the arena come from
`open_rknpu.liveness`; each stage is emitted with the same verified program builders
as the other fan-out profiles (native Conv fields, the relocated depthwise program,
and the elementwise join fields). No vendor capture or RKNN object is read.
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
from .graph import (_align, _join_fields, _native_fields, JOIN_TYPES, join_reference)
from .compose import Binding, ConstantSpec, Stage, TensorSpec, compose
from .liveness import Access, plan
from .pooling import pool_registers
from .quantization import Quantization
from .sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT, ROLE_INTERNAL,
                       ROLE_OUTPUT, decode_sequence)


def _layer_fields(kernel, input_channels, output_channels, w_off, b_off, out_off, in_off, q):
    """Native Conv fields for an arbitrary output-channel count (C1..16)."""
    fields = _native_fields(kernel, input_channels, w_off, b_off, out_off, in_off, q)
    if output_channels != 3:
        fields.update({0x1030: output_channels * 16 * kernel * kernel,
                       0x1038: (kernel << 24) | (kernel << 16) | output_channels,
                       0x403c: ((output_channels - 1) << 16) | 0xf})
    return fields


def _pack_layer_head(data, kernel, input_channels, output_channels, w_off, b_off, q):
    """Pack a dense Conv weight/bias block for an arbitrary output-channel count."""
    for o in range(output_channels):
        weights = q.weights[o].reshape(input_channels, kernel, kernel)
        for kh in range(kernel):
            for kw in range(kernel):
                row = list(weights[:, kh, kw]) + [int(q.weight_zero_points[o])] * (16 - input_channels)
                struct.pack_into('<16b', data, w_off + ((kh * kernel + kw) * output_channels + o) * 16, *row)
        block = b_off + (o // 4) * 32
        lane = o % 4
        struct.pack_into('<i', data, block + lane * 4, int(q.biases[o]))
        struct.pack_into('<h', data, block + 16 + lane * 2, -int(q.weight_zero_points[o]))
        struct.pack_into('<H', data, block + 24 + lane * 2, int(q.channel_multipliers[o]))


def parse_join_dag(nodes):
    """Describe a general join expression, or return None.

    Requires a 1x1 stem, three or more **branches** and two or more joins that each
    combine two *distinct* previously produced tensors. A branch is one to three Conv
    layers chained off the stem (a residual-style block). The simple left-fold chain
    is handled by `compile_join_chain`, so the scheduler checks the chain first.
    """
    ops = [node.op_type for node in nodes]
    if len(ops) < 6 or ops[0] != 'Conv' or any(node.domain not in ('', 'ai.onnx') for node in nodes):
        return None
    stem_end = 2 if len(ops) > 1 and ops[1] == 'Relu' else 1
    for node in nodes[stem_end:]:
        if node.op_type not in ('Conv', 'MaxPool', 'AveragePool') + JOIN_TYPES:
            return None
    stem_out = nodes[stem_end - 1].output[0]
    produced = set()
    branch_of = {}
    branches = []
    joins = []
    pool = None
    for node in nodes[stem_end:]:
        if node.op_type in ('MaxPool', 'AveragePool'):
            if pool is not None or joins and list(node.input) != [joins[-1].output[0]]:
                return None
            pool = node
            produced.add(node.output[0])
            continue
        if pool is not None:
            return None
        if node.op_type == 'Conv':
            source = list(node.input)[:1]
            if source == [stem_out]:
                branches.append([node])
                branch_of[node.output[0]] = len(branches) - 1
            elif source and source[0] in branch_of:
                branches[branch_of[source[0]]].append(node)
                branch_of[node.output[0]] = branch_of[source[0]]
            else:
                return None
            produced.add(node.output[0])
        else:
            if node.attribute or len(node.input) != 2 or len(node.output) != 1:
                return None
            if node.input[0] == node.input[1] or any(name not in produced for name in node.input):
                return None
            joins.append(node)
            produced.add(node.output[0])
    if len(branches) < 3 or len(joins) < 2:
        return None
    if any(len(branch) > 3 for branch in branches):
        return None
    names = [node.output[0] for branch in branches for node in branch]
    if len(set(names)) != len(names):
        return None
    return dict(stem=nodes[:stem_end], branches=branches, joins=joins, stem_out=stem_out,
                branch_of=branch_of, pool=pool)


def _validate(model):
    parts = parse_join_dag(list(model.graph.node))
    if parts is None:
        raise ValueError('join DAG requires a 1x1 stem, three branches and two joins over produced tensors')
    g = model.graph
    stem_nodes = parts['stem']
    branches = parts['branches']
    joins = parts['joins']
    pool = parts['pool']
    final = pool.output[0] if pool is not None else joins[-1].output[0]
    if len(g.input) != 1 or len(g.output) != 1 or final != g.output[0].name:
        raise ValueError('join DAG requires one input, one output and a final join or pool')
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*g.input, *g.output, *g.value_info]}
    expected_output = [1, 3, 4, 4] if pool is not None else [1, 3, 8, 8]
    for value, wanted in ((g.input[0], [1, 3, 8, 8]), (g.output[0], expected_output)):
        if value.type.tensor_type.elem_type != 1 or shapes.get(value.name) != wanted:
            raise ValueError('join DAG external tensors must be float32 %s' % wanted)
    if pool is not None:
        pool_attrs = {a.name: h.get_attribute_value(a) for a in pool.attribute}
        allowed_pool = {'kernel_shape': [2, 2], 'strides': [2, 2], 'auto_pad': b'NOTSET',
                        'ceil_mode': 0, 'count_include_pad': 0, 'storage_order': 0,
                        'dilations': [1, 1]}
        if (pool.domain not in ('', 'ai.onnx') or len(pool.input) != 1 or len(pool.output) != 1
                or any(k not in allowed_pool or v != allowed_pool[k] for k, v in pool_attrs.items())):
            raise ValueError('join DAG terminal pool must be 2x2 stride-2 MaxPool or AveragePool')
    for branch in branches:
        for node in branch:
            declared = shapes.get(node.output[0])
            if declared is None or len(declared) != 4 or declared[0] != 1 or declared[2:] != [8, 8]:
                raise ValueError('join DAG branch outputs must be float32 1xCx8x8 tensors')
    # Join outputs are C3 8x8 by construction (both operands are), and `normalize`
    # does not always carry their inferred shapes, so they are not shape-checked here.
    constants = {t.name: nh.to_array(t) for t in g.initializer}
    stem = stem_nodes[0]
    w1 = constants.get(stem.input[1]) if len(stem.input) > 1 else None
    hidden = w1.shape[0] if w1 is not None and w1.ndim == 4 else 0
    if (w1 is None or w1.dtype != np.float32 or w1.ndim != 4 or w1.shape != (hidden, 3, 1, 1)
            or not 3 <= hidden <= 16 or stem.input[2] not in constants
            or constants[stem.input[2]].shape != (hidden,)):
        raise ValueError('join DAG stem must be a 1x1 Conv with hidden channels 3..16')
    attrs = {a.name: h.get_attribute_value(a) for a in stem.attribute}
    supported = {'kernel_shape': [1, 1], 'pads': [0, 0, 0, 0], 'strides': [1, 1],
                 'dilations': [1, 1], 'group': 1}
    if any(k not in supported or v != supported[k] for k, v in attrs.items()):
        raise ValueError('unsupported stem attributes')
    if len(stem_nodes) == 2 and (stem_nodes[1].attribute or list(stem_nodes[1].input) != list(stem.output)):
        raise ValueError('join DAG stem Relu must consume the stem output')
    specs = []
    for branch in branches:
        layers = []
        input_channels = hidden
        for position, node in enumerate(branch):
            expected_source = parts['stem_out'] if position == 0 else branch[position - 1].output[0]
            if node.input[0] != expected_source:
                raise ValueError('join DAG branch layers must chain off the previous layer')
            w = constants.get(node.input[1]) if len(node.input) > 1 else None
            b = constants.get(node.input[2]) if len(node.input) > 2 else None
            kernel = w.shape[2] if w is not None and w.ndim == 4 else 0
            if w is None or b is None or w.dtype != np.float32 or b.dtype != np.float32:
                raise ValueError('join DAG branch layers require constant float32 weights and bias')
            layer_attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
            group = int(layer_attrs.get('group', 1))
            if group == 1 or (len(branch) > 1 and group == input_channels):
                if group > 1:
                    # A depthwise layer inside a chain is expanded to an equivalent
                    # block-diagonal dense kernel, because the dedicated depthwise
                    # emitter only models a branch that reads the stem directly.
                    if (w.shape != (input_channels, 1, kernel, kernel) or b.shape != (input_channels,)
                            or kernel not in (1, 3) or layer_attrs.get('pads') != [kernel // 2] * 4
                            or layer_attrs.get('strides', [1, 1]) != [1, 1]
                            or layer_attrs.get('dilations', [1, 1]) != [1, 1]):
                        raise ValueError('join DAG chained depthwise layers need group C, 1x1/3x3 and symmetric pad')
                    expanded = np.zeros((input_channels, input_channels, kernel, kernel), np.float32)
                    for channel in range(input_channels):
                        expanded[channel, channel] = w[channel, 0]
                    output_channels = input_channels
                    w = expanded
                else:
                    output_channels = w.shape[0] if w.ndim == 4 else 0
                    if (kernel not in (1, 3) or w.shape != (output_channels, input_channels, kernel, kernel)
                            or b.shape != (output_channels,) or not 1 <= output_channels <= 16):
                        raise ValueError('join DAG dense layers must be C1..16 1x1/3x3 Conv over the previous tensor')
                allowed = {'kernel_shape': [kernel, kernel], 'pads': [kernel // 2] * 4,
                           'strides': [1, 1], 'dilations': [1, 1], 'group': 1}
                if any(k not in allowed or v != allowed[k] for k, v in layer_attrs.items() if k != 'group'):
                    raise ValueError('unsupported dense layer attributes')
                layers.append(dict(kind='dense', node=node, weights=w, bias=b, kernel=kernel,
                                   input_channels=input_channels, output_channels=output_channels,
                                   depthwise_expanded=bool(group > 1)))
                input_channels = output_channels
            else:
                if len(branch) != 1:
                    raise ValueError('join DAG depthwise branches must be a single layer')
                if (group != 3 or hidden != 3 or kernel not in (1, 3, 5)
                        or w.shape != (3, 1, kernel, kernel) or b.shape != (3,)):
                    raise ValueError('join DAG depthwise branches require a three-channel stem and group3 weights')
                allowed = {'kernel_shape': [kernel, kernel], 'pads': [kernel // 2] * 4,
                           'strides': [1, 1], 'dilations': [1, 1], 'group': 3}
                if any(k not in allowed or v != allowed[k] for k, v in layer_attrs.items()):
                    raise ValueError('unsupported depthwise layer attributes')
                layers.append(dict(kind='depthwise', node=node, weights=w, bias=b, kernel=kernel,
                                   input_channels=hidden, output_channels=3))
                input_channels = 3
        if input_channels != 3:
            raise ValueError('join DAG branch outputs must have three channels')
        specs.append(layers)
    # A join operand must be a C3 tensor: only branch finals are guaranteed to be,
    # plus any intermediate layer that happens to keep three channels.
    for entries in specs:
        for position, entry in enumerate(entries):
            if position == len(entries) - 1:
                continue
            if entry['weights'].shape[0] != 3:
                for join in joins:
                    if entry['node'].output[0] in list(join.input):
                        raise ValueError('join DAG joins may only consume three-channel tensors')
    return dict(stem_nodes=stem_nodes, stem=stem, hidden=hidden, branches=specs,
                joins=joins, constants=constants, branch_of=parts['branch_of'], pool=pool)


def _band(params):
    if hasattr(params, 'output_scale'):
        return params.output_scale, params.output_zero_point
    return params['output_scale'], params['output_zero_point']


def _adjusted(params):
    if hasattr(params, 'output_scale'):
        scale, zero = params.output_scale, params.output_zero_point
    else:
        scale, zero = params['output_scale'], params['output_zero_point']
    return float(np.float32(scale * max(128 + zero, 127 - zero) / 127))


def _prepare(model, spec, stem_scale, stem_zp, ranges=None):
    """Quantize every branch layer in order, then propagate the join bands.

    A layer's input band is the previous layer's output band (the stem's for the first
    layer). Intermediate layers keep their natural quantization; each branch's final
    layer is zero-centered so a join can consume it, at its natural adjusted scale or
    at the band an Add/Sub/Max join demands. Mul joins fold two free scales and commit
    both; Add/Sub/Max capture one shared band, re-quantizing an uncommitted branch
    output and rejecting a tensor a later join wants on a different band.

    With `ranges` (an `open_rknpu.calibration.measure` report) every layer whose measured
    tensor has an entry uses that measured band as its natural band instead of the
    analytic one, through the same `measured_range` helper the other profiles use. A
    branch final that a join re-quantizes onto zero point 0 still moves, exactly as it
    does on the analytic path.
    """
    from .calibration import measured_range
    g = model.graph
    t = spec['stem_nodes'][-1].output[0]
    branches = []
    for layers in spec['branches']:
        entries = []
        input_scale, input_zp = stem_scale, stem_zp
        for layer in layers:
            measured = measured_range(ranges, layer['node'].output[0]) if ranges is not None else None
            if layer['kind'] == 'dense':
                natural = native_quantize(layer['weights'], layer['bias'], input_scale, input_zp, measured)
                entry = dict(kind='dense', layer=layer, natural_scale=_adjusted(natural),
                             quantization=natural, standalone=None)
            else:
                standalone = h.make_model(h.make_graph(spec['stem_nodes'] + [layer['node']],
                    'depthwise_branch', [g.input[0]],
                    [h.make_tensor_value_info(layer['node'].output[0], 1, [1, 3, 8, 8])],
                    [nh.from_array(spec['constants'][name], name)
                     for node in (spec['stem'], layer['node']) for name in node.input[1:]],
                    value_info=[h.make_tensor_value_info(t, 1, [1, spec['hidden'], 8, 8])]),
                    opset_imports=list(model.opset_import))
                standalone.ir_version = model.ir_version
                _, natural_meta = compile_depthwise(standalone, output_range=measured)
                entry = dict(kind='depthwise', layer=layer, natural_scale=_adjusted(natural_meta),
                             standalone=standalone, quantization=natural_meta['depthwise'])
            entries.append(entry)
            input_scale, input_zp = _band(entry['quantization'])
        branches.append(entries)

    def requantize(entries, scale):
        entry = entries[-1]
        layer = entry['layer']
        if len(entries) > 1:
            in_scale, in_zp = _band(entries[-2]['quantization'])
        else:
            in_scale, in_zp = stem_scale, stem_zp
        if entry['kind'] == 'dense':
            entry['quantization'] = native_quantize(layer['weights'], layer['bias'], in_scale, in_zp,
                                                    dict(scale=scale, zero_point=0))
        else:
            branch, branch_meta = compile_depthwise(entry['standalone'],
                                                    output_range=dict(scale=scale, zero_point=0))
            entry['quantization'] = branch_meta['depthwise']
            entry['recompiled'] = branch

    # Every branch's final layer must be zero-centered so a join can consume it, at
    # its natural adjusted scale until a join demands a shared band.
    for entries in branches:
        requantize(entries, entries[-1]['natural_scale'])
    by_name = {}
    scales = {}
    committed = set()
    for entries in branches:
        for position, entry in enumerate(entries):
            name = entry['layer']['node'].output[0]
            scales[name] = (_band(entry['quantization'])[0] if position < len(entries) - 1
                            else entries[-1]['natural_scale'])
        # An intermediate layer also feeds the next layer, so its band cannot move.
        for entry in entries[:-1]:
            committed.add(entry['layer']['node'].output[0])
        by_name[entries[-1]['layer']['node'].output[0]] = entries

    def close(a, b):
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b))

    for join in spec['joins']:
        first, second = join.input
        if join.op_type == 'Mul':
            committed.add(first)
            committed.add(second)
            scales[join.output[0]] = float(np.float32(128 * scales[first] * scales[second]))
            committed.add(join.output[0])
            continue
        fixed_first, fixed_second = first in committed, second in committed
        if fixed_first and fixed_second:
            if not close(scales[first], scales[second]):
                raise ValueError('join DAG Add/Sub/Max needs both operands on one scale; '
                                 'reused tensors fixed different bands')
            target = scales[first]
        elif fixed_first or fixed_second:
            target = scales[first] if fixed_first else scales[second]
            free = second if fixed_first else first
            if free not in by_name:
                raise ValueError('join DAG cannot re-quantize a join result onto another band')
            if not close(scales[free], target):
                if free in committed:
                    raise ValueError('join DAG operand was already committed to another band')
                requantize(by_name[free], target)
                scales[free] = float(target)
        else:
            target = float(np.float32(max(scales[first], scales[second])))
            for name in (first, second):
                if name in by_name and not close(scales[name], target):
                    requantize(by_name[name], target)
                scales[name] = float(target)
        committed.add(first)
        committed.add(second)
        scales[join.output[0]] = float(np.float32(2 * target))
        committed.add(join.output[0])
    return branches, by_name, scales


def compile_join_dag(model, output_range=None, serial=True, calibration_ranges=None):
    """Compile a general join expression DAG with multi-layer branches on one stem."""
    onnx.checker.check_model(model)
    if any(n.domain not in ('', 'ai.onnx') for n in model.graph.node):
        raise ValueError('join DAG requires default-domain nodes')
    if calibration_ranges is not None and output_range is not None:
        raise ValueError('calibration and output quantization overrides cannot be combined')
    spec = _validate(model)
    pool_kind = spec['pool'].op_type if spec['pool'] is not None else None
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
        stem_data, stem_meta = compile_model(stem_path, calibration_ranges=calibration_ranges)
    stem_scale = stem_meta['output_scale']
    stem_zp = stem_meta['output_zero_point']
    branches, by_name, scales = _prepare(model, spec, stem_scale, stem_zp, calibration_ranges)
    # Program layout: stem, one 0x440 slot per branch layer, one 78-word slot per join,
    # then packed constants and the liveness-placed arena.
    surface = 8 * 8 * 16
    layer_bases = {}
    cursor = 0x440
    for entries in branches:
        for entry in entries:
            layer_bases[entry['layer']['node'].output[0]] = cursor
            cursor = _align(cursor + 0x440)
    join_bases = {}
    for join in spec['joins']:
        join_bases[join.output[0]] = cursor
        cursor = _align(cursor + (78 + 4) * 8)
    if pool_kind:
        cursor = _align(cursor + (37 + 4) * 8)
    weights1 = _align(cursor)
    weight_size1 = _align(1 * 1 * _align(hidden, 4) * 4)
    bias1 = weights1 + weight_size1
    bias_size1 = ((hidden + 3) // 4) * 32
    cursor = _align(bias1 + bias_size1)
    blocks = {}
    for entries in branches:
        for entry in entries:
            kernel = entry['layer']['kernel']
            if entry['kind'] == 'dense':
                channels = entry['layer']['output_channels']
                wsize, bsize = _align(channels * 16 * kernel * kernel), ((channels + 3) // 4) * 32
            else:
                wsize, bsize = _align(kernel * kernel * 32), ((3 + 3) // 4) * 24
            blocks[entry['layer']['node'].output[0]] = (cursor, cursor + wsize, wsize, bsize)
            cursor = _align(cursor + wsize + bsize)
    payload_size = _align(cursor)
    input_off = _align(payload_size, 4096)
    input_bytes = 8 * 16 * 3
    bindings = [Access(reads=('input0',), writes=('stem',))]
    for entries in branches:
        source = 'stem'
        for entry in entries:
            target = entry['layer']['node'].output[0]
            bindings.append(Access(reads=(source,), writes=(target,)))
            source = target
    for position, join in enumerate(spec['joins']):
        last = position == len(spec['joins']) - 1
        target = join.output[0] if (not last or pool_kind) else 'output'
        bindings.append(Access(reads=tuple(join.input), writes=(target,)))
    if pool_kind:
        bindings.append(Access(reads=(spec['joins'][-1].output[0],), writes=('output',)))
    sizes = {'stem': surface}
    for entries in branches:
        for entry in entries:
            sizes[entry['layer']['node'].output[0]] = surface
    for position, join in enumerate(spec['joins']):
        if position < len(spec['joins']) - 1 or pool_kind:
            sizes[join.output[0]] = surface
    order_idx, intervals, offsets = plan(bindings, sizes, start=_align(input_off + input_bytes),
                                         defined=('input0',))
    # Conservative placement: every internal gets a fresh slot. The board has returned
    # stale data when a slot was written by two different tasks of a mixed task family
    # (see the investigation log), so this profile does not rely on arena reuse.
    offsets = {}
    cursor = _align(input_off + input_bytes)
    for binding in bindings:
        name = binding.writes[0]
        if name in sizes:
            offsets[name] = cursor
            cursor = _align(cursor + sizes[name])
    output_surface = 4 * 4 * 16 if pool_kind else surface
    output_off = _align(max(offsets[name] + sizes[name] for name in sizes))
    layout = dict(offsets)
    layout['input0'] = input_off
    layout['output'] = output_off
    layout_sizes = dict(sizes)
    layout_sizes['input0'] = input_bytes
    layout_sizes['output'] = output_surface
    for external in ('input0', 'output'):
        offset = layout[external]
        size = layout_sizes[external]
        for name in layout:
            if name == external:
                continue
            if offset < layout[name] + layout_sizes[name] and layout[name] < offset + size:
                raise ValueError('join DAG external tensor %s overlaps %s' % (external, name))
    join_scales = {}
    for position, join in enumerate(spec['joins']):
        operands = [scales[join.input[0]], scales[join.input[1]]]
        last = position == len(spec['joins']) - 1
        _, join_scale, join_zero = _join_fields(join.op_type, 8, 8, 3, 0, 0, 0, surface,
                                                operands, (0, 0), output_range if last else None)
        join_scales[join.output[0]] = join_scale
        if last:
            output_scale, output_zero_point = join_scale, join_zero
    stem_src = {w & 65535: (w >> 16) & 0xffffffff for w in struct.unpack_from('<126Q', stem_data)}

    def stem_fill(payload, offset):
        payload[offset:offset + weight_size1] = stem_data[stem_src[0x1110]:stem_src[0x1110] + weight_size1]
        payload[offset + weight_size1:offset + weight_size1 + bias_size1] = stem_data[
            stem_src[0x5020]:stem_src[0x5020] + bias_size1]

    def stem_fields(addresses, constant_offsets):
        values = dict(stem_src)
        values.update({0x1110: constant_offsets['stem'],
                       0x5020: constant_offsets['stem'] + weight_size1,
                       0x4020: addresses['stem'], 0x1070: addresses['input0']})
        return values

    # One stage per task. `open_rknpu.compose` assigns program slots and constant blocks
    # in declared order, places every internal in a fresh slot (this profile does not
    # rely on arena reuse), substitutes the planned addresses and returns the declared
    # binding view. The declaration order is the emitted order, so the container is
    # byte-identical to the hand-assembled one (tests/test_join_dag.py).
    stages = [Stage(name='stem', family='native-conv', reads=('input0',), writes=('stem',),
                    fields=stem_fields,
                    constants=(ConstantSpec('stem', weight_size1 + bias_size1, stem_fill),),
                    bindings=(Binding(0x1070, 'input0', 'read'),
                              Binding(0x4020, 'stem', 'write')))]

    def layer_stage(name, source, source_zp, entry, wsize, bsize):
        kernel = entry['layer']['kernel']
        if entry['kind'] == 'dense':
            input_channels = entry['layer']['input_channels']
            output_channels = entry['layer']['output_channels']
            quantization = entry['quantization']

            def fill(payload, offset, kernel=kernel, input_channels=input_channels,
                     output_channels=output_channels, quantization=quantization, wsize=wsize):
                _pack_layer_head(payload, kernel, input_channels, output_channels,
                                 offset, offset + wsize, quantization)

            def fields(addresses, constant_offsets, kernel=kernel, input_channels=input_channels,
                       output_channels=output_channels, quantization=quantization, wsize=wsize,
                       name=name, source=source, source_zp=source_zp):
                values = _layer_fields(kernel, input_channels, output_channels,
                                       constant_offsets[name], constant_offsets[name] + wsize,
                                       addresses[name], addresses[source], quantization)
                values[0x1184] = int(source_zp) & 0xffff
                return values
        else:
            payload = entry.get('recompiled')
            if payload is None:
                payload, _ = compile_depthwise(entry['standalone'],
                                               output_range=dict(scale=entry['natural_scale'],
                                                                 zero_point=0))
            info = decode_sequence(payload)
            block = payload[96 + 16 * info['task_count']:]
            words = struct.unpack_from('<126Q', block, 0x440)
            regs = {word & 65535: (word >> 16) & 0xffffffff for word in words}
            weight_source = regs[0x1110]
            bias_source = regs[0x5020]

            def fill(payload, offset, block=block, weight_source=weight_source,
                     bias_source=bias_source, wsize=wsize, bsize=bsize):
                payload[offset:offset + wsize] = block[weight_source:weight_source + wsize]
                payload[offset + wsize:offset + wsize + bsize] = block[bias_source:bias_source + bsize]

            def fields(addresses, constant_offsets, regs=regs, name=name, source=source,
                       source_zp=source_zp, wsize=wsize):
                values = dict(regs)
                values.update({0x1070: addresses[source], 0x4020: addresses[name],
                               0x1110: constant_offsets[name],
                               0x5020: constant_offsets[name] + wsize,
                               0x1184: int(source_zp) & 0xffff})
                return values

        return Stage(name=name, family='native-conv', reads=(source,), writes=(name,),
                     fields=fields,
                     constants=(ConstantSpec(name, wsize + bsize, fill),),
                     bindings=(Binding(0x1070, source, 'read'), Binding(0x4020, name, 'write')))

    for entries in branches:
        source = 'stem'
        source_zp = stem_zp
        for entry in entries:
            name = entry['layer']['node'].output[0]
            wsize, bsize = blocks[name][2], blocks[name][3]
            stages.append(layer_stage(name, source, source_zp, entry, wsize, bsize))
            source = name
            source_zp = _band(entry['quantization'])[1]

    for position, join in enumerate(spec['joins']):
        first, second = join.input
        operands = [scales[first], scales[second]]
        last = position == len(spec['joins']) - 1
        target = join.output[0] if (not last or pool_kind) else 'output'
        selected = output_range if last else None

        def join_fields(addresses, constant_offsets, first=first, second=second, target=target,
                        operands=operands, selected=selected, op=join.op_type):
            values, _, _ = _join_fields(op, 8, 8, 3, addresses[first], addresses[second],
                                        addresses[target], surface, operands, (0, 0), selected)
            return values

        stages.append(Stage(name='join_' + join.output[0], family='elementwise',
                            reads=(first, second), writes=(target,), fields=join_fields,
                            bindings=(Binding(0x5018, first, 'read'),
                                      Binding(0x5038, second, 'read'),
                                      Binding(0x4020, target, 'write'))))
    if pool_kind:
        pool_source = spec['joins'][-1].output[0]

        def pool_fields(addresses, constant_offsets, source=pool_source):
            return pool_registers(pool_kind, 8, 8, 4, 4, addresses[source], addresses['output'])

        stages.append(Stage(name='pool', family='pool', reads=(pool_source,), writes=('output',),
                            fields=pool_fields,
                            bindings=(Binding(0x701c, pool_source, 'read'),
                                      Binding(0x6070, 'output', 'write'))))

    tensors = [TensorSpec('input0', ROLE_INPUT, LAYOUT_PACKED_U8, (1, 8, 8, 3), input_bytes, 0),
               TensorSpec('stem', ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, hidden), surface, 0)]
    layer_index = 0
    for entries in branches:
        for entry in entries:
            name = entry['layer']['node'].output[0]
            tensors.append(TensorSpec(name, ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3),
                                      surface, layer_index))
            layer_index += 1
    for position, join in enumerate(spec['joins']):
        if position < len(spec['joins']) - 1 or pool_kind:
            tensors.append(TensorSpec(join.output[0], ROLE_INTERNAL, LAYOUT_NATIVE16,
                                      (1, 8, 8, 3), surface, 0))
    tensors.append(TensorSpec('output', ROLE_OUTPUT, LAYOUT_NATIVE16,
                              (1, 4, 4, 3) if pool_kind else (1, 8, 8, 3), output_surface, 0))
    binary, composed = compose(stages, tensors, input_scale=1.0, input_zero_point=0,
                               output_scale=output_scale, output_zero_point=output_zero_point,
                               serial=serial, reuse=False)
    offsets = composed['tensor_offsets']
    meta = dict(profile='join-dag', engine_runs=composed['engine_runs'],
        join_expression=[dict(name=join.output[0], kind=join.op_type,
                                                          inputs=list(join.input)) for join in spec['joins']],
        head_names=[entries[-1]['layer']['node'].output[0] for entries in branches],
        head_kinds=[entries[-1]['kind'] for entries in branches],
        head_scales=[float(scales[entries[-1]['layer']['node'].output[0]]) for entries in branches],
        branch_names=[[entry['layer']['node'].output[0] for entry in entries] for entries in branches],
        branch_quantization=[[(entry['quantization'].metadata() if entry['kind'] == 'dense'
                               else entry['quantization']) for entry in entries] for entries in branches],
        join_scales=[float(join_scales[join.output[0]]) for join in spec['joins']],
        input_scale=1.0, input_zero_point=0, output_scale=output_scale,
        pool=pool_kind,
        output_zero_point=output_zero_point, shape_nhwc=[1, 8, 8, 3],
        output_shape_nhwc=[1, 4, 4, 3] if pool_kind else [1, 8, 8, 3], output_tensors=['output'],
        schedule=composed['schedule'],
        tensor_lifetimes=composed['tensor_lifetimes'],
        tensor_offsets={name: offsets[name] for name in sorted(offsets)},
        lifetime_bytes=composed['live_bytes'], allocated_bytes=composed['allocated_bytes'],
        stem_quantization=stem_meta['quantization'],
        head_quantization=[(entries[-1]['quantization'].metadata() if entries[-1]['kind'] == 'dense'
                            else entries[-1]['quantization']) for entries in branches],
        depthwise_quantization=[(entries[-1]['quantization'] if entries[-1]['kind'] == 'depthwise' else None)
                                for entries in branches])
    meta.update({key: value for key, value in composed.items()
                 if key not in ('profile', 'engine_runs', 'schedule', 'tensor_lifetimes',
                                'tensor_offsets', 'live_bytes', 'allocated_bytes')})
    return binary, meta


def join_dag_reference(inputs, stem_quantization, head_quantizations, head_names, expression,
                       head_kinds=None, depthwise_quantizations=None,
                       branch_names=None, branch_quantization=None, pool=None):
    """Replay the recorded join expression, folding each branch's layer chain.

    A terminal `pool` (MaxPool or AveragePool) is applied to the final value.
    """
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
    for index, name in enumerate(head_names):
        grid = stem_grid
        zero_point = stem.output_zero_point
        if branch_names:
            for position, layer_name in enumerate(branch_names[index]):
                q = quantize(branch_quantization[index][position])
                if position == 0 and head_kinds and head_kinds[index] == 'depthwise':
                    grid = depthwise_reference(grid, q, zero_point)
                else:
                    grid = native_reference(grid, q, zero_point)
                zero_point = q.output_zero_point
                values[layer_name] = grid
        elif head_kinds and head_kinds[index] == 'depthwise':
            q = quantize(depthwise_quantizations[index])
            grid = depthwise_reference(grid, q, stem.output_zero_point)
        else:
            q = quantize(head_quantizations[index])
            grid = native_reference(grid, q, stem.output_zero_point)
        values[name] = grid
    for step in expression:
        values[step['name']] = join_reference(step['kind'], values[step['inputs'][0]],
                                              values[step['inputs'][1]])
    value = values[expression[-1]['name']]
    if pool:
        height, width, channels = value.shape
        blocked = value.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
        if pool == 'MaxPool':
            value = blocked.max(axis=(1, 3)).astype(np.int8)
        else:
            value = np.rint(blocked.sum(axis=(1, 3)) / 4).astype(np.int8)
    return value
