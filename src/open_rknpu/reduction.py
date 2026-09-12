# SPDX-License-Identifier: MIT
"""Independent experimental Conv[/Relu] followed by three 2x2 pool stages.

This implements the explicit ONNX stages, not a replacement for GlobalAveragePool:
INT8 rounding occurs after each stage.
"""
from pathlib import Path
import struct, tempfile
import onnx
from onnx import helper as h
from .pooling import compile_pool, pool_reference
from .register_profile import REGISTERS


def reduction_reference(inputs,quantization,kind,levels=3):
    """Integer reference for the three-stage 8x8->4x4->2x2->1x1 reduction profile.

    The profiles 5/6 container is three linked 2x2/stride-2 pool tasks after one
    dense Conv, so the reference is the shared `pooling.pool_reference` at three
    levels. AveragePool rounds each stage separately (half to even) and the sum is
    deliberately not an unrounded global mean. Board evidence: `reduction_api_suite`
    profiles 5/6, 256 public-API inferences and 768 exact bytes (`research/README.md`).
    """
    if levels!=3:
        raise ValueError("reduction reference models exactly three 2x2 pooling levels")
    return pool_reference(inputs,quantization,kind,3)

def compile_reduction(path,output_scale=None,output_zero_point=None,calibration_ranges=None):
    m = onnx.load(path)
    onnx.checker.check_model(m)
    g = m.graph
    if not len(g.node) in (4, 5):
        raise ValueError('unsupported staged pooling graph')
    pools = list(g.node[-3:])
    kind = pools[0].op_type
    if not kind in ('MaxPool', 'AveragePool'):
        raise ValueError('unsupported staged pooling graph')
    for i, pool in enumerate(pools):
        if not (pool.op_type == kind and pool.domain in ('', 'ai.onnx')):
            raise ValueError('unsupported staged pooling graph')
        if not {a.name: h.get_attribute_value(a) for a in pool.attribute} == {'kernel_shape': [2, 2], 'strides': [2, 2]}:
            raise ValueError('unsupported staged pooling graph')
        if i:
            if not list(pool.input) == list(pools[i - 1].output):
                raise ValueError('unsupported staged pooling graph')
    if not (len(g.output) == 1 and pools[-1].output[0] == g.output[0].name):
        raise ValueError('unsupported staged pooling graph')
    if g.output[0].type.tensor_type.elem_type!=1:
        raise ValueError('staged pooling requires float32 output')
    if not [d.dim_value for d in g.output[0].type.tensor_type.shape.dim] == [1, 3, 1, 1]:
        raise ValueError('unsupported staged pooling graph')
    sub = h.make_graph(list(g.node[:-2]), 'first_pool', list(g.input), [h.make_tensor_value_info(pools[0].output[0], 1, [1, 3, 4, 4])], list(g.initializer))
    first = h.make_model(sub, opset_imports=list(m.opset_import))
    first.ir_version = m.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'first.onnx'
        onnx.save(first, p)
        source, meta = compile_pool(p,output_scale,output_zero_point,calibration_ranges)
    data = bytearray(8192)
    data[2304:2368] = source[1536:1600]
    data[2368:2496] = source[1600:1728]
    regs = {w & 65535: w >> 16 & 4294967295 for w in struct.unpack_from('<126Q', source)}
    regs.update({4368: 2304, 20512: 2368})
    for i, (r, v, tag) in enumerate(REGISTERS):
        struct.pack_into('<Q', data, i * 8, tag << 48 | regs.get(r, v) << 16 | r)
    data[126 * 8:130 * 8] = source[126 * 8:130 * 8]
    pool_words = struct.unpack_from('<37Q', source, 1088)
    for stage, size, inp, out in ((0, 8, 4096, 5120), (1, 4, 5120, 6144), (2, 2, 6144, 12288)):
        start = 1088 + stage * 384
        output = size // 2
        fields = {24588: size - 1, 24592: size - 1, 24600: output - 1, 24604: output - 1, 24688: out, 24700: output * output * 16, 24708: output * output * 16, 28684: size - 1, 28688: size - 1, 28700: inp, 28708: size * 16, 28712: size * size * 16}
        for i, w in enumerate(pool_words):
            r = w & 65535
            value = fields.get(r, w >> 16 & 4294967295)
            struct.pack_into('<Q', data, start + i * 8, w >> 48 << 48 | value << 16 | r)
        next_offset = start + 384 if stage < 2 else 0
        control = 20 if stage < 2 else 40
        for i, w in enumerate((72339069014638608 | next_offset << 16, 72339069014638612 | control << 16, 18295873486192640, 36310272001966088)):
            struct.pack_into('<Q', data, start + (37 + i) * 8, w)
    meta['pool_levels'] = 3
    meta['profile'] = 5 if kind=='MaxPool' else 6
    meta['output_shape_nhwc'] = [1, 1, 1, 3]
    return (bytes(data), meta)
