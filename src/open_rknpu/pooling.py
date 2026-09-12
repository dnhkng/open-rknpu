# SPDX-License-Identifier: MIT
"""Independent experimental Conv[/Relu] plus 2x2 stride-2 pool emitter."""
from pathlib import Path
import struct, tempfile
import numpy as np
import onnx
from onnx import helper as h
from .compiler import compile_model
from .quantization import reference
from .register_profile import REGISTERS
POOL = [(24580, 14), (28676, 14), (24588, 7), (24592, 7), (24596, 15), (24600, 3), (24604, 3), (24608, 15), (24612, 17), (24628, 1114369), (24632, 0), (24636, 0), (24640, 0), (24644, 524160), (24648, 524160), (24652, 524160), (24656, 524160), (24660, 0), (24664, 0), (24668, 0), (24672, 0), (24676, 0), (24680, 0), (24684, 0), (24688, 12288), (24700, 256), (24708, 256), (24796, 3), (28684, 7), (28688, 7), (28692, 15), (28700, 4096), (28708, 128), (28712, 1024), (28720, 64), (32808, 12), (32812, 4294967295)]

def pool_registers(kind,input_height,input_width,output_height,output_width,
                   input_offset,output_offset):
    """Register values for one verified 2x2/stride-2 pool task.

    `POOL` lists every register the task sets; this fills the geometry and the two
    addresses. AveragePool additionally selects its rounding constants. Shared with
    the sequence lowering and the branch-join composer so the pool program has one
    definition.
    """
    fields={0x600c:input_width-1,0x6010:input_height-1,0x6018:output_width-1,0x601c:output_height-1,
            0x6070:output_offset,0x607c:output_height*output_width*16,0x6084:output_height*output_width*16,
            0x700c:input_width-1,0x7010:input_height-1,0x701c:input_offset,
            0x7024:input_width*16,0x7028:input_height*input_width*16}
    if kind=='AveragePool':
        fields.update({0x6024:0x10,0x6038:0x8000,0x603c:0x8000,
                       0x6048:0x7ff00,0x604c:0x7fe80,0x6050:0x7fe00})
    return fields


def pool_tag(register):
    """Descriptor tag for a pool register word (three address ranges)."""
    return 0x4001 if register<0x7000 else 0x8001 if register<0x8000 else 0x401


def _validate_pool(kind,levels):
    if kind not in ("MaxPool","AveragePool"):
        raise ValueError("pool reference supports MaxPool or AveragePool")
    if not isinstance(levels,int) or levels<1:
        raise ValueError("pool reference requires at least one 2x2 pooling level")


def pool_codes_reference(codes,kind,levels=1):
    """`levels` 2x2/stride-2 pool stages on an already-requantized INT8 grid.

    Shared by `pool_reference` and `network_reference`: MaxPool keeps the signed 2x2
    block maximum, AveragePool rounds the mean of the four codes half to even, and the
    unpaired last row/column of an odd-sized grid is dropped to match the emitted
    floor() geometry.
    """
    _validate_pool(kind,levels)
    codes=np.asarray(codes)
    for _ in range(levels):
        height,width=codes.shape[0]//2,codes.shape[1]//2
        blocks=codes[:height*2,:width*2].reshape(height,2,width,2,codes.shape[2]).astype(np.int64)
        codes=blocks.max(axis=(1,3)) if kind=="MaxPool" else np.rint(blocks.sum(axis=(1,3))/4.0)
        codes=np.clip(codes,-128,127).astype(np.int8)
    return codes


def pool_reference(inputs,quantization,kind,levels=1):
    """Integer reference for the Conv[/Relu] plus `levels` 2x2/stride-2 pool profile.

    Replays the emitted task arithmetic from the container's own quantization
    parameters: the stem is `quantization.reference` (per-channel conversion,
    multiplier/shift and clip), then `pool_codes_reference` applies every pool stage.

    Board evidence (all byte-exact, recorded in `research/README.md`):
    `pool_api_suite` profiles 3/4 (256 public-API inferences, 12,288 bytes),
    `reduction_api_suite` profiles 5/6 (768 bytes) and `scheduled_pool_suite`
    (12 models, 96 runs, 88,320 bytes).
    """
    _validate_pool(kind,levels)
    return pool_codes_reference(reference(inputs,quantization),kind,levels)


def compile_pool(path,output_scale=None,output_zero_point=None,calibration_ranges=None):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    g = model.graph
    if not [n.op_type for n in g.node] in (['Conv', 'MaxPool'], ['Conv', 'AveragePool'], ['Conv', 'Relu', 'MaxPool'], ['Conv', 'Relu', 'AveragePool']):
        raise ValueError('unsupported pooling graph')
    pool = g.node[-1]
    if not pool.domain in ('', 'ai.onnx'):
        raise ValueError('unsupported pooling graph')
    if not (list(pool.input) == list(g.node[-2].output) and len(pool.output) == 1):
        raise ValueError('unsupported pooling graph')
    attrs = {a.name: h.get_attribute_value(a) for a in pool.attribute}
    if not attrs == {'kernel_shape': [2, 2], 'strides': [2, 2]}:
        raise ValueError('unsupported pooling graph')
    if not (len(g.input) == len(g.output) == 1 and pool.output[0] == g.output[0].name):
        raise ValueError('unsupported pooling graph')
    for v, shape in ((g.input[0], [1, 3, 8, 8]), (g.output[0], [1, 3, 4, 4])):
        if not (v.type.tensor_type.elem_type == 1 and [d.dim_value for d in v.type.tensor_type.shape.dim] == shape):
            raise ValueError('unsupported pooling graph')
    sub = h.make_graph(list(g.node[:-1]), 'pool_input', list(g.input), [h.make_tensor_value_info(pool.input[0], 1, [1, 3, 8, 8])], list(g.initializer))
    first = h.make_model(sub, opset_imports=list(model.opset_import))
    first.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'conv.onnx'
        onnx.save(first, p)
        source, meta = compile_model(p,output_scale,output_zero_point,calibration_ranges)
    if not meta['kernel_size'] == 1:
        raise ValueError('unsupported pooling graph')
    data = bytearray(8192)
    regs = {w & 65535: w >> 16 & 4294967295 for w in struct.unpack_from('<126Q', source)}
    data[1536:1600] = source[regs[4368]:regs[4368] + 64]
    data[1600:1728] = source[regs[20512]:regs[20512] + 128]
    regs.update({4368: 1536, 20512: 1600, 16416: 4096})
    for i, (r, v, tag) in enumerate(REGISTERS):
        struct.pack_into('<Q', data, i * 8, tag << 48 | regs.get(r, v) << 16 | r)
    average = pool.op_type == 'AveragePool'
    changes = {24612: 16, 24632: 32768, 24636: 32768, 24648: 524032, 24652: 523904, 24656: 523776} if average else {}
    for i, (r, v) in enumerate(POOL):
        tag = 16385 if r < 28672 else 32769 if r < 32768 else 1025
        struct.pack_into('<Q', data, 1088 + i * 8, tag << 48 | changes.get(r, v) << 16 | r)
    for start, count, next_offset, control, enable in ((0, 126, 1088, 20, 29), (1088, 37, 0, 40, 96)):
        words = (72339069014638608 | next_offset << 16, 72339069014638612 | control << 16, 18295873486192640, 36310271995674632 | enable << 16)
        for i, w in enumerate(words):
            struct.pack_into('<Q', data, start + (count + i) * 8, w)
    meta['pool'] = pool.op_type
    meta['profile'] = 4 if average else 3
    meta['output_shape_nhwc'] = [1, 4, 4, 3]
    return (bytes(data), meta)
