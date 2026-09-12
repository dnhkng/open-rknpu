"""SPDX-License-Identifier: MIT

Independent Conv-Relu-Conv emitter; no RKNN or capture inputs.

Fixed spatial shape 8x8, external input channels 3, hidden channels 3..16,
output channels 3, either kernel 1x1 or padded 3x3. Container profile 2 uses two linked tasks.
"""
from pathlib import Path
import math
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from .compiler import compile_model
from .register_profile import REGISTERS
from .quantization import Quantization,receptive_fields,reference

# Native INT8, 8x8, 1x1, one input tile of 16, three logical outputs.
NATIVE={0x100c:0,0x1010:0x104,0x101c:0,0x1030:0x30,0x1034:0x10,
        0x1038:0x1010003,0x103c:0x40000,0x1044:0x80004,0x104c:9,
        0x1050:0x10001,0x1054:0x10001,0x1058:0,0x105c:0,
        0x107c:0x20,0x1080:0x40,0x1084:0x80008,0x108c:0,
        0x1188:8,0x118c:0x70007,0x301c:0,0x403c:0x2000f}

def native_quantize(weights,bias,input_scale,input_zero_point,output_range=None,*,symmetric=False):
    kernel=np.asarray(weights).shape[2] if np.asarray(weights).ndim==4 else 1
    weights=np.asarray(weights,np.float32)
    outputs=weights.shape[0]
    weights=weights.reshape(outputs,-1)
    bias=np.asarray(bias,np.float32)
    if not (np.isfinite(weights).all() and np.isfinite(bias).all()):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    lower=np.minimum(weights.min(axis=1),0)
    upper=np.maximum(weights.max(axis=1),0)
    scales=(upper-lower)/np.float32(255)
    if symmetric:
        scales=np.maximum(abs(lower),abs(upper))/np.float32(127)
    maximum=float(scales.max())
    if maximum==0:maximum=1.0/32768
    if not (maximum > 0 and np.isfinite(scales).all() and math.isfinite(input_scale) and input_scale>0):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    scales=np.where(scales==0,np.float32(maximum),scales)
    zp=np.clip(np.rint(-128-lower/scales),-128,127).astype(np.int64)
    if symmetric:
        zp=np.zeros(outputs,dtype=np.int64)
    qw=np.clip(np.rint(weights/scales[:,None])+zp[:,None],-128,127).astype(np.int64)
    centered=qw-zp[:,None]
    bias_units=np.rint(bias/(scales*np.float32(input_scale)))
    if not np.isfinite(bias_units).all() or np.any(np.abs(bias_units)>=2**31):
        raise ValueError("native INT32 bias overflow")
    qb=bias_units.astype(np.int64)-input_zero_point*centered.sum(axis=1)
    if not (np.all(np.abs(qb) + 128 * np.abs(centered).sum(axis=1) < 2 ** 31)):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    lo=(-128-input_zero_point)*input_scale
    hi=(127-input_zero_point)*input_scale
    low=min(0,float((np.minimum(weights*lo,weights*hi).sum(axis=1)+bias).min()))
    high=max(0,float((np.maximum(weights*lo,weights*hi).sum(axis=1)+bias).max()))
    output_scale=float(np.float32((high-low)/255)) or 1.0
    if not math.isfinite(output_scale) or output_scale<=0:
        raise ValueError("invalid native output scale")
    output_zp=int(np.clip(np.rint(-128-low/output_scale),-128,127))
    if output_range is not None:
        output_scale=float(output_range["scale"]);output_zp=int(output_range["zero_point"])
        if not math.isfinite(output_scale) or output_scale<=0 or not -128<=output_zp<=127:
            raise ValueError("invalid calibrated native output quantization")
    factor=maximum*input_scale/output_scale
    shift=min(31,math.floor(math.log2(32767/factor)))
    if not (0 <= shift <= 31):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    multiplier=round(factor*2**shift)
    if not 1<=multiplier<=32767:raise ValueError("native output scale outside nonzero multiplier range")
    channel=np.rint(scales/np.float32(maximum)*16384).astype(np.int64)
    return Quantization(qw,zp,scales,qb,channel,multiplier,shift,output_scale,output_zp,kernel)

def native_reference(inputs,q,input_zero_point=-128):
    """Integer reference for one native16 Conv layer, including its activation.

    The engine's activation registers clamp the accumulator (`0x406c`/`0x40e0` lower
    bound, measured in `docs/plans/pipelining-plan.md` S9), so a layer with `q.relu` clamps
    `acc` at zero *before* the output conversion - clamping the converted INT8 value
    would use the wrong origin.
    """
    patches=receptive_fields(inputs.astype(np.int64),q.kernel_size,input_zero_point)
    acc=np.einsum("hwc,oc->hwo",patches,q.weights-q.weight_zero_points[:,None])+q.biases
    if q.relu:acc=np.maximum(acc,0)
    product=acc*q.channel_multipliers
    scaled=(product+8191+((product>>14)&1))>>14
    return np.clip(((scaled*q.multiplier+(1<<(q.shift-1) if q.shift else 0))>>q.shift)+q.output_zero_point,-128,127).astype(np.int8)

def chain_reference(inputs,q1,q2):
    """Integer reference for the two-layer chain, with the border the emitter writes.

    The second Conv reads the first layer's native16 grid, so the CNA border value is the
    first layer's output zero point (`0x1184`); `native_reference` alone would pad with
    zero point 0.
    """
    return native_reference(reference(inputs,q1),q2,int(q1.output_zero_point))


def compile_chain(path,calibration_ranges=None,output_range=None):
    model=onnx.load(path); onnx.checker.check_model(model)
    g=model.graph
    if not ([n.op_type for n in g.node] == ['Conv', 'Relu', 'Conv']):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    a,activation,b=g.node
    if not (all((n.domain in ('', 'ai.onnx') for n in g.node))):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    if not (len(g.input) == len(g.output) == 1 and (not activation.attribute)):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    if not (list(activation.input) == list(a.output) and b.input[0] == activation.output[0]):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    if not (a.input[0] == g.input[0].name and list(b.output) == [g.output[0].name]):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    constants={t.name:nh.to_array(t) for t in g.initializer}
    for n in (a,b):
        if not (len(n.input) == 3):
            raise ValueError("unsupported two-layer graph or quantization parameters")
        attrs={x.name:h.get_attribute_value(x) for x in n.attribute}
        w=constants[n.input[1]]
        if w.ndim!=4:raise ValueError("two-layer weights must have rank four")
        kernel=w.shape[2]
        supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':1}
        if kernel not in (1,3) or any(k not in supported or v!=supported[k] for k,v in attrs.items()) or (kernel==3 and attrs.get('pads')!=[1]*4):
            raise ValueError("unsupported two-layer graph or quantization parameters")
    for v in (g.input[0],g.output[0]):
        if not (v.type.tensor_type.elem_type == 1 and [d.dim_value for d in v.type.tensor_type.shape.dim] == [1, 3, 8, 8]):
            raise ValueError("unsupported two-layer graph or quantization parameters")
    constants={t.name:nh.to_array(t) for t in g.initializer}
    # The loop above already rejected any weight whose rank is not four, for both nodes.
    w1,w2=constants[a.input[1]],constants[b.input[1]]
    c=w1.shape[0]
    kernel=w2.shape[2]
    if not (3 <= c <= 16 and w1.shape == (c,3,w1.shape[2],w1.shape[2]) and w2.shape == (3,c,kernel,kernel)):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    if constants[a.input[2]].shape!=(c,) or constants[b.input[2]].shape!=(3,):
        raise ValueError("each convolution requires a constant bias matching its output channels")
    if not (all((constants[n].dtype == np.float32 for n in (*a.input[1:], *b.input[1:])))):
        raise ValueError("unsupported two-layer graph or quantization parameters")
    first_graph=h.make_graph([a,activation],"first_layer",list(g.input),
        [h.make_tensor_value_info(activation.output[0],1,[1,c,8,8])],
        [nh.from_array(constants[n],n) for n in a.input[1:]])
    first=h.make_model(first_graph,opset_imports=list(model.opset_import)); first.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"first.onnx"; onnx.save(first,p)
        first_data,meta=compile_model(p,calibration_ranges=calibration_ranges)
    if calibration_ranges is not None and output_range is not None:
        raise ValueError('calibration and chain output override cannot be combined')
    from .calibration import measured_range
    selected_range=output_range if output_range is not None else measured_range(calibration_ranges,b.output[0])
    q2=native_quantize(w2,constants[b.input[2]],meta["output_scale"],meta["output_zero_point"],selected_range)
    data=bytearray(8192)
    # Independently repack the first program and its generated constants.
    source={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from("<126Q",first_data)}
    first_weight_size=((source[0x1030]+63)//64)*64
    first_bias=0x880+first_weight_size
    second_weights=first_bias+128
    data[0x880:first_bias]=first_data[source[0x1110]:source[0x1110]+first_weight_size]
    data[first_bias:second_weights]=first_data[source[0x5020]:source[0x5020]+128]
    source.update({0x1110:0x880,0x5020:first_bias,0x4020:0x1000})
    fields=dict(NATIVE)
    second_bias=second_weights+((3*16*kernel*kernel+63)//64)*64
    fields.update({0x1010:0x108 if kernel==3 else 0x104,
                   0x1030:3*16*kernel*kernel,0x1034:16*kernel*kernel,
                   0x1038:(kernel<<24)|(kernel<<16)|3,0x1068:0x101 if kernel==3 else 0,
                   0x1188:8*kernel*kernel})
    fields.update({0x1184:int(meta["output_zero_point"])&0xffff,
                   0x1024:((c-1)<<16)|16,0x1070:0x1000,0x1110:second_weights,
                   0x5020:second_bias,0x4080:q2.output_zero_point&0xffffffff,
                   0x4084:q2.multiplier,0x4088:q2.shift})
    for start,values,next_offset,control in ((0,source,0x440,0x40),(0x440,fields,0,0x28)):
        for i,(reg,value,tag) in enumerate(REGISTERS):
            struct.pack_into("<Q",data,start+i*8,(tag<<48)|(values.get(reg,value)<<16)|reg)
        for i,word in enumerate((0x0101000000000010|(next_offset<<16),
                                  0x0101000000000014|(control<<16),
                                  0x0041000000000000,0x00810000001d0008)):
            struct.pack_into("<Q",data,start+(126+i)*8,word)
    for o in range(3):
        weights=q2.weights[o].reshape(c,kernel,kernel)
        for kh in range(kernel):
            for kw in range(kernel):
                row=list(weights[:,kh,kw])+[int(q2.weight_zero_points[o])]*(16-c)
                struct.pack_into("<16b",data,second_weights+((kh*kernel+kw)*3+o)*16,*row)
        struct.pack_into("<i",data,second_bias+o*4,int(q2.biases[o]))
        struct.pack_into("<h",data,second_bias+16+o*2,-int(q2.weight_zero_points[o]))
        struct.pack_into("<H",data,second_bias+24+o*2,int(q2.channel_multipliers[o]))
    return bytes(data),{"profile":2,"shape_nhwc":[1,8,8,3],"output_shape_nhwc":[1,8,8,3],
                       "register_count":126,"kernel_size":kernel,"relu":True,
                       "output_scale":q2.output_scale,"output_zero_point":q2.output_zero_point,
                       "hidden_channels":c,"first":meta,"second":q2.metadata(),
                       "weights2":w2.tolist(),"bias2":constants[b.input[2]].tolist()}
