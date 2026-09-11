"""MIT. Bounded public RV1103 Sigmoid/Tanh LUT profile."""
import math
import struct,tempfile
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from .compiler import compile_model
from .register_profile import REGISTERS
from .sequence import encode_sequence

LUT_DOMAIN=8.0  # The 1026-entry table spans argument (-8, 8) in steps of 1/64.
BASE_WEIGHT_SCALE=(1/32)/255  # Natural weight scale of the original verified identity stem.


def lut_reference(inputs,weights,bias,kind,input_scale=1.0,input_zero_point=128):
    """Exact integer reference for the sign-split table.

    The hardware advances the table one entry per 1/64 of the argument: the
    positive half uses the dequantized stem value `x`, while the negative half
    uses `x * (BASE_WEIGHT_SCALE/weight_scale)`. The table is built with the
    inverse of that negative gain, so both halves compose to `fn(x)`, and this
    function reproduces the resulting quantized bytes (including the 1/64 index
    grid) so expected bytes are exact by construction.
    """
    values=(np.asarray(inputs,np.float64)-input_zero_point)*input_scale
    kernel=np.asarray(weights,np.float64).reshape(3,3)
    stem=np.einsum('hwc,oc->hwo',values,kernel)+np.asarray(bias,np.float64)
    minimum=np.minimum(kernel.min(axis=1),0)
    maximum=np.maximum(kernel.max(axis=1),0)
    scales=(maximum-minimum)/np.float64(255)
    scale=float(scales.max()) or 1/32768
    hardware_gain=negative_half_gain(scale)
    table_gain=1.0/hardware_gain
    fn=(lambda x:1/(1+np.exp(-x))) if kind=='Sigmoid' else np.tanh
    argument=np.where(stem<0,stem*hardware_gain,stem)
    index=512+np.clip(np.rint(64*argument),-512,512)
    table_argument=(index-512)/64.0
    table_argument=np.where(index<512,table_argument*table_gain,table_argument)
    q15=np.clip(np.rint(fn(table_argument)*32768),-32768,32767)
    scale,zero=(255,-128) if kind=='Sigmoid' else (127,0)
    return (np.rint(q15*scale/32768)+zero).astype(np.int8)


def stem_range(weights,bias,input_scale=1.0,input_zero_point=128):
    """Analytic output interval of the 1x1 stem for input values 0..255."""
    lo=(0-input_zero_point)*input_scale
    hi=(255-input_zero_point)*input_scale
    w=np.asarray(weights,np.float64).reshape(3,3)
    lower=(np.minimum(w*lo,w*hi).sum(axis=1)+np.asarray(bias,np.float64)).min()
    upper=(np.maximum(w*lo,w*hi).sum(axis=1)+np.asarray(bias,np.float64)).max()
    return float(lower),float(upper)


def negative_half_gain(weight_scale):
    """Hardware negative-half gain for a stem weight scale.

    The board measures `H = 2**ceil(log2(BASE_WEIGHT_SCALE/weight_scale))`: the
    accelerator shifts by whole bits, so e.g. weights scaled for 0.02 (ratio
    1.5625) advance the negative table half twice as fast (H = 2), while 0.03
    (ratio 1.0417) also lands on H = 2. Measured for diagonal and mixed stems in
    `research/lut_mixed_gain_probe/` (see `research/lut_domain_suite/README.md`).
    """
    ratio=BASE_WEIGHT_SCALE/float(weight_scale)
    exponent=round(math.log2(ratio))
    if abs(ratio-2**exponent)<=1e-6*ratio:return 2.0**exponent
    return 2.0**math.ceil(math.log2(ratio)-1e-9)


def compile_lut(model,input_scale=1.,input_zero_point=128,declared_scale=1/2048,declared_zero_point=0,
                allow_mixed_stems=False,negative_gain_override=None):
    g=model.graph
    if len(g.node)!=2 or g.node[0].op_type!='Conv' or g.node[1].op_type not in ('Sigmoid','Tanh'):
        raise ValueError('LUT profile requires Conv followed by Sigmoid or Tanh')
    conv,activation=g.node;kind=activation.op_type
    constants={v.name:nh.to_array(v) for v in g.initializer}
    w=constants.get(conv.input[1] if len(conv.input)>1 else '')
    b=constants.get(conv.input[2] if len(conv.input)>2 else '')
    attrs={a.name:h.get_attribute_value(a) for a in conv.attribute}
    shape=lambda v:[d.dim_value for d in v.type.tensor_type.shape.dim]
    if (input_scale!=1 or input_zero_point!=128 or len(g.input)!=1 or len(g.output)!=1
        or conv.domain not in ('','ai.onnx') or activation.domain not in ('','ai.onnx')
        or activation.attribute or list(activation.input)!=list(conv.output)
        or list(activation.output)!=[g.output[0].name] or shape(g.input[0])!=[1,3,8,8]
        or shape(g.output[0])!=[1,3,8,8] or attrs not in ({'kernel_shape':[1,1]}, {})
        or w is None or b is None or w.dtype!=np.float32 or b.dtype!=np.float32
        or w.shape!=(3,3,1,1) or b.shape!=(3,)):
        raise ValueError('bounded LUT profile requires a C3 1x1 Conv stem, 8x8, input scale1 zero point128')
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('LUT stem weights and bias must be finite')
    lower,upper=stem_range(w,b)
    # The hardware's table argument splits by sign: the positive half advances one
    # entry per 1/64 of the dequantized value `x`, while the negative half advances
    # one entry per 1/64 of `x * (weight_scale/BASE_WEIGHT_SCALE)`. The table is
    # therefore sampled with that gain on the negative half so the composed
    # function is `fn(x)` for either sign (measured on the board, see
    # research/lut_domain_suite/README.md).
    weights=w.reshape(3,3)
    diagonal=np.diag(np.diag(weights))
    gain=float(weights[0,0])
    if not allow_mixed_stems and (not np.array_equal(weights,diagonal)
                                  or not np.allclose(np.abs(np.diag(weights)),abs(gain),rtol=0,atol=0)):
        # Mixed input weights are a retained blocker: the gain band is now known
        # (2**ceil(log2(BASE/scale))) but the hardware's argument grid still rounds
        # differently, leaving ±1 on ~12% of values (research/lut_mixed_gain_probe).
        raise ValueError('LUT stem must be a diagonal 1x1 Conv with one scalar magnitude per channel')
    minimum=np.minimum(weights.min(axis=1),0)
    maximum=np.maximum(weights.max(axis=1),0)
    scales=(maximum-minimum)/np.float32(255)
    max_scale=float(scales.max())
    if max_scale==0:max_scale=1/32768
    scales=np.where(scales==0,np.float32(max_scale),scales)
    gains=[negative_half_gain(float(scale)) for scale in scales]
    if len(set(gains))!=1:
        # The negative table half advances by the per-channel weight scale, so one
        # table can serve only stems whose channels share one gain band.
        raise ValueError('LUT stem channels must share one negative-half gain band; got %s'%gains)
    negative_gain=1.0/gains[0]
    # Only the exact band is supportable: when BASE/scale is not a power of two the
    # hardware's whole-bit gain is right but its argument grid still rounds
    # differently, leaving ±1 on ~10% of values (research/lut_mixed_gain_probe/).
    ratio=BASE_WEIGHT_SCALE/max_scale
    if negative_gain_override is None and abs(ratio-2**round(math.log2(ratio)))>1e-6*ratio:
        raise ValueError('LUT stem weight scale must make BASE_WEIGHT_SCALE/scale a power of two; '
                         'other bands advance the negative table half by whole bits and leave ±1 rounding')
    if negative_gain_override is not None:
        negative_gain=float(negative_gain_override)
    if upper>LUT_DOMAIN+1e-6 or lower< -LUT_DOMAIN*negative_gain-1e-6:
        raise ValueError('LUT stem output range [%.3f, %.3f] leaves the table domain [%.3f, %.3f]'
                         %(lower,upper,-LUT_DOMAIN*negative_gain,LUT_DOMAIN))
    sub=h.make_model(h.make_graph([conv],'lut_stem',list(g.input),
        [h.make_tensor_value_info(conv.output[0],1,[1,3,8,8])],list(g.initializer)),opset_imports=list(model.opset_import));sub.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'stem.onnx';onnx.save(sub,path)
        source,meta=compile_model(path,declared_scale,declared_zero_point,input_scale=1,input_zero_point=128)
    q=meta['quantization'];data=bytearray(0x2880)
    regs={v&65535:v>>16&0xffffffff for v in struct.unpack_from('<126Q',source)}
    data[0x2800:0x2840]=source[regs[0x1110]:regs[0x1110]+64]
    data[0x2840:0x2880]=source[regs[0x5020]:regs[0x5020]+64]
    setup={0x400c:0x1e5,0x4018:0,0x4020:0,0x4024:16,0x4030:0,0x4034:0,0x403c:0xf000f,
        0x4040:0x53,0x4048:0,0x4050:0x30000002,0x4054:0,0x4058:3,0x405c:0,0x4060:0x13,
        0x4070:0x10041c1,0x4080:0,0x4084:0x10001,0x4088:0,0x40c0:16,0x4108:1,
        0x500c:0,0x5010:0,0x5014:15,0x5018:0,0x501c:0,0x5020:0,0x5034:1,0x5038:0,0x5040:0,0x5044:0x907809}
    words=[tag<<48|setup.get(reg,default)<<16|reg for reg,default,tag in REGISTERS if reg>=0x4000]
    fn=(lambda x:1/(1+np.exp(-x))) if kind=='Sigmoid' else np.tanh
    for table in (0,1):
        words.append(0x1001<<48|(0x20000+table*0x10000)<<16|0x4100)
        arguments=(np.arange(513)+(table-1)*512)/64
        if table==0:arguments=arguments*negative_gain
        values=np.clip(np.rint(fn(arguments)*32768),-32768,32767).astype(np.int64)
        words.extend(0x1001<<48|(int(v)&65535)<<16|0x4104 for v in values)
    if len(words)!=1106:raise RuntimeError('unexpected LUT setup length')
    for i,word in enumerate(words):struct.pack_into('<Q',data,i*8,word)
    regs.update({0x1004:0x30,0x3004:0x30,0x4004:0x30,0x5004:0x30,0x1070:0x3000,
        0x1110:0x2800,0x5020:0x2840,0x4010:0x804000,0x4020:0x4000,0x4060:0x20000,
        0x4068:q['multiplier']<<16|q['shift']<<8,0x4070:0x1004140,
        0x4080:0xffffff80 if kind=='Sigmoid' else 0,0x4084:255 if kind=='Sigmoid' else 127,
        0x4088:15,0x4108:0x68,0x410c:0x50500,0x4110:0xffffc000,0x4114:0,0x4118:0,0x411c:0x4000})
    for i,(reg,default,tag) in enumerate(REGISTERS):struct.pack_into('<Q',data,0x22c0+i*8,tag<<48|regs.get(reg,default)<<16|reg)
    for base,count,enable,nxt,ctrl in ((0,1106,24,0x22c0,0x40),(0x22c0,126,29,0,0x28)):
        for i,(reg,value,tag) in enumerate(((0x10,nxt,0x101),(0x14,ctrl,0x101),(0,0,0x41),(8,enable,0x81))):
            struct.pack_into('<Q',data,base+(count+i)*8,tag<<48|value<<16|reg)
    outscale,outzp=((1/255,-128) if kind=='Sigmoid' else (1/127,0))
    binary=encode_sequence(data,input_shape=(8,8,3),output_shape=(8,8,3),input_stride=16,
        arena_bytes=24576,input_offset=0x3000,output_offset=0x4000,
        tasks=[(0,1106,24,768),(0x22c0,126,29,768)],input_scale=1,input_zero_point=128,
        output_scale=outscale,output_zero_point=outzp,serial=False)
    return binary,dict(lut_profile=kind.lower()+'-8x8-c3',kind=kind,stem_quantization=q,
        stem_range=[lower,upper],negative_gain=negative_gain,max_weight_scale=max_scale,
        declared_scale=declared_scale,declared_zero_point=declared_zero_point,
        output_scale=outscale,output_zero_point=outzp,
        stem_weights=np.asarray(w,np.float32).reshape(3,3).tolist(),
        stem_bias=[float(v) for v in np.asarray(b,np.float32)])
