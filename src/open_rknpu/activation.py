"""MIT. Fused LeakyRelu after one dense Conv, with separate signed conversion paths."""
from pathlib import Path
import copy,struct,tempfile
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from .compiler import compile_model
from .sequence import encode_sequence
from .quantization import receptive_fields


def leaky_reference(inputs,q,alpha):
    acc=np.einsum('...c,oc->...o',receptive_fields(np.asarray(inputs,np.int64)-128,q.kernel_size,q.input_zero_point-128),q.weights-q.weight_zero_points[:,None])+q.biases
    product=acc*q.channel_multipliers
    negative=(product+8191+((product>>14)&1))>>14
    value=np.where(acc<0,negative*round(alpha*16384),product)*q.multiplier
    shift=q.shift+14
    value=(value+(1<<(shift-1))-1+((value>>shift)&1))>>shift
    return np.clip(value+q.output_zero_point,-128,127).astype(np.int8)


def prelu_reference(inputs,q,slopes):
    acc=np.einsum('...c,oc->...o',receptive_fields(np.asarray(inputs,np.int64)-128,q.kernel_size,q.input_zero_point-128),q.weights-q.weight_zero_points[:,None])+q.biases
    product=acc*q.channel_multipliers;negative=(product+8191+((product>>14)&1))>>14
    value=np.where(acc<0,negative*np.asarray(slopes,np.int64),product)*q.multiplier;shift=q.shift+14
    value=(value+(1<<(shift-1))-1+((value>>shift)&1))>>shift
    return np.clip(value+q.output_zero_point,-128,127).astype(np.int8)


def compile_leaky(model,input_scale=1.0,input_zero_point=0):
    g=model.graph
    if len(g.node)!=2 or len(g.input)!=1 or len(g.output)!=1 or g.node[0].op_type!='Conv':
        raise ValueError('LeakyRelu profile requires one Conv followed by LeakyRelu')
    conv,act=g.node;attrs={a.name:h.get_attribute_value(a) for a in act.attribute};alpha=attrs.get('alpha',.01)
    if (act.domain not in ('','ai.onnx') or act.op_type!='LeakyRelu' or any(k!='alpha' for k in attrs)
        or list(act.input)!=list(conv.output) or list(act.output)!=[g.output[0].name]
        or not np.isfinite(alpha) or not 0<=alpha<=1):raise ValueError('unsupported LeakyRelu parameters/connections')
    lower=copy.deepcopy(model);lower.graph.node[0].output[0]=act.output[0];del lower.graph.node[-1];del lower.graph.value_info[:]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'conv.onnx';onnx.save(lower,p)
        source,meta=compile_model(p,input_scale=input_scale,input_zero_point=input_zero_point)
    q=meta['quantization'];weights=np.array(q['weights'],np.int64)-np.array(q['weight_zero_points'],np.int64)[:,None]
    upper=np.array(q['biases'])+np.maximum(-128*weights,127*weights).sum(axis=1)
    if np.any(np.maximum(upper,0)*np.array(q['channel_multipliers'])>2**31-1):
        raise ValueError('LeakyRelu positive conversion may overflow INT32; smaller-scale lowering required')
    data=bytearray(source);fields={0x4060:0x22,0x4068:round(alpha*16384)<<16,0x4088:meta['quantization']['shift']+14}
    for j in range(126):
        off=j*8;word=struct.unpack_from('<Q',data,off)[0];r=word&65535;old=word>>16&0xffffffff
        if r==0x4040:fields[r]=old&~0x1c00
        if r in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[r]<<16|r)
    _,height,width,ic=meta['shape_nhwc'];oc=meta['output_shape_nhwc'][3]
    result=encode_sequence(data,input_shape=(height,width,ic),output_shape=(height,width,oc),input_stride=meta['input_stride'],
        arena_bytes=meta['arena_bytes'],input_offset=0x2000,output_offset=0x3000,tasks=[(0,126,29,768)],
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=meta['output_scale'],output_zero_point=meta['output_zero_point'],serial=True)
    return result,dict(meta,leaky_alpha=alpha,leaky_alpha_encoded=round(alpha*16384)/16384)


def compile_prelu(model,input_scale=1.0,input_zero_point=0):
    g=model.graph
    if len(g.node)!=2 or len(g.input)!=1 or len(g.output)!=1 or g.node[0].op_type!='Conv':raise ValueError('PRelu profile requires one Conv followed by PRelu')
    conv,act=g.node;constants={v.name:nh.to_array(v) for v in g.initializer};slope=constants.get(act.input[1]) if len(act.input)==2 else None
    if (act.domain not in ('','ai.onnx') or act.op_type!='PRelu' or act.attribute or act.input[0]!=conv.output[0]
        or list(act.output)!=[g.output[0].name] or slope is None or slope.dtype!=np.float32 or not np.isfinite(slope).all() or np.any((slope<0)|(slope>1))):
        raise ValueError('PRelu requires constant float32 slopes in [0,1]')
    lower=copy.deepcopy(model);lower.graph.node[0].output[0]=act.output[0];del lower.graph.node[-1];del lower.graph.value_info[:]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'conv.onnx';onnx.save(lower,p);source,meta=compile_model(p,input_scale=input_scale,input_zero_point=input_zero_point)
    oc=meta['output_shape_nhwc'][3]
    if slope.shape in ((),(1,)):values=np.full(oc,float(slope.reshape(-1)[0]),np.float32)
    elif slope.shape==(oc,1,1):values=slope[:,0,0]
    else:raise ValueError('PRelu slope must be scalar or [C,1,1]')
    encoded=np.rint(values*16384).astype(np.int64);q=meta['quantization'];weights=np.array(q['weights'],np.int64)-np.array(q['weight_zero_points'],np.int64)[:,None];upper=np.array(q['biases'])+np.maximum(-128*weights,127*weights).sum(axis=1)
    if np.any(np.maximum(upper,0)*np.array(q['channel_multipliers'])>2**31-1):raise ValueError('PRelu positive conversion may overflow INT32; smaller-scale lowering required')
    data=bytearray(source);table=0x500;size=((oc*2+7)//8)*8;data[table:table+size]=bytes(size)
    for channel,value in enumerate(encoded):struct.pack_into('<H',data,table+channel*2,int(value))
    fields={0x4060:0x22,0x4068:1,0x4088:q['shift']+14,0x5028:8,0x502c:table}
    for j in range(126):
        off=j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535;old=word>>16&0xffffffff
        if reg==0x4040:fields[reg]=old&~0x1c00
        if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    _,height,width,ic=meta['shape_nhwc'];result=encode_sequence(data,input_shape=(height,width,ic),output_shape=(height,width,oc),input_stride=meta['input_stride'],arena_bytes=meta['arena_bytes'],input_offset=0x2000,output_offset=0x3000,tasks=[(0,126,29,768)],input_scale=input_scale,input_zero_point=input_zero_point,output_scale=meta['output_scale'],output_zero_point=meta['output_zero_point'],serial=True)
    return result,dict(meta,prelu_slopes=values.tolist(),prelu_slopes_q14=encoded.tolist())
