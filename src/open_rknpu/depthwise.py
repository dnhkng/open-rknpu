"""MIT. Independent fixed 8x8, three-channel native depthwise 1/3/5 lowering.

No captured payloads or vendor compiler/runtime are read by this module.
"""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .chain import NATIVE, native_quantize
from .compiler import compile_model
from .register_profile import REGISTERS
from .sequence import encode_sequence,decode_sequence


def depthwise_reference(inputs,q,input_zero_point):
    """Per-channel scalar spatial convolution followed by integer requantization."""
    height,width,channels=inputs.shape;kernel=q.kernel_size;pad=kernel//2
    padded=np.pad(np.asarray(inputs,np.int64),((pad,pad),(pad,pad),(0,0)),constant_values=input_zero_point)
    acc=np.broadcast_to(q.biases,(height,width,channels)).copy()
    weights=q.weights.reshape(channels,kernel,kernel)-q.weight_zero_points[:,None,None]
    for y in range(kernel):
        for x in range(kernel):
            acc+=padded[y:y+height,x:x+width,:]*weights[:,y,x]
    product=acc*q.channel_multipliers
    scaled=(product+8191+((product>>14)&1))>>14
    result=((scaled*q.multiplier+(1<<(q.shift-1) if q.shift else 0))>>q.shift)+q.output_zero_point
    return np.clip(result,-128,127).astype(np.int8)


def compile_depthwise(model,input_scale=1.0,input_zero_point=0,output_range=None,asymmetric_pair=False):
    """Conv[/Relu] -> depthwise Conv; exact supported profile or a clear error."""
    g=model.graph;nodes=list(g.node)
    first_count=2 if len(nodes)>1 and nodes[1].op_type=='Relu' else 1
    if (len(g.input)!=1 or len(g.output)!=1 or len(nodes)!=first_count+1
        or nodes[0].op_type!='Conv' or nodes[-1].op_type!='Conv'):
        raise ValueError('depthwise sequence requires Conv[/Relu] -> depthwise Conv')
    dw=nodes[-1];prev=nodes[first_count-1].output[0]
    constants={t.name:nh.to_array(t) for t in g.initializer}
    attrs={a.name:h.get_attribute_value(a) for a in dw.attribute}
    channels=attrs.get('group',3)
    if not 1<=channels<=16:raise ValueError('depthwise supports 1..16 channels')
    strides=attrs.get('strides',[1,1])
    if strides not in ([1,1],[2,2]):raise ValueError('depthwise supports stride 1 or 2')
    stride=strides[0]
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if len(shape)!=4 or shape[:2]!=[1,3] or not all(5<=v<=8 for v in shape[2:]):raise ValueError('depthwise input must be RGB, H/W 5..8')
    height,width=shape[2:];oh=(height+stride-1)//stride;ow=(width+stride-1)//stride
    kernel=attrs.get('kernel_shape',[3,3])[0]
    if kernel not in (1,3,5):raise ValueError('depthwise kernels 1/3/5 only')
    pad=kernel//2
    allowed={'kernel_shape':[kernel,kernel],'pads':[pad]*4,'strides':strides,'dilations':[1,1],'group':channels}
    if (dw.domain not in ('','ai.onnx') or len(dw.input) not in (2,3) or dw.input[0]!=prev
        or list(dw.output)!=[g.output[0].name] or attrs.get('group')!=channels
        or attrs.get('pads',[0]*4)!=[pad]*4 or any(k not in allowed or v!=allowed[k] for k,v in attrs.items())
        or any(n not in constants for n in dw.input[1:])):
        raise ValueError('depthwise profile requires supported group/kernel/padding/stride and constant weights/bias')
    w=constants[dw.input[1]];b=constants[dw.input[2]] if len(dw.input)==3 else np.zeros(channels,np.float32)
    if w.shape!=(channels,1,kernel,kernel) or b.shape!=(channels,) or w.dtype!=np.float32 or b.dtype!=np.float32:
        raise ValueError('depthwise profile requires matching float32 weights [C,1,K,K] and bias [C]')
    for v,hw,c in [(g.input[0],(height,width),3),(g.output[0],(oh,ow),channels)]:
        if v.type.tensor_type.elem_type!=1 or [d.dim_value for d in v.type.tensor_type.shape.dim]!=[1,c,*hw]:
            raise ValueError('depthwise profile requires matching static input/output shapes')
    values={v.name:v for v in [*g.input,*g.value_info,*g.output]}
    if prev not in values or [d.dim_value for d in values[prev].type.tensor_type.shape.dim]!=[1,channels,height,width]:
        raise ValueError('depthwise input must match group count at 8x8')
    sub=h.make_model(h.make_graph(nodes[:first_count],'stem',list(g.input),[values[prev]],list(g.initializer)),opset_imports=list(model.opset_import))
    sub.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'stem.onnx';onnx.save(sub,path)
        source,meta=compile_model(path,input_scale=input_scale,input_zero_point=input_zero_point)
    if meta['kernel_size'] not in (1,3):
        raise ValueError('depthwise profile supports only 1x1 or 3x3 stem convolution')
    if channels>4 and meta['kernel_size']!=1:raise ValueError('C5..16 requires a 1x1 stem')
    # The symmetric pair (centered weight, 0) is the verified default. The
    # optional asymmetric pair stores (offset code, per-channel zero point), which
    # the depthwise weight pair's second byte appears to carry.
    q=native_quantize(w,b,meta['output_scale'],meta['output_zero_point'],output_range,symmetric=not asymmetric_pair)
    data=bytearray(4096)
    first={v&65535:(v>>16)&0xffffffff for v in struct.unpack_from('<126Q',source)}
    weight_size=(first[0x1030]+63)//64*64
    if weight_size>512:raise ValueError('stem weights exceed depthwise profile allocation')
    data[0x880:0x880+weight_size]=source[first[0x1110]:first[0x1110]+weight_size]
    data[0xa80:0xb00]=source[first[0x5020]:first[0x5020]+128]
    first.update({0x4024:((height*width+3)//4)*64,0x40c0:((height*width+3)//4)*64,0x1110:0x880,0x5020:0xa80,0x1070:0x1000,0x4020:0x2000})
    second=dict(first);second.update(NATIVE)
    second.update({0x100c:7,0x1010:0x108,0x1024:(channels-1)<<16|16,0x1030:288,0x1034:144,
        0x1038:0x3030002,0x1068:0x101,0x1070:0x2000,0x1078:0x400f400f,
        0x1110:0xb00,0x1184:meta['output_zero_point']&65535,0x1188:72,
        0x3010:10,0x3018:31,0x400c:0x1fc,0x4020:0x3000,0x403c:(channels-1)<<16|31,
        0x4040:0x101c92,0x4048:0,0x4050:0x70000125,
        0x4054:0xe000000,
        0x4058:0x10007,0x4060:0x13,0x406c:0x80000000,0x40e0:0x80000000,
        0x4080:q.output_zero_point&0xffffffff,0x4084:q.multiplier,0x4088:q.shift,
        0x40c0:2048,0x5014:31,0x501c:10,0x5020:0xc40,0x5044:0x7816})
    second.update({0x1014:stride*9,0x1028:ow,0x102c:oh*ow,
        0x103c:((width+1)//2)<<16,0x1044:width<<16|((width+1)//2),
        0x107c:width*((height+3)//4*2),0x1080:((height*width+3)//4)*4,0x1084:width<<16|height,
        0x118c:(width-1)<<16|(width-1),
        0x3014:(oh-1)<<16|(ow-1),0x4024:((oh*ow+3)//4)*64,
        0x4030:ow-1,0x4034:oh-1,0x405c:(oh-1)<<16|(ow-1),
        0x40c0:((oh*ow+3)//4)*128,0x500c:ow-1,0x5010:oh-1,
        0x1030:kernel*kernel*32,0x1034:kernel*kernel*16,
        0x1038:kernel<<24|kernel<<16|2,0x1068:pad*0x101,
        0x1188:kernel*kernel*8,0x5020:0xe40 if kernel==5 else 0xc40})
    for base,fields in [(0,first),(0x440,second)]:
        for i,(reg,default,tag) in enumerate(REGISTERS):
            struct.pack_into('<Q',data,base+i*8,tag<<48|fields.get(reg,default)<<16|reg)
        for i,(reg,val,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,29,0x81))):
            struct.pack_into('<Q',data,base+(126+i)*8,tag<<48|val<<16|reg)
    for y in range(kernel):
        for x in range(kernel):
            for c in range(channels):
                # The pair's second byte is the negated weight zero point, as in
                # the dense Conv channel table; stored as a raw byte so -(-128) fits.
                pair_zero_point=((-int(q.weight_zero_points[c])) & 0xff) if asymmetric_pair else 0
                struct.pack_into('<bB',data,0xb00+(y*kernel+x)*32+c*2,int(q.weights[c,y*kernel+x]),pair_zero_point)
    for c in range(channels):
        struct.pack_into('<i',data,second[0x5020]+(c//4)*24+(c%4)*4,int(q.biases[c]))
        struct.pack_into('<H',data,second[0x5020]+(c//4)*24+16+(c%4)*2,int(q.channel_multipliers[c]))
    binary=encode_sequence(data,input_shape=(height,width,3),output_shape=(oh,ow,channels),input_stride=16,
        arena_bytes=16384,input_offset=0x1000,output_offset=0x3000,
        tasks=[(0,126,29,768),(0x440,126,29,768)],input_scale=input_scale,
        input_zero_point=input_zero_point,output_scale=q.output_scale,output_zero_point=q.output_zero_point,serial=True)
    return binary,dict(meta,output_scale=q.output_scale,output_zero_point=q.output_zero_point,
        output_shape_nhwc=[1,oh,ow,channels],depthwise=q.metadata(),first=meta,
        depthwise_asymmetric_pair=bool(asymmetric_pair),
        depthwise_profile=f'{height}x{width}-c{channels}-k{kernel}-s{stride}-pad{pad}')


def compile_depthwise_pointwise(model,input_scale=1.0,input_zero_point=0,output_range=None):
    """Compose the verified chained depthwise task with a dense 1x1 successor."""
    g=model.graph;nodes=list(g.node);first_count=2 if len(nodes)>1 and nodes[1].op_type=='Relu' else 1
    if len(nodes)!=first_count+2 or nodes[-2].op_type!='Conv' or nodes[-1].op_type!='Conv':
        raise ValueError('depthwise-pointwise requires Conv[/Relu] -> depthwise Conv -> pointwise Conv')
    dw,pw=nodes[-2:];attrs={a.name:h.get_attribute_value(a) for a in pw.attribute};constants={t.name:nh.to_array(t) for t in g.initializer}
    w=constants.get(pw.input[1] if len(pw.input)>1 else '');oc=w.shape[0] if w is not None and w.ndim==4 else 0
    values={v.name:v for v in [*g.input,*g.value_info,*g.output]};mid=dw.output[0]
    if (pw.input[0]!=mid or list(pw.output)!=[g.output[0].name] or len(pw.input) not in (2,3)
        or w is None or w.dtype!=np.float32 or w.ndim!=4 or w.shape[2:]!=(1,1)
        or not 1<=w.shape[1]<=16 or not 1<=oc<=128
        or attrs.get('group',1)!=1 or any(k not in {'kernel_shape','group','pads','strides','dilations'} for k in attrs)
        or attrs.get('kernel_shape',[1,1])!=[1,1] or attrs.get('pads',[0]*4)!=[0]*4
        or attrs.get('strides',[1,1])!=[1,1] or attrs.get('dilations',[1,1])!=[1,1]
        or mid not in values):raise ValueError('bounded pointwise successor requires dense 1x1 constant weights')
    sub=h.make_model(h.make_graph(nodes[:-1],'depthwise_prefix',list(g.input),[values[mid]],list(g.initializer),
        value_info=[values[nodes[first_count-1].output[0]]]),opset_imports=list(model.opset_import));sub.ir_version=model.ir_version
    prefix,pmeta=compile_depthwise(sub,input_scale,input_zero_point)
    midshape=[d.dim_value for d in values[mid].type.tensor_type.shape.dim];oshape=[d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    if midshape[0]!=1 or oshape!=[1,oc,*midshape[2:]]:raise ValueError('pointwise successor shape mismatch')
    bias=constants[pw.input[2]] if len(pw.input)==3 else np.zeros(oc,np.float32)
    point=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[1,1])],'pointwise',
        [h.make_tensor_value_info('input',1,midshape)],[h.make_tensor_value_info('output',1,oshape)],
        [nh.from_array(w,'w'),nh.from_array(bias,'b')]),opset_imports=[h.make_opsetid('',13)]);point.ir_version=min(model.ir_version,8)
    from .native import compile_native_input
    successor,smeta=compile_native_input(point,pmeta['output_scale'],pmeta['output_zero_point']+128,output_range)
    pi=decode_sequence(prefix);si=decode_sequence(successor);pdata=bytearray(prefix[96+16*pi['task_count']:]);sdata=successor[96+16*si['task_count']:]
    sregs={v&65535:v>>16&0xffffffff for v in struct.unpack_from('<126Q',sdata)}
    align=lambda x,a=64:(x+a-1)//a*a
    command=align(len(pdata));weight_size=align(sregs[0x1030]);weights=command+align(130*8);bias_offset=weights+weight_size;bias_size=align((oc+3)//4*32);payload=align(bias_offset+bias_size)
    pdata.extend(bytes(payload-len(pdata)));pdata[weights:weights+weight_size]=sdata[sregs[0x1110]:sregs[0x1110]+weight_size];pdata[bias_offset:bias_offset+bias_size]=sdata[sregs[0x5020]:sregs[0x5020]+bias_size]
    new_input=align(payload,4096);delta=new_input-pi['input_offset'];address_regs={0x1070,0x4020,0x5018,0x5038}
    for task in pi['tasks']:
        for j in range(task['register_count']):
            off=task['command_offset']+j*8;word=struct.unpack_from('<Q',pdata,off)[0];reg=word&65535;value=word>>16&0xffffffff
            if reg in address_regs and pi['input_offset']<=value<pi['arena_bytes']:value+=delta
            struct.pack_into('<Q',pdata,off,(word>>48)<<48|value<<16|reg)
    mid_offset=pi['output_offset']+delta;out=align(pi['arena_bytes']+delta,64);out_storage=align(oshape[2]*oshape[3]*16)*((oc+15)//16)
    sregs.update({0x1070:mid_offset,0x4020:out,0x1110:weights,0x5020:bias_offset})
    for j,(reg,default,tag) in enumerate(REGISTERS):struct.pack_into('<Q',pdata,command+j*8,tag<<48|sregs.get(reg,default)<<16|reg)
    for j,(reg,value,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,29,0x81))):struct.pack_into('<Q',pdata,command+(126+j)*8,tag<<48|value<<16|reg)
    tasks=[(t['command_offset'],t['register_count'],t['enable'],t['mask']) for t in pi['tasks']]+[(command,126,29,768)]
    result=encode_sequence(pdata,input_shape=tuple(pi['shape_nhwc'][1:]),output_shape=tuple(si['output_shape_nhwc'][1:]),input_stride=pi['input_stride'],
        arena_bytes=align(out+out_storage,4096),input_offset=new_input,output_offset=out,tasks=tasks,input_scale=input_scale,input_zero_point=input_zero_point,
        output_scale=si['output_scale'],output_zero_point=si['output_zero_point'],serial=True,input_layout=pi['input_layout'])
    return result,dict(pmeta,output_scale=si['output_scale'],output_zero_point=si['output_zero_point'],output_shape_nhwc=si['output_shape_nhwc'],pointwise=smeta['quantization'],depthwise_pointwise=True)
