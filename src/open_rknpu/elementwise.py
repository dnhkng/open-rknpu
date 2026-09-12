"""MIT. Independent 8x8/C3 Add/Mul/Sub/Max between two Conv branches, without broadcast."""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from .compiler import compile_model
from .register_profile import REGISTERS
from .sequence import (encode_sequence, decode_sequence, encode_sequence_v5,
                       LAYOUT_PACKED_U8, LAYOUT_NATIVE16, ROLE_INPUT, ROLE_OUTPUT, ROLE_INTERNAL)
import copy
import math

# Names/positions suggested by Mesa Rocket and RK3588 TRM chapter 36, then
# checked against RV1103 captures and independently generated board probes.
# Only these fields are promoted to named semantics; the remaining profile bits
# are RV1103-specific values, not a wholesale import of the RK3588 register map.
DPU_EW_CFG = 0x4070
EW_ALU_ALGO_SHIFT = 16
EW_ALU_ADD = 2
DPU_OUT_CVT_SHIFT = 0x4088
OUT_CVT_ROUND_AWAY = 1 << 30  # In this Add path: 0 ties-even, 1 ties-away-zero.


def add_reference(a,b):
    """Equal-scale, zero-point-zero inputs; output scale is twice input scale."""
    return np.rint((a.astype(np.int32)+b.astype(np.int32))/2).astype(np.int8)


def sub_reference(a,b):
    return np.clip(np.rint((a.astype(np.int32)-b.astype(np.int32))/2),-128,127).astype(np.int8)


def max_reference(a,b):
    return np.rint(np.maximum(a,b).astype(np.int32)/2).astype(np.int8)


def mul_reference(a,b):
    """Zero-point-zero inputs; output scale is 128 times their scale product."""
    return np.clip(np.rint(a.astype(np.int32)*b.astype(np.int32)/128),-128,127).astype(np.int8)


def mul_output_conversion(product_scale,output_range=None):
    output_scale=float(np.float32(128*product_scale));output_zero_point=0
    if output_range is not None:
        output_scale=float(output_range['scale']);output_zero_point=int(output_range['zero_point'])
        if not math.isfinite(output_scale) or output_scale<=0 or not -128<=output_zero_point<=127:
            raise ValueError('invalid Mul output quantization')
    factor=product_scale/output_scale;shift=min(31,math.floor(math.log2(32767/factor)))
    multiplier=round(factor*2**shift)
    if not 1<=multiplier<=32767 or shift<0:raise ValueError('Mul output scale outside conversion range')
    return output_scale,output_zero_point,0x10000|multiplier,shift


def mul_requant_reference(a,b,multiplier,shift,zero_point,operand_zero_points=(0,0)):
    product=(a.astype(np.int64)-operand_zero_points[0])*(b.astype(np.int64)-operand_zero_points[1])*(multiplier&0xffff)
    if shift:
        product+=zero_point<<shift
        product=(product+(1<<(shift-1))-1+((product>>shift)&1))>>shift
    else:product+=zero_point
    return np.clip(product,-128,127).astype(np.int8)


def compile_mul_relu(model,input_scale=1.0,input_zero_point=0,output_range=None,operand_zero_points=(0,0)):
    """Fuse terminal Relu into the verified Mul DPU task."""
    g=model.graph;nodes=list(g.node);relu=nodes[-1] if nodes else None
    if (relu is None or relu.op_type!='Relu' or relu.domain not in ('','ai.onnx') or relu.attribute
        or len(relu.input)!=1 or len(relu.output)!=1 or nodes[-2].op_type!='Mul'
        or relu.input[0]!=nodes[-2].output[0] or list(relu.output)!=[g.output[0].name]):
        raise ValueError('Mul-to-Relu profile requires a terminal Relu')
    lower=copy.deepcopy(model);product=lower.graph.node[-2].output[0];del lower.graph.node[-1];lower.graph.output[0].name=product
    if len(lower.graph.node)==1:binary,meta=compile_standalone_mul(lower,input_scale,input_zero_point,output_range,operand_zero_points)
    else:binary,meta=compile_elementwise(lower,input_scale,input_zero_point,output_range,operand_zero_points)
    info=decode_sequence(binary);header=96+16*info['task_count'];data=bytearray(binary[header:]);task=info['tasks'][-1]
    for j in range(task['register_count']):
        off=task['command_offset']+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
        if reg in (0x407c,0x40e8):struct.pack_into('<Q',data,off,(word>>48)<<48|reg)
    tasks=[(t['command_offset'],t['register_count'],t['enable'],t['mask']) for t in info['tasks']];shape=info['shape_nhwc'];oshape=info['output_shape_nhwc']
    result=encode_sequence(data,input_shape=tuple(shape[1:]),output_shape=tuple(oshape[1:]),input_stride=info['input_stride'],arena_bytes=info['arena_bytes'],input_offset=info['input_offset'],output_offset=info['output_offset'],tasks=tasks,input_scale=info['input_scale'],input_zero_point=info['input_zero_point'],output_scale=info['output_scale'],output_zero_point=info['output_zero_point'],serial=info['serial'],input_layout=info['input_layout'],batch=info['batch'],input_tensor_count=info['input_tensor_count'])
    meta['fused_activation']='Relu after Mul';return result,meta


def compile_mul_clip(model,input_scale=1.0,input_zero_point=0,output_range=None,operand_zero_points=(0,0)):
    """Append the captured-and-rebuilt INT8 clamp task after Mul."""
    g=model.graph;nodes=list(g.node);activation=nodes[-1] if nodes else None;constants={t.name:nh.to_array(t) for t in g.initializer}
    if (activation is None or activation.op_type!='Clip' or activation.domain not in ('','ai.onnx') or activation.attribute
        or len(activation.input)!=3 or len(activation.output)!=1 or nodes[-2].op_type!='Mul'
        or activation.input[0]!=nodes[-2].output[0] or list(activation.output)!=[g.output[0].name]
        or any(name not in constants or np.asarray(constants[name]).size!=1 for name in activation.input[1:])
        or float(np.asarray(constants[activation.input[1]]).reshape(-1)[0])!=0.0
        or float(np.asarray(constants[activation.input[2]]).reshape(-1)[0])!=6.0):
        raise ValueError('Mul-to-Clip profile requires terminal constant Clip[0,6]')
    clip_range={'scale':float(np.float32(6/255)),'zero_point':-128}
    if output_range is not None and (int(output_range.get('zero_point',999))!=-128 or not np.isclose(float(output_range.get('scale',0)),clip_range['scale'])):
        raise ValueError('Mul Clip[0,6] currently uses output scale 6/255 and zero point -128')
    output_range=clip_range
    lower=copy.deepcopy(model);product=lower.graph.node[-2].output[0];del lower.graph.node[-1];lower.graph.output[0].name=product
    if len(lower.graph.node)==1:binary,meta=compile_standalone_mul(lower,input_scale,input_zero_point,output_range,operand_zero_points)
    else:binary,meta=compile_elementwise(lower,input_scale,input_zero_point,output_range,operand_zero_points)
    info=decode_sequence(binary)
    if info['batch']!=1 or info.get('constant_count'):raise ValueError('Mul Clip currently requires batch1 without mutable descriptors')
    header=96+16*info['task_count'];data=bytearray(binary[header:]);align=lambda n,a=64:(n+a-1)//a*a
    command=align(len(data));payload=align(command+(78+4)*8);data.extend(bytes(payload-len(data)))
    new_input=align(payload,4096);delta=new_input-info['input_offset'];address_regs={0x1070,0x4020,0x5018,0x5038}
    for task in info['tasks']:
        for j in range(task['register_count']):
            off=task['command_offset']+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535;value=word>>16&0xffffffff
            if reg in address_regs and info['input_offset']<=value<info['arena_bytes']:value+=delta
            struct.pack_into('<Q',data,off,(word>>48)<<48|value<<16|reg)
    _,height,width,channels=info['output_shape_nhwc'];surface=align(height*width*16);intermediate=info['output_offset']+delta;final=align(info['arena_bytes']+delta,64)
    lo=info['output_zero_point'];hi=min(127,round(6/info['output_scale'])+lo)
    fields={0x400c:0x1e5,0x4010:0x800000,0x4018:lo&255,0x401c:(hi+128)&255,0x4020:final,0x4024:surface,
        0x4030:width-1,0x4034:height-1,0x403c:0xf000f,0x4040:0x100012,0x4048:0,0x404c:0,
        0x4050:0x30000002,0x4054:0,0x4058:3,0x405c:(height-1)<<16|(width-1),0x4070:0x10041c1,0x4078:1,
        0x4080:lo&0xffffffff,0x4084:0x10001,0x4088:0,0x40c0:surface,0x40d8:0,0x40dc:(hi+128)&255,
        0x500c:width-1,0x5010:height-1,0x5014:15,0x5018:intermediate,0x501c:0,0x5020:0,0x5034:1,0x5044:0x907809}
    layout=[r for r in REGISTERS if r[0]>=0x4000]
    for j,(reg,default,tag) in enumerate(layout):struct.pack_into('<Q',data,command+j*8,tag<<48|fields.get(reg,default)<<16|reg)
    for j,(reg,value,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,24,0x81))):struct.pack_into('<Q',data,command+(78+j)*8,tag<<48|value<<16|reg)
    tasks=[(t['command_offset'],t['register_count'],t['enable'],t['mask']) for t in info['tasks']]+[(command,78,24,768)];shape=info['shape_nhwc'];oshape=info['output_shape_nhwc']
    result=encode_sequence(data,input_shape=tuple(shape[1:]),output_shape=tuple(oshape[1:]),input_stride=info['input_stride'],arena_bytes=align(final+surface,4096),input_offset=new_input,output_offset=final,tasks=tasks,input_scale=info['input_scale'],input_zero_point=info['input_zero_point'],output_scale=info['output_scale'],output_zero_point=info['output_zero_point'],serial=True,input_layout=info['input_layout'],input_tensor_count=info['input_tensor_count'])
    meta['fused_activation']='Clip[0,6] after Mul via clamp task';return result,meta


def compile_mul_add(model,input_scale=1.0,input_zero_point=0,output_range=None,operand_zero_points=(0,0)):
    """Fold a representable scalar Add after Mul into its output zero point."""
    g=model.graph;nodes=list(g.node);add=nodes[-1] if nodes else None;constants={t.name:nh.to_array(t) for t in g.initializer}
    if (add is None or add.op_type!='Add' or add.domain not in ('','ai.onnx') or add.attribute or nodes[-2].op_type!='Mul'
        or len(add.input)!=2 or len(add.output)!=1 or list(add.output)!=[g.output[0].name] or nodes[-2].output[0] not in add.input):
        raise ValueError('Mul-to-Add profile requires a terminal scalar constant Add')
    name=add.input[1] if add.input[0]==nodes[-2].output[0] else add.input[0];value=constants.get(name)
    if value is None or value.dtype!=np.float32 or value.size!=1 or not np.isfinite(value).all() or output_range is None:
        raise ValueError('Mul-to-Add requires a finite float32 scalar and explicit output quantization')
    scale=float(output_range['scale']);zero_point=int(output_range['zero_point']);step=round(float(value.reshape(()))/scale)
    if not np.isclose(float(value.reshape(())),step*scale,rtol=0,atol=max(1e-7,abs(scale)*1e-6)) or not -128<=zero_point+step<=127:
        raise ValueError('scalar Add must be exactly representable without overflowing the internal zero point')
    lower=copy.deepcopy(model);product=lower.graph.node[-2].output[0];del lower.graph.node[-1];lower.graph.output[0].name=product
    internal={'scale':scale,'zero_point':zero_point+step}
    if len(lower.graph.node)==1:binary,meta=compile_standalone_mul(lower,input_scale,input_zero_point,internal,operand_zero_points)
    else:binary,meta=compile_elementwise(lower,input_scale,input_zero_point,internal,operand_zero_points)
    info=decode_sequence(binary);header=96+16*info['task_count'];data=binary[header:];tasks=[(t['command_offset'],t['register_count'],t['enable'],t['mask']) for t in info['tasks']];shape=info['shape_nhwc'];oshape=info['output_shape_nhwc']
    result=encode_sequence(data,input_shape=tuple(shape[1:]),output_shape=tuple(oshape[1:]),input_stride=info['input_stride'],arena_bytes=info['arena_bytes'],input_offset=info['input_offset'],output_offset=info['output_offset'],tasks=tasks,input_scale=info['input_scale'],input_zero_point=info['input_zero_point'],output_scale=scale,output_zero_point=zero_point,serial=info['serial'],input_layout=info['input_layout'],batch=info['batch'],input_tensor_count=info['input_tensor_count'])
    meta.update(output_scale=scale,output_zero_point=zero_point,fused_add=float(value.reshape(())),internal_mul_zero_point=zero_point+step)
    return result,meta


def compile_elementwise(model,input_scale=1.0,input_zero_point=0,output_range=None,operand_zero_points=(0,0)):
    g=model.graph;nodes=list(g.node)
    kind=nodes[-1].op_type if nodes else ''
    if [n.op_type for n in nodes]==['Conv','Add'] and len(g.input)==1:
        import copy
        from onnx import numpy_helper as nh
        conv,add=nodes;external=g.input[0].name;shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
        if (conv.input[0]!=external or list(add.input)!=[conv.output[0],external] or len(shape)!=4 or shape[1]!=3):
            raise ValueError('residual Add requires Conv(input) + matching RGB input')
        lower=copy.deepcopy(model);names={v.name for v in [*g.input,*g.output,*g.initializer]}|{name for n in g.node for name in n.output}
        def unique(base):
            while base in names:base+='_'
            names.add(base);return base
        weight=unique('residual_identity_w');bias=unique('residual_identity_b');converted=unique('residual_converted')
        lower.graph.initializer.extend([nh.from_array(np.eye(3,dtype=np.float32).reshape(3,3,1,1),weight),nh.from_array(np.zeros(3,np.float32),bias)])
        identity=h.make_node('Conv',[external,weight,bias],[converted],kernel_shape=[1,1]);lower.graph.node.insert(1,identity);lower.graph.node[-1].input[1]=converted;lower=onnx.shape_inference.infer_shapes(lower)
        binary,meta=compile_elementwise(lower,input_scale,input_zero_point,output_range,operand_zero_points);meta['elementwise_graph_rewrite']='residual input converted by identity Conv'
        return binary,meta
    if [n.op_type for n in nodes]==['Conv','Mul'] and len(g.input)==2:
        import copy
        from onnx import numpy_helper as nh
        conv,mul=nodes;external=mul.input[1] if mul.input[0]==conv.output[0] else ''
        inputs={v.name:v for v in g.input};shape=[d.dim_value for d in inputs[external].type.tensor_type.shape.dim] if external in inputs else []
        if (conv.input[0]!=g.input[0].name or mul.input[0]!=conv.output[0] or shape!=[1,3,*[d.dim_value for d in g.input[0].type.tensor_type.shape.dim][2:]]):
            raise ValueError('external-plus-intermediate Mul requires Conv(input0) * matching RGB input1')
        lower=copy.deepcopy(model);names={v.name for v in [*g.input,*g.output,*g.initializer]}|{name for n in g.node for name in n.output}
        def unique(base):
            while base in names:base+='_'
            names.add(base);return base
        weight=unique('mul_external_identity_w');bias=unique('mul_external_identity_b');converted=unique('mul_external_converted')
        lower.graph.initializer.extend([nh.from_array(np.eye(3,dtype=np.float32).reshape(3,3,1,1),weight),nh.from_array(np.zeros(3,np.float32),bias)])
        identity=h.make_node('Conv',[external,weight,bias],[converted],kernel_shape=[1,1]);lower.graph.node.insert(1,identity);lower.graph.node[-1].input[1]=converted;lower=onnx.shape_inference.infer_shapes(lower)
        binary,meta=compile_elementwise(lower,input_scale,input_zero_point,output_range,operand_zero_points);meta['mul_graph_rewrite']='external operand converted by identity Conv'
        return binary,meta
    if (len(g.input) not in (1,2) or len(g.output)!=1 or kind not in ('Add','Mul','Sub','Max')
        or any(n.domain not in ('','ai.onnx') for n in nodes)):
        raise ValueError('elementwise profile requires two Conv branches feeding Add, Mul, Sub or Max')
    add=nodes[-1];producers={name:n for n in nodes[:-1] for name in n.output};branches=[]
    for operand in add.input:
        producer=producers.get(operand);branch=[]
        if producer is not None and producer.op_type=='Relu':
            if producer.attribute or len(producer.input)!=1:raise ValueError('invalid branch Relu')
            conv=producers.get(producer.input[0]);branch=[conv,producer]
        else:conv=producer;branch=[conv]
        if conv is None or conv.op_type!='Conv':raise ValueError('elementwise operands must come from Conv[/Relu] branches')
        branches.append(branch)
    if [n for branch in branches for n in branch]!=nodes[:-1]:
        raise ValueError('elementwise branches must be independent and ordered')
    a,b=branches[0][0],branches[1][0]
    if (a.input[0]!=g.input[0].name or b.input[0]!=g.input[-1].name
        or list(add.input)!=[branches[0][-1].output[0],branches[1][-1].output[0]] or list(add.output)!=[g.output[0].name]
        or add.attribute):
        raise ValueError('elementwise profile requires two same-input Conv branches without broadcasting')
    values={v.name:v for v in [*g.input,*g.value_info,*g.output]}
    ishape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    oshape=[d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    if len(ishape)==4 and len(oshape)==4 and (ishape[0]!=1 or ishape[1]!=3 or oshape[1]==1 or not all(5<=v<=8 for v in ishape[2:])):
        from .native_elementwise import compile_native_elementwise
        return compile_native_elementwise(model,input_scale,input_zero_point,output_range,operand_zero_points)
    if (len(ishape)!=4 or len(oshape)!=4 or ishape[:2]!=[1,3]
        or not all(5<=n<=8 for n in ishape[2:]) or oshape[0]!=1
        or not 2<=oshape[1]<=16 or oshape[2:]!=ishape[2:]):
        raise ValueError('elementwise requires RGB input, 2..16 output channels, H/W 5..8')
    if any(v.type.tensor_type.elem_type!=1 or [d.dim_value for d in v.type.tensor_type.shape.dim]!=ishape for v in g.input):raise ValueError('external branch inputs must have matching RGB shapes')
    input_count=len(g.input)
    height,width=ishape[2:];channels=oshape[1];surface=((height*width+3)//4)*64
    for name,shape in [(g.input[0].name,ishape),(a.output[0],oshape),(b.output[0],oshape),(g.output[0].name,oshape)]:
        v=values.get(name)
        if v is None or v.type.tensor_type.elem_type!=1 or [d.dim_value for d in v.type.tensor_type.shape.dim]!=shape:
            raise ValueError('elementwise branch shapes must agree; broadcasting unsupported')
    with tempfile.TemporaryDirectory() as tmp:
        paths=[];initial=[]
        for i,branch in enumerate(branches):
            node=branch[0];branch_output=branch[-1].output[0]
            m=h.make_model(h.make_graph(branch,'branch',[next(v for v in g.input if v.name==node.input[0])],[values[branch_output]],list(g.initializer)),opset_imports=list(model.opset_import))
            m.ir_version=model.ir_version
            path=Path(tmp)/f'branch{i}.onnx';onnx.save(m,path);paths.append(path)
            _,meta=compile_model(path,input_scale=input_scale,input_zero_point=input_zero_point)
            if meta['kernel_size']!=1:raise ValueError('elementwise profile requires 1x1 Conv branches')
            initial.append(meta)
        # Bound each branch by its existing interval estimate, then use a shared
        # symmetric activation scale. This avoids unverified unequal-scale arithmetic.
        scale=float(np.float32(max(m['output_scale']*max(128+m['output_zero_point'],127-m['output_zero_point'])/127 for m in initial)))
        branch_scales=([float(np.float32(m['output_scale']*max(128+m['output_zero_point'],127-m['output_zero_point'])/127)) for m in initial] if kind=='Mul' else [scale,scale])
        if (not isinstance(operand_zero_points,(tuple,list)) or len(operand_zero_points)!=2
            or any(not isinstance(z,int) or not -128<=z<=127 for z in operand_zero_points)):
            raise ValueError('Mul operand zero points must be two INT8 values')
        if kind!='Mul' and tuple(operand_zero_points)!=(0,0):raise ValueError('operand zero points apply only to Mul')
        compiled=[compile_model(p,s,z,input_scale=input_scale,input_zero_point=input_zero_point) for p,s,z in zip(paths,branch_scales,operand_zero_points)]
    data=bytearray(4096)
    for i,(source,meta) in enumerate(compiled):
        regs={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from('<126Q',source)}
        step=192 if channels>8 else 128
        woff,boff=0xc00+i*step,0xc40+i*step
        bias_size=128 if channels>8 else 64
        if regs[0x1030]>64:raise ValueError('elementwise branch weight allocation exceeded')
        data[woff:woff+64]=source[regs[0x1110]:regs[0x1110]+64]
        data[boff:boff+bias_size]=source[regs[0x5020]:regs[0x5020]+bias_size]
        regs.update({0x4024:surface,0x40c0:surface,0x1070:0x1000+(i*height*16*3 if input_count==2 else 0),0x4020:0x2000+i*1024,0x1110:woff,0x5020:boff})
        for j,(reg,default,tag) in enumerate(REGISTERS):
            struct.pack_into('<Q',data,i*0x440+j*8,tag<<48|regs.get(reg,default)<<16|reg)
    regs={0x400c:0x1e5,0x4018:0,0x4020:0x3000,0x403c:0x2000f,
          0x4040:0x100012,0x4048:0x40000000,0x4050:0x30000000,0x4054:0,
          DPU_EW_CFG:0x8000c0c0|(EW_ALU_ADD<<EW_ALU_ALGO_SHIFT),0x4074:0,0x4078:0x4000,0x4080:0,
          0x4084:0x14000,DPU_OUT_CVT_SHIFT:29,0x5018:0x2000,0x501c:0,0x5020:0,
          0x5034:0x40000004,0x5038:0x2400,0x5040:1024,0x5044:0x907809}
    regs.update({0x4024:surface,0x4030:width-1,0x4034:height-1,
        0x403c:(channels-1)<<16|15,0x405c:(height-1)<<16|(width-1),
        0x40c0:surface,0x500c:width-1,0x5010:height-1,0x5040:surface})
    output_scale=2*scale;output_zero_point=0
    if kind=='Sub':
        # Captures differ from Add by signed operand scale -16384.
        regs[0x4078]=0xc000
    elif kind=='Max':
        regs[DPU_EW_CFG]=0x8000c0c0
    elif kind=='Mul':
        # RK3588 names suggest OD_BYPASS and EW_OP_TYPE=MUL. The complete
        # RV1103 profile retains observed bits that those references mark reserved.
        output_scale,output_zero_point,conversion,shift=mul_output_conversion(branch_scales[0]*branch_scales[1],output_range)
        regs.update({0x4048:0,0x4050:0x30000002,DPU_EW_CFG:0x81004094,
                     0x4074:(-operand_zero_points[1])&0xffffffff,0x4078:1,0x4080:output_zero_point&0xffffffff,0x4084:conversion,DPU_OUT_CVT_SHIFT:shift})
        if operand_zero_points[0]:regs.update({0x4040:0x120050,0x4044:(-operand_zero_points[0])&0xffffffff})
    layout=[r for r in REGISTERS if r[0]>=0x4000]
    if len(layout)!=78:raise ValueError('unexpected elementwise register layout')
    for i,(reg,default,tag) in enumerate(layout):
        struct.pack_into('<Q',data,0x880+i*8,tag<<48|regs.get(reg,default)<<16|reg)
    tasks=[(0,126,29,768),(0x440,126,29,768),(0x880,78,24,768)]
    for offset,count,enable,_ in tasks:
        for i,(reg,val,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,enable,0x81))):
            struct.pack_into('<Q',data,offset+(count+i)*8,tag<<48|val<<16|reg)
    binary=encode_sequence(data,input_shape=(height*input_count,width,3),output_shape=(height,width,channels),input_stride=16,
        arena_bytes=16384,input_offset=0x1000,output_offset=0x3000,tasks=tasks,
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=output_scale,output_zero_point=output_zero_point,serial=True,input_tensor_count=input_count)
    return binary,dict(elementwise_profile=kind.lower()+f'-{height}x{width}-c{channels}-'+('independent-scales' if kind=='Mul' else 'equal-scale'),
        branches=[m for _,m in compiled],output_scale=output_scale,output_zero_point=output_zero_point,operand_zero_points=list(operand_zero_points),
        input_tensors=[dict(name=v.name,shape_nhwc=[1,height,width,3],packed_byte_offset=i*height*width*3) for i,v in enumerate(g.input)],
        shape_nhwc=[1,height*input_count,width,3],output_shape_nhwc=[1,height,width,channels])


def compile_standalone_mul(model,input_scale=1.0,input_zero_point=0,output_range=None,operand_zero_points=(0,0)):
    """Lower logical external-input Mul via two identity Conv input conversions."""
    g=model.graph;node=g.node[0]
    if (len(g.node)!=1 or node.op_type!='Mul' or node.domain not in ('','ai.onnx') or node.attribute
        or len(node.input)!=2 or len(g.output)!=1 or list(node.output)!=[g.output[0].name]
        or len(g.input) not in (1,2) or set(node.input)!={v.name for v in g.input}):
        raise ValueError('standalone Mul requires one or two external inputs with matching RGB shapes')
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if len(shape)!=4 or not 1<=shape[1]<=16:raise ValueError('standalone Mul requires C1..16')
    channels=shape[1]
    lower=copy.deepcopy(model);names={v.name for v in g.input}|{v.name for v in g.output}|{v.name for v in g.initializer}
    def unique(base):
        while base in names:base+='_' 
        names.add(base);return base
    nodes=[];outputs=[]
    for i in range(2):
        weight,bias,output=[unique('mul_'+str(i)+'_'+v) for v in ('weight','bias','converted')]
        lower.graph.initializer.extend([nh.from_array(np.eye(channels,dtype=np.float32).reshape(channels,channels,1,1),weight),nh.from_array(np.zeros(channels,np.float32),bias)])
        nodes.append(h.make_node('Conv',[g.input[min(i,len(g.input)-1)].name,weight,bias],[output],kernel_shape=[1,1]));outputs.append(output)
    nodes.append(h.make_node('Mul',outputs,list(node.output)))
    del lower.graph.node[:];lower.graph.node.extend(nodes);lower=onnx.shape_inference.infer_shapes(lower)
    data,meta=compile_elementwise(lower,input_scale,input_zero_point,output_range,operand_zero_points)
    return data,dict(meta,standalone_mul_input_conversion='two identity Conv tasks')


def compile_constant_mul(model,input_scale=1.0,input_zero_point=0,output_range=None,operand_zero_points=(0,0),mutable=False):
    """External RGB tensor times an immutable broadcast constant; two NPU tasks."""
    g=model.graph;node=g.node[0]
    constants={v.name:nh.to_array(v) for v in g.initializer if v.name not in {x.name for x in g.input}}
    if (len(g.node)!=1 or len(g.input)!=1 or len(g.output)!=1 or len(node.input)!=2
        or node.domain not in ('','ai.onnx') or node.attribute or g.input[0].name not in node.input):
        raise ValueError('constant Mul requires one external input and one immutable constant')
    factor=constants.get(next((v for v in node.input if v!=g.input[0].name),''))
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if factor is None or factor.dtype!=np.float32 or not np.isfinite(factor).all() or len(shape)!=4:
        raise ValueError('finite float32 broadcast constant and rank-four input required')
    try:expanded=np.broadcast_to(factor,shape)
    except ValueError:raise ValueError('unsupported Mul broadcast shape') from None
    if shape[0]>1:
        return _compile_batched_constant_mul(model,factor,expanded,input_scale,input_zero_point,output_range,operand_zero_points,mutable)
    if shape[:2]!=[1,3] or not all(5<=v<=8 for v in shape[2:]):
        raise ValueError('single-batch constant Mul requires RGB H/W5..8')
    lower=copy.deepcopy(model);lower.graph.node[0].input[:]=[g.input[0].name]*2
    if tuple(operand_zero_points)!=(0,0):raise ValueError('constant Mul uses a materialized zero-point-zero operand')
    binary,meta=compile_standalone_mul(lower,input_scale,input_zero_point);data=bytearray(binary[144:]);data.extend(bytes(4096))
    _,_,height,width=shape
    scale=float(np.float32(np.max(abs(factor))/127)) or 1.0
    q=np.clip(np.rint(expanded/scale),-128,127).astype(np.int8)[0].transpose(1,2,0)
    per_channel=bool(np.all(q==q[0,0]))
    if per_channel:
        packed=np.zeros(16,np.int8);packed[:3]=q[0,0]
    else:
        packed=np.zeros((height,width,16),np.int8);packed[:,:,:3]=q
    values=packed.tobytes();data[0x1000:0x1000+len(values)]=values
    output_scale,output_zero_point,conversion,shift=mul_output_conversion(
        meta['branches'][0]['output_scale']*scale,output_range)
    fields_by_program={0:{0x1070:0x2000,0x4020:0x3000},0x880:{0x5018:0x3000,0x4020:0x4000,0x5038:0x1000,0x5034:4 if per_channel else 0x40000004,0x5040:16 if per_channel else ((height*width+3)//4)*64,
        0x4080:output_zero_point&0xffffffff,0x4084:conversion,0x4088:shift}}
    for base,fields in fields_by_program.items():
        for j in range(126 if base==0 else 78):
            off=base+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
            if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    result=encode_sequence(data,input_shape=(height,width,3),output_shape=(height,width,3),input_stride=16,arena_bytes=20480,
        input_offset=0x2000,output_offset=0x4000,tasks=[(0,126,29,768),(0x880,78,24,768)],
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=output_scale,output_zero_point=output_zero_point,serial=True,
        constants=([dict(name='mul.factor',offset=0x1000,size=len(values),kind=3)] if mutable else ()))
    meta.update(output_scale=output_scale,output_zero_point=output_zero_point,branches=meta['branches'][:1],constant_scale=scale,constant_zero_point=0,constant_data_mode='per-channel' if per_channel else 'per-pixel')
    meta.pop('standalone_mul_input_conversion',None)
    return result,meta


def compile_per_channel_constant_mul(model,input_scale=1.0,input_zero_point=0,output_range=None):
    """External RGB tensor times an immutable per-channel constant, per-channel quantized.

    The verified EW Mul task carries one output multiplier/shift, so its operand
    stream stores a single constant scale: `[.02,.35,1.9]` becomes `[1,23,127]`
    and the small channel loses most of its precision. A 1x1 depthwise
    convolution computes the same product while reusing the verified native
    per-output-channel weight scale, so each channel of the constant is
    quantized on its own grid. The output grid stays one per-tensor scale.
    """
    g=model.graph;node=g.node[0]
    constants={v.name:nh.to_array(v) for v in g.initializer if v.name not in {x.name for x in g.input}}
    if (len(g.node)!=1 or len(g.input)!=1 or len(g.output)!=1 or len(node.input)!=2
        or node.op_type!='Mul' or node.domain not in ('','ai.onnx') or node.attribute):
        raise ValueError('per-channel constant Mul requires one Mul with one external input')
    external=g.input[0].name
    other=next((v for v in node.input if v!=external),None)
    if other is None or other not in constants:
        raise ValueError('per-channel constant Mul requires one immutable constant operand')
    factor=constants[other]
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if factor.dtype!=np.float32 or not np.isfinite(factor).all() or len(shape)!=4:
        raise ValueError('finite float32 constant and rank-four input required')
    if shape[0]!=1 or shape[1]!=3 or not all(5<=v<=8 for v in shape[2:]):
        raise ValueError('per-channel constant Mul requires single-batch RGB H/W5..8')
    try:expanded=np.broadcast_to(factor,shape)
    except ValueError:raise ValueError('unsupported Mul broadcast shape') from None
    per_channel=expanded[0,:,0,0]
    if not np.all(expanded==per_channel.reshape(3,1,1)):
        raise ValueError('per-channel constant Mul requires a spatially invariant constant')
    if np.unique(per_channel).size<2:
        raise ValueError('per-channel constant Mul requires distinct channel values')
    _,_,height,width=shape
    source=g.input[0].name;result=g.output[0].name
    nodes=[h.make_node('Conv',[source,'stem_weight','stem_bias'],['converted'],kernel_shape=[1,1]),
           h.make_node('Conv',['converted','scale_weight','scale_bias'],[result],kernel_shape=[1,1],
                       pads=[0,0,0,0],strides=[1,1],dilations=[1,1],group=3)]
    graph=h.make_graph(nodes,'per_channel_constant_mul',list(g.input),list(g.output),
        [nh.from_array(np.eye(3,dtype=np.float32).reshape(3,3,1,1),'stem_weight'),
         nh.from_array(np.zeros(3,np.float32),'stem_bias'),
         nh.from_array(per_channel.astype(np.float32).reshape(3,1,1,1),'scale_weight'),
         nh.from_array(np.zeros(3,np.float32),'scale_bias')])
    lowered=copy.deepcopy(model)
    lowered.graph.CopyFrom(graph)
    lowered=onnx.shape_inference.infer_shapes(lowered)
    from .depthwise import compile_depthwise
    binary,meta=compile_depthwise(lowered,input_scale,input_zero_point,output_range)
    meta.update(profile='per-channel-constant-mul',constant_data_mode='per-channel-weight-scales',
        per_channel_constant=[float(v) for v in per_channel],
        constant_weight_scales=[float(v) for v in meta['depthwise']['weight_scales']])
    return binary,meta


def runtime_scale_reference(image,operand_codes,stem_quantization):
    """Integer reference for the runtime per-channel scale Mul."""
    from .quantization import Quantization,reference
    if isinstance(stem_quantization,Quantization):q=stem_quantization
    else:
        values=dict(stem_quantization)
        for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
            values[key]=np.array(values[key])
        q=Quantization(**values)
    stem=reference(image,q)
    codes=np.asarray(operand_codes,np.int32).reshape(1,1,3)
    return mul_reference(stem,codes)


def compile_runtime_scale_mul(model,operand_scale=1/127,input_scale=1.0,input_zero_point=0,output_range=None):
    """Mul(image[1,3,H,W], scale[1,3,1,1]) with the per-channel operand bound at run time.

    The verified per-channel operand mode already reads one int8 operand per channel
    from a 16-byte row, so the second input is exposed as a *named external tensor*
    of shape (1,1,1,3): the runtime packs three bytes into that row and the elementwise
    task picks them up per channel. This is the first profile with two external
    inputs of different shapes, and it gives a runtime affine modulation.
    """
    chained=copy.deepcopy(model)
    g=chained.graph;nodes=list(g.node)
    if (len(nodes)!=1 or nodes[0].op_type!='Mul' or nodes[0].domain not in ('','ai.onnx')
        or nodes[0].attribute or len(nodes[0].input)!=2 or len(g.output)!=1
        or list(nodes[0].output)!=[g.output[0].name] or len(g.input)!=2):
        raise ValueError('runtime scale Mul requires one Mul with two external inputs')
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in g.input}
    image=next((v for v in g.input if len(shapes[v.name])==4 and shapes[v.name][:2]==[1,3]),None)
    scale=next((v for v in g.input if v is not image and shapes[v.name]==[1,3,1,1]),None)
    if image is None or scale is None or scale.name in {t.name for t in g.initializer}:
        raise ValueError('runtime scale Mul requires [1,3,H,W] and [1,3,1,1] external inputs')
    _,_,height,width=shapes[image.name]
    if not all(5<=v<=8 for v in (height,width)):
        raise ValueError('runtime scale Mul requires H/W 5..8')
    if not 0<operand_scale and np.isfinite(operand_scale):
        raise ValueError('operand scale must be a positive finite number')
    image_name=image.name
    # The program comes from the verified two-identity-branch lowering of the image
    # against itself; the operand bytes and conversion are then supplied here, with
    # the operand row exposed as an external tensor instead of payload constants.
    lower=copy.deepcopy(model)
    del lower.graph.input[:]
    lower.graph.input.append(image)
    lower.graph.node[0].input[:]=[image_name,image_name]
    binary,meta=compile_standalone_mul(lower,input_scale,input_zero_point)
    data=bytearray(binary[144:])
    if len(data)!=4096:raise ValueError('unexpected elementwise payload size')
    branch_scale=float(meta['branches'][0]['output_scale'])
    output_scale,output_zero_point,conversion,shift=mul_output_conversion(branch_scale*operand_scale,output_range)
    fields_by_program={0:{0x1070:0x2000,0x4020:0x3000},
        0x880:{0x5018:0x3000,0x4020:0x4000,0x5038:0x1000,0x5034:4,0x5040:16,
               0x4080:output_zero_point&0xffffffff,0x4084:conversion,0x4088:shift}}
    for base,fields in fields_by_program.items():
        for j in range(126 if base==0 else 78):
            off=base+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
            if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    tensors=[
        dict(name='image',role=ROLE_INPUT,layout=LAYOUT_PACKED_U8,index=0,shape=(1,height,width,3),
             offset=0x2000,size=height*16*3),
        dict(name='scale',role=ROLE_INPUT,layout=LAYOUT_NATIVE16,index=1,shape=(1,1,1,3),
             offset=0x1000,size=64),
        dict(name='converted',role=ROLE_INTERNAL,layout=LAYOUT_NATIVE16,index=0,shape=(1,height,width,3),
             offset=0x3000,size=((height*width+3)//4)*64),
        dict(name='output',role=ROLE_OUTPUT,layout=LAYOUT_NATIVE16,index=0,shape=(1,height,width,3),
             offset=0x4000,size=((height*width+3)//4)*64),
    ]
    result=encode_sequence_v5(data,tensors=tensors,tasks=[(0,126,29,768),(0x880,78,24,768)],
        arena_bytes=20480,input_scale=input_scale,input_zero_point=input_zero_point,
        output_scale=output_scale,output_zero_point=output_zero_point,serial=True)
    meta.update(profile='runtime-scale-mul',output_scale=output_scale,output_zero_point=output_zero_point,
        operand_scale=float(operand_scale),branch_scale=branch_scale,branches=meta['branches'][:1],
        shape_nhwc=[1,height,width,3],output_shape_nhwc=[1,height,width,3],
        input_tensors=[dict(name='image',shape_nhwc=[1,height,width,3]),
                       dict(name='scale',shape_nhwc=[1,1,1,3])],
        output_tensors=['output'])
    meta.pop('standalone_mul_input_conversion',None)
    return result,meta


def _compile_batched_constant_mul(model,factor,expanded,input_scale,input_zero_point,output_range,operand_zero_points,mutable=False):
    """Native16 batch tasks with one immutable scalar/channel vector per batch."""
    g=model.graph;node=g.node[0];shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    batch,channels,height,width=shape
    if (not 2<=batch<=16 or not 1<=channels<=16 or not all(1<=v<=32 for v in (height,width))
        or tuple(operand_zero_points)!=(0,0)):
        raise ValueError('batched constant Mul requires N2..16, C1..16, H/W1..32 and zero-centered constant arithmetic')
    # This bounded path keeps constants compressed: every batch/channel value
    # must be spatially invariant, while scalar and per-batch forms broadcast.
    nhwc=expanded.transpose(0,2,3,1)
    if not np.all(nhwc==nhwc[:,0:1,0:1,:]):
        raise ValueError('batched constant Mul supports scalar or [N,C,1,1] constants')
    lower=copy.deepcopy(model);lower.graph.node[0].input[:]=[g.input[0].name]*2
    names={v.name for v in g.input}|{v.name for v in g.output}|{v.name for v in g.initializer}
    def unique(base):
        while base in names:base+='_'
        names.add(base);return base
    nodes=[];outputs=[]
    for i in range(2):
        wn,bn,on=[unique(f'constant_batch_{i}_{x}') for x in ('w','b','out')]
        lower.graph.initializer.extend([nh.from_array(np.eye(channels,dtype=np.float32).reshape(channels,channels,1,1),wn),nh.from_array(np.zeros(channels,np.float32),bn)])
        nodes.append(h.make_node('Conv',[g.input[0].name,wn,bn],[on],kernel_shape=[1,1]));outputs.append(on)
    nodes.append(h.make_node('Mul',outputs,list(node.output)));del lower.graph.node[:];lower.graph.node.extend(nodes)
    lower=onnx.shape_inference.infer_shapes(lower)
    from .native_elementwise import compile_native_elementwise
    source,meta=compile_native_elementwise(lower,input_scale,input_zero_point)
    info=decode_sequence(source);start=96+16*info['task_count'];data=bytearray(source[start:])
    constant_offset=(len(data)+63)//64*64
    scale=float(np.float32(np.max(np.abs(factor))/127)) or 1.0
    q=np.clip(np.rint(nhwc[:,0,0,:]/scale),-128,127).astype(np.int8)
    packed=np.zeros((batch,16),np.int8);packed[:,:channels]=q
    data.extend(bytes(constant_offset+packed.nbytes-len(data)));data[constant_offset:constant_offset+packed.nbytes]=packed.tobytes()
    new_payload=(len(data)+63)//64*64;data.extend(bytes(new_payload-len(data)))
    new_input=(new_payload+4095)//4096*4096;delta=new_input-info['input_offset']
    address_regs={0x1070,0x4020,0x5018,0x5038}
    tasks=[]
    output_scale,output_zero_point,conversion,shift=mul_output_conversion(meta['branches'][0]['output_scale']*scale,output_range)
    for bi in range(batch):
        for old_index in (bi*3,bi*3+2):
            task=info['tasks'][old_index];tasks.append((task['command_offset'],task['register_count'],task['enable'],task['mask']))
            for j in range(task['register_count']):
                off=task['command_offset']+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535;value=word>>16&0xffffffff
                if reg in address_regs and info['input_offset']<=value<info['arena_bytes']:value+=delta
                if old_index==bi*3+2:
                    fields={0x5034:4,0x5038:constant_offset+bi*16,0x5040:16,0x4080:output_zero_point&0xffffffff,0x4084:conversion,0x4088:shift}
                    value=fields.get(reg,value)
                struct.pack_into('<Q',data,off,(word>>48)<<48|value<<16|reg)
    result=encode_sequence(data,input_shape=(height,width,channels),output_shape=(height,width,channels),input_stride=width,
        arena_bytes=info['arena_bytes']+delta,input_offset=new_input,output_offset=info['output_offset']+delta,tasks=tasks,
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=output_scale,output_zero_point=output_zero_point,
        serial=True,input_layout='native16',batch=batch,
        constants=([dict(name='mul.factor',offset=constant_offset,size=packed.nbytes,kind=3)] if mutable else ()))
    meta.update(output_scale=output_scale,output_zero_point=output_zero_point,constant_scale=scale,constant_zero_point=0,
        constant_data_mode='per-batch-channel',shape_nhwc=[batch,height,width,channels],output_shape_nhwc=[batch,height,width,channels])
    return result,meta
