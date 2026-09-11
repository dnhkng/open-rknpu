"""MIT. Native-input Conv branches and elementwise arithmetic with dynamic allocation."""
import struct
import numpy as np
from onnx import helper as h
from .native import compile_native_input
from .sequence import encode_sequence, decode_sequence
from .register_profile import REGISTERS


def compile_native_elementwise(model,input_scale=1.,input_zero_point=0,output_range=None,operand_zero_points=(0,0)):
    g=model.graph;nodes=list(g.node)
    if len(nodes)!=3 or len(g.input) not in (1,2) or len(g.output)!=1:
        raise ValueError('native elementwise requires two Conv branches')
    a,b,op=nodes;kind=op.op_type
    if ([a.op_type,b.op_type]!=['Conv','Conv'] or kind not in ('Add','Mul','Sub','Max')
        or any(n.domain not in ('','ai.onnx') for n in nodes) or op.attribute
        or a.input[0]!=g.input[0].name or b.input[0]!=g.input[-1].name
        or list(op.input)!=[a.output[0],b.output[0]] or list(op.output)!=[g.output[0].name]):
        raise ValueError('invalid native elementwise graph connections')
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    output=[d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    batch=shape[0] if len(shape)==4 else 0
    if (len(shape)!=4 or len(output)!=4 or not 1<=batch<=16 or output[0]!=batch
        or not 1<=shape[1]<=16 or not 1<=output[1]<=16 or output[2:]!=shape[2:] or any(v.type.tensor_type.elem_type!=1 or
        [d.dim_value for d in v.type.tensor_type.shape.dim]!=shape for v in g.input)):
        raise ValueError('native elementwise requires matching static input shapes')
    values={v.name:v for v in (*g.value_info,*g.output)};branches=[];initial=[]
    for node in (a,b):
        if node.output[0] not in values:raise ValueError('missing branch shape')
        sub=h.make_model(h.make_graph([node],'branch',[next(v for v in g.input if v.name==node.input[0])],
            [values[node.output[0]]],list(g.initializer)),opset_imports=list(model.opset_import));sub.ir_version=model.ir_version
        data,meta=compile_native_input(sub,input_scale,input_zero_point)
        if meta['quantization']['kernel_size']!=1 or meta['output_shape_nhwc']!=[batch,shape[2],shape[3],output[1]]:
            raise ValueError('native elementwise branches require same-shape 1x1 Conv')
        branches.append(sub);initial.append(meta)
    scales=[float(np.float32(m['output_scale']*max(128+m['output_zero_point'],127-m['output_zero_point'])/127)) for m in initial]
    if kind!='Mul':scales=[max(scales)]*2
    if (not isinstance(operand_zero_points,(tuple,list)) or len(operand_zero_points)!=2
        or any(not isinstance(z,int) or not -128<=z<=127 for z in operand_zero_points)):
        raise ValueError('Mul operand zero points must be two INT8 values')
    if kind!='Mul' and tuple(operand_zero_points)!=(0,0):raise ValueError('operand zero points apply only to Mul')
    compiled=[compile_native_input(m,input_scale,input_zero_point,dict(scale=s,zero_point=z)) for m,s,z in zip(branches,scales,operand_zero_points)]
    align=lambda x,a=64:(x+a-1)//a*a
    ih,iw=shape[2:];ic=shape[1];oc=output[1];surface=align(ih*iw*16)
    program_group=0xb40;cursor=batch*program_group;parts=[]
    for binary,meta in compiled:
        info=decode_sequence(binary);source=binary[96+info['task_count']*16:]
        regs={w&65535:w>>16&0xffffffff for w in struct.unpack_from('<126Q',source)}
        weight_size=align(regs[0x1030]);bias_size=align((oc+3)//4*32)
        parts.append((source,regs,cursor,cursor+weight_size,weight_size,bias_size))
        cursor+=weight_size+bias_size
    payload=align(cursor);inp=align(payload,4096);input_storage=align(ih*iw*16*len(g.input));act0=align(inp+batch*input_storage);out=act0+batch*2*surface
    data=bytearray(payload)
    for i,(source,r,wo,bo,ws,bs) in enumerate(parts):
        data[wo:wo+ws]=source[r[0x1110]:r[0x1110]+ws];data[bo:bo+bs]=source[r[0x5020]:r[0x5020]+bs]
        for bi in range(batch):
            fields=dict(r);fields.update({0x1110:wo,0x5020:bo,0x1070:inp+bi*input_storage+(ih*iw*16*i if len(g.input)==2 else 0),0x4020:act0+bi*2*surface+i*surface})
            for j,(reg,default,tag) in enumerate(REGISTERS):struct.pack_into('<Q',data,bi*program_group+i*0x440+j*8,tag<<48|fields.get(reg,default)<<16|reg)
    r={0x400c:0x1e5,0x4018:0,0x4020:out,0x4024:surface,0x4030:iw-1,0x4034:ih-1,0x403c:(oc-1)<<16|15,
       0x4040:0x100012,0x4048:0x40000000,0x4050:0x30000000,0x4054:0,0x405c:(ih-1)<<16|(iw-1),
       0x4070:0x8002c0c0,0x4074:0,0x4078:0x4000,0x4080:0,0x4084:0x14000,0x4088:29,0x40c0:surface,
       0x500c:iw-1,0x5010:ih-1,0x5018:act0,0x501c:0,0x5020:0,0x5034:0x40000004,0x5038:act0+surface,0x5040:surface,0x5044:0x907809}
    if kind=='Sub':r[0x4078]=0xc000
    elif kind=='Max':r[0x4070]=0x8000c0c0
    elif kind=='Mul':
        from .elementwise import mul_output_conversion
        output_scale,output_zero_point,conversion,shift=mul_output_conversion(scales[0]*scales[1],output_range)
        r.update({0x4048:0,0x4050:0x30000002,0x4070:0x81004094,0x4074:(-operand_zero_points[1])&0xffffffff,0x4078:1,0x4080:output_zero_point&0xffffffff,0x4084:conversion,0x4088:shift})
        if operand_zero_points[0]:r.update({0x4040:0x120050,0x4044:(-operand_zero_points[0])&0xffffffff})
    tasks=[]
    for bi in range(batch):
        base=bi*program_group;fields=dict(r);fields.update({0x4020:out+bi*surface,0x5018:act0+bi*2*surface,0x5038:act0+bi*2*surface+surface})
        for j,(reg,default,tag) in enumerate(v for v in REGISTERS if v[0]>=0x4000):
            struct.pack_into('<Q',data,base+0x880+j*8,tag<<48|fields.get(reg,default)<<16|reg)
        tasks.extend([(base,126,29,768),(base+0x440,126,29,768),(base+0x880,78,24,768)])
    for base,count,enable,_ in tasks:
        for j,(reg,value,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,enable,0x81))):
            struct.pack_into('<Q',data,base+(count+j)*8,tag<<48|value<<16|reg)
    if kind!='Mul':output_scale=float(np.float32(2*scales[0]));output_zero_point=0
    binary=encode_sequence(data,input_shape=(ih*len(g.input),iw,ic),output_shape=(ih,iw,oc),input_stride=iw,
        arena_bytes=align(out+batch*surface,4096),input_offset=inp,output_offset=out,tasks=tasks,
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=output_scale,output_zero_point=output_zero_point,serial=True,input_layout='native16',batch=batch,input_tensor_count=len(g.input))
    return binary,dict(elementwise_profile=kind.lower()+'-native16-input',branches=[m for _,m in compiled],
        input_tensors=[dict(name=v.name,shape_nhwc=[batch,ih,iw,ic],packed_byte_offset=i*batch*ih*iw*ic) for i,v in enumerate(g.input)],
        shape_nhwc=[batch,ih*len(g.input),iw,ic],output_shape_nhwc=[batch,ih,iw,oc],output_scale=output_scale,output_zero_point=output_zero_point,operand_zero_points=list(operand_zero_points))
