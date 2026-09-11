"""SPDX-License-Identifier: MIT

Bounded DAG emission for RV1103 using the version-5 named-tensor container.

Supported shapes:

    input -> Conv(1x1) -> Relu -> t -> Conv_head_a -> outputA
                                    -> Conv_head_b -> outputB      (fan-out)

    input -> Conv(1x1) -> Relu -> t -> Conv_head_a -+
                                    -> Conv_head_b -+-> Op -> output
                                                                   (diamond)

`t` is produced once and consumed twice (fan-out), and the diamond additionally
*fans in*: both head outputs feed one elementwise task. The diamond's arena is
placed by `open_rknpu.liveness` from the task read/write sets, so the tensor
lifetimes are explicit rather than hand-computed. All commands are independently
generated; no vendor capture or RKNN object is read. Bounds: RGB 8x8 external
tensors, stem input channels 3, hidden channels 3..16, 1x1 stem, dense 1x1/3x3
heads with three output channels each, join in Add/Mul/Sub/Max.
"""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from .chain import NATIVE,native_quantize
from .compiler import compile_model
from .liveness import Access,plan
from .compose import Binding,ConstantSpec,Stage,TensorSpec,compose
from .register_profile import REGISTERS
from .sequence import encode_sequence_v5,decode_sequence,LAYOUT_PACKED_U8,LAYOUT_NATIVE16,ROLE_INPUT,ROLE_OUTPUT,ROLE_INTERNAL

def _align(n,a=64):
    return (n+a-1)//a*a

def _terminal(data,offset,enable):
    for i,(reg,value,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,enable,0x81))):
        struct.pack_into('<Q',data,offset+(126+i)*8,tag<<48|value<<16|reg)

def _terminals(data,offset,count,enable):
    """Task descriptor tail for a program of `count` register words."""
    _tail(data,offset,count,enable)

def _tail(data,offset,count,enable,link=0,control=0x28):
    """Task tail with an explicit next-program link and engine hand-off control.

    A non-serial submission is one ioctl per maximal linked run (S10): a task whose
    link is zero ends its run. `link` is the payload-relative offset of the next
    program and `control` is the measured transition (`0x40` inside a run, `0x14`
    CNA->DPU, `0x28` terminal).
    """
    for i,(reg,value,tag) in enumerate(((0x10,link,0x101),(0x14,control,0x101),(0,0,0x41),(8,enable,0x81))):
        struct.pack_into('<Q',data,offset+(count+i)*8,tag<<48|value<<16|reg)

def handoff_tails(data,tasks,serial=True):
    """Link an ordered task list for a non-serial submission; serial tails stay terminal.

    `tasks` is the emitted order as `(program offset, register words, enable, mask)`.
    Each tail's control word is the *successor's* fetch amount
    (`compose.amount_control`), which is what the front end needs to load it, so the
    whole list is one job (S8/S10). Returns `True` when the tails were rewritten.
    """
    if serial:
        return False
    from .compose import amount_control
    for position,(base,words,enable,_) in enumerate(tasks):
        if position+1<len(tasks):
            successor=tasks[position+1]
            _tail(data,base,words,enable,link=successor[0],
                  control=amount_control(successor[1]))
        else:
            _tail(data,base,words,enable)
    return True

def _native_fields(kernel,hidden,weight_offset,bias_offset,output_offset,input_offset,q):
    fields=dict(NATIVE)
    fields.update({0x1010:0x108 if kernel==3 else 0x104,
                   0x1030:3*16*kernel*kernel,0x1034:16*kernel*kernel,
                   0x1038:(kernel<<24)|(kernel<<16)|3,0x1068:0x101 if kernel==3 else 0,
                   0x1188:8*kernel*kernel,
                   0x1024:((hidden-1)<<16)|16,
                   0x1070:input_offset,0x1110:weight_offset,0x5020:bias_offset,0x4020:output_offset,
                   0x4080:q.output_zero_point&0xffffffff,0x4084:q.multiplier,0x4088:q.shift})
    return fields

def _pack_native_head(data,kernel,hidden,weight_offset,bias_offset,q):
    for o in range(3):
        w=q.weights[o].reshape(hidden,kernel,kernel)
        for kh in range(kernel):
            for kw in range(kernel):
                row=list(w[:,kh,kw])+[int(q.weight_zero_points[o])]*(16-hidden)
                struct.pack_into('<16b',data,weight_offset+((kh*kernel+kw)*3+o)*16,*row)
        struct.pack_into('<i',data,bias_offset+o*4,int(q.biases[o]))
        struct.pack_into('<h',data,bias_offset+16+o*2,-int(q.weight_zero_points[o]))
        struct.pack_into('<H',data,bias_offset+24+o*2,int(q.channel_multipliers[o]))

def compile_two_head(model,input_scale=1.0,input_zero_point=0):
    onnx.checker.check_model(model)
    g=model.graph;nodes=list(g.node)
    if [n.op_type for n in nodes]!=['Conv','Relu','Conv','Conv'] or any(n.domain not in ('','ai.onnx') for n in nodes):
        raise ValueError('two-head profile requires Conv, Relu, Conv, Conv')
    stem,activation,head_a,head_b=nodes
    if activation.attribute or list(activation.input)!=list(stem.output):
        raise ValueError('two-head profile requires a valid stem Relu')
    t=activation.output[0]
    if head_a.input[0]!=t or head_b.input[0]!=t:
        raise ValueError('two-head profile requires both heads to consume the stem output')
    if len(g.input)!=1 or len(g.output)!=2:
        raise ValueError('two-head profile requires one input and two outputs')
    outs={n.output[0]:n for n in (head_a,head_b)}
    if g.output[0].name not in outs or g.output[1].name not in outs or outs[g.output[0].name] is outs[g.output[1].name]:
        raise ValueError('two-head outputs must be the two head outputs in order')
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*g.input,*g.output,*g.value_info]}
    for v in (g.input[0],g.output[0],g.output[1]):
        if v.type.tensor_type.elem_type!=1 or shapes.get(v.name)!=[1,3,8,8]:
            raise ValueError('two-head external tensors must be float32 [1,3,8,8]')
    constants={x.name:nh.to_array(x) for x in g.initializer}
    for node in (stem,head_a,head_b):
        if len(node.input)!=3 or any(name not in constants for name in node.input[1:]):
            raise ValueError('two-head convolutions require constant float32 weights and bias')
        for name in node.input[1:]:
            if constants[name].dtype!=np.float32:raise ValueError('two-head constants must be float32')
    w1=constants[stem.input[1]];hidden=w1.shape[0] if w1.ndim==4 else 0
    if (w1.ndim!=4 or w1.shape!=(hidden,3,1,1) or not 3<=hidden<=16
        or constants[stem.input[2]].shape!=(hidden,)):
        raise ValueError('two-head stem must be a 1x1 Conv with hidden channels 3..16')
    stem_supported={'kernel_shape':[1,1],'pads':[0,0,0,0],'strides':[1,1],'dilations':[1,1],'group':1}
    stem_attrs={a.name:h.get_attribute_value(a) for a in stem.attribute}
    if any(k not in stem_supported or v!=stem_supported[k] for k,v in stem_attrs.items()):
        raise ValueError('unsupported stem attributes')
    heads=[]
    for head in (head_a,head_b):
        w=constants[head.input[1]];kernel=w.shape[2] if w.ndim==4 else 0
        if (w.ndim!=4 or kernel not in (1,3) or w.shape!=(3,hidden,kernel,kernel)
            or constants[head.input[2]].shape!=(3,)):
            raise ValueError('two-head heads must be dense 3-output 1x1/3x3 Conv with hidden inputs')
        attrs={a.name:h.get_attribute_value(a) for a in head.attribute}
        supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':1}
        if any(k not in supported or v!=supported[k] for k,v in attrs.items()):
            raise ValueError('unsupported head attributes')
        if kernel==3 and attrs.get('pads')!=[1,1,1,1]:
            raise ValueError('two-head 3x3 heads require symmetric pad1')
        heads.append((w,constants[head.input[2]],kernel))
    # Stem program and constants, independently compiled for the single-stem graph.
    first_graph=h.make_graph([stem,activation],'stem',list(g.input),
        [h.make_tensor_value_info(t,1,[1,hidden,8,8])],
        [nh.from_array(constants[name],name) for name in stem.input[1:]])
    first=h.make_model(first_graph,opset_imports=list(model.opset_import));first.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        stem_path=Path(tmp)/'stem.onnx';onnx.save(first,stem_path)
        stem_data,stem_meta=compile_model(stem_path,input_scale=input_scale,input_zero_point=input_zero_point)
    stem_scale=stem_meta['output_scale'];stem_zp=stem_meta['output_zero_point']
    # Consumer quantization is independent per head and analytic.
    q_heads=[native_quantize(w,b,stem_scale,stem_zp,None) for w,b,_ in heads]
    # Layout: three 126-word programs, then packed constants, then arena tensors.
    program_ends=[0x880+_align((126+4)*8)]
    weights1=_align(program_ends[0]);weight_size1=_align(1*1*_align(hidden,4)*4)
    bias1=weights1+weight_size1;bias_size1=((hidden+3)//4)*32
    head_specs=[];cursor=_align(bias1+bias_size1)
    for w,b,kernel in heads:
        wsize=_align(3*16*kernel*kernel);bsize=32
        head_specs.append((cursor,cursor+wsize,wsize,bsize,kernel))
        cursor=_align(cursor+wsize+bsize)
    payload_size=_align(cursor)
    stem_out=_align(payload_size);stem_bytes=8*8*16
    input_off=_align(stem_out+stem_bytes,4096);input_bytes=8*16*3
    out_a=_align(input_off+input_bytes);out_b=_align(out_a+stem_bytes)
    arena=_align(out_b+stem_bytes,4096)
    data=bytearray(payload_size)
    # Stem program with relocated constants and native intermediate destination.
    stem_src={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from('<126Q',stem_data)}
    registers=dict(stem_src);registers.update({0x1110:weights1,0x5020:bias1,0x4020:stem_out,0x1070:input_off})
    for i,(reg,default,tag) in enumerate(REGISTERS):
        struct.pack_into('<Q',data,i*8,tag<<48|registers.get(reg,default)<<16|reg)
    _terminal(data,0,29)
    data[weights1:weights1+weight_size1]=stem_data[stem_src[0x1110]:stem_src[0x1110]+weight_size1]
    data[bias1:bias1+bias_size1]=stem_data[stem_src[0x5020]:stem_src[0x5020]+bias_size1]
    for index,((w,b,kernel),q,(w_off,b_off,wsize,bsize,_),output) in enumerate(zip(heads,q_heads,head_specs,(out_a,out_b))):
        fields=_native_fields(kernel,hidden,w_off,b_off,output,stem_out,q)
        base=0x440+index*0x440
        for i,(reg,default,tag) in enumerate(REGISTERS):
            struct.pack_into('<Q',data,base+i*8,tag<<48|fields.get(reg,default)<<16|reg)
        _terminal(data,base,29)
        _pack_native_head(data,kernel,hidden,w_off,b_off,q)
    tensors=[
        dict(name='input0',role=ROLE_INPUT,layout=LAYOUT_PACKED_U8,index=0,shape=(1,8,8,3),
             offset=input_off,size=8*16*3),
        dict(name='stem',role=ROLE_INTERNAL,layout=LAYOUT_NATIVE16,index=0,shape=(1,8,8,hidden),
             offset=stem_out,size=8*8*16),
        dict(name='output_a',role=ROLE_OUTPUT,layout=LAYOUT_NATIVE16,index=0,shape=(1,8,8,3),
             offset=out_a,size=8*8*16),
        dict(name='output_b',role=ROLE_OUTPUT,layout=LAYOUT_NATIVE16,index=1,shape=(1,8,8,3),
             offset=out_b,size=8*8*16),
    ]
    tasks=[(0,126,29,768),(0x440,126,29,768),(0x880,126,29,768)]
    binary=encode_sequence_v5(data,tensors=tensors,tasks=tasks,arena_bytes=arena,
        input_scale=input_scale,input_zero_point=input_zero_point,
        output_scale=q_heads[0].output_scale,output_zero_point=q_heads[0].output_zero_point,serial=True)
    meta=dict(profile='two-head-fanout',hidden_channels=hidden,stem_kernel=1,
        head_kernels=[k for _,_,k in heads],input_scale=input_scale,input_zero_point=input_zero_point,
        output_scales=[q.output_scale for q in q_heads],output_zero_points=[q.output_zero_point for q in q_heads],
        shape_nhwc=[1,8,8,3],output_shape_nhwc=[1,8,8,3],output_tensors=['output_a','output_b'],
        stem_quantization=stem_meta['quantization'],
        head_quantization=[q.metadata() for q in q_heads],
        weights=[w.tolist() for w,_,_ in heads],bias=[b.tolist() for _,b,_ in heads],
        stem_weights=w1.tolist(),stem_bias=constants[stem.input[2]].tolist())
    return binary,meta

JOIN_TYPES=('Add','Mul','Sub','Max')

def join_reference(kind,first,second):
    """Integer reference for the join task, matching the emitted registers."""
    from .elementwise import add_reference,mul_reference,sub_reference,max_reference
    return {'Add':add_reference,'Mul':mul_reference,'Sub':sub_reference,
            'Max':max_reference}[kind](first,second)

def _join_fields(kind,height,width,channels,primary,secondary,output,surface,scales,operand_zero_points,output_range):
    """Elementwise join registers, from the verified native elementwise profile.

    `primary` is the accumulator-side tensor (0x5018) and `secondary` is the
    ERDMA-side tensor (0x5038) walked with `surface` as its plane stride. For
    Add/Sub/Max both operands share one scale and the output scale is twice it;
    Mul keeps the per-branch scales and folds them into its own conversion.
    """
    fields={0x400c:0x1e5,0x4018:0,0x4020:output,0x4024:surface,0x4030:width-1,0x4034:height-1,
            0x403c:(channels-1)<<16|15,0x4040:0x100012,0x4048:0x40000000,0x4050:0x30000000,0x4054:0,
            0x405c:(height-1)<<16|(width-1),0x4070:0x8002c0c0,0x4074:0,0x4078:0x4000,0x4080:0,
            0x4084:0x14000,0x4088:29,0x40c0:surface,0x500c:width-1,0x5010:height-1,0x5018:primary,
            0x501c:0,0x5020:0,0x5034:0x40000004,0x5038:secondary,0x5040:surface,0x5044:0x907809}
    if kind=='Sub':
        fields[0x4078]=0xc000
    elif kind=='Max':
        fields[0x4070]=0x8000c0c0
    elif kind=='Mul':
        from .elementwise import mul_output_conversion
        output_scale,output_zero_point,conversion,shift=mul_output_conversion(scales[0]*scales[1],output_range)
        fields.update({0x4048:0,0x4050:0x30000002,0x4070:0x81004094,
                       0x4074:(-operand_zero_points[1])&0xffffffff,0x4078:1,
                       0x4080:output_zero_point&0xffffffff,0x4084:conversion,0x4088:shift})
        if operand_zero_points[0]:
            fields.update({0x4040:0x120050,0x4044:(-operand_zero_points[0])&0xffffffff})
        return fields,output_scale,output_zero_point
    return fields,float(np.float32(2*scales[0])),0

def compile_diamond(model,output_range=None,operand_zero_points=(0,0),serial=True):
    """Shared stem, two consuming heads, one elementwise join and an optional Conv tail.

    The arena comes from `open_rknpu.liveness`: the stem is live until both heads
    have read it, each head output is live until the join, the join output until the
    first tail layer, and the external output is placed after every internal tensor
    so the version-5 no-overlap rule holds. The tail is a `[Conv, Relu]* Conv` chain
    whose first layer consumes the join output by name, i.e. the elementwise and
    native emitters are composed through the v5 tensor table.
    """
    onnx.checker.check_model(model)
    g=model.graph;nodes=list(g.node)
    if any(n.domain not in ('','ai.onnx') for n in nodes):
        raise ValueError('diamond profile requires default-domain nodes')
    join_index=next((index for index,node in enumerate(nodes) if node.op_type in JOIN_TYPES),None)
    if join_index is None or join_index<3:
        raise ValueError('diamond profile requires stem[,Relu], two heads and one join')
    join=nodes[join_index];kind=join.op_type
    if join.attribute:
        raise ValueError('diamond join must be Add, Mul, Sub or Max without attributes')
    head_a,head_b=nodes[join_index-2],nodes[join_index-1];stem_nodes=nodes[:join_index-2]
    tail_nodes=nodes[join_index+1:]
    if tail_nodes and (len(tail_nodes)%2!=1 or any(node.op_type!=('Conv' if index%2==0 else 'Relu')
                                                   for index,node in enumerate(tail_nodes))):
        raise ValueError('diamond tail must be [Conv, Relu]* Conv')
    if any(node.attribute for index,node in enumerate(tail_nodes) if index%2==1):
        raise ValueError('diamond tail Relu must not carry attributes')
    if [n.op_type for n in stem_nodes] not in (['Conv'],['Conv','Relu']):
        raise ValueError('diamond stem must be Conv or Conv,Relu')
    if any(n.op_type!='Conv' for n in (head_a,head_b)):
        raise ValueError('diamond heads must be Conv')
    stem=stem_nodes[0];t=stem_nodes[-1].output[0]
    if len(stem_nodes)==2 and (stem_nodes[1].attribute or list(stem_nodes[1].input)!=list(stem.output)):
        raise ValueError('diamond stem Relu must consume the stem output')
    if head_a.input[0]!=t or head_b.input[0]!=t:
        raise ValueError('diamond heads must consume the stem output')
    if list(join.input)!=[head_a.output[0],head_b.output[0]]:
        raise ValueError('diamond join must consume both head outputs in order')
    if tail_nodes:
        if tail_nodes[0].input[0]!=join.output[0]:
            raise ValueError('diamond tail must consume the join output')
        for index,node in enumerate(tail_nodes):
            if index and node.input[0]!=tail_nodes[index-1].output[0]:
                raise ValueError('diamond tail nodes must be connected in order')
    final=(tail_nodes[-1].output[0] if tail_nodes else join.output[0])
    if len(g.input)!=1 or len(g.output)!=1 or final!=g.output[0].name:
        raise ValueError('diamond profile requires one input and one output')
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*g.input,*g.output,*g.value_info]}
    for v in (g.input[0],g.output[0]):
        if v.type.tensor_type.elem_type!=1 or shapes.get(v.name)!=[1,3,8,8]:
            raise ValueError('diamond external tensors must be float32 [1,3,8,8]')
    for head in (head_a,head_b):
        if shapes.get(head.output[0])!=[1,3,8,8]:
            raise ValueError('diamond heads must produce float32 [1,3,8,8]')
    for node in tail_nodes:
        if node.op_type=='Conv' and shapes.get(node.output[0])!=[1,3,8,8]:
            raise ValueError('diamond tail must produce float32 [1,3,8,8]')
    constants={x.name:nh.to_array(x) for x in g.initializer}
    for node in (stem,head_a,head_b,*[n for n in tail_nodes if n.op_type=='Conv']):
        if len(node.input)!=3 or any(name not in constants for name in node.input[1:]):
            raise ValueError('diamond convolutions require constant float32 weights and bias')
        for name in node.input[1:]:
            if constants[name].dtype!=np.float32:
                raise ValueError('diamond constants must be float32')
    w1=constants[stem.input[1]];hidden=w1.shape[0] if w1.ndim==4 else 0
    if (w1.ndim!=4 or w1.shape!=(hidden,3,1,1) or not 3<=hidden<=16
        or constants[stem.input[2]].shape!=(hidden,)):
        raise ValueError('diamond stem must be a 1x1 Conv with hidden channels 3..16')
    stem_supported={'kernel_shape':[1,1],'pads':[0,0,0,0],'strides':[1,1],'dilations':[1,1],'group':1}
    stem_attrs={a.name:h.get_attribute_value(a) for a in stem.attribute}
    if any(k not in stem_supported or v!=stem_supported[k] for k,v in stem_attrs.items()):
        raise ValueError('unsupported stem attributes')
    heads=[]
    for head in (head_a,head_b):
        w=constants[head.input[1]];kernel=w.shape[2] if w.ndim==4 else 0
        if (w.ndim!=4 or kernel not in (1,3) or w.shape!=(3,hidden,kernel,kernel)
            or constants[head.input[2]].shape!=(3,)):
            raise ValueError('diamond heads must be dense 3-output 1x1/3x3 Conv with hidden inputs')
        attrs={a.name:h.get_attribute_value(a) for a in head.attribute}
        supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':1}
        if any(k not in supported or v!=supported[k] for k,v in attrs.items()):
            raise ValueError('unsupported head attributes')
        if kernel==3 and attrs.get('pads')!=[1,1,1,1]:
            raise ValueError('diamond 3x3 heads require symmetric pad1')
        heads.append((w,constants[head.input[2]],kernel))
    if (not isinstance(operand_zero_points,(tuple,list)) or len(operand_zero_points)!=2
        or any(not isinstance(z,int) or not -128<=z<=127 for z in operand_zero_points)):
        raise ValueError('Mul operand zero points must be two INT8 values')
    if kind!='Mul' and tuple(operand_zero_points)!=(0,0):
        raise ValueError('operand zero points apply only to the Mul join')
    if kind!='Mul' and output_range is not None and not tail_nodes:
        raise ValueError('the diamond output override requires the Mul join or a Conv tail')
    tails=[]
    for node in [n for n in tail_nodes if n.op_type=='Conv']:
        w=constants[node.input[1]];kernel=w.shape[2] if w.ndim==4 else 0
        if (w.ndim!=4 or kernel not in (1,3) or w.shape!=(3,3,kernel,kernel)
            or constants[node.input[2]].shape!=(3,)):
            raise ValueError('diamond tail layers must be dense 3-output 1x1/3x3 Conv with three inputs')
        attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
        supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':1}
        if any(k not in supported or v!=supported[k] for k,v in attrs.items()):
            raise ValueError('unsupported tail attributes')
        if kernel==3 and attrs.get('pads')!=[1,1,1,1]:
            raise ValueError('diamond 3x3 tail layers require symmetric pad1')
        tails.append((w,constants[node.input[2]],kernel))
    first_graph=h.make_graph(stem_nodes,'stem',list(g.input),
        [h.make_tensor_value_info(t,1,[1,hidden,8,8])],
        [nh.from_array(constants[name],name) for name in stem.input[1:]])
    first=h.make_model(first_graph,opset_imports=list(model.opset_import));first.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        stem_path=Path(tmp)/'stem.onnx';onnx.save(first,stem_path)
        stem_data,stem_meta=compile_model(stem_path)
    stem_scale=stem_meta['output_scale'];stem_zp=stem_meta['output_zero_point']
    natural=[native_quantize(w,b,stem_scale,stem_zp,None) for w,b,_ in heads]
    adjusted=[float(np.float32(q.output_scale*max(128+q.output_zero_point,127-q.output_zero_point)/127))
              for q in natural]
    scales=[float(np.float32(v)) for v in adjusted] if kind=='Mul' else [float(np.float32(max(adjusted)))]*2
    zeros=list(operand_zero_points) if kind=='Mul' else [0,0]
    q_heads=[native_quantize(w,b,stem_scale,stem_zp,dict(scale=s,zero_point=z))
             for (w,b,_),s,z in zip(heads,scales,zeros)]
    if kind=='Mul':
        from .elementwise import mul_output_conversion
        join_scale,join_zero=mul_output_conversion(scales[0]*scales[1],output_range)[:2]
    else:
        join_scale,join_zero=(float(np.float32(2*scales[0])),0)
    q_tails=[];previous=(join_scale,join_zero)
    for index,(w,b,_) in enumerate(tails):
        selected=output_range if index==len(tails)-1 else None
        q=native_quantize(w,b,previous[0],previous[1],selected)
        # The tail is `[Conv, Relu]* Conv`, so every non-final layer applies its
        # activation: the emitter writes 0x4060/0x406c/0x40e0 and the reference reads
        # `q.relu`. Before this the diamond tail dropped the graph's Relu entirely.
        q.relu=index<len(tails)-1
        q_tails.append(q);previous=(q.output_scale,q.output_zero_point)
    surface=8*8*16
    input_bytes=8*16*3
    weight_size1=_align(1*1*_align(hidden,4)*4)
    bias_size1=((hidden+3)//4)*32
    stem_block=weight_size1+bias_size1
    tail_names=[f'tail{index}' for index in range(len(tails))]
    stem_src={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from('<126Q',stem_data)}

    def stem_fill(payload,offset):
        payload[offset:offset+weight_size1]=stem_data[stem_src[0x1110]:stem_src[0x1110]+weight_size1]
        payload[offset+weight_size1:offset+stem_block]=stem_data[stem_src[0x5020]:stem_src[0x5020]+bias_size1]

    def stem_fields(addresses,constant_offsets):
        values=dict(stem_src)
        values.update({0x1110:constant_offsets['stem'],0x5020:constant_offsets['stem']+weight_size1,
                       0x4020:addresses['stem'],0x1070:addresses['input0']})
        return values

    # One stage per task. `open_rknpu.compose` assigns the program slots and constant
    # blocks in declared order, places the arena from these read/write sets, substitutes
    # the planned addresses into the register words and returns the declared binding
    # view. The declaration order is the emitted order, so the container is
    # byte-identical to the hand-assembled one (tests/test_diamond.py).
    stages=[Stage(name='stem',family='native-conv',reads=('input0',),writes=('stem',),
                  fields=stem_fields,constants=(ConstantSpec('stem',stem_block,stem_fill),),
                  bindings=(Binding(0x1070,'input0','read'),Binding(0x4020,'stem','write')))]

    def conv_stage(name,source,target,kernel,channels,quantization,input_zero):
        weight_size=_align(3*16*kernel*kernel);bias_size=32

        def fill(payload,offset):
            _pack_native_head(payload,kernel,channels,offset,offset+weight_size,quantization)

        def fields(addresses,constant_offsets):
            values=_native_fields(kernel,channels,constant_offsets[name],
                                  constant_offsets[name]+weight_size,addresses[target],
                                  addresses[source],quantization)
            # The CNA border path injects the input tensor's zero point, so a layer must
            # declare the grid it actually reads.
            values[0x1184]=int(input_zero)&0xffff
            # A hidden tail layer carries its Relu in its own program (S9), the same
            # way the chain family does.
            relu=bool(getattr(quantization,'relu',False))
            values.update({0x4060:0x12 if relu else 0x13,
                           0x406c:0 if relu else 0x80000000,
                           0x40e0:0 if relu else 0x80000000})
            return values

        return Stage(name=name,family='native-conv',reads=(source,),writes=(target,),
                     fields=fields,
                     constants=(ConstantSpec(name,weight_size+bias_size,fill),),
                     bindings=(Binding(0x1070,source,'read'),Binding(0x4020,target,'write')))

    for index,(_,_,kernel) in enumerate(heads):
        name=('head_a','head_b')[index]
        stages.append(conv_stage(name,'stem',name,kernel,hidden,q_heads[index],stem_zp))

    _,output_scale,output_zero_point=_join_fields(kind,8,8,3,0,0,0,surface,scales,
                                                  operand_zero_points,output_range)

    def join_fields(addresses,constant_offsets):
        values,_,_=_join_fields(kind,8,8,3,addresses['head_a'],addresses['head_b'],
                                addresses['join'] if tails else addresses['output'],
                                surface,scales,operand_zero_points,output_range)
        return values

    stages.append(Stage(name='join',family='elementwise',reads=('head_a','head_b'),
                        writes=('join',) if tails else ('output',),fields=join_fields,
                        bindings=(Binding(0x5018,'head_a','read'),Binding(0x5038,'head_b','read'),
                                  Binding(0x4020,'join' if tails else 'output','write'))))
    for index,(_,_,kernel) in enumerate(tails):
        name=tail_names[index]
        source='join' if index==0 else tail_names[index-1]
        target='output' if index==len(tails)-1 else name
        stages.append(conv_stage(name,source,target,kernel,3,q_tails[index],
                                 q_tails[index-1].output_zero_point if index else join_zero))
    if tails:
        output_scale=q_tails[-1].output_scale;output_zero_point=q_tails[-1].output_zero_point
    tensors=[TensorSpec('input0',ROLE_INPUT,LAYOUT_PACKED_U8,(1,8,8,3),input_bytes,0),
             TensorSpec('stem',ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,hidden),surface,0),
             TensorSpec('head_a',ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,3),surface,0),
             TensorSpec('head_b',ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,3),surface,1)]
    if tails:
        tensors.append(TensorSpec('join',ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,3),surface,0))
        for name in tail_names[:-1]:
            tensors.append(TensorSpec(name,ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,3),surface,0))
    tensors.append(TensorSpec('output',ROLE_OUTPUT,LAYOUT_NATIVE16,(1,8,8,3),surface,0))
    binary,composed=compose(stages,tensors,input_scale=1.0,input_zero_point=0,
                            output_scale=output_scale,output_zero_point=output_zero_point,
                            serial=serial)
    offsets=composed['tensor_offsets']
    if offsets['head_b']!=offsets['head_a']+surface:
        raise ValueError('diamond join requires adjacent head buffers')
    meta=dict(profile='diamond-join' if not tails else 'diamond-tail',
        engine_runs=composed['engine_runs'],join=kind,hidden_channels=hidden,
        stem_kernel=1,join_scale=float(join_scale),join_zero_point=int(join_zero),
        head_kernels=[k for _,_,k in heads],tail_kernels=[k for _,_,k in tails],
        operand_zero_points=list(operand_zero_points),
        input_scale=1.0,input_zero_point=0,output_scale=output_scale,output_zero_point=output_zero_point,
        head_scales=[q.output_scale for q in q_heads],head_zero_points=[q.output_zero_point for q in q_heads],
        shape_nhwc=[1,8,8,3],output_shape_nhwc=[1,8,8,3],output_tensors=['output'],
        schedule=composed['schedule'],
        tensor_lifetimes=composed['tensor_lifetimes'],
        tensor_offsets={name:offsets[name] for name in sorted(offsets)},
        lifetime_bytes=composed['live_bytes'],allocated_bytes=composed['allocated_bytes'],
        stem_quantization=stem_meta['quantization'],head_quantization=[q.metadata() for q in q_heads],
        tail_quantization=[q.metadata() for q in q_tails],
        tail_weights=[w.tolist() for w,_,_ in tails],tail_bias=[b.tolist() for _,b,_ in tails],
        weights=[w.tolist() for w,_,_ in heads],bias=[b.tolist() for _,b,_ in heads],
        stem_weights=w1.tolist(),stem_bias=constants[stem.input[2]].tolist())
    meta.update({key:value for key,value in composed.items()
                 if key not in ('profile','engine_runs','schedule','tensor_lifetimes',
                                'tensor_offsets','live_bytes','allocated_bytes')})
    return binary,meta

def _classify_join_chain(nodes,externals=()):
    """Split a candidate chain into its parts, or return None.

    Heads are Conv nodes reading the shared stem output, joins are elementwise
    nodes folding them, the tail is the optional `[Conv, Relu]* Conv` chain and
    `scale` is an optional final `Mul(result, external)` runtime per-channel scale.
    Both topological node orders seen in practice are accepted: heads and their
    joins interleaved, or all heads first.
    """
    ops=[node.op_type for node in nodes]
    if len(ops)<6 or ops[0]!='Conv' or any(node.domain not in ('','ai.onnx') for node in nodes):
        return None
    if ops[1]=='Relu':
        if len(ops)<7:return None
        stem_end=2
    else:
        stem_end=1
    stem_out=nodes[stem_end-1].output[0]
    heads=[];joins=[];tail=[];scale=None
    for node in nodes[stem_end:]:
        if node.op_type=='Conv' and not joins and not tail and scale is None and list(node.input)[:1]==[stem_out]:
            heads.append(node)
        elif node.op_type in JOIN_TYPES and not tail and heads:
            if scale is None and len(node.input)==2 and node.input[1] in externals:
                # The runtime per-channel scale reads a declared graph input.
                scale=node
            else:
                joins.append(node)
        else:
            tail.append(node)
    if len(heads)<3 or len(joins)!=len(heads)-1:
        return None
    if scale is not None and tail:
        return None
    if tail and not (len(tail)%2==1 and all(node.op_type==('Conv' if index%2==0 else 'Relu')
                                           for index,node in enumerate(tail))):
        return None
    return dict(stem=nodes[:stem_end],heads=heads,joins=joins,tail=tail,scale=scale)

def parse_join_chain(nodes,externals=()):
    """Recognise a stem fan-out folded by chained joins and describe it.

    Every head consumes the shared stem output and each extra head feeds a join
    that also reads the previous join result, so `n` heads need `n-1` joins. An
    optional final `Mul(result, external)` adds a runtime per-channel scale. The
    two-head single-join diamond stays with `compile_diamond`, so only
    ``head_count >= 3`` is recognised. Returns a shape dict, or ``None`` when the
    nodes are not a chain (the caller then falls through to the other bounded
    profiles).
    """
    parts=_classify_join_chain(nodes,externals)
    if parts is None:return None
    # Only the left fold `((h0 op h1) op h2) ...` belongs to the chain emitter; any
    # other expression over produced tensors is the general join DAG.
    heads=parts['heads'];joins=parts['joins']
    for position,join in enumerate(joins):
        primary=heads[0].output[0] if position==0 else joins[position-1].output[0]
        if list(join.input)!=[primary,heads[position+1].output[0]]:
            return None
    tail_len=len(parts['tail'])+(1 if parts['scale'] is not None else 0)
    return dict(head_count=len(heads),join_count=len(joins),
                stem_end=len(parts['stem']),tail_start=len(nodes)-tail_len,
                runtime_scale=parts['scale'] is not None)

def compile_join_chain(model,output_range=None,operand_zero_points=(0,0),asymmetric_depthwise=False,
                       operand_scale=1/127,serial=True):
    """A shared stem fanned out to 3..8 heads, folded by chained elementwise joins.

    This is the scheduler's first variable fan-out: `n` heads all read one stem
    output and `n-1` independently generated elementwise joins fold them left to
    right before an optional `[Conv, Relu]* Conv` tail. Join kinds may be mixed:
    a Mul join folds two free operand scales into its conversion, while
    Add/Sub/Max require a shared scale, so the next head is re-quantized onto the
    running result. The arena is placed by `open_rknpu.liveness` from the task
    read/write sets, so a dead buffer is reused by a later join when the lifetimes
    are disjoint. A head may be dense or a group-3 depthwise Conv: the depthwise task
    is the verified standalone program with its four addresses relocated, exactly as
    in `open_rknpu.depthwise_join`, so mixed head families compose by tensor name.
    Bounds: RGB 8x8 external tensors, 3..8 heads with three output channels, dense
    hidden 3..16 (depthwise heads require a three-channel stem), depthwise kernels
    1x1/3x3/5x5, zero-point-zero join operands.
    """
    onnx.checker.check_model(model)
    if any(n.domain not in ('','ai.onnx') for n in model.graph.node):
        raise ValueError('join chain requires default-domain nodes')
    parts=_classify_join_chain(list(model.graph.node),{v.name for v in model.graph.input})
    if parts is None:
        raise ValueError('join chain requires stem[,Relu] then head,Mul pairs')
    if tuple(operand_zero_points)!=(0,0):
        raise ValueError('chained Mul joins require zero-centered operands')
    g=model.graph
    stem_nodes=parts['stem'];head_nodes=parts['heads'];join_nodes=parts['joins']
    tail_nodes=parts['tail'];scale_node=parts['scale']
    runtime_scale=scale_node is not None
    head_count=len(head_nodes);join_count=len(join_nodes)
    if head_count>8:
        raise ValueError('join chain supports 3..8 heads')
    stem=stem_nodes[0];t=stem_nodes[-1].output[0]
    if len(stem_nodes)==2 and (stem_nodes[1].attribute or list(stem_nodes[1].input)!=list(stem.output)):
        raise ValueError('join chain stem Relu must consume the stem output')
    for head in head_nodes:
        if head.op_type!='Conv' or head.input[0]!=t:
            raise ValueError('join chain heads must be Conv nodes reading the stem output')
    if any(join.attribute or join.op_type not in JOIN_TYPES for join in join_nodes):
        raise ValueError('join chain joins must be attribute-free Add/Mul/Sub/Max nodes')
    joined=[]
    for position,join in enumerate(join_nodes):
        primary=head_nodes[0].output[0] if position==0 else joined[position-1]
        if list(join.input)!=[primary,head_nodes[position+1].output[0]]:
            raise ValueError('join chain joins must consume the previous result and the next head in order')
        joined.append(join.output[0])
    if any(node.attribute for index,node in enumerate(tail_nodes) if index%2==1):
        raise ValueError('join chain tail Relu must not carry attributes')
    if tail_nodes:
        if tail_nodes[0].input[0]!=joined[-1]:
            raise ValueError('join chain tail must consume the last join output')
        for index,node in enumerate(tail_nodes):
            if index and node.input[0]!=tail_nodes[index-1].output[0]:
                raise ValueError('join chain tail nodes must be connected in order')
    final=tail_nodes[-1].output[0] if tail_nodes else joined[-1]
    scale_input=None
    runtime_residual=False
    if runtime_scale:
        scale_names=[v for v in g.input if v.name!=g.input[0].name]
        if (scale_node.op_type not in JOIN_TYPES or scale_node.attribute or len(scale_names)!=1
            or list(scale_node.input)!=[joined[-1],scale_names[0].name]):
            raise ValueError('join chain runtime tail must combine the last join result with one external input')
        scale_input=scale_names[0]
        runtime_residual=scale_node.op_type!='Mul'
        final=scale_node.output[0]
    expected_inputs=2 if runtime_scale else 1
    if len(g.input)!=expected_inputs or len(g.output)!=1 or final!=g.output[0].name:
        raise ValueError('join chain requires one image input and one output')
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*g.input,*g.output,*g.value_info]}
    for v in (g.input[0],g.output[0]):
        if v.type.tensor_type.elem_type!=1 or shapes.get(v.name)!=[1,3,8,8]:
            raise ValueError('join chain external tensors must be float32 [1,3,8,8]')
    if runtime_scale:
        wanted=[1,3,8,8] if runtime_residual else [1,3,1,1]
        if scale_input.type.tensor_type.elem_type!=1 or shapes.get(scale_input.name)!=wanted:
            raise ValueError('join chain runtime tail requires a float32 %s input'%wanted)
    # Internal native16 storage of the runtime operand (the caller's api view is
    # the logical HWC byte count; the loader packs it into the 16-byte rows).
    tail_operand_bytes=8*8*16 if runtime_residual else 64
    for head in head_nodes:
        if shapes.get(head.output[0])!=[1,3,8,8]:
            raise ValueError('join chain heads must produce float32 [1,3,8,8]')
    for node in tail_nodes:
        if node.op_type=='Conv' and shapes.get(node.output[0])!=[1,3,8,8]:
            raise ValueError('join chain tail must produce float32 [1,3,8,8]')
    constants={x.name:nh.to_array(x) for x in g.initializer}
    for node in (stem,*head_nodes,*[n for n in tail_nodes if n.op_type=='Conv']):
        if len(node.input)!=3 or any(name not in constants for name in node.input[1:]):
            raise ValueError('join chain convolutions require constant float32 weights and bias')
        for name in node.input[1:]:
            if constants[name].dtype!=np.float32:
                raise ValueError('join chain constants must be float32')
    w1=constants[stem.input[1]];hidden=w1.shape[0] if w1.ndim==4 else 0
    if (w1.ndim!=4 or w1.shape!=(hidden,3,1,1) or not 3<=hidden<=16
        or constants[stem.input[2]].shape!=(hidden,)):
        raise ValueError('join chain stem must be a 1x1 Conv with hidden channels 3..16')
    stem_supported={'kernel_shape':[1,1],'pads':[0,0,0,0],'strides':[1,1],'dilations':[1,1],'group':1}
    stem_attrs={a.name:h.get_attribute_value(a) for a in stem.attribute}
    if any(k not in stem_supported or v!=stem_supported[k] for k,v in stem_attrs.items()):
        raise ValueError('unsupported stem attributes')
    heads=[]
    for head in head_nodes:
        w=constants[head.input[1]];kernel=w.shape[2] if w.ndim==4 else 0
        attrs={a.name:h.get_attribute_value(a) for a in head.attribute}
        group=int(attrs.get('group',1))
        if group==1:
            if (w.ndim!=4 or kernel not in (1,3) or w.shape!=(3,hidden,kernel,kernel)
                or constants[head.input[2]].shape!=(3,)):
                raise ValueError('join chain heads must be dense 3-output 1x1/3x3 Conv with hidden inputs')
            supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':1}
            if any(k not in supported or v!=supported[k] for k,v in attrs.items()):
                raise ValueError('unsupported head attributes')
            if kernel==3 and attrs.get('pads')!=[1,1,1,1]:
                raise ValueError('join chain 3x3 heads require symmetric pad1')
            heads.append(dict(kind='dense',weights=w,bias=constants[head.input[2]],kernel=kernel))
        else:
            if (group!=3 or hidden!=3 or w.ndim!=4 or kernel not in (1,3,5)
                or w.shape!=(3,1,kernel,kernel) or constants[head.input[2]].shape!=(3,)):
                raise ValueError('join chain depthwise heads require a three-channel stem and group3 1x1/3x3/5x5 weights')
            supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':3}
            if any(k not in supported or v!=supported[k] for k,v in attrs.items()):
                raise ValueError('unsupported depthwise head attributes')
            heads.append(dict(kind='depthwise',weights=w,bias=constants[head.input[2]],
                              kernel=kernel,node=head))
    tails=[]
    for node in [n for n in tail_nodes if n.op_type=='Conv']:
        w=constants[node.input[1]];kernel=w.shape[2] if w.ndim==4 else 0
        if (w.ndim!=4 or kernel not in (1,3) or w.shape!=(3,3,kernel,kernel)
            or constants[node.input[2]].shape!=(3,)):
            raise ValueError('join chain tail layers must be dense 3-output 1x1/3x3 Conv with three inputs')
        attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
        supported={'kernel_shape':[kernel,kernel],'pads':[kernel//2]*4,'strides':[1,1],'dilations':[1,1],'group':1}
        if any(k not in supported or v!=supported[k] for k,v in attrs.items()):
            raise ValueError('unsupported tail attributes')
        if kernel==3 and attrs.get('pads')!=[1,1,1,1]:
            raise ValueError('join chain 3x3 tail layers require symmetric pad1')
        tails.append((w,constants[node.input[2]],kernel))
    first_graph=h.make_graph(stem_nodes,'stem',[g.input[0]],
        [h.make_tensor_value_info(t,1,[1,hidden,8,8])],
        [nh.from_array(constants[name],name) for name in stem.input[1:]])
    first=h.make_model(first_graph,opset_imports=list(model.opset_import));first.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        stem_path=Path(tmp)/'stem.onnx';onnx.save(first,stem_path)
        stem_data,stem_meta=compile_model(stem_path)
    stem_scale=stem_meta['output_scale'];stem_zp=stem_meta['output_zero_point']
    # Join kinds decide the operand scales. A Mul join folds two free operand
    # scales into its own conversion; Add/Sub/Max need both operands on one shared
    # scale (the running result), so the next head is re-quantized onto it. Every
    # operand uses zero point zero, which is what the verified join tasks read.
    kinds=[join.op_type for join in join_nodes]
    if output_range is not None and not tails and kinds[-1]!='Mul':
        raise ValueError('the join chain output override requires a Mul join or a Conv tail')
    if output_range is not None and runtime_residual:
        raise ValueError('the join chain output override is unsupported for a runtime residual tail')
    # A dense head quantizes directly; a depthwise head's natural range comes from
    # the verified standalone depthwise emitter, which also supplies the task
    # program this container relocates.
    from .depthwise import compile_depthwise

    def adjusted_scale(scale,zero_point):
        return float(np.float32(scale*max(128+zero_point,127-zero_point)/127))

    natural=[];adjusted=[];branch_models=[]
    for head in heads:
        if head['kind']=='dense':
            q=native_quantize(head['weights'],head['bias'],stem_scale,stem_zp,None)
            natural.append(q);adjusted.append(adjusted_scale(q.output_scale,q.output_zero_point))
            branch_models.append(None)
        else:
            standalone=h.make_model(h.make_graph(stem_nodes+[head['node']],'depthwise_head',[g.input[0]],
                [h.make_tensor_value_info(head['node'].output[0],1,[1,3,8,8])],
                [nh.from_array(constants[name],name) for node in (stem,head['node']) for name in node.input[1:]],
                value_info=[h.make_tensor_value_info(t,1,[1,hidden,8,8])]),
                opset_imports=list(model.opset_import))
            standalone.ir_version=model.ir_version
            _,meta=compile_depthwise(standalone)
            natural.append(meta)
            adjusted.append(adjusted_scale(meta['output_scale'],meta['output_zero_point']))
            branch_models.append(standalone)
    final_scales=[None]*head_count
    if kinds[0]=='Mul':
        final_scales[0],final_scales[1]=adjusted[0],adjusted[1]
        running=float(np.float32(128*adjusted[0]*adjusted[1]))
    else:
        shared=float(np.float32(max(adjusted[0],adjusted[1])))
        final_scales[0]=final_scales[1]=shared
        running=float(np.float32(2*shared))
    for position in range(1,join_count):
        head_index=position+1
        if kinds[position]=='Mul':
            final_scales[head_index]=adjusted[head_index]
            running=float(np.float32(128*running*adjusted[head_index]))
        else:
            final_scales[head_index]=running
            running=float(np.float32(2*running))
    q_heads=[]
    for head,scale,standalone in zip(heads,final_scales,branch_models):
        if head['kind']=='dense':
            q_heads.append(native_quantize(head['weights'],head['bias'],stem_scale,stem_zp,
                                           dict(scale=scale,zero_point=0)))
        else:
            branch,branch_meta=compile_depthwise(standalone,output_range=dict(scale=scale,zero_point=0),
                                                 asymmetric_pair=asymmetric_depthwise)
            info=decode_sequence(branch)
            payload=branch[96+16*info['task_count']:]
            words=struct.unpack_from('<126Q',payload,0x440)
            regs={word&65535:(word>>16)&0xffffffff for word in words}
            q_heads.append(dict(quantization=branch_meta['depthwise'],program=regs,payload=payload,
                                weight_source=regs[0x1110],bias_source=regs[0x5020],
                                weight_size=head['kernel']*head['kernel']*32,bias_size=((3+3)//4)*24))
    # Program layout: stem, one 126-word program per head, one 78-word program per
    # join, one program per tail layer, then the packed weights/biases and the
    # liveness-placed arena. Each 126-word program owns 0x440 bytes, including its
    # four-word task descriptor, so the next program starts after that.
    join_bases=[];cursor=_align(0x440*(head_count+1))
    for _ in range(join_count):
        join_bases.append(cursor);cursor=_align(cursor+(78+4)*8)
    tail_bases=[_align(join_bases[-1]+(78+4)*8)+index*0x440 for index in range(len(tails))]
    scale_basis=_align(join_bases[-1]+(78+4)*8) if runtime_scale else None
    if runtime_scale:
        weights1=_align(scale_basis+(78+4)*8)
    else:
        weights1=_align((tail_bases[-1]+0x410) if tail_bases else cursor)
    weight_size1=_align(1*1*_align(hidden,4)*4)
    bias1=weights1+weight_size1;bias_size1=((hidden+3)//4)*32
    head_specs=[];cursor=_align(bias1+bias_size1)
    for head in heads:
        kernel=head['kernel']
        if head['kind']=='dense':
            wsize=_align(3*16*kernel*kernel);bsize=32
        else:
            wsize=_align(kernel*kernel*32);bsize=((3+3)//4)*24
        head_specs.append((cursor,cursor+wsize,wsize,bsize,kernel));cursor=_align(cursor+wsize+bsize)
    tail_specs=[]
    for w,b,kernel in tails:
        wsize=_align(3*16*kernel*kernel);bsize=32
        tail_specs.append((cursor,cursor+wsize,wsize,bsize,kernel));cursor=_align(cursor+wsize+bsize)
    payload_size=_align(cursor)
    surface=8*8*16
    input_off=_align(payload_size,4096);input_bytes=8*16*3
    tail_names=[f'tail{index}' for index in range(len(tails))]
    join_names=[f'join{index}' for index in range(join_count)]
    bindings=[Access(reads=('input0',),writes=('stem',))]
    for index in range(head_count):
        bindings.append(Access(reads=('stem',),writes=(f'head{index}',)))
    for position in range(join_count):
        primary='head0' if position==0 else join_names[position-1]
        target=join_names[position] if (position<join_count-1 or tails or runtime_scale) else 'output'
        bindings.append(Access(reads=(primary,f'head{position+1}'),writes=(target,)))
    for index in range(len(tails)):
        source=join_names[-1] if index==0 else tail_names[index-1]
        target=tail_names[index] if index<len(tails)-1 else 'output'
        bindings.append(Access(reads=(source,),writes=(target,)))
    if runtime_scale:
        bindings.append(Access(reads=(join_names[-1],'scale'),writes=('output',)))
    sizes={'stem':surface}
    for index in range(head_count):sizes[f'head{index}']=surface
    for position in range(join_count):
        if position<join_count-1 or tails or runtime_scale:sizes[join_names[position]]=surface
    for name in tail_names[:-1]:sizes[name]=surface
    defined=('input0','scale') if runtime_scale else ('input0',)
    order,intervals,offsets=plan(bindings,sizes,start=_align(input_off+input_bytes),defined=defined)
    if runtime_scale:
        # Conservative placement: the elementwise scale task reading a slot that an
        # earlier task also wrote returned stale data on the board (see the
        # investigation log), so every internal of this profile gets a fresh slot.
        offsets={}
        cursor=_align(input_off+input_bytes)
        for binding in bindings:
            name=binding.writes[0]
            if name in sizes:
                offsets[name]=cursor;cursor=_align(cursor+sizes[name])
    internals_end=max(offsets[name]+sizes[name] for name in sizes)
    scale_off=_align(internals_end) if runtime_scale else None
    output_off=_align(scale_off+tail_operand_bytes) if runtime_scale else _align(internals_end)
    layout=dict(offsets);layout['input0']=input_off;layout['output']=output_off
    layout_sizes=dict(sizes);layout_sizes['input0']=input_bytes;layout_sizes['output']=surface
    if runtime_scale:
        layout['scale']=scale_off;layout_sizes['scale']=tail_operand_bytes
    for external in (('input0','scale','output') if runtime_scale else ('input0','output')):
        offset=layout[external];size=layout_sizes[external]
        for name in layout:
            if name==external:continue
            other=layout[name];other_size=layout_sizes[name]
            if offset<other+other_size and other<offset+size:
                raise ValueError('join chain external tensor %s overlaps %s'%(external,name))
    # Output quantization is address-independent, so resolve it before composing.
    join_scales=[];join_zeros=[];previous=None
    for position,join in enumerate(join_nodes):
        running=final_scales[0] if position==0 else previous
        operands=[running,final_scales[position+1]] if kinds[position]=='Mul' else [running,running]
        last=position==join_count-1
        _,join_scale,join_zero=_join_fields(kinds[position],8,8,3,0,0,0,surface,operands,(0,0),
                                            output_range if (last and not tails) else None)
        join_scales.append(join_scale);join_zeros.append(join_zero);previous=join_scale
    q_tails=[];tail_previous=(join_scales[-1],join_zeros[-1])
    for index,(w,b,_) in enumerate(tails):
        selected=output_range if index==len(tails)-1 else None
        q=native_quantize(w,b,tail_previous[0],tail_previous[1],selected)
        # The tail is `[Conv, Relu]* Conv`: every non-final layer applies its activation
        # in its own program (registers 0x4060/0x406c/0x40e0), like the chain family, and
        # the reference follows `q.relu`. Before this the join chain dropped the Relu.
        q.relu=index<len(tails)-1
        q_tails.append(q);tail_previous=(q.output_scale,q.output_zero_point)
    scale_words=None;conversion=shift=None
    if runtime_residual:
        _,output_scale,output_zero_point=_join_fields(scale_node.op_type,8,8,3,0,0,0,surface,
                                                      [join_scales[-1]]*2,(0,0),None)
    elif runtime_scale:
        from .elementwise import compile_standalone_mul,mul_output_conversion
        probe=h.make_model(h.make_graph([h.make_node('Mul',['probe_in','probe_in'],['probe_out'])],'probe',
            [h.make_tensor_value_info('probe_in',1,[1,3,8,8])],
            [h.make_tensor_value_info('probe_out',1,[1,3,8,8])]),opset_imports=[h.make_opsetid('',13)])
        probe.ir_version=8
        probe_binary,_=compile_standalone_mul(probe)
        probe_info=decode_sequence(probe_binary)
        scale_words=struct.unpack_from('<78Q',probe_binary[96+16*probe_info['task_count']:],0x880)
        output_scale,output_zero_point,conversion,shift=mul_output_conversion(
            join_scales[-1]*operand_scale,output_range)
    elif tails:
        output_scale=q_tails[-1].output_scale;output_zero_point=q_tails[-1].output_zero_point
    else:
        output_scale=join_scales[-1];output_zero_point=join_zeros[-1]
    stem_src={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from('<126Q',stem_data)}

    def stem_fill(payload,offset):
        payload[offset:offset+weight_size1]=stem_data[stem_src[0x1110]:stem_src[0x1110]+weight_size1]
        payload[offset+weight_size1:offset+weight_size1+bias_size1]=stem_data[
            stem_src[0x5020]:stem_src[0x5020]+bias_size1]

    def stem_fields(addresses,constant_offsets):
        values=dict(stem_src)
        values.update({0x1110:constant_offsets['stem'],0x5020:constant_offsets['stem']+weight_size1,
                       0x4020:addresses['stem'],0x1070:addresses['input0']})
        return values

    # One stage per task. `open_rknpu.compose` assigns program slots and constant
    # blocks in declared order, places the arena (fresh slots for the runtime-scale
    # profile, liveness reuse otherwise), substitutes the planned addresses and returns
    # the declared binding view. The declaration order is the emitted order, so the
    # container is byte-identical to the hand-assembled one (tests/test_composer_emitters.py).
    stages=[Stage(name='stem',family='native-conv',reads=('input0',),writes=('stem',),
                  fields=stem_fields,constants=(ConstantSpec('stem',weight_size1+bias_size1,stem_fill),),
                  bindings=(Binding(0x1070,'input0','read'),Binding(0x4020,'stem','write')))]
    for index,(head,q) in enumerate(zip(heads,q_heads)):
        name=f'head{index}';kernel=head['kernel']
        if head['kind']=='dense':
            wsize=head_specs[index][2];bsize=head_specs[index][3]

            def fill(payload,offset,kernel=kernel,q=q,wsize=wsize):
                _pack_native_head(payload,kernel,hidden,offset,offset+wsize,q)

            def fields(addresses,constant_offsets,kernel=kernel,q=q,wsize=wsize,name=name):
                values=_native_fields(kernel,hidden,constant_offsets[name],
                                      constant_offsets[name]+wsize,addresses[name],
                                      addresses['stem'],q)
                # The CNA border path injects the input tensor's zero point, so a head
                # must declare the stem grid it actually reads.
                values[0x1184]=int(stem_zp)&0xffff
                return values
        else:
            # The relocated section's slot comes from `head_specs` (a 64-byte aligned
            # weight slot) while only the verified program's own block is copied into it.
            program=q['program'];payload=q['payload']
            weight_source=q['weight_source'];bias_source=q['bias_source']
            wsize=head_specs[index][2];bsize=head_specs[index][3]
            copy_weight=q['weight_size'];copy_bias=q['bias_size']

            def fill(payload_out,offset,payload=payload,weight_source=weight_source,
                     bias_source=bias_source,wsize=wsize,copy_weight=copy_weight,
                     copy_bias=copy_bias):
                payload_out[offset:offset+copy_weight]=payload[weight_source:weight_source+copy_weight]
                payload_out[offset+wsize:offset+wsize+copy_bias]=payload[bias_source:bias_source+copy_bias]

            def fields(addresses,constant_offsets,program=program,name=name,wsize=wsize):
                # Verified depthwise program with its addresses and input zero point
                # relocated into this container.
                values=dict(program)
                values.update({0x1070:addresses['stem'],0x4020:addresses[name],
                               0x1110:constant_offsets[name],
                               0x5020:constant_offsets[name]+wsize,
                               0x1184:int(stem_zp)&0xffff})
                return values

        stages.append(Stage(name=name,family='native-conv',reads=('stem',),writes=(name,),
                            fields=fields,constants=(ConstantSpec(name,wsize+bsize,fill),),
                            bindings=(Binding(0x1070,'stem','read'),Binding(0x4020,name,'write'))))
    for position,join in enumerate(join_nodes):
        primary='head0' if position==0 else join_names[position-1]
        target=join_names[position] if (position<join_count-1 or tails or runtime_scale) else 'output'
        last=position==join_count-1
        running=final_scales[0] if position==0 else (join_scales[position-1] if position else 0.0)
        operands=[running,final_scales[position+1]] if kinds[position]=='Mul' else [running,running]
        selected=output_range if (last and not tails) else None

        def join_fields(addresses,constant_offsets,primary=primary,position=position,target=target,
                        operands=operands,selected=selected,kind=kinds[position]):
            second=f'head{position+1}'
            values,_,_=_join_fields(kind,8,8,3,addresses[primary],addresses[second],
                                    addresses[target],surface,operands,(0,0),selected)
            return values

        stages.append(Stage(name=f'join{position}',family='elementwise',
                            reads=(primary,f'head{position+1}'),writes=(target,),fields=join_fields,
                            bindings=(Binding(0x5018,primary,'read'),
                                      Binding(0x5038,f'head{position+1}','read'),
                                      Binding(0x4020,target,'write'))))
    for index,(w,b,kernel) in enumerate(tails):
        name=tail_names[index];source=join_names[-1] if index==0 else tail_names[index-1]
        target=tail_names[index] if index<len(tails)-1 else 'output'
        q=q_tails[index];wsize=tail_specs[index][2];bsize=tail_specs[index][3]
        input_zero=q_tails[index-1].output_zero_point if index else join_zeros[-1]

        def fill(payload,offset,kernel=kernel,q=q,wsize=wsize):
            _pack_native_head(payload,kernel,3,offset,offset+wsize,q)

        def fields(addresses,constant_offsets,kernel=kernel,q=q,wsize=wsize,name=name,
                   source=source,target=target,input_zero=input_zero):
            values=_native_fields(kernel,3,constant_offsets[name],constant_offsets[name]+wsize,
                                  addresses[target],addresses[source],q)
            values[0x1184]=int(input_zero)&0xffff
            relu=bool(getattr(q,'relu',False))
            values.update({0x4060:0x12 if relu else 0x13,
                           0x406c:0 if relu else 0x80000000,
                           0x40e0:0 if relu else 0x80000000})
            return values

        stages.append(Stage(name=name,family='native-conv',reads=(source,),writes=(target,),
                            fields=fields,constants=(ConstantSpec(name,wsize+bsize,fill),),
                            bindings=(Binding(0x1070,source,'read'),Binding(0x4020,target,'write'))))
    if runtime_scale:
        if runtime_residual:
            def fields(addresses,constant_offsets):
                values,_,_=_join_fields(scale_node.op_type,8,8,3,addresses[join_names[-1]],
                                        addresses['scale'],addresses['output'],surface,
                                        [join_scales[-1]]*2,(0,0),None)
                return values
        else:
            def fields(addresses,constant_offsets):
                values={word&65535:(word>>16)&0xffffffff for word in scale_words}
                values.update({0x5018:addresses[join_names[-1]],0x4020:addresses['output'],
                               0x5038:addresses['scale'],0x5034:4,0x5040:16,
                               0x4080:output_zero_point&0xffffffff,0x4084:conversion,0x4088:shift})
                return values

        stages.append(Stage(name='scale',family='elementwise',
                            reads=(join_names[-1],'scale'),writes=('output',),fields=fields,
                            bindings=(Binding(0x5018,join_names[-1],'read'),
                                      Binding(0x5038,'scale','read'),
                                      Binding(0x4020,'output','write'))))
    tensors=[TensorSpec('input0',ROLE_INPUT,LAYOUT_PACKED_U8,(1,8,8,3),input_bytes,0)]
    if runtime_scale:
        tensors.append(TensorSpec('scale',ROLE_INPUT,LAYOUT_NATIVE16,
                                  (1,8,8,3) if runtime_residual else (1,1,1,3),
                                  tail_operand_bytes,1))
    tensors.append(TensorSpec('stem',ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,hidden),surface,0))
    for index in range(head_count):
        tensors.append(TensorSpec(f'head{index}',ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,3),surface,index))
    for position in range(join_count):
        if position<join_count-1 or tails or runtime_scale:
            tensors.append(TensorSpec(join_names[position],ROLE_INTERNAL,LAYOUT_NATIVE16,
                                      (1,8,8,3),surface,0))
    for name in tail_names[:-1]:
        tensors.append(TensorSpec(name,ROLE_INTERNAL,LAYOUT_NATIVE16,(1,8,8,3),surface,0))
    tensors.append(TensorSpec('output',ROLE_OUTPUT,LAYOUT_NATIVE16,(1,8,8,3),surface,0))
    binary,composed=compose(stages,tensors,input_scale=1.0,input_zero_point=0,
                            output_scale=output_scale,output_zero_point=output_zero_point,
                            serial=serial,reuse=not runtime_scale,
                            late_inputs=('scale',) if runtime_scale else ())
    offsets=composed['tensor_offsets']
    profile='join-chain'
    if tails:profile='join-chain-tail'
    if runtime_scale:profile='join-chain-residual' if runtime_residual else 'join-chain-scale'
    meta=dict(profile=profile,engine_runs=composed['engine_runs'],
        join=kinds[0] if join_count==1 else 'mixed',
        runtime_tail=(None if not runtime_scale else
                      (dict(kind=scale_node.op_type,residual=True) if runtime_residual else
                       dict(kind='Mul',operand_scale=float(operand_scale),conversion=conversion,shift=shift))),
        runtime_scale=(dict(operand_scale=float(operand_scale),conversion=conversion,shift=shift)
                       if runtime_scale and not runtime_residual else None),
        join_kinds=list(kinds),head_count=head_count,join_count=join_count,
        hidden_channels=hidden,stem_kernel=1,
        join_scales=[float(v) for v in join_scales],join_zero_points=[int(v) for v in join_zeros],
        join_zero_point=int(join_zeros[-1]),
        head_kernels=[head['kernel'] for head in heads],tail_kernels=[k for _,_,k in tails],
        operand_zero_points=[0,0],input_scale=1.0,input_zero_point=0,
        output_scale=output_scale,output_zero_point=output_zero_point,
        head_kinds=[head['kind'] for head in heads],
        head_scales=[(q.output_scale if head['kind']=='dense' else q['quantization']['output_scale'])
                     for head,q in zip(heads,q_heads)],
        head_zero_points=[(q.output_zero_point if head['kind']=='dense' else q['quantization']['output_zero_point'])
                          for head,q in zip(heads,q_heads)],
        depthwise_quantization=[q['quantization'] if head['kind']=='depthwise' else None
                                for head,q in zip(heads,q_heads)],
        shape_nhwc=[1,8,8,3],output_shape_nhwc=[1,8,8,3],output_tensors=['output'],
        schedule=composed['schedule'],
        tensor_lifetimes=composed['tensor_lifetimes'],
        tensor_offsets={name:offsets[name] for name in sorted(offsets)},
        lifetime_bytes=composed['live_bytes'],allocated_bytes=composed['allocated_bytes'],
        stem_quantization=stem_meta['quantization'],
        head_quantization=[(q.metadata() if head['kind']=='dense' else q['quantization'])
                           for head,q in zip(heads,q_heads)],
        tail_quantization=[q.metadata() for q in q_tails],
        tail_weights=[w.tolist() for w,_,_ in tails],tail_bias=[b.tolist() for _,b,_ in tails],
        weights=[head['weights'].tolist() for head in heads],bias=[head['bias'].tolist() for head in heads],
        stem_weights=w1.tolist(),stem_bias=constants[stem.input[2]].tolist())
    meta.update({key:value for key,value in composed.items()
                 if key not in ('profile','engine_runs','schedule','tensor_lifetimes',
                                'tensor_offsets','live_bytes','allocated_bytes')})
    return binary,meta

def join_chain_scale_reference(inputs,stem_quantization,head_quantizations,kinds,operand_codes,
                              head_kinds=None,depthwise_quantizations=None,runtime_scale=None,
                              output_zero_point=0):
    """Composed integer output for a fan-out chain plus a runtime per-channel scale.

    The scale step uses the hardware requantization formula with the container's own
    multiplier and shift: the folded scale product is not exactly 1/128 in float32,
    so `rint(a*b/128)` differs by one on a small fraction of boundary values.
    """
    from .elementwise import mul_requant_reference
    grid=diamond_reference(inputs,stem_quantization,head_quantizations,kinds,(),0,
                           head_kinds,depthwise_quantizations)
    codes=np.asarray(operand_codes,np.int32).reshape(1,1,3)
    if runtime_scale is None:
        from .elementwise import mul_reference
        return mul_reference(grid,codes)
    multiplier=runtime_scale['conversion']&0xffff
    shift=int(runtime_scale['shift'])
    return mul_requant_reference(grid,codes,multiplier,shift,output_zero_point)


def join_chain_residual_reference(inputs,stem_quantization,head_quantizations,kinds,tail_kind,residual,
                                  head_kinds=None,depthwise_quantizations=None):
    """Composed integer output for a fan-out chain plus a runtime residual operand."""
    grid=diamond_reference(inputs,stem_quantization,head_quantizations,kinds,(),0,
                           head_kinds,depthwise_quantizations)
    other=np.asarray(residual,np.int64).reshape(grid.shape)
    return join_reference(tail_kind,grid,other)


def diamond_reference(inputs,stem_quantization,head_quantizations,kind,tail_quantizations=(),join_zero_point=0,
                      head_kinds=None,depthwise_quantizations=None):
    """Composed integer output for one uint8 HWC input (stem, all heads, joins, tail).

    `kind` is either one join op for the two-head diamond or a sequence of join
    ops applied left to right for a fan-out chain. `head_kinds` marks depthwise
    heads, whose grids come from the depthwise reference instead of the dense one.
    """
    from .quantization import reference
    from .chain import native_reference
    def quantize(params):
        from .quantization import Quantization
        if isinstance(params,Quantization):
            return params
        values=dict(params)
        for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
            values[key]=np.array(values[key])
        return Quantization(**values)
    stem=quantize(stem_quantization)
    stem_int8=reference(inputs,stem)
    from .depthwise import depthwise_reference
    grids=[]
    for index,params in enumerate(head_quantizations):
        if head_kinds and head_kinds[index]=='depthwise':
            grids.append(depthwise_reference(stem_int8,quantize(depthwise_quantizations[index]),
                                             stem.output_zero_point))
        else:
            grids.append(native_reference(stem_int8,quantize(params),stem.output_zero_point))
    kinds=(kind,) if isinstance(kind,str) else tuple(kind)
    value=join_reference(kinds[0],grids[0],grids[1])
    for position,join_kind in enumerate(kinds[1:],start=2):
        value=join_reference(join_kind,value,grids[position])
    zero_point=join_zero_point
    for params in tail_quantizations:
        q=quantize(params)
        value=native_reference(value,q,zero_point)
        zero_point=q.output_zero_point
    return value
