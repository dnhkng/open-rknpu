"""MIT. Bounded native depthwise ConvTranspose, K2/K3 stride1/2, 8x8/C3."""
import copy,struct
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from .depthwise import compile_depthwise
from .chain import native_quantize
from .sequence import encode_sequence


def _compile_dense_transposed(model,input_scale,input_zero_point,output_range):
    g=model.graph;nodes=list(g.node);node=nodes[-1];attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
    first_count=2 if len(nodes)>1 and nodes[1].op_type=='Relu' else 1;prev=nodes[first_count-1].output[0] if nodes else ''
    values={v.name:v for v in [*g.input,*g.value_info,*g.output]};shape=[d.dim_value for d in values[prev].type.tensor_type.shape.dim] if prev in values else []
    constants={v.name:nh.to_array(v) for v in g.initializer};w=constants.get(node.input[1]) if len(node.input)>=2 else None
    if (len(nodes)!=first_count+1 or len(g.input)!=1 or len(g.output)!=1 or node.domain not in ('','ai.onnx')
        or node.op_type!='ConvTranspose' or node.input[0]!=prev or list(node.output)!=[g.output[0].name]
        or shape[:1]!=[1] or shape[2:]!=[8,8] or not 1<=shape[1]<=16 or w is None or w.ndim!=4):
        raise ValueError('dense ConvTranspose requires Conv[/Relu] stem with static 8x8 C1..16 output')
    ic=shape[1];oc=w.shape[1];bias=constants.get(node.input[2]) if len(node.input)==3 else np.zeros(oc,np.float32)
    kernel=w.shape[2] if w.ndim==4 and w.shape[2]==w.shape[3] else 0
    if kernel not in (3,5):raise ValueError('dense ConvTranspose supports square K3 or K5')
    half=kernel//2
    strides=attrs.get('strides',[1,1]);output_padding=attrs.get('output_padding',[0,0]);output_shape=attrs.get('output_shape');auto=attrs.get('auto_pad',b'NOTSET');pads=attrs.get('pads',[0,0,0,0])
    if len(strides)!=2 or any(v not in (1,2) for v in strides) or len(output_padding)!=2 or any(not 0<=output_padding[i]<strides[i] for i in range(2)):raise ValueError('dense ConvTranspose requires per-axis stride1/2 and legal output_padding')
    if output_shape is not None:
        if len(output_shape)!=2 or 'pads' in attrs:raise ValueError('dense ConvTranspose output_shape requires two dimensions and no explicit pads')
        totals=[7*strides[i]+kernel+output_padding[i]-output_shape[i] for i in range(2)]
        if any(not 0<=v<=2*kernel-2 for v in totals):raise ValueError('dense ConvTranspose output_shape outside bounded profile')
        begins=[(v+(auto==b'SAME_LOWER'))//2 for v in totals];pads=[begins[0],begins[1],totals[0]-begins[0],totals[1]-begins[1]]
    elif auto in (b'SAME_UPPER',b'SAME_LOWER'):
        totals=[kernel+output_padding[i]-strides[i] for i in range(2)];begins=[(v+(auto==b'SAME_LOWER'))//2 for v in totals];pads=[begins[0],begins[1],totals[0]-begins[0],totals[1]-begins[1]]
    elif auto==b'VALID':pads=[0,0,0,0]
    elif auto!=b'NOTSET':raise ValueError('unsupported dense ConvTranspose auto_pad')
    oh=7*strides[0]+kernel-pads[0]-pads[2]+output_padding[0];ow=7*strides[1]+kernel-pads[1]-pads[3]+output_padding[1]
    allowed={'group':1,'kernel_shape':[kernel,kernel],'strides':strides,'pads':attrs.get('pads',[0,0,0,0]),'dilations':[1,1],'output_padding':output_padding,'auto_pad':auto,'output_shape':output_shape}
    if (not 1<=oc<=16 or w.shape!=(ic,oc,kernel,kernel) or w.dtype!=np.float32 or bias is None or bias.dtype!=np.float32 or bias.shape!=(oc,)
        or any(name not in allowed or value!=allowed[name] for name,value in attrs.items())
        or attrs.get('group',1)!=1 or attrs.get('kernel_shape')!=[kernel,kernel] or attrs.get('dilations',[1,1])!=[1,1]
        or len(pads)!=4 or any(not 0<=v<kernel for v in pads)
        or [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]!=[1,oc,oh,ow]):
        raise ValueError('dense ConvTranspose supports C1..16 to C1..16, K3/K5 and per-axis stride1/2')
    lower=copy.deepcopy(model);dw=lower.graph.node[-1];names={v.name for v in lower.graph.initializer}|{v.name for v in lower.graph.input}|{n for op in lower.graph.node for n in op.output}
    weight_name='transpose_dense_proxy_w'
    while weight_name in names:weight_name+='_'
    names.add(weight_name);bias_name='transpose_dense_proxy_b'
    while bias_name in names:bias_name+='_'
    lower.graph.initializer.extend([nh.from_array(np.ones((ic,1,kernel,kernel),np.float32),weight_name),nh.from_array(np.zeros(ic,np.float32),bias_name)])
    del dw.input[:];dw.input.extend([prev,weight_name,bias_name]);dw.op_type='Conv';del dw.attribute[:];dw.attribute.extend([h.make_attribute('group',ic),h.make_attribute('kernel_shape',[kernel,kernel]),h.make_attribute('pads',[half]*4)])
    for dim,value in zip(lower.graph.output[0].type.tensor_type.shape.dim,[1,ic,8,8]):dim.dim_value=value
    del lower.graph.value_info[:];lower=onnx.shape_inference.infer_shapes(lower)
    binary,meta=compile_depthwise(lower,input_scale,input_zero_point);first=meta['first'];dense=np.transpose(w,(1,0,2,3))
    q=native_quantize(dense,bias,first['output_scale'],first['output_zero_point'],output_range).metadata();data=bytearray(binary[128:]);align=lambda value,a=64:(value+a-1)//a*a
    weights=0xb00;weight_bytes=oc*16*kernel*kernel;bias_offset=align(weights+weight_bytes);bias_bytes=((oc+3)//4)*32;payload=align(bias_offset+bias_bytes,4096)
    if len(data)<payload:data.extend(bytes(payload-len(data)))
    external=payload;intermediate=external+0x1000;output=intermediate+0x1000;surface=((oh*ow+3)//4)*64;arena=align(output+surface,4096)
    def patch(base,fields):
        for j in range(126):
            off=base+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
            if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    patch(0,{0x1070:external,0x4020:intermediate})
    fields={0x100c:0,0x1010:0x3ff,0x1014:(strides[0]-1)<<11|(strides[1]-1)<<8|9,0x1024:(ic-1)<<16|16,0x1028:ow,0x102c:oh*ow,
        0x1030:weight_bytes,0x1034:16*kernel*kernel,0x1038:(kernel<<24)|(kernel<<16)|oc,0x1048:0x1c000000,0x1068:((kernel-1-pads[1])&0xff)<<8|((kernel-1-pads[0])&0xff),0x1070:intermediate,0x1110:weights,
        0x1184:first['output_zero_point']&65535,0x1188:8*kernel*kernel,0x3010:1,0x3014:(oh-1)<<16|(ow-1),0x3018:15,0x301c:0,
        0x400c:0x1e4,0x4020:output,0x4024:surface,0x4030:ow-1,0x4034:oh-1,0x403c:(oc-1)<<16|15,
        0x4040:0x121d80,0x4048:1,0x4050:0x30000001,0x4054:0x8e000000,0x4058:3,0x405c:(oh-1)<<16|(ow-1),
        0x4080:q['output_zero_point']&0xffffffff,0x4084:q['multiplier'],0x4088:q['shift'],0x40c0:surface,
        0x500c:ow-1,0x5010:oh-1,0x5014:15,0x501c:14,0x5020:bias_offset,0x5044:0x7810}
    patch(0x440,fields);data[weights:bias_offset+bias_bytes]=bytes(bias_offset+bias_bytes-weights)
    qw=np.array(q['weights']).reshape(oc,ic,kernel,kernel);weight_zp=np.array(q['weight_zero_points']);_ = weight_zp
    for y in range(kernel):
        for x in range(kernel):
            for o in range(oc):
                row=list(qw[o,:,kernel-1-y,kernel-1-x])+[int(weight_zp[o])]*(16-ic);off=weights+((y*kernel+x)*oc+o)*16;struct.pack_into('<16b',data,off,*row)
    hardware_bias=np.array(q['biases'])
    for o in range(oc):
        block=bias_offset+(o//4)*32;lane=o%4;struct.pack_into('<i',data,block+lane*4,int(hardware_bias[o]));struct.pack_into('<h',data,block+16+lane*2,-int(weight_zp[o]));struct.pack_into('<H',data,block+24+lane*2,int(q['channel_multipliers'][o]))
    result=encode_sequence(data,input_shape=(8,8,3),output_shape=(oh,ow,oc),input_stride=16,arena_bytes=arena,input_offset=external,output_offset=output,tasks=[(0,126,29,768),(0x440,126,29,768)],input_scale=input_scale,input_zero_point=input_zero_point,output_scale=q['output_scale'],output_zero_point=q['output_zero_point'],serial=True)
    meta.update(output_scale=q['output_scale'],output_zero_point=q['output_zero_point'],transposed_quantization=q,output_shape_nhwc=[1,oh,ow,oc],transposed_profile=f'8x8-dense-c{ic}-to-c{oc}-k{kernel}-s{strides[0]}x{strides[1]}-pads{pads}')
    meta.pop('depthwise_profile',None);return result,meta


def compile_transposed(model,input_scale=1.0,input_zero_point=0,output_range=None):
    g=model.graph;node=g.node[-1];attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
    constants={v.name:nh.to_array(v) for v in g.initializer};candidate=constants.get(node.input[1]) if len(node.input)>1 else None
    group=attrs.get('group',1)
    if (isinstance(group,int) and group>1 and candidate is not None and candidate.ndim==4
        and (candidate.shape[1]>1 or group!=candidate.shape[0])):
        ic,per_group,kh,kw=candidate.shape;oc=per_group*group
        if ic%group or not 1<=ic<=16 or not 1<=oc<=16:raise ValueError('grouped ConvTranspose dense rewrite supports input/output C1..16')
        lower=copy.deepcopy(model);lower_node=lower.graph.node[-1];tensor=next(v for v in lower.graph.initializer if v.name==lower_node.input[1]);expanded=np.zeros((ic,oc,kh,kw),np.float32);inputs_per_group=ic//group
        for i in range(ic):
            gindex=i//inputs_per_group;expanded[i,gindex*per_group:(gindex+1)*per_group]=candidate[i]
        tensor.CopyFrom(nh.from_array(expanded,tensor.name));lowered=dict(attrs,group=1);del lower_node.attribute[:];lower_node.attribute.extend(h.make_attribute(name,value) for name,value in lowered.items())
        binary,meta=compile_transposed(lower,input_scale,input_zero_point,output_range);meta['transposed_rewrite']=f'group{group} expanded to zero-filled dense C{ic}-to-C{oc}'
        return binary,meta
    if (attrs.get('group',1)==1 and candidate is not None and candidate.ndim==4
        and candidate.shape[:2]!=(1,1) and attrs.get('dilations',[1,1])==[1,1]):
        # Dilated dense weights take the sparse-kernel rewrite below and come back here
        # with a unit dilation, so the dense emitter only ever sees dilation 1.
        return _compile_dense_transposed(model,input_scale,input_zero_point,output_range)
    if attrs.get('kernel_shape')==[2,2] and attrs.get('dilations',[1,1])==[2,2]:
        lower=copy.deepcopy(model);lower_node=lower.graph.node[-1];constants={v.name:v for v in lower.graph.initializer};tensor=constants.get(lower_node.input[1]);channels=attrs.get('group')
        if tensor is None:raise ValueError('constant depthwise square weights required')
        raw=nh.to_array(tensor)
        if not isinstance(channels,int) or raw.dtype!=np.float32 or raw.shape!=(channels,1,2,2):raise ValueError('constant depthwise square weights required')
        expanded=np.zeros((channels,1,3,3),np.float32);expanded[:,:,::2,::2]=raw;tensor.CopyFrom(nh.from_array(expanded,tensor.name))
        lowered=dict(attrs,kernel_shape=[3,3],dilations=[1,1]);del lower_node.attribute[:];lower_node.attribute.extend(h.make_attribute(name,value) for name,value in lowered.items())
        binary,meta=compile_transposed(lower,input_scale,input_zero_point,output_range);meta['transposed_rewrite']='K2 dilation2 expanded to sparse K3'
        return binary,meta
    if attrs.get('kernel_shape')==[3,3] and attrs.get('dilations',[1,1])==[2,2]:
        # The dilated K3 support spans five taps, so the operation is exactly a K5
        # ConvTranspose with zeros at the odd positions: zero-stuff the weights and
        # emit the K5 form (verified for depthwise and dense geometry) instead of
        # splitting the kernel. The stuffing is layout independent, so a depthwise
        # `(C,1,3,3)` and a dense `(C_in,C_out/group,3,3)` weight take the same path.
        lower=copy.deepcopy(model);lower_node=lower.graph.node[-1];constants_l={v.name:v for v in lower.graph.initializer};tensor=constants_l.get(lower_node.input[1])
        if tensor is None:raise ValueError('constant K3 dilation2 weights required')
        raw=nh.to_array(tensor)
        if raw.dtype!=np.float32 or raw.ndim!=4 or raw.shape[2:]!=(3,3):
            raise ValueError('K3 dilation2 ConvTranspose requires constant float32 (C_in,C_out/group,3,3) weights')
        expanded=np.zeros(raw.shape[:2]+(5,5),np.float32);expanded[:,:,::2,::2]=raw;tensor.CopyFrom(nh.from_array(expanded,tensor.name))
        lowered=dict(attrs,kernel_shape=[5,5],dilations=[1,1]);del lower_node.attribute[:];lower_node.attribute.extend(h.make_attribute(name,value) for name,value in lowered.items())
        binary,meta=compile_transposed(lower,input_scale,input_zero_point,output_range);meta['transposed_rewrite']='K3 dilation2 expanded to sparse K5'
        return binary,meta
    if attrs.get('dilations',[1,1])!=[1,1]:
        raise ValueError('ConvTranspose dilation is supported only for square K2 or depthwise K3 with dilation2')
    kernel_shape=attrs.get('kernel_shape')
    if (isinstance(kernel_shape,list) and len(kernel_shape)==2 and all(1<=v<=3 for v in kernel_shape)
        and kernel_shape not in ([2,2],[3,3]) and attrs.get('dilations',[1,1])==[1,1]
        and attrs.get('auto_pad',b'NOTSET')==b'NOTSET' and 'output_shape' not in attrs):
        kh,kw=kernel_shape;pads=attrs.get('pads',[0,0,0,0])
        if len(pads)!=4 or any(not 0<=pads[i]<kernel_shape[i%2] for i in range(4)):
            raise ValueError('rectangular ConvTranspose rewrite requires per-axis padding below its kernel')
        lower=copy.deepcopy(model);lower_node=lower.graph.node[-1];constants={v.name:v for v in lower.graph.initializer};tensor=constants.get(lower_node.input[1]);channels=attrs.get('group')
        if tensor is None:raise ValueError('constant depthwise rectangular weights required')
        raw=nh.to_array(tensor)
        if not isinstance(channels,int) or raw.dtype!=np.float32 or raw.shape!=(channels,1,kh,kw):raise ValueError('constant depthwise rectangular weights required')
        expanded=np.zeros((channels,1,3,3),np.float32);expanded[:,:,:kh,:kw]=raw;tensor.CopyFrom(nh.from_array(expanded,tensor.name))
        lowered=dict(attrs,kernel_shape=[3,3],dilations=[1,1],pads=[pads[0],pads[1],pads[2]+3-kh,pads[3]+3-kw]);del lower_node.attribute[:];lower_node.attribute.extend(h.make_attribute(name,value) for name,value in lowered.items())
        binary,meta=compile_transposed(lower,input_scale,input_zero_point,output_range);meta['transposed_rewrite']=f'K{kh}x{kw} expanded to sparse K3'
        return binary,meta
    kernel=attrs.get('kernel_shape',[0,0]);k=kernel[0] if len(kernel)==2 and kernel[0]==kernel[1] else 0
    strides=attrs.get('strides',[1,1]);stride_ok=len(strides)==2 and all(v in (1,2) for v in strides)
    sy,sx=strides if stride_ok else (0,0)
    pads=attrs.get('pads',[0]*4);output_padding=attrs.get('output_padding',[0,0]);output_shape=attrs.get('output_shape');auto=attrs.get('auto_pad',b'NOTSET')
    if auto not in (b'NOTSET',b'VALID',b'SAME_UPPER',b'SAME_LOWER'):raise ValueError('unsupported ConvTranspose auto_pad')
    if output_shape is not None:
        if len(output_shape)!=2 or 'pads' in attrs:raise ValueError('ConvTranspose output_shape requires two dimensions and no explicit pads')
        totals=[7*strides[i]+k+output_padding[i]-output_shape[i] for i in range(2)]
        if any(not 0<=v<=2*(k-1) for v in totals):raise ValueError('ConvTranspose output_shape outside bounded profile')
        begins=[(v+(auto==b'SAME_LOWER'))//2 for v in totals];pads=[begins[0],begins[1],totals[0]-begins[0],totals[1]-begins[1]]
    elif auto in (b'SAME_UPPER',b'SAME_LOWER'):
        totals=[k+output_padding[i]-strides[i] for i in range(2)];begins=[(v+(auto==b'SAME_LOWER'))//2 for v in totals];pads=[begins[0],begins[1],totals[0]-begins[0],totals[1]-begins[1]]
    elif auto==b'VALID':pads=[0]*4
    channels=attrs.get('group',0)
    allowed={'group':channels,'kernel_shape':[k,k],'strides':strides,'dilations':[1,1],'output_padding':output_padding,
             'pads':attrs.get('pads',[0]*4),'auto_pad':auto,'output_shape':output_shape}
    if not 1<=channels<=16 or k not in (2,3,5) or not stride_ok or len(pads)!=4 or any(not 0<=v<k for v in pads) or len(output_padding)!=2 or any(not 0<=output_padding[i]<strides[i] for i in range(2)):
        raise ValueError('ConvTranspose K2/K3/K5 stride1/2 requires padding below K and output_padding below its axis stride')
    oh=7*sy+k-pads[0]-pads[2]+output_padding[0];ow=7*sx+k-pads[1]-pads[3]+output_padding[1]
    constants={v.name:nh.to_array(v) for v in g.initializer}
    if (node.domain not in ('','ai.onnx') or len(node.input) not in (2,3) or len(g.output)!=1
        or list(node.output)!=[g.output[0].name] or attrs.get('group')!=channels
        or any(k not in allowed or v!=allowed[k] for k,v in attrs.items())
        or [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]!=[1,channels,oh,ow]):
        raise ValueError('ConvTranspose profile requires depthwise C1..16, K2/K3/K5, per-axis stride1/2 and matching output geometry')
    w=constants.get(node.input[1])
    if w is None or w.dtype!=np.float32 or w.shape!=(channels,1,k,k):raise ValueError('constant depthwise square weights required')
    bias=constants.get(node.input[2]) if len(node.input)==3 else np.zeros(channels,np.float32)
    if bias is None or bias.dtype!=np.float32 or bias.shape!=(channels,):raise ValueError('ConvTranspose bias must be float32 and match channels')
    # Reuse verified stem/quantization allocation with a zero-extended spatial kernel.
    # Zero taps preserve the symmetric quantization and interval bounds.
    lower=copy.deepcopy(model);dw=lower.graph.node[-1];dw.op_type='Conv'
    del dw.attribute[:];dw.attribute.extend([h.make_attribute('group',channels),h.make_attribute('kernel_shape',[3,3]),h.make_attribute('pads',[1]*4)])
    names={v.name for v in g.initializer}|{v.name for v in g.input}|{n for op in g.node for n in op.output}
    if len(dw.input)==2:
        name='transposed_zero_bias'
        while name in names:name+='_' 
        names.add(name);lower.graph.initializer.append(nh.from_array(bias,name));dw.input.append(name)
    if k==2:
        name='transposed_extended_weights'
        while name in names:name+='_' 
        padded=np.zeros((channels,1,3,3),np.float32);padded[:,:,:2,:2]=w
        lower.graph.initializer.append(nh.from_array(padded,name));dw.input[1]=name
    if k==5:
        # The base compile only supplies the verified stem/arena skeleton; the K5
        # tap table is written later, so a 3x3 placeholder satisfies the depthwise
        # Conv profile without changing the emitted task fields.
        name='transposed_placeholder_weights'
        while name in names:name+='_'
        names.add(name)
        placeholder=np.zeros((channels,1,3,3),np.float32);placeholder[:,:,:,:]=w[:,:,1:4,1:4]
        lower.graph.initializer.append(nh.from_array(placeholder,name));dw.input[1]=name
    for d in lower.graph.output[0].type.tensor_type.shape.dim[2:]:d.dim_value=8
    del lower.graph.value_info[:];lower=onnx.shape_inference.infer_shapes(lower)
    binary,meta=compile_depthwise(lower,input_scale,input_zero_point);data=bytearray(binary[128:])
    if meta['shape_nhwc']!=[1,8,8,3]:raise ValueError('ConvTranspose requires 8x8 RGB input')
    first=meta['first']
    if k==5:
        # K5 follows the vendor capture: asymmetric per-channel weight zero points
        # in the 32-byte tap lanes, 25 taps, and the vendor's phase/feature fields.
        q=native_quantize(w,bias,first['output_scale'],first['output_zero_point'],output_range,symmetric=False).metadata()
    else:
        q=native_quantize(w,bias,first['output_scale'],first['output_zero_point'],output_range,symmetric=True).metadata()
    qw=np.array(q['weights']).reshape(channels,k,k);stored=4 if k==2 else k;taps=stored*stored
    bias_offset=(0xb00+taps*32+31)//32*32 if k==5 else (0xd00 if k==2 else 0xc40)
    surface=((oh*ow+3)//4)*64
    fields={0x4080:q['output_zero_point']&0xffffffff,0x4084:q['multiplier'],0x4088:q['shift'],0x1014:(sy-1)<<11|(sx-1)<<8|9,0x1028:ow,0x102c:oh*ow,0x1030:taps*32,0x1034:taps*16,0x1038:(k if k==5 else stored)<<24|(k if k==5 else stored)<<16|2,
        0x1048:0x1c000000,0x1068:(((k-1-pads[1])<<8)|(k-1-pads[0])) if k==5 else ((2-pads[1])<<8|(2-pads[0])),0x1188:taps*8,0x3014:(oh-1)<<16|(ow-1),
        0x4024:surface,0x4030:ow-1,0x4034:oh-1,0x4058:7,0x405c:(oh-1)<<16|(ow-1),
        0x40c0:surface*2,0x500c:ow-1,0x5010:oh-1,0x5020:bias_offset}
    if k==5:fields.update({0x1010:0x3ff,0x3010:0xa})
    for j in range(126):
        off=0x440+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
        if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    bias_bytes=((channels+3)//4)*24;data[0xb00:bias_offset+bias_bytes]=bytes(bias_offset+bias_bytes-0xb00)
    zero_points=[int(v) for v in q['weight_zero_points']]
    for y in range(k):
        for x in range(k):
            for c in range(channels):
                tap=((y+1)*4+x+1) if k==2 else y*k+x
                pair=(-zero_points[c])&0xff if k==5 else 0
                struct.pack_into('<bB',data,0xb00+tap*32+c*2,int(qw[c,k-1-y,k-1-x]),pair)
    for c in range(channels):
        block=bias_offset+(c//4)*24;struct.pack_into('<i',data,block+(c%4)*4,q['biases'][c]);struct.pack_into('<H',data,block+16+(c%4)*2,q['channel_multipliers'][c])
        if k==5:struct.pack_into('<h',data,block+24+(c%4)*2,-zero_points[c])
    result=encode_sequence(data,input_shape=(8,8,3),output_shape=(oh,ow,channels),input_stride=16,
        arena_bytes=24576,input_offset=0x1000,output_offset=0x3000,tasks=[(0,126,29,768),(0x440,126,29,768)],
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=q['output_scale'],output_zero_point=q['output_zero_point'],serial=True)
    meta.update(output_scale=q['output_scale'],output_zero_point=q['output_zero_point'],transposed_quantization=q,output_shape_nhwc=[1,oh,ow,channels],transposed_profile=f'8x8-c{channels}-k{k}-s{sy}x{sx}-pads{pads}-output-padding{output_padding}')
    meta.pop('depthwise_profile',None)
    return result,meta
