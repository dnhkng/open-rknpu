"""MIT. Native INT8 convolution with channel planes, independently generated."""
import struct
import numpy as np
from onnx import helper as h, numpy_helper as nh
from .chain import NATIVE, native_quantize
from .quantization import receptive_fields
from .register_profile import REGISTERS
from .sequence import encode_sequence


def native_input_reference(inputs, q, zero_point, pads=None, strides=(1,1),dilations=(1,1),
                           upper_code=None):
    """Integer output of one image-input Conv task.

    `upper_code` models the fused `Clip[0,6]`/ReLU6 upper clamp: the container programs the
    *accumulator* limit `round(6 / (max(weight_scales) * input_scale))` in registers
    `0x4028`/`0x40e4` (see `native_fields`), while the board-verified reference clamps the
    resulting code to `clip(rint(6 / output_scale) + output_zero_point, -128, 127)`, which is
    what `compile_native_input` returns as `meta["clip_upper_code"]`. Pass that value here to
    reproduce a Clip container exactly; leave it `None` for Conv/Relu.
    """
    if pads is not None:
        pt,pl,pb,pr=pads;k=q.kernel_size;dy,dx=dilations;ekh=(k-1)*dy+1;ekw=(k-1)*dx+1
        x=np.pad(inputs.astype(np.int64)-128,((pt,pb),(pl,pr),(0,0)),constant_values=zero_point-128)
        patches=np.lib.stride_tricks.sliding_window_view(x,(ekh,ekw),axis=(0,1))[::strides[0],::strides[1],:,::dy,::dx]
        patches=patches.reshape(*patches.shape[:2],-1)
    else:
        patches=receptive_fields(inputs.astype(np.int64)-128,q.kernel_size,zero_point-128)
    acc=np.einsum('hwc,oc->hwo',patches,q.weights-q.weight_zero_points[:,None])+q.biases
    if q.relu:acc=np.maximum(acc,0)
    product=acc*q.channel_multipliers
    scaled=(product+8191+((product>>14)&1))>>14
    product=scaled*q.multiplier
    if q.shift:
        product+=q.output_zero_point<<q.shift
        result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
    else:result=product+q.output_zero_point
    result=np.clip(result,-128,127)
    if upper_code is not None:
        result=np.minimum(result,upper_code)
    return result.astype(np.int8)


def native_fields(iw, tih, ic, lanes, tiles, ow, toh, oc, k, pl, tpt, sy, sx, dy, dx,
                  inp, out, weights, bias, surface, data_entries, feature_grains,
                  scan_flags, input_zero_point, q, relu, clip6, ih, input_scale):
    """Register fields for one native16 Conv task (extracted from the verified emitter).

    Geometry is explicit so the same builder serves a full image and a height strip:
    `tih` is the task's input rows, `ih` the height of the surface it reads, `surface`
    the containing output plane stride.
    """
    align=lambda x,a=64:(x+a-1)//a*a
    f=dict(NATIVE)
    f.update({0x1010:(feature_grains<<4)|scan_flags,0x1020:iw<<16|tih,0x1024:(ic-1)<<16|lanes,
            0x1014:(dy-1)<<21|(dx-1)<<16|sy<<3|sx,0x1028:ow,0x102c:toh*ow,0x1030:oc*lanes*k*k,0x1034:lanes*k*k,
            0x1038:k<<24|k<<16|oc,0x103c:data_entries<<16,0x1044:iw<<16|data_entries,
            0x1048:((tih*iw*tiles+2047)//2048)*0x04000000,0x1088:lanes,
            0x1068:pl<<8|tpt,0x1070:inp,0x107c:iw*min(4,(tih+3)//4*2),
            0x1080:((ih*iw+3)//4)*4,0x1084:iw<<16|tih,0x1110:weights,
            0x1184:(input_zero_point-128)&65535,0x1188:8*k*k*tiles,0x118c:(iw-1)<<16|(iw-1),
            0x3014:(toh-1)<<16|(ow-1),0x4020:out,0x4024:surface,0x4030:ow-1,0x4034:toh-1,
            0x3018:align(oc,16)-1,0x403c:(oc-1)<<16|(align(oc,16)-1),0x4058:((oc-1)//16)<<16|3,0x5014:align(oc,16)-1,0x405c:(toh-1)<<16|(ow-1),0x40c0:surface,
            0x4080:q.output_zero_point&0xffffffff,0x4084:q.multiplier,0x4088:q.shift,
            0x4060:0x12 if relu else 0x13,0x406c:0 if relu else 0x80000000,
            0x40e0:0 if relu else 0x80000000,0x500c:ow-1,0x5010:toh-1,0x5020:bias})
    if clip6:
        upper=round(6/(float(np.max(q.weight_scales))*input_scale))
        if not 0<upper<2**31:raise ValueError('Clip upper clamp outside INT32 accumulator range')
        f.update({0x4028:upper,0x40e4:upper})
    return f


def compile_native_input(model,input_scale=1.,input_zero_point=0,output_range=None,*,quantization=None,expose_constants=False):
    g=model.graph
    ops=[node.op_type for node in g.node]
    if len(g.input)!=1 or len(g.output)!=1 or ops not in (['Conv'],['Conv','Relu'],['Conv','Clip']):
        raise ValueError('native input profile requires one Conv')
    n=g.node[0];constants={t.name:nh.to_array(t) for t in g.initializer if t.name!=g.input[0].name};relu=len(g.node)==2;clip6=ops==['Conv','Clip']
    if relu and (g.node[1].domain not in ('','ai.onnx') or g.node[1].attribute or list(g.node[1].input)!=list(n.output)):
        if not clip6:raise ValueError('invalid native Relu connection')
    if clip6:
        activation=g.node[1]
        if (activation.domain not in ('','ai.onnx') or activation.attribute or len(activation.input)!=3 or activation.input[0]!=n.output[0] or any(name not in constants for name in activation.input[1:])
            or any(np.asarray(constants[name]).size!=1 for name in activation.input[1:])
            or float(np.asarray(constants[activation.input[1]]).reshape(-1)[0])!=0.0
            or float(np.asarray(constants[activation.input[2]]).reshape(-1)[0])!=6.0):
            raise ValueError('native Clip fusion requires constant scalar range [0,6]')
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    if (n.domain not in ('','ai.onnx') or len(n.input) not in (2,3) or n.input[0]!=g.input[0].name
        or list(g.node[-1].output)!=[g.output[0].name] or any(v not in constants for v in n.input[1:])
        or len(shape)!=4 or not 1<=shape[0]<=16 or not 1<=shape[1]<=128 or not all(1<=x<=128 for x in shape[2:])):
        raise ValueError('native Conv requires static batch1..16, H/W1..128, input C1..128, constant weights/bias')
    batch,ic,ih,iw=shape;w=constants[n.input[1]]
    b=constants[n.input[2]] if len(n.input)==3 else np.zeros(w.shape[0],np.float32)
    if w.ndim!=4:raise ValueError('native Conv weights must have rank four')
    oc,_,k,_=w.shape
    attrs={a.name:h.get_attribute_value(a) for a in n.attribute}
    pads=attrs.get('pads',[0]*4);strides=attrs.get('strides',[1,1]);dilations=attrs.get('dilations',[1,1])
    if (len(pads)!=4 or any(not 0<=v<=255 for v in pads) or len(strides)!=2 or any(v not in (1,2,3,4) for v in strides)
        or len(dilations)!=2 or any(not 1<=v<=17 for v in dilations)):
        raise ValueError('native padding/stride/dilation unsupported')
    pt,pl,pb,pr=pads;sy,sx=strides;dy,dx=dilations;ekh=(k-1)*dy+1;ekw=(k-1)*dx+1
    oh=(ih+pt+pb-ekh)//sy+1;ow=(iw+pl+pr-ekw)//sx+1
    if oh<1 or ow<1 or pt>=ekh or pb>=ekh or pl>=ekw or pr>=ekw:raise ValueError('invalid native Conv output geometry')
    allowed={'kernel_shape':[k,k],'pads':pads,'strides':strides,'dilations':dilations,'group':1}
    if (not 1<=oc<=128 or not 1<=k<=31 or k%2==0 or w.shape!=(oc,ic,k,k) or b.shape!=(oc,)
        or w.dtype!=np.float32 or b.dtype!=np.float32
        or any(key not in allowed or value!=allowed[key] for key,value in attrs.items())
        or any(v.type.tensor_type.elem_type!=1 for v in (*g.input,*g.output))
        or [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]!=[batch,oc,oh,ow]):
        raise ValueError('native Conv supports odd K1..31, explicit padding, stride 1..4, input C1..128/output C1..128')
    if not isinstance(input_zero_point,int) or not 0<=input_zero_point<=255:raise ValueError('invalid input zero point')
    selected_range=({'scale':6/255,'zero_point':-128} if clip6 and output_range is None else output_range)
    q=native_quantize(w,b,input_scale,input_zero_point-128,selected_range) if quantization is None else quantization
    if quantization is not None and (clip6 or relu or output_range is not None):
        raise ValueError('prequantized native Conv already carries output quantization')
    q.relu=relu
    q.input_scale=input_scale;q.input_zero_point=input_zero_point
    align=lambda x,a=64:(x+a-1)//a*a
    lanes=align(ic,16);tiles=lanes//16
    def tile_geometry():
        """Split output rows so each CNA task stays within the observed 6144-atom limit."""
        if ih*iw*tiles<=6144:return [(0,oh,0,ih,pt,pb)]
        result=[];oy0=0;max_input_rows=6144//(iw*tiles)
        if max_input_rows<ekh:raise ValueError('native Conv geometry cannot be safely height-tiled')
        while oy0<oh:
            chosen=None
            for oy1 in range(oh,oy0,-1):
                first=oy0*sy-pt;last=(oy1-1)*sy-pt+ekh-1
                iy0=max(0,first);iy1=min(ih,last+1)
                if iy1-iy0>max_input_rows:continue
                # CNA/DPU base addresses are 64-byte aligned native16 rows.
                if (iy0*iw)%4 or (oy0*ow)%4:continue
                if oy1<oh and ((max(0,oy1*sy-pt)*iw)%4 or (oy1*ow)%4):continue
                chosen=(oy1,iy0,iy1,max(0,-first),max(0,last-ih+1));break
            if chosen is None:raise ValueError('native Conv geometry has no aligned height tiling')
            oy1,iy0,iy1,tpt,tpb=chosen
            if oy1==oy0:raise ValueError('native Conv height tiling made no progress')
            result.append((oy0,oy1,iy0,iy1,tpt,tpb));oy0=oy1
        return result
    tile_specs=tile_geometry();task_specs=[(bi,*tile) for bi in range(batch) for tile in tile_specs]
    program_bytes=align(130*8);weights=program_bytes*len(task_specs);bias=weights+align(oc*lanes*k*k)
    payload=align(bias+((oc+3)//4)*32);inp=align(payload,4096)
    input_storage=align(ih*iw*16)*tiles;surface=align(oh*ow*16);output_storage=surface*((oc+15)//16)
    out=inp+input_storage*batch;arena=align(out+output_storage*batch,4096)
    data=bytearray(payload)
    data_entries=(iw*tiles+1)//2
    plane_grains=16//(1<<(tiles.bit_length()-1))
    feature_grains=max(min(plane_grains,(64+data_entries-1)//data_entries),(k+1)//2+1)
    for task_index,(bi,oy0,oy1,iy0,iy1,tpt,tpb) in enumerate(task_specs):
        tih=iy1-iy0;toh=oy1-oy0
        scan_flags=(4 if k==1 else 8) if tih*iw*tiles<=2048 else 0
        f=native_fields(iw,tih,ic,lanes,tiles,ow,toh,oc,k,pl,tpt,sy,sx,dy,dx,
                        inp+bi*input_storage+iy0*iw*16,out+bi*output_storage+oy0*ow*16,
                        weights,bias,surface,data_entries,feature_grains,scan_flags,
                        input_zero_point,q,relu,clip6,ih,input_scale)
        command=task_index*program_bytes
        for i,(reg,default,tag) in enumerate(REGISTERS):
            struct.pack_into('<Q',data,command+i*8,tag<<48|f.get(reg,default)<<16|reg)
        for i,(reg,value,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,29,0x81))):
            struct.pack_into('<Q',data,command+(126+i)*8,tag<<48|value<<16|reg)
    if lanes%16 or not 16<=lanes<=128:raise ValueError('native input channel tiling unsupported')
    def weight_byte(o,t,c):
        """Byte offset of one quantized weight (output o, tap t, input channel c).

        Inputs are packed in 16-lane planes and the planes are grouped in 32-lane
        parts: all taps of the first 32 lanes, then all taps of the next 32, and a
        final part holding the remaining 16 or 32 lanes. Inside a part the layout
        is [tap][lane][member]. Verified by the vendor marker captures for C48/C64
        (two parts, `capture_native_c48_*`/`capture_native_c64_channels`) and for
        C65/C128 (three and four parts, `capture_native_c65_channels`/
        `capture_native_c128_channels`).
        """
        block=o//16;lane=o%16;OCb=min(16,oc-block*16)
        part,within=divmod(c,32)
        part_size=min(32,lanes-part*32)
        base=block*16*k*k*lanes
        for previous in range(part):
            base+=k*k*min(32,lanes-previous*32)*OCb
        return base+t*part_size*OCb+lane*part_size+within
    for o in range(oc):
        rows=q.weights[o].reshape(ic,k,k)
        zero_point=int(q.weight_zero_points[o])
        for t in range(k*k):
            y,x=divmod(t,k)
            for c in range(lanes):
                value=int(rows[c,y,x]) if c<ic else zero_point
                struct.pack_into('<b',data,weights+weight_byte(o,t,c),value)
        block=bias+(o//4)*32;lane=o%4
        struct.pack_into('<i',data,block+lane*4,int(q.biases[o]))
        struct.pack_into('<h',data,block+16+lane*2,-int(q.weight_zero_points[o]))
        struct.pack_into('<H',data,block+24+lane*2,int(q.channel_multipliers[o]))
    binary=encode_sequence(data,input_shape=(ih,iw,ic),output_shape=(oh,ow,oc),input_stride=iw,
        arena_bytes=arena,input_offset=inp,output_offset=out,tasks=[(i*program_bytes,126,29,768) for i in range(len(task_specs))],
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=q.output_scale,
        output_zero_point=q.output_zero_point,serial=True,input_layout='native16',batch=batch,
        constants=([dict(name='conv.parameters',offset=weights,size=payload-weights,kind=1)] if expose_constants else ()))
    clip_upper_code=(int(np.clip(round(6/float(q.output_scale))+q.output_zero_point,-128,127))
                     if clip6 else None)
    return binary,dict(clip_upper_code=clip_upper_code,shape_nhwc=[batch,ih,iw,ic],output_shape_nhwc=[batch,oh,ow,oc],
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=q.output_scale,
        output_zero_point=q.output_zero_point,quantization=q.metadata(),profile='native16-input',fused_activation='Clip[0,6]' if clip6 else ('Relu' if relu else None),conv_pads=pads,conv_strides=strides,conv_dilations=dilations,
        spatial_tiles=[dict(output_rows=[a,b],input_rows=[c,d],pads=[e,pl,f,pr]) for a,b,c,d,e,f in tile_specs],
        mutable_constants=['conv.parameters'] if expose_constants else [])
