"""MIT. Hardware-verified 8x8/C3 stride-2 dense Conv profile."""
from pathlib import Path
import copy
import struct
import tempfile
import onnx
from onnx import helper as h
from .compiler import compile_model
from .sequence import encode_sequence


def compile_strided(model,input_scale=1.0,input_zero_point=0):
    g=model.graph
    if len(g.node) not in (1,2) or len(g.input)!=1 or len(g.output)!=1:
        raise ValueError('geometry profile requires Conv with optional Relu')
    attrs={a.name:h.get_attribute_value(a) for a in g.node[0].attribute}
    shape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    output=[d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    if len(shape)!=4 or len(output)!=4:raise ValueError('static NCHW tensors required')
    weights={v.name:onnx.numpy_helper.to_array(v) for v in g.initializer}.get(g.node[0].input[1])
    if weights is None or weights.ndim!=4:raise ValueError('constant 2D kernel required')
    kernel=weights.shape[2];strides=attrs.get('strides',[1,1]);pads=attrs.get('pads',[0]*4)
    if len(strides)!=2 or any(v not in (1,2,3,4) for v in strides):raise ValueError('strides must be 1..4 per axis')
    if len(pads)!=4 or any(v<0 or v>=kernel for v in pads):raise ValueError('padding must be 0..K-1 on each side')
    ih,iw=shape[2:];oh=(ih+pads[0]+pads[2]-kernel)//strides[0]+1;ow=(iw+pads[1]+pads[3]-kernel)//strides[1]+1
    if oh<1 or ow<1 or output!=[1,weights.shape[0],oh,ow]:raise ValueError('incorrect convolution output shape')
    dense=copy.deepcopy(model)
    kept=[a for a in dense.graph.node[0].attribute if a.name not in ('strides','pads')]
    del dense.graph.node[0].attribute[:];dense.graph.node[0].attribute.extend(kept)
    dense.graph.node[0].attribute.extend([h.make_attribute('strides',[1,1]),h.make_attribute('pads',[kernel//2]*4)])
    for d,size in zip(dense.graph.output[0].type.tensor_type.shape.dim[2:],(ih,iw)):d.dim_value=size
    del dense.graph.value_info[:]
    # compile_model validates remaining attributes, connections, constants and dtype.
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'dense.onnx';onnx.save(dense,path)
        source,meta=compile_model(path,input_scale=input_scale,input_zero_point=input_zero_point)
    data=bytearray(source)
    # Mesa/TRM stride fields plus RV1103 capture cross-check: 1028/102c are
    # output width / output pixel count here, despite the input-size register group.
    fields={0x1014:strides[0]<<3|strides[1],0x1028:ow,0x102c:oh*ow,
            0x1068:pads[1]<<8|pads[0],0x3014:(oh-1)<<16|(ow-1),0x4024:oh*ow*16,
            0x4030:ow-1,0x4034:oh-1,0x405c:(oh-1)<<16|(ow-1),
            0x40c0:oh*ow*16,0x500c:ow-1,0x5010:oh-1}
    for j in range(126):
        word=struct.unpack_from('<Q',data,j*8)[0];reg=word&65535
        if reg in fields:struct.pack_into('<Q',data,j*8,(word>>48)<<48|fields[reg]<<16|reg)
    binary=encode_sequence(data,input_shape=(ih,iw,shape[1]),output_shape=(oh,ow,output[1]),input_stride=meta['input_stride'],
        arena_bytes=((0x3000+oh*ow*16+4095)//4096)*4096,input_offset=0x2000,output_offset=0x3000,tasks=[(0,126,29,768)],
        input_scale=input_scale,input_zero_point=input_zero_point,output_scale=meta['output_scale'],
        output_zero_point=meta['output_zero_point'],serial=True)
    meta.update(output_shape_nhwc=[1,oh,ow,output[1]],conv_strides=strides,conv_pads=pads)
    return binary,meta
