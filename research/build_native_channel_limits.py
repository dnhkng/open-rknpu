"""MIT. Generate native16 channel-plane cases beyond the former C32/C64 bounds."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'native_channel_limits_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110364128);manifest=[]
cases=[(5,5,33,1,1),(6,5,48,17,3),(5,7,63,65,1),(6,6,64,128,3),
       (7,6,1,65,5),(8,7,17,80,3),(8,8,32,127,7)]
for i,(ih,iw,ic,oc,k) in enumerate(cases):
    pad=k//2;strides=((1,1),(1,2),(2,1))[i%3]
    oh=(ih+2*pad-k)//strides[0]+1;ow=(iw+2*pad-k)//strides[1]+1
    w=rng.uniform(-.12,.12,(oc,ic,k,k)).astype(np.float32)
    b=rng.uniform(-.5,.5,oc).astype(np.float32);relu=bool(i%2)
    nodes=[h.make_node('Conv',['input','w','b'],['conv' if relu else 'output'],kernel_shape=[k,k],pads=[pad]*4,strides=list(strides))]
    if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
    model=h.make_model(h.make_graph(nodes,'native_channel_limits',
        [h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),
        opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_native_input(model,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(value,q,zp,[pad]*4,strides) for value in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],kernel=k,strides=list(strides),
        input_scale=scale,input_zero_point=zp,relu=relu))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
