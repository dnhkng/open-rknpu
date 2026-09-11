"""MIT. Independently generate broad native16 height-tiling cases."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'native_spatial_tiling_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(1103306144);manifest=[]
# H,W,Cin,Cout,K,strides,pads,relu. Every case exceeds 6144 input pixel-planes.
cases=[
    (128,128,1,1,1,(1,1),(0,0,0,0),False),
    (127,96,1,3,3,(1,1),(1,2,1,0),True),
    (96,96,3,1,7,(2,2),(3,3,3,3),False),
    (64,64,17,1,3,(1,2),(1,1,1,1),True),
    (48,96,32,1,5,(2,3),(2,2,2,2),False),
    (128,64,1,17,3,(4,1),(0,1,2,1),True),
]
for i,(ih,iw,ic,oc,k,strides,pads,relu) in enumerate(cases):
    sy,sx=strides;pt,pl,pb,pr=pads
    oh=(ih+pt+pb-k)//sy+1;ow=(iw+pl+pr-k)//sx+1
    w=rng.uniform(-.18,.18,(oc,ic,k,k)).astype(np.float32)
    b=rng.uniform(-.8,.8,oc).astype(np.float32)
    mid='conv' if relu else 'output'
    nodes=[h.make_node('Conv',['input','w','b'],[mid],kernel_shape=[k,k],pads=list(pads),strides=list(strides))]
    if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
    model=h.make_model(h.make_graph(nodes,'native_spatial_tiling',
        [h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],
        [nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)])
    model.ir_version=8;path=root/f'model{i:03}.onnx';onnx.save(model,path)
    scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_native_input(model,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(2,ih,iw,ic),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(value,q,zp,list(pads),strides) for value in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],kernel=k,
        strides=list(strides),pads=list(pads),relu=relu,input_scale=scale,input_zero_point=zp,
        tiles=meta['spatial_tiles']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
