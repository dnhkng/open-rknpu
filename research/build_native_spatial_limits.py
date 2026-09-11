"""MIT. Probe native Conv spatial geometry beyond 32 pixels."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'native_spatial_limits_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110332);manifest=[]
cases=[(33,33,1,1),(33,48,3,3),(48,33,17,1),(48,64,1,17),(64,64,3,3),(96,64,1,1),(128,128,1,1)]
for i,(ih,iw,ic,oc) in enumerate(cases):
    k=(1,3,5,7)[i%4];sy,sx=((1,1),(1,2),(2,3),(4,4))[i%4];pads=[k//2]*4
    oh=(ih+2*pads[0]-k)//sy+1;ow=(iw+2*pads[1]-k)//sx+1
    w=rng.uniform(-.2,.2,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-1,1,oc).astype(np.float32)
    relu=bool(i%2);out='conv' if relu else 'output'
    nodes=[h.make_node('Conv',['input','w','b'],[out],kernel_shape=[k,k],pads=pads,strides=[sy,sx])]
    if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
    model=h.make_model(h.make_graph(nodes,'spatial_limits',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_native_input(model,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(2,ih,iw,ic),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(value,q,zp,pads,[sy,sx]) for value in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kernel=k,strides=[sy,sx],pads=pads,input_shape=[ih,iw,ic],output_channels=oc,
        relu=relu,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
