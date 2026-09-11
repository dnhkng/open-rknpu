"""MIT. Probe independently generated native Conv kernels beyond 15x15."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'native_kernel_limits_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110330);manifest=[]
cases=[(17,1,1),(17,17,3),(21,3,3),(21,32,1),(25,1,3),(25,17,1),(31,1,1),(31,3,3)]
for i,(k,ic,oc) in enumerate(cases):
    ih=iw=32;sy,sx=((1,1),(1,2),(2,3),(4,4))[i%4]
    pads=([0,0,0,0],[k//2]*4,[k-1,0,1,k-2],[1,k-1,k-2,0])[i%4]
    oh=(ih+pads[0]+pads[2]-k)//sy+1;ow=(iw+pads[1]+pads[3]-k)//sx+1
    w=rng.uniform(-.15,.15,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-1,1,oc).astype(np.float32)
    relu=bool(i%2);out='conv' if relu else 'output'
    nodes=[h.make_node('Conv',['input','w','b'],[out],kernel_shape=[k,k],pads=pads,strides=[sy,sx])]
    if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
    model=h.make_model(h.make_graph(nodes,'kernel_limits',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_native_input(model,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=0;x[1]=255;x[2]=zp
    y=np.stack([native_input_reference(value,q,zp,pads,[sy,sx]) for value in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kernel=k,strides=[sy,sx],pads=pads,input_shape=[ih,iw,ic],output_channels=oc,
        relu=relu,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
