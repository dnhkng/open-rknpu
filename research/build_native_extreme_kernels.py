"""MIT. Probe the native kernel dimension and weight-table range."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'native_extreme_kernels_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110331);manifest=[]
for i,(k,ic) in enumerate(((33,1),(33,17),(63,1),(63,17),(127,1),(127,17))):
    ih=iw=32;oc=1;sy,sx=((1,1),(1,2),(2,3))[i%3];pads=[k//2]*4
    oh=(ih+pads[0]+pads[2]-k)//sy+1;ow=(iw+pads[1]+pads[3]-k)//sx+1
    w=rng.uniform(-.05,.05,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-1,1,oc).astype(np.float32)
    node=h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],pads=pads,strides=[sy,sx])
    model=h.make_model(h.make_graph([node],'extreme_kernel',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_native_input(model,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(2,ih,iw,ic),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(value,q,zp,pads,[sy,sx]) for value in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kernel=k,strides=[sy,sx],pads=pads,input_shape=[ih,iw,ic],output_channels=oc,
        input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
