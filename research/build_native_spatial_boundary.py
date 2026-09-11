"""MIT. Locate the first unsupported native Conv feature-map geometry."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.native import compile_native_input, native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'native_spatial_boundary_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110333);manifest=[]
cases=[(height,width) for height in (34,35,36,39,40,41,47,48,49,63,64,65,96,128) for width in (17,33)]
for i,(ih,iw) in enumerate(cases):
    ic=oc=k=1;w=rng.uniform(-.2,.2,(1,1,1,1)).astype(np.float32);b=rng.uniform(-1,1,1).astype(np.float32)
    node=h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[1,1])
    model=h.make_model(h.make_graph([node],'spatial_boundary',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,ih,iw])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_native_input(model,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(2,ih,iw,ic),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(value,q,zp,[0]*4,[1,1]) for value in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_channels=oc,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
