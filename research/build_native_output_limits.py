"""MIT. Generate native16 output-channel blocks C65..128."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.native import compile_native_input,native_input_reference
from open_rknpu.quantization import Quantization
root=Path(__file__).resolve().parent/'native_output_limits_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(1103128);manifest=[]
for i,(ih,iw,ic,oc,k) in enumerate(((7,6,1,65,5),(8,7,17,80,3),(8,8,32,127,7),(6,5,32,128,3))):
 pad=k//2;strides=((1,1),(1,2),(2,1),(2,2))[i];oh=(ih+2*pad-k)//strides[0]+1;ow=(iw+2*pad-k)//strides[1]+1
 w=rng.uniform(-.12,.12,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32);relu=bool(i%2)
 nodes=[h.make_node('Conv',['input','w','b'],['conv' if relu else 'output'],kernel_shape=[k,k],pads=[pad]*4,strides=list(strides))]
 if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
 m=h.make_model(h.make_graph(nodes,'native_output_limits',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
  [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
 p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3];binary,meta=compile_native_input(m,scale,zp);p.with_suffix('.bin').write_bytes(binary)
 q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()});x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=zp
 y=np.stack([native_input_reference(v,q,zp,[pad]*4,strides) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],kernel=k,strides=list(strides),input_scale=scale,input_zero_point=zp,relu=relu))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
