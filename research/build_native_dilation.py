"""MIT. Generate native unequal/larger dilation and tiled halo cases."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.native import compile_native_input,native_input_reference
from open_rknpu.quantization import Quantization
root=Path(__file__).resolve().parent/'native_dilation_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(11033232);manifest=[]
cases=[(8,9,3,5,3,(2,3),(1,1),(2,3,2,3)),(16,17,17,3,3,(3,2),(2,1),(3,2,3,2)),
 (24,25,32,1,5,(2,2),(1,2),(4,4,4,4)),(40,33,1,17,3,(8,4),(3,1),(8,4,8,4)),
 (80,80,3,1,3,(32,16),(4,4),(15,15,15,15)),(128,128,1,3,7,(3,2),(2,3),(9,6,9,6))]
cases += [(70,80,1,1,3,(31,16),(1,1),(15,15,15,15)),(70,80,1,1,3,(32,16),(1,1),(15,15,15,15))]
cases += [(50,52,1,1,3,(16,16),(1,1),(15,15,15,15)),(50,52,1,1,3,(17,16),(1,1),(15,15,15,15))]
cases += [(60,64,1,1,3,(24,16),(1,1),(15,15,15,15)),(60,64,1,1,3,(25,16),(1,1),(15,15,15,15))]
cases += [(60,64,1,1,3,(18,16),(1,1),(15,15,15,15)),(60,64,1,1,3,(20,16),(1,1),(15,15,15,15)),
          (60,64,1,1,3,(22,16),(1,1),(15,15,15,15)),(60,64,1,1,3,(23,16),(1,1),(15,15,15,15))]
for i,(ih,iw,ic,oc,k,dil,strides,pads) in enumerate(cases):
 eh=(k-1)*dil[0]+1;ew=(k-1)*dil[1]+1;oh=(ih+pads[0]+pads[2]-eh)//strides[0]+1;ow=(iw+pads[1]+pads[3]-ew)//strides[1]+1
 w=rng.uniform(-.14,.14,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32);relu=bool(i%2)
 nodes=[h.make_node('Conv',['input','w','b'],['conv' if relu else 'output'],kernel_shape=[k,k],dilations=list(dil),pads=list(pads),strides=list(strides))]
 if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
 m=h.make_model(h.make_graph(nodes,'native_dilation',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
 p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3];binary,meta=compile_native_input(m,scale,zp);p.with_suffix('.bin').write_bytes(binary)
 q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()});x=rng.integers(0,256,(3,ih,iw,ic),dtype=np.uint8);x[0]=zp
 y=np.stack([native_input_reference(v,q,zp,list(pads),strides,dil) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],kernel=k,dilations=list(dil),strides=list(strides),pads=list(pads),input_scale=scale,input_zero_point=zp,relu=relu,tiles=meta['spatial_tiles']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
