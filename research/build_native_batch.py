"""MIT. Generate static native16 Conv batches as independent serial tasks."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.native import compile_native_input,native_input_reference
from open_rknpu.quantization import Quantization
root=Path(__file__).resolve().parent/'native_batch_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110316);manifest=[]
cases=[(2,5,6,3,17,3,(1,1)),(3,8,7,32,65,1,(1,2)),(4,33,48,1,3,5,(2,3)),(2,128,128,1,1,3,(2,2)),(16,1,1,1,1,1,(1,1))]
for i,(batch,ih,iw,ic,oc,k,strides) in enumerate(cases):
 pad=k//2;oh=(ih+2*pad-k)//strides[0]+1;ow=(iw+2*pad-k)//strides[1]+1;w=rng.uniform(-.15,.15,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32)
 m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],pads=[pad]*4,strides=list(strides))],'native_batch',
  [h.make_tensor_value_info('input',1,[batch,ic,ih,iw])],[h.make_tensor_value_info('output',1,[batch,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
 p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3];binary,meta=compile_native_input(m,scale,zp);p.with_suffix('.bin').write_bytes(binary)
 q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()});x=rng.integers(0,256,(2,batch,ih,iw,ic),dtype=np.uint8);x[0]=zp
 y=np.stack([[native_input_reference(sample,q,zp,[pad]*4,strides) for sample in run] for run in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,batch=batch,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],kernel=k,strides=list(strides),input_scale=scale,input_zero_point=zp,tasks=len(meta['spatial_tiles'])*batch))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
