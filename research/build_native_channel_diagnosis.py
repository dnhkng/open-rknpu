"""MIT. Isolate native Conv input-plane versus output-block channel failures."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.native import compile_native_input,native_input_reference
from open_rknpu.quantization import Quantization
root=Path(__file__).resolve().parent/'native_channel_diagnosis_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(11034817);manifest=[]
for i,(ic,oc) in enumerate(((48,1),(32,17),(48,16),(33,17),(48,17),(64,1))):
 ih,iw,k=6,5,3;w=rng.uniform(-.12,.12,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32)
 model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[3,3],pads=[1]*4)],'channel_diagnosis',
  [h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,ih,iw])],
  [nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
 p=root/f'model{i:03}.onnx';onnx.save(model,p);binary,meta=compile_native_input(model);p.with_suffix('.bin').write_bytes(binary)
 q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
 x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=0
 y=np.stack([native_input_reference(v,q,0,[1]*4) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_channels=oc,input_scale=1.,input_zero_point=0))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
