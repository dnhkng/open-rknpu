"""MIT. Verify Conv plus matching external-input residual Add."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import add_reference

base=Path(__file__).resolve().parent;root=base/'residual_add_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110364);manifest=[]
for index,(height,width,input_scale,input_zp) in enumerate(((5,8,1.,0),(8,5,.5,128),(7,7,.25,255),(8,8,2.,17))):
 w=rng.uniform(-.3,.3,(3,3,1,1)).astype(np.float32);b=rng.uniform(-.5,.5,3).astype(np.float32);model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['learned'],kernel_shape=[1,1]),h.make_node('Add',['learned','input'],['output'])],'residual_add',[h.make_tensor_value_info('input',1,[1,3,height,width])],[h.make_tensor_value_info('output',1,[1,3,height,width])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8;path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,input_scale,input_zp);path.with_suffix('.bin').write_bytes(binary);qs=[Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in branch['quantization'].items()}) for branch in meta['branches']];x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);expected=np.stack([add_reference(reference(sample,qs[0]),reference(sample,qs[1])) for sample in x]);x.tofile(root/f'input{index:03}.u8');expected.tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,input_scale=input_scale,input_zero_point=input_zp,shape=[height,width,3],rewrite=meta['elementwise_graph_rewrite']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
