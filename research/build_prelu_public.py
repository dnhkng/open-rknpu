"""MIT. Verify per-channel fused PRelu slope tables."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.activation import prelu_reference

base=Path(__file__).resolve().parent;root=base/'prelu_public_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110362);manifest=[]
for index,(channels,height,width,input_scale,input_zp) in enumerate(((3,8,8,1.,0),(5,5,8,.5,128),(8,8,5,.25,255),(16,7,7,2.,17))):
 w=rng.uniform(-.3,.3,(channels,3,1,1)).astype(np.float32);b=rng.uniform(-.5,.5,channels).astype(np.float32);slopes=np.linspace(0,1,channels,dtype=np.float32).reshape(channels,1,1)
 model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['conv'],kernel_shape=[1,1]),h.make_node('PRelu',['conv','slopes'],['output'])],'prelu',[h.make_tensor_value_info('input',1,[1,3,height,width])],[h.make_tensor_value_info('output',1,[1,channels,height,width])],[nh.from_array(w,'w'),nh.from_array(b,'b'),nh.from_array(slopes,'slopes')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8;path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,input_scale,input_zp);path.with_suffix('.bin').write_bytes(binary);q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()});x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);expected=np.stack([prelu_reference(sample,q,np.array(meta['prelu_slopes_q14'])) for sample in x]);x.tofile(root/f'input{index:03}.u8');expected.tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,input_scale=input_scale,input_zero_point=input_zp,channels=channels,shape=[height,width,3],slopes=slopes[:,0,0].tolist(),encoded=meta['prelu_slopes_q14']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
