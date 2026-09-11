"""MIT. Verify fused Conv Clip[0,6]/ReLU6 accumulator clamps."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference

base=Path(__file__).resolve().parent;root=base/'conv_clip_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110361);manifest=[]
cases=((3,3,8,8,1,[1,1],[0,0,0,0],{'scale':.1,'zero_point':0}),(3,5,7,8,3,[1,1],[1,1,1,1],{'scale':.1,'zero_point':0}),(17,8,5,6,3,[2,1],[1,1,1,1],{'scale':.05,'zero_point':-100}),(32,16,6,5,1,[2,2],[0,0,0,0],{'scale':.025,'zero_point':-128}))
for index,(ic,oc,ih,iw,k,strides,pads,out_range) in enumerate(cases):
 oh=(ih+pads[0]+pads[2]-k)//strides[0]+1;ow=(iw+pads[1]+pads[3]-k)//strides[1]+1;w=rng.uniform(-.3,.3,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32);lo=np.array(0,np.float32);hi=np.array(6,np.float32)
 nodes=[h.make_node('Conv',['input','w','b'],['conv'],kernel_shape=[k,k],strides=strides,pads=pads),h.make_node('Clip',['conv','lo','hi'],['output'])];model=h.make_model(h.make_graph(nodes,'conv_clip',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b'),nh.from_array(lo,'lo'),nh.from_array(hi,'hi')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
 path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,1.0,128,output_range=out_range);path.with_suffix('.bin').write_bytes(binary);q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()});x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);upper=int(np.clip(np.rint(6/q.output_scale)+q.output_zero_point,-128,127));expected=np.stack([np.minimum(native_input_reference(sample,q,128,pads,strides),upper) for sample in x])
 x.tofile(root/f'input{index:03}.u8');expected.tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,input_scale=1.0,input_zero_point=128,compile_output_range=out_range,input_channels=ic,output_channels=oc,kernel=k,strides=strides,pads=pads,upper_code=upper))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
