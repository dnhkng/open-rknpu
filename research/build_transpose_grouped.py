"""MIT. Verify grouped and multiplier ConvTranspose dense rewrites."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'transpose_grouped_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110357);manifest=[]
cases=((4,8,2),(6,4,2),(6,9,3),(8,4,4),(16,16,8),(4,8,4))
for index,(ic,oc,group) in enumerate(cases):
 w1=rng.uniform(-.3,.3,(ic,3,1,1)).astype(np.float32);b1=rng.uniform(-.5,.5,ic).astype(np.float32);w=rng.uniform(-.3,.3,(ic,oc//group,3,3)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32)
 nodes=[h.make_node('Conv',['input','w1','b1'],['stem'],kernel_shape=[1,1]),h.make_node('ConvTranspose',['stem','w','b'],['output'],group=group,kernel_shape=[3,3],strides=[2,2],pads=[1]*4)]
 model=h.make_model(h.make_graph(nodes,f'transpose_group_{group}',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,oc,15,15])],[nh.from_array(w1,'w1'),nh.from_array(b1,'b1'),nh.from_array(w,'w'),nh.from_array(b,'b')],value_info=[h.make_tensor_value_info('stem',1,[1,ic,8,8])]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
 path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary);q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['transposed_quantization'].items()});q1=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['first']['quantization'].items()});x=rng.integers(0,256,(4,8,8,3),dtype=np.uint8);qw=q.weights.reshape(oc,ic,3,3)-q.weight_zero_points[:,None,None,None];bias_units=q.biases+q1.output_zero_point*qw.reshape(oc,-1).sum(axis=1);ys=[]
 for sample in x:
  a=reference(sample,q1).astype(np.int64)-q1.output_zero_point;acc=np.broadcast_to(bias_units,(15,15,oc)).copy()
  for iy in range(8):
   for ix in range(8):
    for ky in range(3):
     for kx in range(3):
      oy=iy*2+ky-1;ox=ix*2+kx-1
      if 0<=oy<15 and 0<=ox<15:acc[oy,ox]+=a[iy,ix]@qw[:,:,ky,kx].T
  product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;product=scaled*q.multiplier
  if q.shift:product+=q.output_zero_point<<q.shift;result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
  else:result=product+q.output_zero_point
  ys.append(np.clip(result,-128,127).astype(np.int8))
 x.tofile(root/f'input{index:03}.u8');np.stack(ys).tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,input_channels=ic,output_channels=oc,group=group,rewrite=meta['transposed_rewrite']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
