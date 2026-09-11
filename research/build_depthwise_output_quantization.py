"""MIT. Verify independent output conversion for chained depthwise Conv."""
from pathlib import Path
import json,numpy as np,onnx
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'depthwise_output_quantization_suite';root.mkdir(exist_ok=True);manifest=[]
cases=[(4,{'scale':0.125,'zero_point':-128}),(5,{'scale':0.5,'zero_point':127}),(8,{'scale':1.25,'zero_point':-31}),(16,{'scale':2.0,'zero_point':73})]
for index,(channels,out_range) in enumerate(cases):
 source=base/f'depthwise_c{channels}_suite/model000.onnx';model=onnx.load(source);path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,output_range=out_range);path.with_suffix('.bin').write_bytes(binary)
 q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['depthwise'].items()});q1=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['first']['quantization'].items()});x=np.fromfile(base/f'depthwise_c{channels}_suite/input000.u8',np.uint8).reshape(-1,8,8,3)[:4];weights=q.weights.reshape(channels,3,3);ys=[]
 for sample in x:
  a=reference(sample,q1).astype(np.int64);padded=np.pad(a,((1,1),(1,1),(0,0)),constant_values=q1.output_zero_point);acc=np.broadcast_to(q.biases,(8,8,channels)).copy()
  for y in range(3):
   for xoff in range(3):acc+=padded[y:y+8,xoff:xoff+8]*weights[:,y,xoff]
  product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;product=scaled*q.multiplier
  if q.shift:product+=q.output_zero_point<<q.shift;result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
  else:result=product+q.output_zero_point
  ys.append(np.clip(result,-128,127).astype(np.int8))
 x.tofile(root/f'input{index:03}.u8');np.stack(ys).tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,channels=channels,input_scale=1.0,input_zero_point=0,compile_output_range=out_range))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
