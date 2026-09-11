"""MIT. Test learned overlapping depthwise K3/S2 ConvTranspose."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
root=Path(__file__).resolve().parent;out=root/'transpose_overlap_suite';out.mkdir(exist_ok=True);rng=np.random.default_rng(11033203);manifest=[]
for i,pads in enumerate(([1,1,1,1],[0,0,0,0],[2,2,2,2],[1,0,2,1])):
 m=onnx.load(root/'depthwise_suite'/f'model{i:03}.onnx');n=m.graph.node[-1];n.op_type='ConvTranspose';del n.attribute[:];n.attribute.extend([h.make_attribute('group',3),h.make_attribute('kernel_shape',[3,3]),h.make_attribute('strides',[2,2]),h.make_attribute('pads',pads)])
 w=rng.uniform(-.3,.3,(3,1,3,3)).astype(np.float32);b=rng.uniform(-1,1,3).astype(np.float32)
 for t in m.graph.initializer:
  if t.name==n.input[1]:t.CopyFrom(nh.from_array(w,t.name))
  elif t.name==n.input[2]:t.CopyFrom(nh.from_array(b,t.name))
 oh=17-pads[0]-pads[2];ow=17-pads[1]-pads[3]
 for d,v in zip(m.graph.output[0].type.tensor_type.shape.dim[2:],[oh,ow]):d.dim_value=v
 p=out/f'model{i:03}.onnx';onnx.save(m,p);binary,meta=compile_sequence(p);p.with_suffix('.bin').write_bytes(binary);q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['transposed_quantization'].items()});q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()});x=np.fromfile(root/'depthwise_suite'/f'input{i:03}.u8',np.uint8).reshape(16,8,8,3);ys=[];bias_units=q.biases+q1.output_zero_point*q.weights.sum(axis=1);qw=q.weights.reshape(3,3,3)
 for v in x:
  a=reference(v,q1).astype(np.int64)-q1.output_zero_point;acc=np.broadcast_to(bias_units,(oh,ow,3)).copy()
  for iy in range(8):
   for ix in range(8):
    for ky in range(3):
     for kx in range(3):
      oy=iy*2+ky-pads[0];ox=ix*2+kx-pads[1]
      if 0<=oy<oh and 0<=ox<ow:acc[oy,ox]+=a[iy,ix]*qw[:,ky,kx]
  product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;ys.append(np.clip(((scaled*q.multiplier+(1<<(q.shift-1)))>>q.shift)+q.output_zero_point,-128,127).astype(np.int8))
 x.tofile(out/f'input{i:03}.u8');np.stack(ys).tofile(out/f'expected{i:03}.i8');manifest.append(dict(index=i,pads=pads,output_shape=[oh,ow,3],weights=w.tolist(),bias=b.tolist()))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
