"""Compare retained K5 hardware output with candidate bias/weight interpretations."""
import numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

_,meta=compile_sequence('research/transpose_k5_suite/model000.onnx')
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['transposed_quantization'].items()})
q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()})
x=np.fromfile('research/transpose_k5_suite/input000.u8',np.uint8)[:192].reshape(8,8,3)
actual=np.fromfile('/tmp/k5out.i8',np.int8).reshape(15,15,3)
a=reference(x,q1).astype(np.int64)-q1.output_zero_point;qw=q.weights.reshape(3,5,5)
base=np.zeros((15,15,3),np.int64)
for iy in range(8):
 for ix in range(8):
  for ky in range(5):
   for kx in range(5):
    oy=iy*2+ky-2;ox=ix*2+kx-2
    if 0<=oy<15 and 0<=ox<15:base[oy,ox]+=a[iy,ix]*qw[:,ky,kx]

def convert(acc):
 product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;product=scaled*q.multiplier
 if q.shift:product+=q.output_zero_point<<q.shift;result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
 else:result=product+q.output_zero_point
 return np.clip(result,-128,127).astype(np.int8)

def alternate(acc,mode):
 product=acc*q.channel_multipliers
 scaled=(product+(8192 if mode=='half_up' else 8191+((product>>14)&1)))>>14
 product=scaled*q.multiplier
 if mode=='offset_after':result=((product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift)+q.output_zero_point
 elif mode=='half_up':result=((product+(1<<(q.shift-1)))>>q.shift)+q.output_zero_point
 else:raise ValueError(mode)
 return np.clip(result,-128,127).astype(np.int8)

for label,bias in [('corrected',q.biases+q1.output_zero_point*q.weights.sum(axis=1)),('native',q.biases),('zero',np.zeros(3,np.int64))]:
 result=convert(base+bias);delta=actual.astype(int)-result.astype(int)
 print(label,'mismatches',np.count_nonzero(delta),'max',np.abs(delta).max(),'mae',np.abs(delta).mean())
 if label=='corrected':
  for mode in ('offset_after','half_up'):
   candidate=alternate(base+bias,mode);print(mode,'mismatches',np.count_nonzero(candidate!=actual))
for channel in range(3):
 best=(226,None)
 for delta in range(-10000,10001):
  bias=q.biases+q1.output_zero_point*q.weights.sum(axis=1);bias[channel]+=delta
  mismatch=np.count_nonzero(convert(base+bias)[:,:,channel]!=actual[:,:,channel])
  if mismatch<best[0]:best=(mismatch,delta)
 print('best fixed bias delta',channel,best)
 for py in range(2):
  for px in range(2):
   mask=np.zeros((15,15),bool);mask[py::2,px::2]=True;phase=(226,None)
   for delta in range(-10000,10001):
    bias=q.biases+q1.output_zero_point*q.weights.sum(axis=1);bias[channel]+=delta
    mismatch=np.count_nonzero(convert(base+bias)[:,:,channel][mask]!=actual[:,:,channel][mask])
    if mismatch<phase[0]:phase=(mismatch,delta)
   print(' phase',py,px,phase)
