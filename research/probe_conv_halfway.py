"""Discriminating Conv final-rounding probe from independent branch commands."""
import struct,json
from pathlib import Path
import numpy as np
from open_rknpu.quantization import Quantization
from open_rknpu.sequence import encode_sequence
root=Path(__file__).resolve().parent;out=root/'conv_halfway_suite';out.mkdir(exist_ok=True)
m=json.loads((root/'mul_geometry_suite/manifest.json').read_text())[7];b=m['branches'][1]
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()})
data=(root/'mul_geometry_suite/model007.bin').read_bytes()[144:]
(out/'model000.bin').write_bytes(encode_sequence(data,input_shape=(5,8,3),output_shape=(5,8,16),input_stride=16,arena_bytes=16384,input_offset=0x1000,output_offset=0x2400,tasks=[(0x440,126,29,768)],input_scale=m['input_scale'],input_zero_point=m['input_zero_point'],output_scale=q.output_scale,output_zero_point=q.output_zero_point,serial=True))
rng=np.random.default_rng(110323);candidates=rng.integers(0,256,(20000,3),dtype=np.uint8)
acc=np.einsum('nc,oc->no',candidates.astype(np.int64)-128,q.weights-q.weight_zero_points[:,None])+q.biases
p=acc*q.channel_multipliers;s=(p+8191+((p>>14)&1))>>14
v=s*q.multiplier/2**q.shift;half=(abs(v-np.floor(v)-.5)<1e-10)&(abs(v)<127)
chosen=[]
for sign in (-1,1):
 for parity in (0,1):
  rows=np.where(np.any(half&(np.sign(v)==sign)&(np.floor(v).astype(np.int64)%2==parity),axis=1))[0]
  chosen.extend(rows[:8].tolist())
chosen=chosen[:32];assert len(chosen)==32
x=np.broadcast_to(candidates[chosen,None,None,:],(32,5,8,3)).copy();x.tofile(out/'input000.u8')
for name,values in [('even',np.rint(v)),('away',np.sign(v)*np.floor(abs(v)+.5)),('up',np.floor(v+.5))]:
 y=np.clip(values[chosen]+q.output_zero_point,-128,127).astype(np.int8)
 np.broadcast_to(y[:,None,None,:],(32,5,8,16)).copy().tofile(out/f'expected_{name}.i8')
(out/'expected000.i8').write_bytes((out/'expected_even.i8').read_bytes())
print('Rounding candidates differ:',np.count_nonzero(np.rint(v[chosen])!=np.floor(v[chosen]+.5)))
