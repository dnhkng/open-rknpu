"""Independent LeakyRelu 0.5 fused conversion hypothesis."""
from pathlib import Path
import struct
import numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,receptive_fields
from open_rknpu.model import checksum
root=Path(__file__).resolve().parent;out=root/'leaky_independent_suite';out.mkdir(exist_ok=True)
data,meta=compile_sequence(root/'stride2_suite/model000.onnx');data=bytearray(data)
for j in range(126):
 off=112+j*8;word=struct.unpack_from('<Q',data,off)[0];r=word&65535
 if r in (0x4040,0x4060,0x4068,0x4088):struct.pack_into('<Q',data,off,(word>>48)<<48|({0x4040:0x120180,0x4060:0x22,0x4068:0x20000000,0x4088:meta['quantization']['shift']+14}[r])<<16|r)
data[80:84]=bytes(4);struct.pack_into('<I',data,80,checksum(data));(out/'model000.bin').write_bytes(data)
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
x=np.fromfile(root/'stride2_suite/input000.u8',np.uint8).reshape(32,8,8,3);x.tofile(out/'input000.u8');ys=[]
for v in x:
 acc=np.einsum('...c,oc->...o',receptive_fields(v.astype(np.int64)-128,1),q.weights-q.weight_zero_points[:,None])+q.biases
 product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14
 value=np.where(acc<0,scaled*8192,product)
 y=np.rint(value*q.multiplier/2**(q.shift+14)).astype(np.int64)+q.output_zero_point
 ys.append(np.clip(y,-128,127).astype(np.int8))
np.stack(ys).tofile(out/'expected000.i8')
