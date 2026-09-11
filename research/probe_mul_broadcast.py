"""Mesa/TRM-guided per-channel ERDMA broadcast, independent generated commands."""
from pathlib import Path
import struct
import numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_reference
from open_rknpu.model import checksum
root=Path(__file__).resolve().parent;out=root/'mul_broadcast_probe';out.mkdir(exist_ok=True)
data,meta=compile_sequence(root/'mul_suite/model000.onnx');data=bytearray(data)
for j in range(78):
 off=144+0x880+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
 if reg in (0x5034,0x5040):struct.pack_into('<Q',data,off,(word>>48)<<48|{0x5034:4,0x5040:16}[reg]<<16|reg)
data[80:84]=bytes(4);struct.pack_into('<I',data,80,checksum(data));(out/'model000.bin').write_bytes(data)
qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()}) for b in meta['branches']]
x=np.fromfile(root/'mul_suite/input000.u8',np.uint8).reshape(32,8,8,3);x.tofile(out/'input000.u8')
np.stack([mul_reference(reference(v,qs[0]),reference(v,qs[1])[0,0]) for v in x]).tofile(out/'expected000.i8')
