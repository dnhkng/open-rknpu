"""Independent depthwise ConvTranspose nearest-repeat hypothesis."""
from pathlib import Path
import struct
import numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.sequence import encode_sequence
root=Path(__file__).resolve().parent;out=root/'transpose_independent_suite';out.mkdir(exist_ok=True)
binary,meta=compile_sequence(root/'depthwise_suite/model000.onnx');data=bytearray(binary[128:])
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()})
fields={0x1014:0x909,0x1028:16,0x102c:256,0x1030:512,0x1034:256,0x1038:0x4040002,0x1048:0x1c000000,0x1068:0x202,0x1188:128,
    0x3014:15<<16|15,0x4024:4096,0x4030:15,0x4034:15,0x4058:7,0x405c:15<<16|15,0x4080:q.output_zero_point&0xffffffff,
    0x4084:round(2**21/127),0x4088:21,0x40c0:8192,0x500c:15,0x5010:15,0x5020:0xd00}
for j in range(126):
 off=0x440+j*8;word=struct.unpack_from('<Q',data,off)[0];r=word&65535
 if r in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[r]<<16|r)
data[0xb00:0xd40]=bytes(0x240)
for tap in (5,6,9,10):
 for c in range(3):struct.pack_into('<bb',data,0xb00+tap*32+c*2,127,0)
for c in range(3):
 struct.pack_into('<i',data,0xd00+c*4,-q.output_zero_point*127*4);struct.pack_into('<H',data,0xd10+c*2,16384)
(out/'model000.bin').write_bytes(encode_sequence(data,input_shape=(8,8,3),output_shape=(16,16,3),input_stride=16,arena_bytes=24576,input_offset=0x1000,output_offset=0x3000,tasks=[(0,126,29,768),(0x440,126,29,768)],output_scale=q.output_scale,output_zero_point=q.output_zero_point,serial=True))
x=np.fromfile(root/'depthwise_suite/input000.u8',np.uint8).reshape(16,8,8,3);x.tofile(out/'input000.u8')
np.stack([np.repeat(np.repeat(reference(v,q),2,0),2,1) for v in x]).tofile(out/'expected000.i8')
