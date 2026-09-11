"""Independent learned depthwise 2x2 stride2 transposed Conv hypothesis."""
from pathlib import Path
import struct,json
import numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.chain import native_quantize
from open_rknpu.sequence import encode_sequence
root=Path(__file__).resolve().parent;out=root/'transpose_learned_suite';out.mkdir(exist_ok=True);rng=np.random.default_rng(110322)
for i in range(6):
    binary,meta=compile_sequence(root/'depthwise_suite'/f'model{i:03}.onnx');data=bytearray(binary[128:])
    q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()})
    w=rng.uniform(-.4,.4,(3,1,2,2)).astype(np.float32);b=rng.uniform(-1,1,3).astype(np.float32)
    q=native_quantize(w,b,q1.output_scale,q1.output_zero_point,symmetric=True)
    fields={0x1014:0x909,0x1028:16,0x102c:256,0x1030:512,0x1034:256,0x1038:0x4040002,0x1048:0x1c000000,0x1068:0x202,0x1188:128,
        0x3014:15<<16|15,0x4024:4096,0x4030:15,0x4034:15,0x4058:7,0x405c:15<<16|15,0x4080:q.output_zero_point&0xffffffff,
        0x4084:q.multiplier,0x4088:q.shift,0x40c0:8192,0x500c:15,0x5010:15,0x5020:0xd00}
    for j in range(126):
        off=0x440+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
        if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    data[0xb00:0xd40]=bytes(0x240)
    for y in range(2):
        for x in range(2):
            for c in range(3):struct.pack_into('<bb',data,0xb00+((y+1)*4+x+1)*32+c*2,int(q.weights[c,(1-y)*2+1-x]),0)
    for c in range(3):
        struct.pack_into('<i',data,0xd00+c*4,int(q.biases[c]));struct.pack_into('<H',data,0xd10+c*2,int(q.channel_multipliers[c]))
    (out/f'model{i:03}.bin').write_bytes(encode_sequence(data,input_shape=(8,8,3),output_shape=(16,16,3),input_stride=16,arena_bytes=24576,input_offset=0x1000,output_offset=0x3000,tasks=[(0,126,29,768),(0x440,126,29,768)],output_scale=q.output_scale,output_zero_point=q.output_zero_point,serial=True))
    x=np.fromfile(root/'depthwise_suite'/f'input{i:03}.u8',np.uint8).reshape(16,8,8,3);x.tofile(out/f'input{i:03}.u8');ys=[]
    for v in x:
        a=reference(v,q1).astype(np.int64)-q1.output_zero_point;acc=np.empty((16,16,3),np.int64)
        bias=q.biases+q1.output_zero_point*q.weights.sum(axis=1)
        for yy in range(2):
            for xx in range(2):acc[yy::2,xx::2]=a*q.weights[:,yy*2+xx]+bias
        product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14
        ys.append(np.clip(((scaled*q.multiplier+(1<<(q.shift-1)))>>q.shift)+q.output_zero_point,-128,127).astype(np.int8))
    np.stack(ys).tofile(out/f'expected{i:03}.i8')
    (out/f'model{i:03}.json').write_text(json.dumps(dict(weights=w.tolist(),bias=b.tolist(),first=meta['first'],quantization=q.metadata())))
