"""Independent stride-2 register hypothesis; not public compiler support."""
from pathlib import Path
import struct
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.quantization import Quantization,reference
from open_rknpu.sequence import encode_sequence
root=Path(__file__).resolve().parent/'stride2_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110316)
for i,k in enumerate((1,3,5)):
    w=rng.uniform(-.3,.3,(3,3,k,k)).astype(np.float32);b=rng.uniform(-2,2,3).astype(np.float32)
    m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],pads=[k//2]*4)],'stride',
        [h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,8,8])],
        [nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)])
    m.ir_version=8;p=root/f'model{i:03}.onnx';onnx.save(m,p)
    data,meta=compile_model(p);data=bytearray(data)
    fields={0x1014:18,0x1028:4,0x102c:16,0x3014:3<<16|3,0x4024:256,0x4030:3,0x4034:3,0x405c:3<<16|3,0x40c0:256,0x500c:3,0x5010:3}
    for j in range(126):
        word=struct.unpack_from('<Q',data,j*8)[0];reg=word&65535
        if reg in fields:struct.pack_into('<Q',data,j*8,(word>>48)<<48|fields[reg]<<16|reg)
    binary=encode_sequence(data,input_shape=(8,8,3),output_shape=(4,4,3),input_stride=16,arena_bytes=16384,
        input_offset=0x2000,output_offset=0x3000,tasks=[(0,126,29,768)],output_scale=meta['output_scale'],output_zero_point=meta['output_zero_point'],serial=True)
    p.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=rng.integers(0,256,(32,8,8,3),dtype=np.uint8);x[0]=0;x[1]=255
    x.tofile(root/f'input{i:03}.u8');np.stack([reference(v,q)[::2,::2] for v in x]).tofile(root/f'expected{i:03}.i8')
print('Built three stride-2 hypotheses')
