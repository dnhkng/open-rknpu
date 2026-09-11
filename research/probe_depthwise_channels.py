"""Independent C4 depthwise hypothesis; not public support."""
from pathlib import Path
import struct,sys
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.chain import native_quantize
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.quantization import Quantization,reference
from open_rknpu.sequence import encode_sequence
root=Path(__file__).resolve().parent;c=int(sys.argv[1]) if len(sys.argv)>1 else 4;out=root/f'depthwise_c{c}_suite';out.mkdir(exist_ok=True)
rng=np.random.default_rng(110318)
w=rng.uniform(-.3,.3,(c,3,1,1)).astype(np.float32);bias=rng.uniform(-2,2,c).astype(np.float32)
m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[1,1])],'stem',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,c,8,8])],[nh.from_array(w,'w'),nh.from_array(bias,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
onnx.save(m,out/'stem.onnx');source,meta=compile_model(out/'stem.onnx')
data=bytearray((root/'depthwise_suite/model000.bin').read_bytes()[128:])
data[:1040]=source[:1040];data[0x880:0x8c0]=source[0x440:0x480];data[0xa80:0xb00]=source[0x480:0x500]
w2=rng.uniform(-.3,.3,(c,1,3,3)).astype(np.float32);b2=rng.uniform(-2,2,c).astype(np.float32)
q=native_quantize(w2,b2,meta['output_scale'],meta['output_zero_point'],symmetric=True)
m.graph.node[0].output[0]='stem'
m.graph.node.append(h.make_node('Conv',['stem','w2','b2'],['output'],group=c,kernel_shape=[3,3],pads=[1]*4))
m.graph.initializer.extend([nh.from_array(w2,'w2'),nh.from_array(b2,'b2')]);onnx.save(m,out/'model000.onnx')
for base,fields in [(0,{0x1110:0x880,0x5020:0xa80,0x1070:0x1000,0x4020:0x2000}),(0x440,{0x1024:(c-1)<<16|16,0x403c:(c-1)<<16|31,0x1184:meta['output_zero_point']&65535,0x4080:q.output_zero_point&0xffffffff,0x4084:q.multiplier,0x4088:q.shift})]:
 for j in range(126):
  off=base+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
  if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
for tap in range(9):
 for channel in range(c):struct.pack_into('<bb',data,0xb00+tap*32+channel*2,int(q.weights[channel,tap]),0)
for channel in range(c):
 struct.pack_into('<i',data,0xc40+(channel//4)*24+(channel%4)*4,int(q.biases[channel]));struct.pack_into('<H',data,0xc50+(channel//4)*24+(channel%4)*2,int(q.channel_multipliers[channel]))
binary=encode_sequence(data,input_shape=(8,8,3),output_shape=(8,8,c),input_stride=16,arena_bytes=16384,input_offset=0x1000,output_offset=0x3000,tasks=[(0,126,29,768),(0x440,126,29,768)],output_scale=q.output_scale,output_zero_point=q.output_zero_point,serial=True)
(out/'model000.bin').write_bytes(binary)
a=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
x=rng.integers(0,256,(16,8,8,3),dtype=np.uint8);x.tofile(out/'input000.u8')
np.stack([depthwise_reference(reference(v,a),q,a.output_zero_point) for v in x]).tofile(out/'expected000.i8')
