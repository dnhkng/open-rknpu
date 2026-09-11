"""Independent LUT tables and register programs; no capture inputs."""
from pathlib import Path
import sys,struct
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.register_profile import REGISTERS
root=Path(__file__).resolve().parent;kind=sys.argv[1] if len(sys.argv)>1 else 'sigmoid';out=root/(kind+'_lut_probe');out.mkdir(exist_ok=True)
w=np.eye(3,dtype=np.float32).reshape(3,3,1,1)/32
m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[1,1])],'lut',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,8,8])],[nh.from_array(w,'w'),nh.from_array(np.zeros(3,np.float32),'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;onnx.save(m,out/'stem.onnx')
source,meta=compile_model(out/'stem.onnx',1/2048,0,input_scale=1,input_zero_point=128);q=meta['quantization']
data=bytearray(24576);regs={v&65535:v>>16&0xffffffff for v in struct.unpack_from('<126Q',source)}
data[0x2800:0x2840]=source[regs[0x1110]:regs[0x1110]+64];data[0x2840:0x2880]=source[regs[0x5020]:regs[0x5020]+64]
setup={0x400c:0x1e5,0x4018:0,0x4020:0,0x4024:16,0x4030:0,0x4034:0,0x403c:0xf000f,0x4040:0x53,0x4048:0,0x4050:0x30000002,0x4054:0,0x4058:3,0x405c:0,0x4060:0x13,0x4070:0x10041c1,0x4080:0,0x4084:0x10001,0x4088:0,0x40c0:16,0x4108:1,0x500c:0,0x5010:0,0x5014:15,0x5018:0,0x501c:0,0x5020:0,0x5034:1,0x5038:0,0x5040:0,0x5044:0x907809}
words=[tag<<48|setup.get(reg,default)<<16|reg for reg,default,tag in REGISTERS if reg>=0x4000]
fn=(lambda x:1/(1+np.exp(-x))) if kind=='sigmoid' else np.tanh
for table in (0,1):
 words.append(0x1001<<48|(0x20000+table*0x10000)<<16|0x4100)
 values=np.clip(np.rint(fn((np.arange(513)+(table-1)*512)/64)*32768),-32768,32767).astype(np.int64)
 words.extend(0x1001<<48|(int(v)&65535)<<16|0x4104 for v in values)
assert len(words)==1106
for i,word in enumerate(words):struct.pack_into('<Q',data,i*8,word)
regs.update({0x1004:0x30,0x3004:0x30,0x4004:0x30,0x5004:0x30,0x1070:0x3000,0x1110:0x2800,0x5020:0x2840,0x4010:0x804000,0x4020:0x4000,0x4060:0x20000,0x4068:q['multiplier']<<16|q['shift']<<8,0x4070:0x1004140,0x4080:0xffffff80 if kind=='sigmoid' else 0,0x4084:255 if kind=='sigmoid' else 127,0x4088:15,0x4108:0x68,0x410c:0x50500,0x4110:0xffffc000,0x4114:0,0x4118:0,0x411c:0x4000})
for i,(reg,default,tag) in enumerate(REGISTERS):struct.pack_into('<Q',data,0x22c0+i*8,tag<<48|regs.get(reg,default)<<16|reg)
for base,count,enable,nxt,ctrl in ((0,1106,24,0x22c0,0x40),(0x22c0,126,29,0,0x28)):
 for i,(reg,value,tag) in enumerate(((0x10,nxt,0x101),(0x14,ctrl,0x101),(0,0,0x41),(8,enable,0x81))):struct.pack_into('<Q',data,base+(count+i)*8,tag<<48|value<<16|reg)
bundle=bytearray();rng=np.random.default_rng(110328)
for run in range(4):
 x=rng.integers(0,256,(8,8,3),dtype=np.uint8)
 if run==0:x=np.arange(192,dtype=np.uint8).reshape(8,8,3)
 if run==1:x[:]=128
 if run==2:x=(np.arange(192,dtype=np.uint16)+64).astype(np.uint8).reshape(8,8,3)
 inp=np.full((8,16,3),128,np.uint8);inp[:,:8]=x;data[0x3000:0x3180]=inp.tobytes()
 expected=np.zeros((8,8,16),np.int8)
 real=(x.astype(np.float64)-128)/32
 expected[:,:,:3]=np.rint(np.clip(np.rint(fn(real)*32768),-32768,32767)*(255 if kind=='sigmoid' else 127)/32768).astype(np.int16)-(128 if kind=='sigmoid' else 0)
 bundle+=struct.pack('<5I',2,len(data),0x4000,1024,1)+struct.pack('<10I',0,1106,24,768,1,0x22c0,126,29,768,1)+data+expected.tobytes()
(out/'test.replay').write_bytes(bundle)
