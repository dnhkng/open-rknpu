"""Independent 14x14 native 8->16 5x5 Conv experiment; no capture inputs.

Not a public graph profile. Establishes lowering before general task scheduling.
"""
from pathlib import Path
import json
import struct
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.chain import NATIVE, native_quantize, native_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.register_profile import REGISTERS
from open_rknpu.sequence import encode_sequence

root=Path(__file__).resolve().parent/"generated"
rng=np.random.default_rng(110322)
w1=rng.uniform(-.2,.6,(8,1,1,1)).astype(np.float32)
b1=rng.uniform(-2,2,8).astype(np.float32)
w2=rng.uniform(-.4,.4,(16,8,5,5)).astype(np.float32)
b2=rng.uniform(-3,3,16).astype(np.float32)
model=h.make_model(h.make_graph([
    h.make_node("Conv",["input","w","b"],["hidden"]),
    h.make_node("Relu",["hidden"],["output"])],"first",
    [h.make_tensor_value_info("input",1,[1,1,14,14])],
    [h.make_tensor_value_info("output",1,[1,8,14,14])],
    [nh.from_array(w1,"w"),nh.from_array(b1,"b")]),opset_imports=[h.make_opsetid("",13)])
path=root/"native14_first.onnx";onnx.save(model,path)
first,meta=compile_model(path)
q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta["quantization"].items()})
q2=native_quantize(w2,b2,q1.output_scale,q1.output_zero_point)
data=bytearray(16384)
first_regs={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from("<126Q",first)}
data[0x880:0x8c0]=first[first_regs[0x1110]:first_regs[0x1110]+64]
data[0x900:0x940]=first[first_regs[0x5020]:first_regs[0x5020]+64]
first_regs.update({0x1110:0x880,0x5020:0x900,0x1070:0x4000,0x4020:0x5000})
second=dict(first_regs)
second.update(NATIVE)
second.update({0x1010:0x9c,0x1020:14<<16|14,0x1024:7<<16|16,
    0x1028:14,0x102c:196,0x1030:6400,0x1034:400,0x1038:0x5050010,
    0x103c:7<<16,0x1044:14<<16|7,0x1068:0x202,0x1070:0x5000,
    0x1078:0x400f400f,0x107c:56,0x1080:196,0x1084:14<<16|14,0x1110:0x1000,
    0x1188:200,0x118c:13<<16|13,0x4020:0x6000,0x403c:15<<16|15,
    0x4060:0x13,0x406c:0x80000000,0x40e0:0x80000000,
    0x4080:q2.output_zero_point&0xffffffff,0x4084:q2.multiplier,
    0x4088:q2.shift,0x5020:0x2900})
for base,fields,next_command,control in ((0,first_regs,0x440,0x40),(0x440,second,0,0x28)):
    for i,(reg,default,tag) in enumerate(REGISTERS):
        struct.pack_into("<Q",data,base+i*8,tag<<48|fields.get(reg,default)<<16|reg)
    for i,(reg,value,tag) in enumerate(((0x10,next_command,0x101),(0x14,control,0x101),(0,0,0x41),(8,29,0x81))):
        struct.pack_into("<Q",data,base+(126+i)*8,tag<<48|value<<16|reg)
for o in range(16):
    weights=q2.weights[o].reshape(8,5,5)
    for kh in range(5):
        for kw in range(5):
            row=list(weights[:,kh,kw])+[int(q2.weight_zero_points[o])]*8
            struct.pack_into("<16b",data,0x1000+((kh*5+kw)*16+o)*16,*row)
    block=0x2900+o//4*32;lane=o%4
    struct.pack_into("<i",data,block+lane*4,int(q2.biases[o]))
    struct.pack_into("<h",data,block+16+lane*2,-int(q2.weight_zero_points[o]))
    struct.pack_into("<H",data,block+24+lane*2,int(q2.channel_multipliers[o]))
(root/"native14.bin").write_bytes(data)
expected=[];inputs=[]
for k in range(36):
    x=np.zeros((14,14,1),np.uint8)
    if k==1:x[:]=128
    elif k==2:x[:]=np.arange(196,dtype=np.uint8).reshape(14,14,1)
    elif k==3:x[3,4,0]=255
    elif k>=4:
        state=(1103+k*0x9e3779b9)&0xffffffff
        for j in range(x.size):
            state=(1664525*state+1013904223)&0xffffffff;x.flat[j]=state>>24
    inputs.append(x)
    expected.append(native_reference(reference(x,q1),q2,q1.output_zero_point))
np.stack(expected).tofile(root/"native14.expected.i8")
(root/"native14.json").write_text(json.dumps(dict(first=q1.metadata(),second=q2.metadata()),indent=2)+"\n")
print("Emitted independent native14 program and 36 expected cases")
suite=root.parent/"sequence_suite";suite.mkdir(exist_ok=True)
(suite/"model000.bin").write_bytes(encode_sequence(data,input_shape=(14,14,1),
    output_shape=(14,14,16),input_stride=16,arena_bytes=32768,
    input_offset=0x4000,output_offset=0x6000,tasks=[(0,126,29,768),(0x440,126,29,768)],
    output_scale=q2.output_scale,output_zero_point=q2.output_zero_point))
np.stack(inputs).tofile(suite/"input000.u8")
np.stack(expected).tofile(suite/"expected000.i8")
