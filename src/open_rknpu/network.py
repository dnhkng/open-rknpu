"""SPDX-License-Identifier: MIT
Spatial Conv-Relu-Conv followed by three explicit pooling stages.
"""
from pathlib import Path
import struct,tempfile
import onnx
from onnx import helper as h
from .chain import compile_chain
from .pooling import POOL
from .register_profile import REGISTERS

def compile_network(path,calibration_ranges=None):
    model=onnx.load(path);onnx.checker.check_model(model);g=model.graph
    if len(g.node)!=6 or [n.op_type for n in g.node[:3]]!=["Conv","Relu","Conv"]:
        raise ValueError("expected Conv-Relu-Conv followed by three pools")
    kind=g.node[3].op_type
    for i in range(3,6):
        n=g.node[i]
        if (kind not in ("MaxPool","AveragePool") or n.op_type!=kind or n.domain not in ("","ai.onnx") or
            list(n.input)!=list(g.node[i-1].output) or len(n.output)!=1 or
            {a.name:h.get_attribute_value(a) for a in n.attribute}!={"kernel_shape":[2,2],"strides":[2,2]}):
            raise ValueError("three matching 2x2 stride-2 pools required")
    if (len(g.output)!=1 or g.output[0].name!=g.node[-1].output[0] or
        g.output[0].type.tensor_type.elem_type!=1 or
        [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]!=[1,3,1,1]):
        raise ValueError("network output must be float32 [1,3,1,1]")
    sub=h.make_graph(list(g.node[:3]),"spatial_layers",list(g.input),
        [h.make_tensor_value_info(g.node[2].output[0],1,[1,3,8,8])],list(g.initializer))
    first=h.make_model(sub,opset_imports=list(model.opset_import));first.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"chain.onnx";onnx.save(first,p);source,meta=compile_chain(p,calibration_ranges)
    data=bytearray(8192);cursor=0xd00
    for layer,start in enumerate((0,0x440)):
        regs={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from("<126Q",source,start)}
        for register,length in ((0x1110,((regs[0x1030]+63)//64)*64),(0x5020,128)):
            data[cursor:cursor+length]=source[regs[register]:regs[register]+length]
            regs[register]=cursor;cursor+=length
        regs[0x1070]=0x2000 if layer==0 else 0x1400
        regs[0x4020]=0x1400 if layer==0 else 0x1800
        for i,(r,v,tag) in enumerate(REGISTERS):
            struct.pack_into("<Q",data,start+i*8,(tag<<48)|(regs.get(r,v)<<16)|r)
        terminal(data,start+126*8,0x440 if layer==0 else 0x880,0x40 if layer==0 else 0x14,29)
    if cursor>0x1400:raise ValueError("network constants exceed memory layout")
    for stage,size,inp,out in ((0,8,0x1800,0x1c00),(1,4,0x1c00,0x1d00),(2,2,0x1d00,0x3000)):
        start=0x880+stage*0x180;output=size//2
        fields={0x600c:size-1,0x6010:size-1,0x6018:output-1,0x601c:output-1,
                0x6070:out,0x607c:output*output*16,0x6084:output*output*16,
                0x700c:size-1,0x7010:size-1,0x701c:inp,0x7024:size*16,0x7028:size*size*16}
        if kind=="AveragePool":fields.update({0x6024:0x10,0x6038:0x8000,0x603c:0x8000,0x6048:0x7ff00,0x604c:0x7fe80,0x6050:0x7fe00})
        for i,(r,v) in enumerate(POOL):
            tag=0x4001 if r<0x7000 else 0x8001 if r<0x8000 else 0x0401
            struct.pack_into("<Q",data,start+i*8,(tag<<48)|(fields.get(r,v)<<16)|r)
        terminal(data,start+37*8,start+0x180 if stage<2 else 0,0x14 if stage<2 else 0x28,96)
    meta.update(profile=7 if kind=="MaxPool" else 8,pool=kind,pool_levels=3,output_shape_nhwc=[1,1,1,3])
    return bytes(data),meta

def terminal(data,offset,next_offset,control,enable):
    words=(0x0101000000000010|(next_offset<<16),0x0101000000000014|(control<<16),
           0x0041000000000000,0x0081000000000008|(enable<<16))
    for i,w in enumerate(words):struct.pack_into("<Q",data,offset+i*8,w)
