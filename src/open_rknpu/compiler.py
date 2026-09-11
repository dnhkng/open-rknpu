"""SPDX-License-Identifier: MIT

Experimental ONNX compiler for a deliberately narrow RV1103 convolution profile.
No RKNN imports or captured binary inputs. Unsupported graphs fail explicitly.
"""
import argparse
from pathlib import Path
import struct
import json
import numpy as np
import onnx
from onnx import numpy_helper
from .register_profile import REGISTERS
from .quantization import quantize

def compile_model(path, output_scale=None, output_zero_point=None, calibration_ranges=None, input_scale=1.0, input_zero_point=0):
    if calibration_ranges is not None and (output_scale is not None or output_zero_point is not None):
        raise ValueError("calibration and output quantization overrides cannot be combined")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph
    if (input_scale,input_zero_point)!=(1.0,0) and (len(graph.node) not in (1,2) or graph.node[-1].op_type not in ("Conv","Relu")):
        raise ValueError("input quantization overrides currently require a single Conv[/Relu]")
    if len(graph.node)==6:
        if output_scale is not None or output_zero_point is not None:
            raise ValueError("network output overrides require calibration instead")
        from .network import compile_network
        return compile_network(path,calibration_ranges)
    if len(graph.node)>=4 and all(n.op_type in ("MaxPool","AveragePool") for n in graph.node[-3:]):
        from .reduction import compile_reduction
        return compile_reduction(path,output_scale,output_zero_point,calibration_ranges)
    if graph.node and graph.node[-1].op_type in ("MaxPool","AveragePool"):
        from .pooling import compile_pool
        return compile_pool(path,output_scale,output_zero_point,calibration_ranges)
    if len(graph.node)==3:
        if output_scale is not None or output_zero_point is not None:
            raise ValueError("two-layer output quantization overrides are not supported yet")
        from .chain import compile_chain
        return compile_chain(path,calibration_ranges)
    if len(graph.node) not in (1,2) or graph.node[0].op_type!="Conv" or graph.node[0].domain not in ("", "ai.onnx"):
        raise ValueError("one Conv with optional following Relu is supported")
    op=graph.node[0]
    relu=len(graph.node)==2
    if relu:
        activation=graph.node[1]
        if activation.op_type!="Relu" or activation.domain not in ("", "ai.onnx") or activation.attribute or list(activation.input)!=list(op.output):
            raise ValueError("only a directly connected standard Relu may follow Conv")
    if len(op.input) not in (2,3) or len(graph.input)!=1 or len(graph.output)!=1:
        raise ValueError("one input/output and optional constant bias required")
    final_node=graph.node[-1]
    if op.input[0]!=graph.input[0].name or len(final_node.output)!=1 or final_node.output[0]!=graph.output[0].name:
        raise ValueError("unexpected graph connections")
    attributes={a.name:onnx.helper.get_attribute_value(a) for a in op.attribute}
    constants={x.name:numpy_helper.to_array(x) for x in graph.initializer}
    weights=constants[op.input[1]]
    if (weights.ndim!=4 or weights.shape[1:] not in ((1,1,1),(1,3,3),(1,5,5),(3,1,1),(3,3,3),(3,5,5)) or
        not 1<=weights.shape[0]<=16):
        raise ValueError("weights must be [O,I,K,K], I=1 or 3, 1<=O<=16, K=1, 3 or 5")
    input_channels=weights.shape[1]
    kernel=weights.shape[2]
    channels=weights.shape[0]
    aligned_channels=(channels+3)//4*4
    supported={"kernel_shape":[kernel,kernel], "strides":[1,1], "pads":[kernel//2]*4, "dilations":[1,1], "group":1}
    for key,value in attributes.items():
        if key not in supported or value!=supported[key]:
            raise ValueError("unsupported Conv attribute: "+key)
    if kernel>1 and attributes.get("pads")!=[kernel//2]*4:
        raise ValueError("spatial Conv requires explicit symmetric padding K//2")
    shape=[d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    if len(shape)!=4 or shape[:2]!=[1,input_channels] or not all(5<=d<=(32 if input_channels==1 else 8) for d in shape[2:]):
        raise ValueError("only static NCHW [1,I,H,W], I=1 or 3, 5 <= H,W <= 8")
    output_shape=[d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    if output_shape!=[1,channels,*shape[2:]] or any(v.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT for v in (graph.input[0],graph.output[0])):
        raise ValueError("input/output must be float32 tensors with matching spatial shapes")
    height,width=shape[2:]
    input_stride=(width+15)//16*16
    arena_size=12288+((height*width*16+4095)//4096)*4096
    if weights.dtype!=np.float32:
        raise ValueError("float32 weights required")
    bias=constants[op.input[2]] if len(op.input)==3 else np.zeros(channels,dtype=np.float32)
    if bias.shape!=(channels,) or bias.dtype!=np.float32:
        raise ValueError("bias must be float32 [output_channels]")
    if calibration_ranges is not None:
        entry=calibration_ranges[final_node.output[0]]
        output_scale,output_zero_point=entry["scale"],entry["zero_point"]
    quantization=quantize(weights,bias,output_scale,output_zero_point,relu,input_scale,input_zero_point)
    # Keep the active command program only. Pack weights immediately after it,
    # unlike the vendor arena, which also holds unused conversion programs.
    weight_offset=0x440
    bias_offset=weight_offset+((kernel*kernel*aligned_channels*4+63)//64)*64
    fields={
        0x1020:width<<16|height, 0x1028:width, 0x102c:height*width,
        0x1030:kernel*kernel*aligned_channels*4,0x1034:kernel*kernel*4,
        0x1038:(kernel<<24)|(kernel<<16)|aligned_channels,
        0x1068:(kernel//2)*0x101,
        0x1188:kernel*kernel*2,
        0x1184:(input_zero_point-128)&0xffff,
        0x1070:0x2000, 0x107c:height*3, 0x1080:height*3,
        0x1084:(height*3)<<16|1, 0x1110:weight_offset,
        0x118c:((height*2-1)<<16)|(height*2-1),
        0x3014:(height-1)<<16|(width-1),
        0x4020:0x3000, 0x4024:height*width*16,
        0x4030:width-1, 0x4034:height-1,
        0x403c:((aligned_channels-1)<<16)|15,
        0x405c:(height-1)<<16|(width-1), 0x40c0:height*width*16,
        0x4080:quantization.output_zero_point & 0xffffffff,
        0x4084:quantization.multiplier, 0x4088:quantization.shift,
        0x4060:0x12 if relu else 0x13,
        0x406c:0 if relu else 0x80000000,
        0x40e0:0 if relu else 0x80000000,
        0x500c:width-1, 0x5010:height-1, 0x5020:bias_offset,
    }
    if input_channels==1:
        fields.update({0x100c:0x20008000,0x101c:0,0x1024:0x10,
                       0x104c:0xe0,0x1050:0x14000,0x1054:0x10001,
                       0x105c:0,0x1078:0xc00f300f,
                       0x103c:input_stride<<16,0x1044:input_stride<<16|input_stride//8,
                       0x107c:height*input_stride//8,0x1080:height*input_stride//8,
                       0x1084:(height*input_stride//8)<<16|1,
                       0x118c:((height*input_stride//8-1)<<16)|(height*input_stride//8-1)})
    data=bytearray(8192)
    for i,(reg,value,tag) in enumerate(REGISTERS):
        struct.pack_into("<Q",data,i*8,(tag<<48)|(fields.get(reg,value)<<16)|reg)
    # Single task terminal sequence: no next command program.
    for i,word in enumerate((0x0101000000000010,0x0101000000280014,0x0041000000000000,0x00810000001d0008)):
        struct.pack_into("<Q",data,(len(REGISTERS)+i)*8,word)
    for c in range(channels):
        channel_weights=quantization.weights[c].reshape(input_channels,kernel,kernel)
        for kh in range(kernel):
            for kw in range(kernel):
                row=list(channel_weights[:,kh,kw])+[int(quantization.weight_zero_points[c])]*(4-input_channels)
                offset=weight_offset+(kh*kernel+kw)*aligned_channels*4+c*4
                struct.pack_into("<4b",data,offset,*row)
        block=bias_offset+(c//4)*32
        lane=c%4
        struct.pack_into("<i",data,block+lane*4,int(quantization.biases[c]))
        struct.pack_into("<h",data,block+16+lane*2,-int(quantization.weight_zero_points[c]))
        struct.pack_into("<H",data,block+24+lane*2,int(quantization.channel_multipliers[c]))
    metadata={"shape_nhwc":[1,height,width,input_channels],
              "output_shape_nhwc":[1,height,width,channels],
              "float_weights":weights.tolist(),"float_bias":bias.tolist(),"kernel_size":kernel,"relu":relu,
              "input_stride":input_stride,"arena_bytes":arena_size,
              "input_scale":input_scale,"input_zero_point":input_zero_point,
              "output_scale":quantization.output_scale,"output_zero_point":quantization.output_zero_point,
              "quantization":quantization.metadata(),"register_count":len(REGISTERS),
              "limitations":"experimental one/three-input-channel INT8 convolution profile; register semantics partially recovered"}
    if kernel==1 and np.all(np.count_nonzero(weights[:,:,0,0],axis=1)==1):
        metadata["input_channel_for_output"]=np.argmax(np.abs(weights[:,:,0,0]),axis=1).tolist()
    return bytes(data),metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("-o","--output",type=Path,required=True)
    parser.add_argument("--output-scale",type=float)
    parser.add_argument("--output-zero-point",type=int)
    args=parser.parse_args()
    data,metadata=compile_model(args.model,args.output_scale,args.output_zero_point)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(data)
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Emitted",len(data),"bytes:",args.output)

if __name__=="__main__":
    main()
