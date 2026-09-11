"""SPDX-License-Identifier: MIT
Executable container; payloads are trusted compiler output.
Version 2 assigns bytes 88/92 to float32 input scale and UINT8 zero point.
Version 1 retains zero reserved fields and input scale 1, zero point 0.
"""
import math
import struct

HEADER=struct.Struct("<8s17Iif3I")
MAGIC=b"ORNPUBIN"

def checksum(data):
    value=2166136261
    for byte in data:
        value=((value^byte)*16777619)&0xffffffff
    return value

def encode(payload,metadata):
    _,height,width,channels=metadata["shape_nhwc"]
    out_channels=metadata.get("output_shape_nhwc",metadata["shape_nhwc"])[3]
    profile=metadata.get("profile",1)
    input_scale=metadata.get("input_scale",1.0);input_zp=metadata.get("input_zero_point",0)
    version=1 if (input_scale,input_zp)==(1.0,0) else 2
    input_bits=0 if version==1 else struct.unpack("<I",struct.pack("<f",input_scale))[0]
    header=HEADER.pack(MAGIC,version,HEADER.size,1103,profile,height,width,channels,out_channels,
                       metadata.get("input_stride",16),16,len(payload),metadata["register_count"],8192,12288,metadata.get("arena_bytes",16384),
                       metadata["kernel_size"],int(metadata.get("relu",False)),
                       metadata["output_zero_point"],metadata["output_scale"],0,input_bits,input_zp)
    data=bytearray(header+payload)
    struct.pack_into("<I",data,84,checksum(data))
    decode(data)
    return bytes(data)

def decode(data):
    if data[:8]==b"ORNPUSEQ":
        from .sequence import decode_sequence
        return decode_sequence(data)
    if len(data)<HEADER.size: raise ValueError("truncated model header")
    fields=HEADER.unpack_from(data)
    (magic,version,header_size,target,profile,height,width,in_channels,out_channels,
     in_stride,out_stride,payload_size,register_count,in_offset,out_offset,arena,
     kernel,relu,zero_point,scale,stored_checksum,reserved0,reserved1)=fields
    if magic!=MAGIC or version not in (1,2) or header_size!=96: raise ValueError("unsupported model format")
    if target!=1103 or profile not in (1,2,3,4,5,6,7,8): raise ValueError("unsupported NPU target/profile")
    if profile in (3,4,5,6) and (height,width,in_channels,out_channels,kernel)!=(8,8,3,3,1):
        raise ValueError("unsupported pooling profile")
    if profile in (2,7,8) and ((height,width,in_channels,out_channels,relu)!=(8,8,3,3,1) or kernel not in (1,3)):
        raise ValueError("unsupported two-layer profile")
    limit=32 if in_channels==1 and profile==1 else 8
    if not (5<=height<=limit and 5<=width<=limit) or in_channels not in (1,3) or not 1<=out_channels<=16:
        raise ValueError("unsupported tensor shape")
    if (in_stride,out_stride,payload_size,register_count,in_offset,out_offset,arena)!=((width+15)//16*16,16,8192,126,8192,12288,12288+((height*width*16+4095)//4096)*4096):
        raise ValueError("unsupported memory layout")
    if kernel not in (1,3,5) or relu not in (0,1):
        raise ValueError("unsupported model options")
    input_scale=1.0 if version==1 else struct.unpack("<f",struct.pack("<I",reserved0))[0]
    input_zp=0 if version==1 else reserved1
    if version==1 and (reserved0 or reserved1): raise ValueError("nonzero reserved fields")
    if version==2 and (profile!=1 or not math.isfinite(input_scale) or input_scale<=0 or input_zp>255):
        raise ValueError("invalid input quantization")
    if not -128<=zero_point<=127 or not math.isfinite(scale) or scale<=0:
        raise ValueError("invalid output quantization")
    if len(data)!=header_size+payload_size: raise ValueError("incorrect model length")
    checked=bytearray(data)
    checked[84:88]=b"\0"*4
    if checksum(checked)!=stored_checksum: raise ValueError("model checksum mismatch")
    output_height,output_width=(1,1) if profile>=5 else (4,4) if profile>=3 else (height,width)
    operators=["Conv","Relu"] if relu else ["Conv"]
    if profile==2: operators.append("Conv")
    elif profile in (3,4): operators.append("MaxPool" if profile==3 else "AveragePool")
    elif profile in (5,6): operators.extend(["MaxPool" if profile==5 else "AveragePool"]*3)
    elif profile in (7,8): operators.extend(["Conv"]+["MaxPool" if profile==7 else "AveragePool"]*3)
    return {"target":"rv1103","format_version":version,"profile":profile,
            "task_count":5 if profile>=7 else 4 if profile>=5 else 1 if profile==1 else 2,
            "operators":operators,
            "shape_nhwc":[1,height,width,in_channels],"input_dtype":"uint8","input_scale":input_scale,
            "output_shape_nhwc":[1,output_height,output_width,out_channels],
            "input_zero_point":input_zp,"output_dtype":"int8","output_scale":scale,
            "output_zero_point":zero_point,"kernel_size":kernel,"relu":bool(relu),
            "payload_bytes":payload_size,"arena_bytes":arena,"task_bytes":4096,
            "input_bytes":height*width*in_channels,"output_bytes":output_height*output_width*out_channels}
