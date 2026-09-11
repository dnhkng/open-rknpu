"""SPDX-License-Identifier: MIT

Explicit task-table container for compiler-generated NPU command programs.
Payloads are trusted code, as in the legacy format; validation is not a sandbox.
"""
import math
import struct
from .model import checksum

MAGIC=b"ORNPUSEQ"
HEADER=struct.Struct("<8s22I")
TASK=struct.Struct("<4I")
CONSTANT=struct.Struct("<24s4I")
# Version 5 adds a named tensor table so one executable can describe fan-out,
# several external inputs of unequal shape and several external outputs.
EXTENSION=struct.Struct("<4I")
TENSOR=struct.Struct("<24s10I")
V5_TENSOR_SIZE=64
ROLE_INPUT,ROLE_OUTPUT,ROLE_INTERNAL=0,1,2
LAYOUT_PACKED_U8,LAYOUT_NATIVE16,LAYOUT_PACKED_I8=0,1,2
LAYOUT_NAMES={LAYOUT_PACKED_U8:"packed",LAYOUT_NATIVE16:"native16",LAYOUT_PACKED_I8:"packed-int8"}


def float_bits(value):
    return struct.unpack("<I",struct.pack("<f",value))[0]


def tensor_native_bytes(layout,batch,height,width,channels):
    """Arena storage bytes for one tensor descriptor.

    Packed layouts keep the CNA row stride (width aligned to16), so a packed
    descriptor is larger than the flat API buffer of batch*H*W*C bytes.
    """
    if layout in (LAYOUT_PACKED_U8,LAYOUT_PACKED_I8):
        return batch*height*((width+15)//16*16)*channels
    if layout==LAYOUT_NATIVE16:
        return batch*((height*width+3)//4)*64*((channels+15)//16)
    raise ValueError('unknown tensor layout')


def _pack_tensors(tensors):
    packed=[];names=set();inputs=[];outputs=[]
    for entry in tensors:
        name=str(entry['name']);encoded=name.encode('utf-8')
        role=int(entry['role']);layout=int(entry['layout']);index=int(entry.get('index',0))
        batch,height,width,channels=(int(v) for v in entry['shape'])
        offset=int(entry['offset']);size=int(entry['size'])
        if (not name or len(encoded)>23 or name in names or role not in (ROLE_INPUT,ROLE_OUTPUT,ROLE_INTERNAL)
            or layout not in (LAYOUT_PACKED_U8,LAYOUT_NATIVE16,LAYOUT_PACKED_I8)
            or not 1<=batch<=16 or not all(1<=v<=1024 for v in (height,width))
            or not 1<=channels<=128
            or offset<0 or size<1):
            raise ValueError('invalid tensor descriptor')
        expected=tensor_native_bytes(layout,batch,height,width,channels)
        if size!=expected:raise ValueError('tensor descriptor size mismatch')
        names.add(name)
        if role==ROLE_INPUT:inputs.append((index,name))
        elif role==ROLE_OUTPUT:outputs.append((index,name))
        packed.append(TENSOR.pack(encoded+b'\0'*(24-len(encoded)),role,layout,index,
                                  batch,height,width,channels,offset,size,0))
    inputs.sort();outputs.sort()
    if not inputs or not outputs:raise ValueError('v5 requires at least one external input and output')
    if [i for i,_ in inputs]!=list(range(len(inputs))) or [o for o,_ in outputs]!=list(range(len(outputs))):
        raise ValueError('external tensor indices must be contiguous from zero')
    return b"".join(packed),inputs,outputs


def encode_sequence_v5(payload, *, tensors, tasks, arena_bytes,
                       input_scale=1.0, input_zero_point=0,
                       output_scale=1.0, output_zero_point=0,
                       serial=False, constants=()):
    """Named-tensor container for bounded DAG graphs (fan-out, multi-IO)."""
    if len(constants)>64:raise ValueError('too many constant descriptors')
    descriptors=[];seen=set()
    for entry in constants:
        name=str(entry['name']);encoded=name.encode('utf-8')
        offset=int(entry['offset']);size=int(entry['size']);kind=int(entry.get('kind',1))
        if not name or len(encoded)>23 or name in seen or offset<0 or size<1 or offset+size>len(payload) or kind not in (1,2,3):
            raise ValueError('invalid constant descriptor')
        seen.add(name);descriptors.append(CONSTANT.pack(encoded+b'\0'*(24-len(encoded)),offset,size,kind,0))
    tensor_blob,inputs,outputs=_pack_tensors(tensors)
    primary_in=next(t for t in tensors if t['role']==ROLE_INPUT and int(t.get('index',0))==0)
    primary_out=next(t for t in tensors if t['role']==ROLE_OUTPUT and int(t.get('index',0))==0)
    _,ih,iw,ic=primary_in['shape'];_,oh,ow,oc=primary_out['shape']
    stride=iw if primary_in['layout']==LAYOUT_NATIVE16 else (iw+15)//16*16
    flags=int(serial)
    header=HEADER.pack(MAGIC,5,96,ih,iw,ic,oh,ow,oc,stride,len(payload),arena_bytes,
        int(primary_in['offset']),int(primary_out['offset']),len(tasks),
        float_bits(input_scale),input_zero_point,float_bits(output_scale),output_zero_point&0xffffffff,
        0,flags,int(primary_in['layout']==LAYOUT_NATIVE16),int(primary_in['shape'][0])-1)
    extension=EXTENSION.pack(len(tensors),V5_TENSOR_SIZE,len(inputs),len(outputs))
    data=bytearray(header+extension+b"".join(TASK.pack(*t) for t in tasks)+b''.join(descriptors)+tensor_blob+payload)
    struct.pack_into("<I",data,80,checksum(data))
    decode_sequence(data)
    return bytes(data)


def encode_sequence(payload, *, input_shape, output_shape, input_stride,
                    arena_bytes, input_offset, output_offset, tasks,
                    input_scale=1.0, input_zero_point=0,
                    output_scale=1.0, output_zero_point=0, serial=False, input_layout="packed", batch=1,input_tensor_count=1,
                    constants=()):
    """Shapes are HWC; tasks are (command_offset, register_count, enable, mask)."""
    if input_layout not in ('packed','native16'):raise ValueError('unknown input layout')
    if not isinstance(batch,int) or not 1<=batch<=16:raise ValueError('batch must be 1..16')
    if input_tensor_count not in (1,2) or (input_tensor_count==2 and input_shape[0]%2):
        raise ValueError('logical input tensor count unsupported')
    if len(constants)>64:raise ValueError('too many constant descriptors')
    descriptors=[];seen=set()
    for entry in constants:
        name=str(entry['name']);encoded=name.encode('utf-8')
        offset=int(entry['offset']);size=int(entry['size']);kind=int(entry.get('kind',1))
        if not name or len(encoded)>23 or name in seen or offset<0 or size<1 or offset+size>len(payload) or kind not in (1,2,3):
            raise ValueError('invalid constant descriptor')
        seen.add(name);descriptors.append(CONSTANT.pack(encoded+b'\0'*(24-len(encoded)),offset,size,kind,0))
    version=4 if descriptors else 3;flags=int(serial)|((input_tensor_count-1)<<1)|(len(descriptors)<<8)
    header=HEADER.pack(MAGIC,version,96,*input_shape,*output_shape,input_stride,len(payload),
        arena_bytes,input_offset,output_offset,len(tasks),float_bits(input_scale),
        input_zero_point,float_bits(output_scale),output_zero_point&0xffffffff,0,flags,int(input_layout=="native16"),batch-1)
    data=bytearray(header+b"".join(TASK.pack(*t) for t in tasks)+b''.join(descriptors)+payload)
    struct.pack_into("<I",data,80,checksum(data))
    decode_sequence(data)
    return bytes(data)


def _decode_v5(data):
    (magic,version,header_size,ih,iw,ic,oh,ow,oc,stride,payload_size,arena,
     input_offset,output_offset,count,inscale,inzp,outscale,outzp,stored,r0,r1,r2)=HEADER.unpack_from(data)
    if magic!=MAGIC or version!=5 or header_size!=96 or r1>1 or r2>15 or (r0&255)>1 or (r0>>8)!=0:
        raise ValueError("invalid v5 sequence header")
    if len(data)<112: raise ValueError("truncated v5 extension")
    tensor_count,tensor_size,input_count,output_count=EXTENSION.unpack_from(data,96)
    if tensor_size!=V5_TENSOR_SIZE or not 1<=tensor_count<=64 or not 1<=input_count<=8 or not 1<=output_count<=8:
        raise ValueError("invalid v5 extension")
    if not 1<=count<=64: raise ValueError("invalid task count")
    if not 0<payload_size<=1048576 or payload_size%64 or not 0<arena<=4194304 or arena%4096:
        raise ValueError("invalid sequence allocation")
    task_bytes=112+count*16;tensor_start=task_bytes;payload_start=tensor_start+tensor_count*V5_TENSOR_SIZE
    if len(data)!=payload_start+payload_size: raise ValueError("incorrect sequence length")
    inscale=struct.unpack("<f",struct.pack("<I",inscale))[0]
    outscale=struct.unpack("<f",struct.pack("<I",outscale))[0]
    outzp=outzp-(1<<32) if outzp>=1<<31 else outzp
    if not (math.isfinite(inscale) and inscale>0 and inzp<=255 and
            math.isfinite(outscale) and outscale>0 and -128<=outzp<=127):
        raise ValueError("invalid sequence quantization")
    tasks=[]
    for i in range(count):
        offset,amount,enable,mask=TASK.unpack_from(data,112+i*16)
        # 0x300|0xc00 is the union mask a batched job needs when it mixes engines.
        valid_kind=(enable in (29,96) and mask in (768,3072,3840) and amount<=256) or \
                   (enable==24 and mask in (768,3840) and amount in (78,1106))
        if (offset%8 or not 1<=amount<=1106 or offset+(amount+4)*8>payload_size or not valid_kind):
            raise ValueError("invalid task descriptor")
        tasks.append(dict(command_offset=offset,register_count=amount,enable=enable,mask=mask))
    tensors=[];names=set();external=[]
    for i in range(tensor_count):
        raw,role,layout,index,batch,height,width,channels,offset,size,reserved=TENSOR.unpack_from(data,tensor_start+i*V5_TENSOR_SIZE)
        name_bytes=raw.split(b'\0',1)[0]
        try:name=name_bytes.decode('utf-8')
        except UnicodeDecodeError:raise ValueError('invalid tensor name') from None
        if (not name or len(name_bytes)>=24 or raw[len(name_bytes)]!=0 or reserved or name in names
            or role not in (ROLE_INPUT,ROLE_OUTPUT,ROLE_INTERNAL) or layout not in LAYOUT_NAMES
            or not 1<=batch<=16 or not all(1<=v<=1024 for v in (height,width))
            or not 1<=channels<=128):
            raise ValueError('invalid tensor descriptor')
        expected=tensor_native_bytes(layout,batch,height,width,channels)
        if size!=expected:raise ValueError('tensor descriptor size mismatch')
        if offset<payload_size or offset+size>arena:raise ValueError('tensor outside arena')
        names.add(name)
        entry=dict(name=name,role=role,role_name=("input","output","internal")[role],layout=layout,
                   layout_name=LAYOUT_NAMES[layout],index=index,batch=batch,height=height,width=width,
                   channels=channels,byte_offset=offset,bytes=size)
        tensors.append(entry)
        if role!=ROLE_INTERNAL:external.append(entry)
    inputs=sorted((t for t in tensors if t['role']==ROLE_INPUT),key=lambda t:t['index'])
    outputs=sorted((t for t in tensors if t['role']==ROLE_OUTPUT),key=lambda t:t['index'])
    if [t['index'] for t in inputs]!=list(range(input_count)) or [t['index'] for t in outputs]!=list(range(output_count)):
        raise ValueError('external tensor indices must be contiguous from zero')
    for i,a in enumerate(external):
        for b in external[i+1:]:
            if a['byte_offset']<b['byte_offset']+b['bytes'] and b['byte_offset']<a['byte_offset']+a['bytes']:
                raise ValueError('overlapping external tensors')
    for t in tensors:
        if t['role']==ROLE_INTERNAL:
            for a in external:
                if t['byte_offset']<a['byte_offset']+a['bytes'] and a['byte_offset']<t['byte_offset']+t['bytes']:
                    raise ValueError('internal tensor overlaps external tensor')
    pin,pout=inputs[0],outputs[0]
    stride_expected=iw if pin['layout']==LAYOUT_NATIVE16 else (iw+15)//16*16
    if ((pin['height'],pin['width'],pin['channels'])!=(ih,iw,ic)
        or (pout['height'],pout['width'],pout['channels'])!=(oh,ow,oc)
        or pin['batch']!=r2+1 or stride!=stride_expected
        or pin['byte_offset']!=input_offset or pout['byte_offset']!=output_offset):
        raise ValueError('v5 primary tensor mismatch')
    checked=bytearray(data);checked[80:84]=b"\0"*4
    if checksum(checked)!=stored: raise ValueError("sequence checksum mismatch")
    return dict(target="rv1103",format_version=5,task_count=count,tasks=tasks,constants=[],constant_count=0,
                serial=bool(r0&1),input_tensor_count=input_count,output_tensor_count=output_count,
                tensors=tensors,input_tensors=inputs,output_tensors=outputs,tensor_count=tensor_count,
                shape_nhwc=[inputs[0]['batch'],ih,iw,ic],output_shape_nhwc=[outputs[0]['batch'],oh,ow,oc],
                batch=inputs[0]['batch'],input_layout=LAYOUT_NAMES[inputs[0]['layout']],
                input_dtype="uint8",output_dtype="int8",input_scale=inscale,input_zero_point=inzp,
                output_scale=outscale,output_zero_point=outzp,
                input_bytes=inputs[0]['batch']*ih*iw*ic,output_bytes=outputs[0]['batch']*oh*ow*oc,
                input_stride=stride,payload_bytes=payload_size,arena_bytes=arena,
                input_offset=input_offset,output_offset=output_offset,task_bytes=4096)


def decode_sequence(data):
    if len(data)<96: raise ValueError("truncated sequence header")
    (magic,version,header_size,ih,iw,ic,oh,ow,oc,stride,payload_size,arena,
     input_offset,output_offset,count,inscale,inzp,outscale,outzp,stored,r0,r1,r2)=HEADER.unpack_from(data)
    if magic!=MAGIC: raise ValueError("invalid sequence magic")
    if version==5: return _decode_v5(data)
    constant_count=r0>>8;flags=r0&255
    if version not in (3,4) or header_size!=96 or flags>3 or r1>1 or r2>15 or constant_count>64 or (version==3)!=(constant_count==0):
        raise ValueError("invalid sequence header")
    batch=r2+1
    input_tensor_count=(flags>>1)+1
    if input_tensor_count==2 and ih%2:raise ValueError('invalid two-input tensor layout')
    if batch!=1 and not r1:raise ValueError("batched sequence requires native16 layout")
    if not (all(1<=n<=1024 for n in (ih,iw,oh,ow)) and ((1<=ic<=128) if r1 else ic in (1,3)) and 1<=oc<=128):
        raise ValueError("invalid sequence shape")
    if stride!=(iw if r1 else (iw+15)//16*16) or not 1<=count<=64:
        raise ValueError("invalid stride/task count")
    if not 0<payload_size<=1048576 or payload_size%64 or not 0<arena<=4194304 or arena%4096:
        raise ValueError("invalid sequence allocation")
    input_bytes=batch*(((ih*iw+3)//4)*64*((ic+15)//16) if r1 else ih*stride*ic);output_bytes=batch*((oh*ow+3)//4)*64*((oc+15)//16)
    if (input_offset%64 or output_offset%64 or input_offset<payload_size or output_offset<payload_size
        or input_offset+input_bytes>arena or output_offset+output_bytes>arena
        or not (input_offset+input_bytes<=output_offset or output_offset+output_bytes<=input_offset)):
        raise ValueError("overlapping or out-of-bounds IO buffers")
    inscale=struct.unpack("<f",struct.pack("<I",inscale))[0]
    outscale=struct.unpack("<f",struct.pack("<I",outscale))[0]
    outzp=outzp-(1<<32) if outzp>=1<<31 else outzp
    if not (math.isfinite(inscale) and inscale>0 and inzp<=255 and
            math.isfinite(outscale) and outscale>0 and -128<=outzp<=127):
        raise ValueError("invalid sequence quantization")
    descriptor_bytes=constant_count*CONSTANT.size
    if len(data)!=96+count*16+descriptor_bytes+payload_size: raise ValueError("incorrect sequence length")
    tasks=[]
    for i in range(count):
        offset,amount,enable,mask=TASK.unpack_from(data,96+i*16)
        # 0x300|0xc00 is the union mask a batched job needs when it mixes engines.
        valid_kind=(enable in (29,96) and mask in (768,3072,3840) and amount<=256) or \
                   (enable==24 and mask in (768,3840) and amount in (78,1106))
        if (offset%8 or not 1<=amount<=1106 or offset+(amount+4)*8>payload_size
            or not valid_kind):
            raise ValueError("invalid task descriptor")
        tasks.append(dict(command_offset=offset,register_count=amount,enable=enable,mask=mask))
    constants=[]
    for i in range(constant_count):
        raw,offset,size,kind,reserved=CONSTANT.unpack_from(data,96+count*16+i*CONSTANT.size)
        name_bytes=raw.split(b'\0',1)[0]
        try:name=name_bytes.decode('utf-8')
        except UnicodeDecodeError:raise ValueError('invalid constant name') from None
        if (not name or len(name_bytes)>=24 or raw[len(name_bytes)]!=0 or reserved or kind not in (1,2,3) or not size or offset+size>payload_size
            or name in {x['name'] for x in constants}):raise ValueError('invalid constant descriptor')
        if any(offset < t['command_offset']+(t['register_count']+4)*8 and t['command_offset'] < offset+size for t in tasks):
            raise ValueError('constant overlaps command program')
        constants.append(dict(name=name,offset=offset,size=size,kind=kind))
    checked=bytearray(data);checked[80:84]=b"\0"*4
    if checksum(checked)!=stored: raise ValueError("sequence checksum mismatch")
    return dict(target="rv1103",format_version=version,task_count=count,tasks=tasks,constants=constants,constant_count=constant_count,serial=bool(flags&1),input_tensor_count=input_tensor_count,
                shape_nhwc=[batch,ih,iw,ic],output_shape_nhwc=[batch,oh,ow,oc],batch=batch,
                input_layout="native16" if r1 else "packed",input_dtype="uint8",output_dtype="int8",input_scale=inscale,input_zero_point=inzp,
                output_scale=outscale,output_zero_point=outzp,
                input_bytes=batch*ih*iw*ic,output_bytes=batch*oh*ow*oc,input_stride=stride,
                payload_bytes=payload_size,arena_bytes=arena,input_offset=input_offset,
                output_offset=output_offset,task_bytes=4096)


# --- submission shape ------------------------------------------------------
# A non-serial container links every task and the tail's control word carries the
# *successor program's* fetch amount: the value the driver itself programs into
# PC_DATA_AMOUNT, `(regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT + scale - 1) / scale - 1`
# (extra 4, scale 2 on RV1106). A terminal tail keeps the 0x28 sentinel and a zero link.
# docs/plans/pipelining-plan.md S8/S10.
TERMINAL_CONTROL = 0x28


def amount_control(words):
    """The control word that links to a program of `words` register words."""
    return (words + 5) // 2 - 1


def payload_base(info):
    """Offset of the payload (where program offsets are relative to) in a container."""
    if info["format_version"] == 5:
        return 112 + 16 * info["task_count"] + 64 * info["tensor_count"]
    return 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)


def relink_for_batched(data):
    """Submit an existing sequence container as one linked job, without touching programs.

    Every descriptor's tail is rewritten to link to the next program with that
    program's fetch amount (`amount_control`), the last task becomes terminal and header
    flag bit 0 is cleared; the checksum is recomputed. Already-linked containers are
    returned unchanged, and a legacy `ORNPUBIN` container (one program, no task table)
    has no submission shape to change, so it is returned unchanged as well - which is
    correct for it: one program is one ioctl either way.
    """
    if data[:8] != MAGIC:
        return data
    info = decode_sequence(data)
    base = payload_base(info)
    out = bytearray(data)
    tasks = info["tasks"]
    for position, task in enumerate(tasks):
        start = base + task["command_offset"] + task["register_count"] * 8
        if position + 1 < len(tasks):
            successor = tasks[position + 1]
            link = successor["command_offset"]
            control = amount_control(successor["register_count"])
        else:
            link, control = 0, TERMINAL_CONTROL
        struct.pack_into("<Q", out, start, 0x101 << 48 | link << 16 | 0x10)
        struct.pack_into("<Q", out, start + 8, 0x101 << 48 | control << 16 | 0x14)
    flags = struct.unpack_from("<I", out, 84)[0] & ~1
    struct.pack_into("<I", out, 84, flags)
    out[80:84] = b"\0" * 4
    struct.pack_into("<I", out, 80, checksum(bytes(out)))
    return bytes(out)
