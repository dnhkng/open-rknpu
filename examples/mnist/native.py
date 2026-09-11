"""MIT licensed. Model-specific extension using the verified native14 Conv profile."""
import struct
from open_rknpu.chain import NATIVE, native_quantize
from open_rknpu.register_profile import REGISTERS
from open_rknpu.sequence import decode_sequence, encode_sequence


def extend(binary, weights, bias, output_range=None):
    if weights.shape != (16, 8, 5, 5) or bias.shape != (16,):
        raise ValueError("native MNIST profile requires 16x8x5x5 weights")
    meta = decode_sequence(binary)
    if meta['output_shape_nhwc'] != [1, 14, 14, 8] or meta['task_count'] != 2:
        raise ValueError("expected the trained MNIST two-task prefix")
    q = native_quantize(weights, bias, meta['output_scale'], meta['output_zero_point'], output_range)
    payload = binary[96+16*meta['task_count']:]
    data = bytearray(12288)
    data[:len(payload)] = payload
    # Constants/programs remain below 12 KiB; shift existing activations above it.
    shift = 12288
    tasks = []
    addresses = {0x1070, 0x4020, 0x6070, 0x701c}
    first = {}
    for task in meta['tasks']:
        off, count = task['command_offset'], task['register_count']
        tasks.append((off, count, task['enable'], task['mask']))
        for i in range(count):
            word = struct.unpack_from('<Q', data, off+8*i)[0]
            reg, value, tag = word & 65535, (word >> 16) & 0xffffffff, word >> 48
            if reg in addresses:
                value += shift
                struct.pack_into('<Q', data, off+8*i, tag << 48 | value << 16 | reg)
            if off == 0:
                first[reg] = value
    command, weight_offset, bias_offset = 2496, 3584, 9984
    if len(payload) > command:
        raise ValueError("prefix payload exceeds model-specific allocation")
    input_offset = meta['output_offset'] + shift
    output_offset = meta['arena_bytes'] + shift
    fields = dict(first)
    fields.update(NATIVE)
    fields.update({0x1010:0x9c,0x1020:14<<16|14,0x1024:7<<16|16,
        0x1028:14,0x102c:196,0x1030:6400,0x1034:400,0x1038:0x5050010,
        0x103c:7<<16,0x1044:14<<16|7,0x1068:0x202,0x1070:input_offset,
        0x1078:0x400f400f,0x107c:56,0x1080:196,0x1084:14<<16|14,
        0x1110:weight_offset,0x1184:meta['output_zero_point']&65535,
        0x1188:200,0x118c:13<<16|13,0x4020:output_offset,0x403c:15<<16|15,
        0x4060:0x13,0x406c:0x80000000,0x40e0:0x80000000,
        0x4080:q.output_zero_point&0xffffffff,0x4084:q.multiplier,
        0x4088:q.shift,0x5020:bias_offset,
        0x3014:13<<16|13,0x4024:3136,0x4030:13,0x4034:13,
        0x405c:13<<16|13,0x40c0:3136,0x500c:13,0x5010:13})
    for i,(reg,default,tag) in enumerate(REGISTERS):
        struct.pack_into('<Q',data,command+i*8,tag<<48|fields.get(reg,default)<<16|reg)
    for i,(reg,value,tag) in enumerate(((0x10,0,0x101),(0x14,0x28,0x101),(0,0,0x41),(8,29,0x81))):
        struct.pack_into('<Q',data,command+(126+i)*8,tag<<48|value<<16|reg)
    for oc in range(16):
        w=q.weights[oc].reshape(8,5,5)
        for y in range(5):
            for x in range(5):
                row=list(w[:,y,x])+[int(q.weight_zero_points[oc])]*8
                struct.pack_into('<16b',data,weight_offset+((y*5+x)*16+oc)*16,*row)
        block=bias_offset+oc//4*32; lane=oc%4
        struct.pack_into('<i',data,block+lane*4,int(q.biases[oc]))
        struct.pack_into('<h',data,block+16+lane*2,-int(q.weight_zero_points[oc]))
        struct.pack_into('<H',data,block+24+lane*2,int(q.channel_multipliers[oc]))
    tasks.append((command,126,29,768))
    result=encode_sequence(data,input_shape=(28,28,1),output_shape=(14,14,16),input_stride=32,
        arena_bytes=output_offset+4096,input_offset=meta['input_offset']+shift,
        output_offset=output_offset,tasks=tasks,input_scale=meta['input_scale'],
        input_zero_point=meta['input_zero_point'],output_scale=q.output_scale,
        output_zero_point=q.output_zero_point,serial=True)
    return result,q
