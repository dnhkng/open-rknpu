"""Independent depthwise stride-2 hypothesis from our compiler output."""
from pathlib import Path
import struct
import numpy as np
from open_rknpu.model import checksum
root=Path(__file__).resolve().parent
out=root/'depthwise_stride2_suite';out.mkdir(exist_ok=True)
for i in range(12):
    data=bytearray((root/'depthwise_suite'/f'model{i:03}.bin').read_bytes())
    struct.pack_into('<II',data,28,4,4)
    fields={0x1014:18,0x1028:4,0x102c:16,0x3014:3<<16|3,0x4024:256,
            0x4030:3,0x4034:3,0x405c:3<<16|3,0x40c0:512,0x500c:3,0x5010:3}
    for j in range(126):
        off=96+2*16+0x440+j*8;word=struct.unpack_from('<Q',data,off)[0];reg=word&65535
        if reg in fields:struct.pack_into('<Q',data,off,(word>>48)<<48|fields[reg]<<16|reg)
    data[80:84]=b'\0'*4;struct.pack_into('<I',data,80,checksum(data))
    (out/f'model{i:03}.bin').write_bytes(data)
    (out/f'input{i:03}.u8').write_bytes((root/'depthwise_suite'/f'input{i:03}.u8').read_bytes())
    expected=np.fromfile(root/'depthwise_suite'/f'expected{i:03}.i8',np.int8).reshape(16,8,8,3)
    expected[:,::2,::2].tofile(out/f'expected{i:03}.i8')
