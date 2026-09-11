"""MIT. RK3588 TRM/Mesa hypotheses tested on independently generated RV1103 Add.
No vendor binaries or captured commands are consumed.
"""
from pathlib import Path
import json
import struct
import numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.model import checksum
from open_rknpu.quantization import Quantization,reference

root=Path(__file__).resolve().parent
out=root/'add_register_suite';out.mkdir(exist_ok=True)
binary,meta=compile_sequence(root/'add_suite/model000.onnx')
qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()}) for b in meta['branches']]
inputs=np.fromfile(root/'add_suite/input000.u8',np.uint8).reshape(-1,8,8,3)
a,b=[np.stack([reference(x,q) for x in inputs]) for q in qs]
summed=a.astype(np.int32)+b.astype(np.int32)
baseline=np.rint((a.astype(np.int32)+b.astype(np.int32))/2).astype(np.int8)
report=[]
for i,(name,reg,mask,value,expected) in enumerate([
    ('out_cvt_round_bit30',0x4088,1<<30,1<<30,(np.sign(summed)*((abs(summed)+1)//2)).astype(np.int8)),
    ('ew_alu_algo_max',0x4070,15<<16,0,np.rint(np.maximum(a,b).astype(np.int32)/2).astype(np.int8))]):
    data=bytearray(binary);start=96+3*16+0x880;found=False
    for j in range(78):
        off=start+j*8;word=struct.unpack_from('<Q',data,off)[0]
        if word&65535==reg:
            old=(word>>16)&0xffffffff;new=(old&~mask)|value
            struct.pack_into('<Q',data,off,(word>>48)<<48|new<<16|reg);found=True
    assert found
    data[80:84]=b'\0'*4;struct.pack_into('<I',data,80,checksum(data))
    (out/f'model{i:03}.bin').write_bytes(data)
    inputs.tofile(out/f'input{i:03}.u8');expected.tofile(out/f'expected{i:03}.i8')
    changes=int(np.count_nonzero(expected!=baseline));assert changes
    report.append(dict(index=i,hypothesis=name,predicted_changed_outputs=changes))
(out/'hypotheses.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
