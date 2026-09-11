"""MIT. Patch the RV1103 elementwise offset field for discriminating experiments."""
from pathlib import Path
import json,struct
from open_rknpu.model import checksum
from open_rknpu.sequence import decode_sequence

source=Path(__file__).resolve().parent/'mul_output_quantization_suite'
root=Path(__file__).resolve().parent/'mul_offset_probe';root.mkdir(exist_ok=True)
original=(source/'model000.bin').read_bytes();info=decode_sequence(original)
payload=96+16*info['task_count'];command=payload+info['tasks'][-1]['command_offset']
values=[0,1,0xffff,0x10000,0xffff0000,0x00010001,0xffffffff,0x00800080]
for i,value in enumerate(values):
    data=bytearray(original)
    for pos in range(command,command+info['tasks'][-1]['register_count']*8,8):
        word=struct.unpack_from('<Q',data,pos)[0]
        if word&0xffff==0x4074:
            struct.pack_into('<Q',data,pos,(word&0xffff00000000ffff)|(value<<16));break
    else:raise RuntimeError('missing 0x4074')
    data[80:84]=b'\0'*4;struct.pack_into('<I',data,80,checksum(data))
    (root/f'model{i:03}.bin').write_bytes(data)
inputs=(source/'input000.u8').read_bytes()
(root/'input.u8').write_bytes(inputs[2*info['input_bytes']:3*info['input_bytes']])
(root/'manifest.json').write_text(json.dumps(values)+'\n')
