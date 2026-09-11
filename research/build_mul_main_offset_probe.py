"""MIT. Probe BS ALU as the primary Mul operand zero-point control."""
from pathlib import Path
import json,struct
from open_rknpu.model import checksum
from open_rknpu.sequence import decode_sequence

source=Path(__file__).resolve().parent/'standalone_mul_suite';root=Path(__file__).resolve().parent/'mul_main_offset_probe';root.mkdir(exist_ok=True)
original=(source/'model000.bin').read_bytes();info=decode_sequence(original);payload=96+16*info['task_count'];command=payload+info['tasks'][-1]['command_offset']
cases=[('base',0x100012,0),('operand_only',0x100012,1),('mesa_add_pos',0x120150,1),('mesa_add_neg',0x120150,0xffffffff),('source0_pos',0x120050,1),('source0_neg',0x120050,0xffffffff)]
for i,(_,cfg,operand) in enumerate(cases):
    data=bytearray(original)
    for pos in range(command,command+info['tasks'][-1]['register_count']*8,8):
        word=struct.unpack_from('<Q',data,pos)[0];reg=word&65535
        value={0x4040:cfg,0x4044:operand}.get(reg)
        if value is not None:struct.pack_into('<Q',data,pos,(word&0xffff00000000ffff)|(value<<16))
    data[80:84]=b'\0'*4;struct.pack_into('<I',data,80,checksum(data));(root/f'model{i:03}.bin').write_bytes(data)
inputs=(source/'input000.u8').read_bytes();(root/'input.u8').write_bytes(inputs[2*info['input_bytes']:3*info['input_bytes']])
(root/'manifest.json').write_text(json.dumps(cases,indent=2)+'\n')
