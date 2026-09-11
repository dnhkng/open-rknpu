"""MIT. Summarize observed tasks and package captures for vendor-free replay."""
from pathlib import Path
import json
import re
import struct
from probe_primitives import NAMES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'primitive_survey'
report={}
for name in NAMES:
    cap=ROOT/('capture_primitive_'+name)
    logfile=ROOT/('primitive_'+name+'_capture.log')
    if not logfile.exists() or not (cap/'run3_submit.bin').exists():continue
    log=logfile.read_text()
    alloc={int(i):dict(size=int(s),dma=int(d,16)) for i,s,d in re.findall(r'ALLOC (\d+).*? size=(\d+).*? dma=([0-9a-f]+)',log)}
    outputs=[bytes.fromhex(v) for v in re.findall(r'^OUTPUT \w+ ([0-9a-f]+)$',log,re.M)]
    if len(outputs)!=4:continue
    entries=[];bundle=bytearray()
    for run in range(4):
        submit=(cap/f'run{run}_submit.bin').read_bytes()
        start,count=struct.unpack_from('<II',submit,8)
        base=struct.unpack_from('<Q',submit,40)[0]
        desc=(cap/f'run{run}_before_mem0.bin').read_bytes()
        if start:raise ValueError('partial submission requires separate analysis')
        memories={i:a for i,a in alloc.items() if i!=0 and a['dma']>=base}
        arena=max(a['dma']-base+a['size'] for a in memories.values())
        if arena>262144:raise ValueError('capture exceeds small replay bound')
        data=bytearray(arena)
        for i,a in memories.items():
            b=(cap/f'run{run}_before_mem{i}.bin').read_bytes()
            off=a['dma']-base;data[off:off+len(b)]=b
        # board_probe allocates its external input/output last, in that order.
        output_offset=alloc[max(alloc)]['dma']-base
        specs=[];tasks=[]
        for i in range(count):
            flags,op,enable,mask,clear,status,amount,offset,command=struct.unpack_from('<8IQ',desc,i*40)
            command-=base
            specs.append((command,amount,enable,mask,op))
            regs={}
            for w in struct.unpack_from('<'+str(amount)+'Q',data,command):
                regs[f'{w&65535:04x}']=f'{(w>>16)&0xffffffff:08x}'
            tasks.append(dict(enable=enable,mask=mask,register_count=amount,
                blocks=sorted({r[0]+'xxx' for r in regs}),registers=regs))
        bundle+=struct.pack('<5I',count,arena,output_offset,len(outputs[run]),struct.unpack_from('<I',submit)[0])
        bundle+=b''.join(struct.pack('<5I',*t) for t in specs)+data+outputs[run]
        if run==0:entries=tasks
    (OUT/(name+'.replay')).write_bytes(bundle)
    report[name]=dict(tasks=entries,task_count=len(entries),replay_bytes=len(bundle),
                      status='vendor captured; independent generation not implemented')
(OUT/'tasks.json').write_text(json.dumps(report,indent=2)+'\n')
for n,r in report.items():print(n,r['task_count'],[(t['enable'],t['register_count']) for t in r['tasks']],r['replay_bytes'])
