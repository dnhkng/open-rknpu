"""Evaluate candidate fixed-point arithmetic against complete oracle outputs."""
from pathlib import Path
import struct
import re
import numpy as np

ROOT=Path(__file__).resolve().parent
def read_case(name):
    capture=ROOT/("capture_"+name)
    data=(capture/"run0_before_mem1.bin").read_bytes()
    regs={}
    for offset in range(0,126*8,8):
        word=struct.unpack_from("<Q",data,offset)[0]
        regs[word&65535]=(word>>16)&0xffffffff
    w=np.array(struct.unpack_from("<12b",data,regs[0x1110]),dtype=np.int64).reshape(3,4)[:,:3]
    p=regs[0x5020]
    bias=np.array(struct.unpack_from("<3i",data,p),dtype=np.int64)
    z=np.array(struct.unpack_from("<3h",data,p+16),dtype=np.int64)
    scales=np.array(struct.unpack_from("<3H",data,p+24),dtype=np.int64)
    zp=np.int32(regs[0x4080]&0x7fffffff)-(0x80000000 if regs[0x4080]&0x80000000 else 0)
    # After-SUBMIT mmap snapshots precede the client's FROM_DEVICE cache sync;
    # use the runner's synchronized readback as the correctness oracle.
    lines=(ROOT/(name+"_capture.log")).read_text().splitlines()
    log_outputs=[bytes.fromhex(line.split()[2]) for line in lines if line.startswith("OUTPUT ")]
    cases=[]
    for run in range(4):
        inputs=np.frombuffer((capture/("run%d_before_mem2.bin"%run)).read_bytes(),dtype=np.uint8)[:384].reshape(8,16,3)[:,:8,:].astype(np.int64)
        outputs=np.frombuffer(log_outputs[run],dtype=np.int8).reshape(8,8,16)[:,:,:3].astype(np.int64)
        acc=np.einsum("hwc,oc->hwo",inputs-128,w+z[:,None])+bias
        cases.append((acc,outputs))
    return regs,scales,zp,cases

if __name__=="__main__":
    for name in ("identity","dense","signed","biased"):
        regs,scales,zp,cases=read_case(name)
        multiplier,shift=regs[0x4084],regs[0x4088]
        print(name,"channel",scales,"global",multiplier,shift,"zp",zp)
        for rounding in ("floor","nearest"):
            for stage in ("combined","separate"):
                differences=[]
                for acc,out in cases:
                    if stage=="combined":
                        rshift=shift+14
                        numerator=acc*scales*multiplier
                    else:
                        rshift=shift
                        numerator=((acc*scales+(8192 if rounding=="nearest" else 0))>>14)*multiplier
                    result=(numerator+((1<<(rshift-1)) if rounding=="nearest" else 0))>>rshift
                    result=np.clip(result+zp,-128,127)
                    differences.extend((result-out).ravel())
                d=np.array(differences)
                print(rounding,stage,"mismatches",np.count_nonzero(d),"maxerr",np.max(np.abs(d)))
