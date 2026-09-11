"""Test native-intermediate arithmetic against a two-layer vendor capture."""
from pathlib import Path
import struct
import numpy as np

ROOT=Path(__file__).resolve().parent

def verify(name="chain4",channels=4,size=8,input_channels=3,output_channels=3):
    capture=ROOT/f"capture_{name}"
    data=(capture/"run0_before_mem1.bin").read_bytes()
    regs=[{w&65535:(w>>16)&0xffffffff for w in struct.unpack_from("<126Q",data,o)}
          for o in (0,0x440)]
    def layer(inputs,r,outputs,input_channels,stride,relu):
        kernel=(r[0x1038]>>24)&255
        weights=np.frombuffer(data,dtype=np.int8,count=((outputs+3)//4*4 if stride==4 else outputs)*stride*kernel*kernel,
                              offset=r[0x1110]).reshape(kernel,kernel,((outputs+3)//4*4 if stride==4 else outputs),stride)
        weights=weights.transpose(2,3,0,1)[:outputs,:input_channels].reshape(outputs,-1).astype(np.int64)
        bias=[]; offsets=[]; multipliers=[]
        for c in range(outputs):
            block=r[0x5020]+c//4*32; lane=c%4
            bias.append(struct.unpack_from("<i",data,block+lane*4)[0])
            offsets.append(struct.unpack_from("<h",data,block+16+lane*2)[0])
            multipliers.append(struct.unpack_from("<H",data,block+24+lane*2)[0])
        if kernel>1:
            pad=kernel//2
            padded=np.pad(inputs,((pad,pad),(pad,pad),(0,0)),constant_values=-128)
            inputs=np.lib.stride_tricks.sliding_window_view(padded,(kernel,kernel),axis=(0,1)).reshape(size,size,-1)
        acc=np.einsum("hwc,oc->hwo",inputs,weights+np.array(offsets)[:,None])+bias
        if relu: acc=np.maximum(acc,0)
        product=acc*multipliers
        scaled=(product+8191+((product>>14)&1))>>14
        shift=r[0x4088]
        zp=r[0x4080]-(1<<32 if r[0x4080]>=1<<31 else 0)
        return np.clip(((scaled*r[0x4084]+(1<<(shift-1)))>>shift)+zp,-128,127).astype(np.int64)
    outputs=[bytes.fromhex(x.split()[2]) for x in (ROOT/f"{name}_capture.log").read_text().splitlines()
             if x.startswith("OUTPUT ")]
    assert len(outputs)==4
    for i,raw in enumerate(outputs):
        inputs=np.frombuffer((capture/f"run{i}_before_mem2.bin").read_bytes(),dtype=np.uint8)[:size*((size+15)//16*16)*input_channels].reshape(size,((size+15)//16*16),input_channels)[:,:size].astype(np.int64)
        hidden=layer(inputs-128,regs[0],channels,input_channels,4,True)
        predicted=layer(hidden,regs[1],output_channels,channels,16,False)
        observed=np.frombuffer(raw,dtype=np.int8).reshape(size,size,16)[:,:,:output_channels]
        np.testing.assert_array_equal(predicted,observed)
    print(f"{name}: {4*size*size*output_channels} bytes match; native signed INT8 intermediate consumed directly")

if __name__=="__main__":
    verify()
