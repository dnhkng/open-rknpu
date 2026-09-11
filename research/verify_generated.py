"""Compare synchronized hardware output to integer and original float operators."""
from pathlib import Path
import argparse
import json
import numpy as np
from quantization import Quantization,reference,receptive_fields

def inputs_for(name,height,width):
    data=np.zeros((height,width,3),dtype=np.uint8)
    if name=="constant": data.fill(128)
    elif name=="ramp": data[:]=np.arange(data.size,dtype=np.uint8).reshape(data.shape)
    elif name=="impulse": data[3,4,1]=255
    elif name.startswith("random"):
        run=int(name[6:])+4
        state=(1103+run*0x9e3779b9)&0xffffffff
        for i in range(data.size):
            state=(1664525*state+1013904223)&0xffffffff
            data.flat[i]=state>>24
    elif name!="zero": raise ValueError("unknown input case "+name)
    return data

def verify(log,metadata):
    meta=json.loads(Path(metadata).read_text())
    params=meta["quantization"]
    for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
        params[key]=np.array(params[key])
    q=Quantization(**params)
    _,height,width,_=meta["shape_nhwc"]
    channels=len(q.weights)
    counts={"combined":0,"separate":0}
    max_errors={"combined":0,"separate":0}
    float_errors=[]
    seen=[]
    for line in Path(log).read_text().splitlines():
        if not line.startswith("OUTPUT "): continue
        _,name,hexbytes=line.split()
        seen.append(name)
        raw=np.frombuffer(bytes.fromhex(hexbytes),dtype=np.int8)
        assert raw.size==height*width*16
        output=raw.reshape(height,width,16)[:,:,:channels]
        inputs=inputs_for(name,height,width)
        for mode in counts:
            expected=reference(inputs,q,mode)
            error=expected.astype(int)-output.astype(int)
            counts[mode]+=np.count_nonzero(error)
            max_errors[mode]=max(max_errors[mode],int(np.max(np.abs(error))))
        patches=receptive_fields(inputs.astype(float),q.kernel_size)
        target=np.einsum("hwc,oc->hwo",patches,np.array(meta["float_weights"]).reshape(channels,-1))+np.array(meta["float_bias"])
        if q.relu: target=np.maximum(target,0)
        decoded=(output.astype(float)-q.output_zero_point)*q.output_scale
        float_errors.extend(np.abs(target-decoded).ravel())
    assert len(seen)==len(set(seen)),"duplicate cases"
    assert set(seen)>={"zero","constant","ramp","impulse"},"missing base cases"
    print("cases",len(seen),"logical values",len(seen)*height*width*channels)
    print("integer mismatches",counts,"max error",max_errors)
    print("float absolute error: max %.6g, mean %.6g; output scale %.6g"%(max(float_errors),np.mean(float_errors),q.output_scale))
    return counts

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",type=Path)
    parser.add_argument("metadata",type=Path)
    args=parser.parse_args()
    counts=verify(args.log,args.metadata)
    if counts["separate"]: raise SystemExit("hardware differs from the established two-stage rounding reference")
