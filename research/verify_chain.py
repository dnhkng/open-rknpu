"""Verify independently emitted chain output against integer and float models."""
from pathlib import Path
import argparse
import json
import numpy as np
from emit_chain import Quantization,reference,native_reference
from verify_generated import inputs_for

def quantization(params):
    params=dict(params)
    for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
        params[key]=np.array(params[key])
    return Quantization(**params)

def verify(log,metadata):
    meta=json.loads(Path(metadata).read_text())
    q1=quantization(meta["first"]["quantization"]);q2=quantization(meta["second"])
    names=set();count=0;errors=[]
    for line in Path(log).read_text().splitlines():
        if not line.startswith("OUTPUT "):continue
        _,name,raw=line.split();assert name not in names;names.add(name)
        inputs=inputs_for(name,8,8)
        expected=native_reference(reference(inputs,q1),q2)
        observed=np.frombuffer(bytes.fromhex(raw),dtype=np.int8).reshape(8,8,16)[:,:,:3]
        np.testing.assert_array_equal(observed,expected)
        hidden=np.maximum(np.einsum("hwc,oc->hwo",inputs.astype(float),
            np.array(meta["first"]["float_weights"]).reshape(meta["hidden_channels"],3))+
            np.array(meta["first"]["float_bias"]),0)
        target=np.einsum("hwc,oc->hwo",hidden,np.array(meta["weights2"]).reshape(3,-1))+np.array(meta["bias2"])
        decoded=(observed.astype(float)-q2.output_zero_point)*q2.output_scale
        errors.extend(np.abs(decoded-target).ravel());count+=observed.size
    assert names>={"zero","constant","ramp","impulse"}
    print(f"PASS: {len(names)} two-layer inferences, {count} exact bytes; float max error={max(errors):.6g}, mean={np.mean(errors):.6g}, output scale={q2.output_scale:.6g}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("log");p.add_argument("metadata")
    args=p.parse_args();verify(args.log,args.metadata)
