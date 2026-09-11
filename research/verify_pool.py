"""Compare independently emitted pooling against its quantized computation."""
import argparse,json
from pathlib import Path
import numpy as np
from verify_chain import quantization,reference
from verify_generated import inputs_for

def verify(log,metadata):
    meta=json.loads(Path(metadata).read_text());q=quantization(meta["quantization"])
    names=set();count=0
    for line in Path(log).read_text().splitlines():
        if not line.startswith("OUTPUT "):continue
        _,name,raw=line.split();assert name not in names;names.add(name)
        expected=reference(inputs_for(name,8,8),q).astype(np.int64)
        size=8
        for stage in range(meta.get("pool_levels",1)):
            size//=2;x=expected.reshape(size,2,size,2,3)
            expected=x.max(axis=(1,3)) if meta["pool"]=="MaxPool" else np.rint(x.sum(axis=(1,3))/4).astype(np.int64)
        observed=np.frombuffer(bytes.fromhex(raw),np.int8).reshape(size,size,16)[:,:,:3]
        np.testing.assert_array_equal(observed,expected)
        count+=observed.size
    assert names>={"zero","constant","ramp","impulse"}
    print(f"{meta['pool']}: {len(names)} inputs, {count} exact pooled bytes")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("log");p.add_argument("metadata");a=p.parse_args();verify(a.log,a.metadata)
