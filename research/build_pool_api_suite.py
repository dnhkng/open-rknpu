"""Wrap independent pooling graphs for persistent public C API checks."""
from pathlib import Path
import json
import argparse
import numpy as np
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.quantization import Quantization,reference

root=Path(__file__).resolve().parents[1]/"research"
parser=argparse.ArgumentParser();parser.add_argument("--levels",type=int,choices=(1,3),default=1)
levels=parser.parse_args().levels
out=root/("pool_api_suite" if levels==1 else "reduction_api_suite");out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110315)
manifest=[]
for kind in ("max","average"):
    for relu in (0,1):
        index=len(manifest)
        prefix="pool" if levels==1 else "reduce"
        payload,meta=compile_model(root/f"generated/{prefix}_{kind}_r{relu}.onnx")
        (out/f"model{index:03}.bin").write_bytes(encode(payload,meta))
        params=meta["quantization"]
        for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
            params[key]=np.array(params[key])
        q=Quantization(**params)
        inputs=rng.integers(0,256,(64,8,8,3),dtype=np.uint8)
        inputs[0]=0;inputs[1]=255;inputs[2]=128
        outputs=[]
        for x in inputs:
            y=reference(x,q).astype(np.int64);size=8
            for stage in range(levels):
                size//=2;x=y.reshape(size,2,size,2,3)
                y=x.max(axis=(1,3)) if kind=="max" else np.rint(x.sum(axis=(1,3))/4).astype(np.int64)
            outputs.append(y.astype(np.int8))
        inputs.tofile(out/f"input{index:03}.u8")
        np.stack(outputs).tofile(out/f"expected{index:03}.i8")
        manifest.append({"index":index,"pool":kind,"relu":relu,"inputs":64})
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
