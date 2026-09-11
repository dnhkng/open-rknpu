"""Reproducible independent 1x1 output-channel coverage for the public C API."""
from pathlib import Path
import json
import argparse
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.quantization import Quantization, reference

parser=argparse.ArgumentParser();parser.add_argument("--kernel",type=int,choices=(1,3,5),default=1)
parser.add_argument("--input-channels",type=int,choices=(1,3),default=3)
args=parser.parse_args();kernel=args.kernel;input_channels=args.input_channels
root=Path(__file__).resolve().parents[1]/("research/wide_suite" if kernel==1 else f"research/wide_k{kernel}_suite")
if input_channels==1: root=root.parent/f"gray_k{kernel}_suite"
root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110310)
manifest=[]
for channels in range(2,17):
    for relu in (False,True):
        index=len(manifest)
        height,width=5+channels%4,5+(channels+1)%4
        weights=rng.uniform(-.7,.7,(channels,input_channels,kernel,kernel)).astype(np.float32)
        bias=rng.uniform(-4,4,channels).astype(np.float32)
        nodes=[h.make_node("Conv",["input","weights","bias"],
                           ["hidden" if relu else "output"],kernel_shape=[kernel,kernel],pads=[kernel//2]*4)]
        if relu: nodes.append(h.make_node("Relu",["hidden"],["output"]))
        graph=h.make_graph(nodes,"wide_channels",
            [h.make_tensor_value_info("input",1,[1,input_channels,height,width])],
            [h.make_tensor_value_info("output",1,[1,channels,height,width])],
            [nh.from_array(weights,"weights"),nh.from_array(bias,"bias")])
        model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)])
        model.ir_version=8
        path=root/f"model{index:03}.onnx"
        onnx.save(model,path)
        payload,metadata=compile_model(path)
        path.with_suffix(".bin").write_bytes(encode(payload,metadata))
        params=metadata["quantization"]
        for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
            params[key]=np.array(params[key])
        q=Quantization(**params)
        inputs=rng.integers(0,256,(16,height,width,input_channels),dtype=np.uint8)
        inputs[0]=0; inputs[1]=255; inputs[2]=128
        inputs.tofile(root/f"input{index:03}.u8")
        np.stack([reference(x,q) for x in inputs]).tofile(root/f"expected{index:03}.i8")
        manifest.append({"index":index,"height":height,"width":width,
                         "output_channels":channels,"relu":relu,"inputs":16})
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Built {len(manifest)} independent models")
