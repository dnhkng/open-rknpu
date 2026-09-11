"""Independent public-API models covering every supported hidden channel count."""
from pathlib import Path
import json
import argparse
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.chain import native_reference
from open_rknpu.quantization import Quantization,reference

parser=argparse.ArgumentParser();parser.add_argument("--kernel",type=int,choices=(1,3),default=1)
parser.add_argument("--first-kernel",type=int,choices=(1,3),default=1)
args=parser.parse_args();kernel=args.kernel;first_kernel=args.first_kernel
root=Path(__file__).resolve().parents[1]/("research/chain_suite" if kernel==1 else "research/chain_k3_suite")
if first_kernel==3: root=root.with_name(f"chain_spatial_{kernel}_suite")
root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110313)
manifest=[]
for c in range(3,17):
    index=c-3
    tensors=[]
    for name,shape in (("w1",(c,3,first_kernel,first_kernel)),("w2",(3,c,kernel,kernel)),("b1",(c,)),("b2",(3,))):
        lo,hi=(-.7,.8) if len(shape)==4 else (-4,4)
        tensors.append(nh.from_array(rng.uniform(lo,hi,shape).astype(np.float32),name))
    nodes=[h.make_node("Conv",["input","w1","b1"],["conv1"],kernel_shape=[first_kernel,first_kernel],pads=[first_kernel//2]*4),
           h.make_node("Relu",["conv1"],["hidden"]),
           h.make_node("Conv",["hidden","w2","b2"],["output"],kernel_shape=[kernel,kernel],pads=[kernel//2]*4)]
    graph=h.make_graph(nodes,"public_chain",
        [h.make_tensor_value_info("input",1,[1,3,8,8])],
        [h.make_tensor_value_info("output",1,[1,3,8,8])],tensors)
    model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)]);model.ir_version=8
    path=root/f"model{index:03}.onnx";onnx.save(model,path)
    payload,meta=compile_model(path);path.with_suffix(".bin").write_bytes(encode(payload,meta))
    quantizers=[]
    for params in (meta["first"]["quantization"],meta["second"]):
        params=dict(params)
        for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
            params[key]=np.array(params[key])
        quantizers.append(Quantization(**params))
    inputs=rng.integers(0,256,(32,8,8,3),dtype=np.uint8)
    inputs[0]=0;inputs[1]=255;inputs[2]=128
    inputs.tofile(root/f"input{index:03}.u8")
    np.stack([native_reference(reference(x,quantizers[0]),quantizers[1]) for x in inputs]).tofile(root/f"expected{index:03}.i8")
    manifest.append({"index":index,"hidden_channels":c,"inputs":32})
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Built {len(manifest)} two-layer application models")
