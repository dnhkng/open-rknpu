"""Two-head fan-out suite: shared stem, two consuming heads, two outputs.

Independently generated commands (no RKNN, no captures). Expected outputs come
from the documented integer references: the legacy stem reference followed by
the native-input reference for each head. Board runner: tests/board_io.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.chain import native_reference

parser=argparse.ArgumentParser()
parser.add_argument("--directory",default="research/two_head_suite")
parser.add_argument("--cases",type=int,default=32)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/args.directory
root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110320)
kernel_pairs=[(1,3),(3,1),(1,1),(3,3)]
manifest=[]

def quant_from(params):
    params=dict(params)
    for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
        params[key]=np.array(params[key])
    return Quantization(**params)

for hidden in range(3,17):
    index=hidden-3
    kernel_a,kernel_b=kernel_pairs[index%len(kernel_pairs)]
    tensors=[
        nh.from_array(rng.uniform(-.7,.8,(hidden,3,1,1)).astype(np.float32),"w1"),
        nh.from_array(rng.uniform(-4,4,(hidden,)).astype(np.float32),"b1"),
        nh.from_array(rng.uniform(-.7,.8,(3,hidden,kernel_a,kernel_a)).astype(np.float32),"wa"),
        nh.from_array(rng.uniform(-4,4,(3,)).astype(np.float32),"ba"),
        nh.from_array(rng.uniform(-.7,.8,(3,hidden,kernel_b,kernel_b)).astype(np.float32),"wb"),
        nh.from_array(rng.uniform(-4,4,(3,)).astype(np.float32),"bb"),
    ]
    nodes=[
        h.make_node("Conv",["input","w1","b1"],["stem"],kernel_shape=[1,1]),
        h.make_node("Relu",["stem"],["relu1"]),
        h.make_node("Conv",["relu1","wa","ba"],["outputA"],kernel_shape=[kernel_a,kernel_a],pads=[kernel_a//2]*4),
        h.make_node("Conv",["relu1","wb","bb"],["outputB"],kernel_shape=[kernel_b,kernel_b],pads=[kernel_b//2]*4),
    ]
    graph=h.make_graph(nodes,"two_head",
        [h.make_tensor_value_info("input",1,[1,3,8,8])],
        [h.make_tensor_value_info("outputA",1,[1,3,8,8]),h.make_tensor_value_info("outputB",1,[1,3,8,8])],
        tensors)
    graph.value_info.append(h.make_tensor_value_info("relu1",1,[1,hidden,8,8]))
    model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)]);model.ir_version=8
    path=root/f"model{index:03}.onnx";onnx.save(model,path)
    binary,meta=compile_sequence(path)
    path.with_suffix(".bin").write_bytes(binary)
    stem_q=quant_from(meta["stem_quantization"])
    head_q=[quant_from(params) for params in meta["head_quantization"]]
    stem_zp=int(meta["stem_quantization"]["output_zero_point"])
    inputs=rng.integers(0,256,(args.cases,8,8,3),dtype=np.uint8)
    inputs[0]=0;inputs[1]=255;inputs[2]=128
    inputs.tofile(root/f"input{index:03}.u8")
    expected=[]
    for x in inputs:
        stem=reference(x,stem_q)
        expected.append(b"".join(native_reference(stem,q,stem_zp).tobytes() for q in head_q))
    (root/f"expected{index:03}.i8").write_bytes(b"".join(expected))
    manifest.append({"index":index,"hidden_channels":hidden,"head_kernels":[kernel_a,kernel_b],
                     "cases":args.cases,"outputs":2,"output_bytes":384})
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Built {len(manifest)} two-head fan-out models in {root}")
