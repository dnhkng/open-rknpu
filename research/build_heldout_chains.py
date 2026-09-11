"""Build new two-layer ONNX graphs with no vendor compiler invocation."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh

root=Path(__file__).resolve().parent/"generated"
root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110312)
for c in (3,4,8,16):
    tensors=[]
    for name,shape in (("w1",(c,3,1,1)),("w2",(3,c,1,1)),("b1",(c,)),("b2",(3,))):
        lo,hi=(-.6,.8) if len(shape)==4 else (-5,5)
        tensors.append(nh.from_array(rng.uniform(lo,hi,shape).astype(np.float32),name))
    nodes=[h.make_node("Conv",["input","w1","b1"],["conv1"],kernel_shape=[1,1]),
           h.make_node("Relu",["conv1"],["hidden"]),
           h.make_node("Conv",["hidden","w2","b2"],["output"],kernel_shape=[1,1])]
    graph=h.make_graph(nodes,"independent_chain",
        [h.make_tensor_value_info("input",1,[1,3,8,8])],
        [h.make_tensor_value_info("output",1,[1,3,8,8])],tensors)
    model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)]);model.ir_version=8
    onnx.checker.check_model(model)
    onnx.save(model,root/f"chain_heldout{c}.onnx")
print("Built four independently specified Conv-Relu-Conv graphs")
