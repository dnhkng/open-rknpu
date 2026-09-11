"""Independent ONNX fixtures for signed pooling experiments; no RKNN."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh

root=Path(__file__).resolve().parent/"generated"
rng=np.random.default_rng(110314)
for kind in ("max","average"):
    for relu in (False,True):
        w=rng.uniform(-.7,.7,(3,3,1,1)).astype(np.float32)
        b=rng.uniform(-4,4,3).astype(np.float32)
        nodes=[h.make_node("Conv",["input","weights","bias"],["conv" if relu else "hidden"],kernel_shape=[1,1])]
        if relu:nodes.append(h.make_node("Relu",["conv"],["hidden"]))
        nodes.append(h.make_node("MaxPool" if kind=="max" else "AveragePool",["hidden"],["output"],kernel_shape=[2,2],strides=[2,2]))
        g=h.make_graph(nodes,"independent_pool",[h.make_tensor_value_info("input",1,[1,3,8,8])],
                       [h.make_tensor_value_info("output",1,[1,3,4,4])],
                       [nh.from_array(w,"weights"),nh.from_array(b,"bias")])
        m=h.make_model(g,opset_imports=[h.make_opsetid("",13)]);m.ir_version=8
        onnx.checker.check_model(m);onnx.save(m,root/f"pool_{kind}_r{int(relu)}.onnx")
        m.graph.node[-1].output[0]="pool4"
        op=m.graph.node[-1].op_type
        m.graph.node.extend([h.make_node(op,["pool4"],["pool2"],kernel_shape=[2,2],strides=[2,2]),
                             h.make_node(op,["pool2"],["output"],kernel_shape=[2,2],strides=[2,2])])
        m.graph.output[0].type.tensor_type.shape.dim[2].dim_value=1
        m.graph.output[0].type.tensor_type.shape.dim[3].dim_value=1
        onnx.checker.check_model(m);onnx.save(m,root/f"reduce_{kind}_r{int(relu)}.onnx")
