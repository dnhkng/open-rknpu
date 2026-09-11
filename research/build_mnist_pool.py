"""Compile and reference-check the trained MNIST Conv/Relu/MaxPool prefix."""
from pathlib import Path
import json
import numpy as np
import onnx
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

root=Path(__file__).resolve().parent
suite=root/"mnist_pool_suite";suite.mkdir(exist_ok=True)
model=onnx.load(root/"pretrained/mnist/normalized.onnx");g=model.graph
nodes=list(g.node[:3]);output=next(v for v in g.value_info if v.name==nodes[-1].output[0])
prefix=onnx.helper.make_model(onnx.helper.make_graph(nodes,"mnist_first_pool",list(g.input),
    [output],list(g.initializer)),opset_imports=list(model.opset_import))
prefix.ir_version=model.ir_version
path=suite/"model000.onnx";onnx.save(prefix,path)
input_meta=json.loads((root/"mnist_first_suite/manifest.json").read_text())[0]
data,meta=compile_sequence(path,input_meta["input_scale"],input_meta["input_zero_point"])
path.with_suffix(".bin").write_bytes(data)
inputs=np.fromfile(root/"mnist_first_suite/input000.u8",np.uint8).reshape(-1,28,28,1)
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta["quantization"].items()})
expected=np.stack([reference(x,q).reshape(14,2,14,2,8).max(axis=(1,3)) for x in inputs])
inputs.tofile(suite/"input000.u8");expected.tofile(suite/"expected000.i8")
print("Built trained MNIST prefix with serial NPU tasks")
