"""Compile pretrained MNIST's first Conv/Relu with affine UINT8 inputs.

No vendor compiler; input range derived from the one supplied upstream fixture.
This is first-layer verification, not dataset calibration or full inference.
"""
from pathlib import Path
import copy
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.quantization import Quantization, reference

root=Path(__file__).resolve().parent/"mnist_first_suite"
root.mkdir(exist_ok=True)
source=root.parent/"pretrained/mnist"
model=onnx.load(source/"normalized.onnx")
nodes=[copy.deepcopy(n) for n in model.graph.node[:2]]
assert [n.op_type for n in nodes]==["Conv","Relu"]
first=h.make_model(h.make_graph(nodes,"trained_mnist_first",[model.graph.input[0]],
    [h.make_tensor_value_info(nodes[-1].output[0],1,[1,8,28,28])],model.graph.initializer),
    opset_imports=list(model.opset_import))
first.ir_version=model.ir_version
x=nh.to_array(onnx.load_tensor(source/"mnist-12/test_data_set_0/input_0.pb"))
scale=float(np.float32((x.max()-x.min())/255))
zp=int(np.clip(np.rint(-x.min()/scale),0,255))
rng=np.random.default_rng(110321)
report=[]
for i,(input_scale,input_zp) in enumerate(((scale,zp),(.25,0),(.5,128),(.25,255))):
    path=root/f"model{i:03}.onnx";onnx.save(first,path)
    payload,meta=compile_model(path,input_scale=input_scale,input_zero_point=input_zp)
    path.with_suffix(".bin").write_bytes(encode(payload,meta))
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta["quantization"].items()})
    inputs=rng.integers(0,256,(8,28,28,1),dtype=np.uint8)
    inputs[0]=np.clip(np.rint(x[0].transpose(1,2,0)/input_scale)+input_zp,0,255).astype(np.uint8)
    inputs[1]=input_zp;inputs[2]=0;inputs[3]=255
    expected=np.stack([reference(v,q) for v in inputs])
    inputs.tofile(root/f"input{i:03}.u8");expected.tofile(root/f"expected{i:03}.i8")
    float_original=ReferenceEvaluator(first).run(None,{first.graph.input[0].name:x})[0][0].transpose(1,2,0)
    dequantized=(expected[0].astype(float)-q.output_zero_point)*q.output_scale
    error=abs(dequantized-float_original)
    report.append(dict(index=i,input_scale=input_scale,input_zero_point=input_zp,
                       output_scale=q.output_scale,fixture_mae=float(error.mean()),
                       fixture_max_error=float(error.max()),inputs=8))
(root/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report[0],indent=2))
