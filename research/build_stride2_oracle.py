"""Development-only stride-2 oracle."""
from pathlib import Path
import onnx
from onnx import helper as h
from rknn.api import RKNN
root=Path(__file__).resolve().parent
out=root/'fixtures/stride2';out.mkdir(exist_ok=True)
m=onnx.load(root/'stride2_suite/model000.onnx')
m.graph.node[0].attribute.append(h.make_attribute('strides',[2,2]))
for d in m.graph.output[0].type.tensor_type.shape.dim[2:]:d.dim_value=4
onnx.save(m,out/'model.onnx')
r=RKNN(verbose=False)
try:
    assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
    assert r.load_onnx(model=str(out/'model.onnx'))==0
    assert r.build(do_quantization=True,dataset=str(root/'fixtures/primitive_base/dataset.txt'))==0
    assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
