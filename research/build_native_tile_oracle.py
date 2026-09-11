"""Development-only oracle for a C17/K5 geometry that needs tiling."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from rknn.api import RKNN

root=Path(__file__).resolve().parent;out=root/'fixtures/native_tile';out.mkdir(exist_ok=True)
rng=np.random.default_rng(4817)
nodes=[h.make_node('Conv',['input','w1','b1'],['hidden']),h.make_node('Relu',['hidden'],['active']),
       h.make_node('Conv',['active','w2','b2'],['output'],kernel_shape=[5,5],pads=[2]*4,strides=[2,3])]
init=[nh.from_array(rng.uniform(-.2,.2,shape).astype(np.float32),name) for name,shape in
      [('w1',(17,3,1,1)),('b1',(17,)),('w2',(1,17,5,5)),('b2',(1,))]]
model=h.make_model(h.make_graph(nodes,'native_tile',[h.make_tensor_value_info('input',1,[1,3,48,33])],
    [h.make_tensor_value_info('output',1,[1,1,24,11])],init),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
onnx.save(model,out/'model.onnx');paths=[]
for i in range(8):
    path=out/f'calibration{i}.npy';np.save(path,rng.uniform(0,255,(1,3,48,33)).astype(np.float32));paths.append(str(path))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
r=RKNN(verbose=False)
try:
    assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
    assert r.load_onnx(model=str(out/'model.onnx'))==0
    assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
    assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
