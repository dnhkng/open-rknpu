"""Development-only native geometry oracle for H64/W33."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from rknn.api import RKNN

root=Path(__file__).resolve().parent;out=root/'fixtures/native_h64w33';out.mkdir(exist_ok=True)
rng=np.random.default_rng(6433)
nodes=[h.make_node('Conv',['input','w1','b1'],['hidden']),h.make_node('Relu',['hidden'],['active']),
       h.make_node('Conv',['active','w2','b2'],['output'])]
init=[nh.from_array(rng.uniform(-.3,.3,shape).astype(np.float32),name) for name,shape in
      [('w1',(5,3,1,1)),('b1',(5,)),('w2',(3,5,1,1)),('b2',(3,))]]
model=h.make_model(h.make_graph(nodes,'native_h64w33',[h.make_tensor_value_info('input',1,[1,3,64,33])],
    [h.make_tensor_value_info('output',1,[1,3,64,33])],init),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
onnx.save(model,out/'model.onnx');paths=[]
for i in range(8):
    path=out/f'calibration{i}.npy';np.save(path,rng.uniform(0,255,(1,3,64,33)).astype(np.float32));paths.append(str(path))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
r=RKNN(verbose=False)
try:
    assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
    assert r.load_onnx(model=str(out/'model.onnx'))==0
    assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
    assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
