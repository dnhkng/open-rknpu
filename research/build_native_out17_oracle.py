"""Development-only C5 oracle for register/packing comparison."""
from pathlib import Path
import shutil
import numpy as np
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/native_out17';out.mkdir(exist_ok=True)
import onnx
from onnx import helper as h,numpy_helper as nh
rng=np.random.default_rng(9)
nodes=[h.make_node('Conv',['input','w1','b1'],['hidden']),h.make_node('Relu',['hidden'],['active']),h.make_node('Conv',['active','w2','b2'],['output'])]
init=[nh.from_array(rng.uniform(-.3,.3,shape).astype(np.float32),name) for name,shape in [('w1',(5,3,1,1)),('b1',(5,)),('w2',(17,5,1,1)),('b2',(17,))]]
m=h.make_model(h.make_graph(nodes,'native9',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,17,8,8])],init),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;onnx.save(m,out/'model.onnx')
paths=[]
for i in range(8):
 p=out/f'calibration{i}.npy';np.save(p,np.random.default_rng(i).uniform(0,255,(1,3,8,8)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
