"""Development-only vendor oracle for a C48 intermediate feeding K3 Conv."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/native_c48';out.mkdir(exist_ok=True)
rng=np.random.default_rng(481703)
nodes=[h.make_node('Conv',['input','w1','b1'],['hidden']),h.make_node('Relu',['hidden'],['active']),
       h.make_node('Conv',['active','w2','b2'],['output'],kernel_shape=[3,3],pads=[1]*4)]
init=[nh.from_array(rng.uniform(-.12,.12,s).astype(np.float32),n) for n,s in
      [('w1',(48,3,1,1)),('b1',(48,)),('w2',(17,48,3,3)),('b2',(17,))]]
model=h.make_model(h.make_graph(nodes,'native_c48',[h.make_tensor_value_info('input',1,[1,3,6,5])],
 [h.make_tensor_value_info('output',1,[1,17,6,5])],init),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
onnx.save(model,out/'model.onnx');paths=[]
for i in range(8):
 p=out/f'calibration{i}.npy';np.save(p,rng.uniform(0,255,(1,3,6,5)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n');r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
