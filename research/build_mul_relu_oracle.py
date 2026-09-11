"""Development-only two-branch Mul followed by Relu oracle."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/mul_relu';out.mkdir(exist_ok=True);rng=np.random.default_rng(110360);constants=[];nodes=[]
for i in range(2):
 w=rng.uniform(-.3,.3,(3,3,1,1)).astype(np.float32);b=rng.uniform(-.5,.5,3).astype(np.float32);constants.extend([nh.from_array(w,f'w{i}'),nh.from_array(b,f'b{i}')]);nodes.append(h.make_node('Conv',['input',f'w{i}',f'b{i}'],[f'branch{i}'],kernel_shape=[1,1]))
nodes.extend([h.make_node('Mul',['branch0','branch1'],['product']),h.make_node('Relu',['product'],['output'])]);m=h.make_model(h.make_graph(nodes,'mul_relu',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,8,8])],constants),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;onnx.save(m,out/'model.onnx');paths=[]
for i in range(8):p=out/f'calibration{i}.npy';np.save(p,rng.uniform(-1,1,(1,3,8,8)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n');r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103')==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
