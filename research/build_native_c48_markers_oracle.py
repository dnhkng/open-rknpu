"""Development-only one-hot marker oracle for C48 weight byte ordering."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/native_c48_markers';out.mkdir(exist_ok=True)
w=np.zeros((17,48,3,3),np.float32);channels=[0,1,2,7,8,15,16,17,23,24,31,32,33,39,40,46,47]
for o,c in enumerate(channels):w[o,c,(o//3)%3,o%3]=1
b=np.zeros(17,np.float32)
model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[3,3],pads=[1]*4)],'native_c48_markers',
 [h.make_tensor_value_info('input',1,[1,48,6,5])],[h.make_tensor_value_info('output',1,[1,17,6,5])],
 [nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8;onnx.save(model,out/'model.onnx')
rng=np.random.default_rng(4817);paths=[]
for i in range(8):p=out/f'calibration{i}.npy';np.save(p,rng.uniform(-1,1,(1,48,6,5)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n');r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103')==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
