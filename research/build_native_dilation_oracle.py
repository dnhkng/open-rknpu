"""Development-only vendor oracle for unequal native atrous dilation."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/native_dilation';out.mkdir(exist_ok=True);rng=np.random.default_rng(110323)
w=rng.uniform(-.2,.2,(5,3,3,3)).astype(np.float32);b=rng.uniform(-.5,.5,5).astype(np.float32)
m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[3,3],dilations=[2,3],pads=[2,3,2,3])],'native_dilation',
 [h.make_tensor_value_info('input',1,[1,3,8,9])],[h.make_tensor_value_info('output',1,[1,5,8,9])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;onnx.save(m,out/'model.onnx')
paths=[]
for i in range(8):p=out/f'calibration{i}.npy';np.save(p,rng.uniform(-1,1,(1,3,8,9)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n');r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103')==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
