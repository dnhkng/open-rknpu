"""Development-only depthwise ConvTranspose dilation-2 oracles."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from rknn.api import RKNN

root=Path(__file__).resolve().parent;rng=np.random.default_rng(110348)
for k in (2,3):
 out=root/'fixtures'/f'transpose_dilation_k{k}';out.mkdir(exist_ok=True);effective=2*(k-1)+1
 w=rng.uniform(-.25,.25,(3,1,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,3).astype(np.float32)
 m=h.make_model(h.make_graph([h.make_node('ConvTranspose',['input','w','b'],['output'],kernel_shape=[k,k],strides=[2,2],dilations=[2,2],group=3)],f'transpose_dilation_k{k}',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,14+effective,14+effective])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;onnx.save(m,out/'model.onnx');paths=[]
 for i in range(8):p=out/f'calibration{i}.npy';np.save(p,rng.uniform(-1,1,(1,3,8,8)).astype(np.float32));paths.append(str(p))
 (out/'dataset.txt').write_text('\n'.join(paths)+'\n');r=RKNN(verbose=False)
 try:
  assert r.config(target_platform='rv1103')==0
  assert r.load_onnx(model=str(out/'model.onnx'))==0
  assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
  assert r.export_rknn(str(out/'model.rknn'))==0
 finally:r.release()
