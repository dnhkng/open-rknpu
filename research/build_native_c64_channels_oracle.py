"""Development-only C64 input-channel order oracle: one tap, channels valued 1+2c. Development oracle only; never read by the open emitters."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper as h, numpy_helper as nh
from rknn.api import RKNN
out=Path(__file__).resolve().parent/'fixtures/native_c64_channels';out.mkdir(parents=True,exist_ok=True)
w=np.zeros((1,64,3,3),np.float32)
for c in range(64): w[0,c,0,0]=1+2*c
model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[3,3],pads=[1]*4)],
 'native_c64_channels',[h.make_tensor_value_info('input',1,[1,64,6,5])],[h.make_tensor_value_info('output',1,[1,1,6,5])],
 [nh.from_array(w,'w'),nh.from_array(np.zeros(1,np.float32),'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
onnx.save(model,out/'model.onnx')
rng=np.random.default_rng(480031);paths=[]
for i in range(8):
    p=out/f'calibration{i}.npy';np.save(p,rng.uniform(0,255,(1,64,6,5)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
r=RKNN(verbose=False)
try:
    assert r.config(target_platform='rv1103',mean_values=[[0]*64],std_values=[[1]*64])==0
    assert r.load_onnx(model=str(out/'model.onnx'))==0
    assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
    assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
print('built',(out/'model.rknn').stat().st_size)
