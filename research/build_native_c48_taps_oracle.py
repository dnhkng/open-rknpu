"""Development-only C48 tap-order oracle: one input channel, nine taps valued 1..9. Development oracle only; never read by the open emitters."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper as h, numpy_helper as nh
from rknn.api import RKNN
out=Path(__file__).resolve().parent/'fixtures/native_c48_taps';out.mkdir(parents=True,exist_ok=True)
w=np.zeros((1,48,3,3),np.float32)
for ky in range(3):
    for kx in range(3):
        w[0,0,ky,kx]=1+ky*3+kx   # values 1..9, only input channel 0
model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[3,3],pads=[1]*4)],
 'native_c48_taps',[h.make_tensor_value_info('input',1,[1,48,6,5])],[h.make_tensor_value_info('output',1,[1,1,6,5])],
 [nh.from_array(w,'w'),nh.from_array(np.zeros(1,np.float32),'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
onnx.save(model,out/'model.onnx')
rng=np.random.default_rng(480023);paths=[]
for i in range(8):
    p=out/f'calibration{i}.npy';np.save(p,rng.uniform(0,255,(1,48,6,5)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
r=RKNN(verbose=False)
try:
    assert r.config(target_platform='rv1103',mean_values=[[0]*48],std_values=[[1]*48])==0
    assert r.load_onnx(model=str(out/'model.onnx'))==0
    assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
    assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
print('built',out/'model.rknn',(out/'model.rknn').stat().st_size)
