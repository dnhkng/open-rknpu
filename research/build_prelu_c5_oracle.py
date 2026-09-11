"""Development-only C5 PRelu table oracle."""
from pathlib import Path
import numpy as np,onnx
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/prelu_c5';out.mkdir(exist_ok=True);model=onnx.load(root/'prelu_public_suite/model001.onnx');onnx.save(model,out/'model.onnx');rng=np.random.default_rng(110363);paths=[]
for i in range(8):p=out/f'calibration{i}.npy';np.save(p,rng.uniform(-1,1,(1,3,5,8)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n');r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103')==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
