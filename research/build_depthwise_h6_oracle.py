"""Development-only C5 oracle for register/packing comparison."""
from pathlib import Path
import shutil
import numpy as np
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/depthwise_h6';out.mkdir(exist_ok=True)
shutil.copyfile(root/'depthwise_expansion_suite/model001.onnx',out/'model.onnx')
paths=[]
for i in range(8):
 p=out/f'calibration{i}.npy';np.save(p,np.random.default_rng(i).uniform(0,255,(1,3,6,6)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
