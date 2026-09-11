"""Development-only matched vendor capture for terminal Mul Clip[0,6]."""
from pathlib import Path
import numpy as np,onnx
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/mul_clip';out.mkdir(parents=True,exist_ok=True)
model=root/'mul_clip_suite/model000.onnx';base=out/'base.onnx';m=onnx.load(model);product=m.graph.node[-2].output[0];del m.graph.node[-1];m.graph.output[0].name=product;onnx.save(m,base);rng=np.random.default_rng(110379);paths=[]
for i in range(4):
    p=out/f'calibration_{i}.npy';np.save(p,rng.integers(0,256,(1,3,5,8)).astype(np.float32));paths.append(str(p))
(out/'dataset.txt').write_text('\n'.join(paths)+'\n')
for source,name in ((model,'model.rknn'),(base,'base.rknn')):
    r=RKNN(verbose=False)
    try:
        assert r.config(target_platform='rv1103',mean_values=[[0,0,0]],std_values=[[1,1,1]])==0
        assert r.load_onnx(model=str(source))==0
        assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
        assert r.export_rknn(str(out/name))==0
    finally:r.release()
