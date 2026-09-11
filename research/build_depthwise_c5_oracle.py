"""Development-only C5 oracle for register/packing comparison."""
from pathlib import Path
import shutil
from rknn.api import RKNN
root=Path(__file__).resolve().parent;out=root/'fixtures/depthwise_c5';out.mkdir(exist_ok=True)
shutil.copyfile(root/'depthwise_c5_suite/model000.onnx',out/'model.onnx')
r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(root/'fixtures/primitive_base/dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
