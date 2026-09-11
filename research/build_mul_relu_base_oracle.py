"""Development-only unfused companion for the Mul-to-Relu capture."""
from pathlib import Path
import onnx
from rknn.api import RKNN
root=Path(__file__).resolve().parent;source=root/'fixtures/mul_relu';out=root/'fixtures/mul_relu_base';out.mkdir(exist_ok=True);model=onnx.load(source/'model.onnx');del model.graph.node[-1];model.graph.output[0].name='product';onnx.save(model,out/'model.onnx');(out/'dataset.txt').write_text((source/'dataset.txt').read_text());r=RKNN(verbose=False)
try:
 assert r.config(target_platform='rv1103')==0
 assert r.load_onnx(model=str(out/'model.onnx'))==0
 assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
 assert r.export_rknn(str(out/'model.rknn'))==0
finally:r.release()
