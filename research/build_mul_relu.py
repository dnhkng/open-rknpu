"""MIT. Verify fused Relu after external-plus-intermediate Mul."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

base=Path(__file__).resolve().parent;source_root=base/'mul_external_intermediate_suite';root=base/'mul_relu_suite';root.mkdir(exist_ok=True);source_manifest=json.loads((source_root/'manifest.json').read_text());manifest=[]
for entry in source_manifest:
 index=entry['index'];model=onnx.load(source_root/f'model{index:03}.onnx');mul=model.graph.node[-1];mul.output[0]='product';model.graph.node.append(h.make_node('Relu',['product'],['output']));path=root/f'model{index:03}.onnx';onnx.save(model,path);out_range=entry['compile_output_range'];zps=tuple(entry['mul_operand_zero_points']);binary,meta=compile_sequence(path,output_range=out_range,mul_operand_zero_points=zps);path.with_suffix('.bin').write_bytes(binary)
 height,width,_=entry['shape'];packed=np.fromfile(source_root/f'input{index:03}.u8',np.uint8).reshape(-1,2*height,width,3);a=packed[:,:height];b=packed[:,height:];qs=[Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in branch['quantization'].items()}) for branch in meta['branches']];_,output_zp,conversion,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale,out_range);expected=np.stack([np.maximum(mul_requant_reference(reference(x,qs[0]),reference(y,qs[1]),conversion,shift,output_zp,zps),output_zp) for x,y in zip(a,b)])
 packed.tofile(root/f'input{index:03}.u8');expected.tofile(root/f'expected{index:03}.i8');manifest.append(dict(entry,fused_activation='Relu'))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
