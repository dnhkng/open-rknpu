"""MIT. Verify batched same-input Mul within the single-tensor native16 ABI."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

root=Path(__file__).resolve().parent/'mul_batch_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110339);manifest=[]
for i,(batch,c,ih,iw,zps) in enumerate(((2,1,5,7,(0,0)),(4,5,7,6,(17,-23)),(16,16,4,5,(-128,127)))):
    model=h.make_model(h.make_graph([h.make_node('Mul',['input','input'],['output'])],'mul_batch',[h.make_tensor_value_info('input',1,[batch,c,ih,iw])],[h.make_tensor_value_info('output',1,[batch,c,ih,iw])]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i]
    binary,meta=compile_sequence(path,scale,zp,mul_operand_zero_points=zps);path.with_suffix('.bin').write_bytes(binary)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in meta['branches']]
    _,outzp,mult,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale)
    x=rng.integers(0,256,(3,batch,ih,iw,c),dtype=np.uint8);x[0]=zp
    y=np.stack([[mul_requant_reference(reference(v,qs[0]),reference(v,qs[1]),mult,shift,outzp,zps) for v in run] for run in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_scale=scale,input_zero_point=zp,mul_operand_zero_points=list(zps),input_shape=[batch,ih,iw,c]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
