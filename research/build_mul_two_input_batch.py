"""MIT. Verify tensor-contiguous two-input batched Mul packing."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

root=Path(__file__).resolve().parent/'mul_two_input_batch_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110341);manifest=[]
for i,(batch,c,ih,iw,zps) in enumerate(((2,3,5,7,(17,-23)),(4,8,7,6,(-64,63)),(8,16,4,5,(-128,127)))):
    model=h.make_model(h.make_graph([h.make_node('Mul',['a','b'],['output'])],'mul_two_batch',[h.make_tensor_value_info(v,1,[batch,c,ih,iw]) for v in ('a','b')],[h.make_tensor_value_info('output',1,[batch,c,ih,iw])]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i];binary,meta=compile_sequence(path,scale,zp,mul_operand_zero_points=zps);path.with_suffix('.bin').write_bytes(binary)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in meta['branches']];_,outzp,mult,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale)
    a=rng.integers(0,256,(3,batch,ih,iw,c),dtype=np.uint8);b=rng.integers(0,256,a.shape,dtype=np.uint8);a[0]=zp;b[0]=255-zp
    y=np.stack([[mul_requant_reference(reference(x,qs[0]),reference(v,qs[1]),mult,shift,outzp,zps) for x,v in zip(ar,br)] for ar,br in zip(a,b)])
    np.concatenate([a,b],axis=1).tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_scale=scale,input_zero_point=zp,mul_operand_zero_points=list(zps),input_shape=[batch,ih,iw,c]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
