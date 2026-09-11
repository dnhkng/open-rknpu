"""MIT. Verify independent Mul output scales and zero points."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference
root=Path(__file__).resolve().parent/'mul_output_quantization_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(1103280);manifest=[]
cases=[(5,5,1.,0,.5,-128),(6,5,.25,128,2.,37),(7,6,.5,255,8.,127),(8,7,1.,128,.125,-37),(5,8,.125,17,16.,0)]
for i,(ih,iw,inscale,inzp,outscale,outzp) in enumerate(cases):
 m=h.make_model(h.make_graph([h.make_node('Mul',['a','b'],['output'])],'mul_output_quantization',[h.make_tensor_value_info(v,1,[1,3,ih,iw]) for v in ('a','b')],[h.make_tensor_value_info('output',1,[1,3,ih,iw])]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;p=root/f'model{i:03}.onnx';onnx.save(m,p)
 binary,meta=compile_sequence(p,inscale,inzp,dict(scale=outscale,zero_point=outzp));p.with_suffix('.bin').write_bytes(binary);qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()}) for b in meta['branches']];_,_,mult,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale,dict(scale=outscale,zero_point=outzp))
 a=rng.integers(0,256,(16,ih,iw,3),dtype=np.uint8);b=rng.integers(0,256,a.shape,dtype=np.uint8);a[0]=0;b[0]=255;a[1]=255;b[1]=0
 np.concatenate([a,b],axis=1).tofile(root/f'input{i:03}.u8');y=np.stack([mul_requant_reference(reference(x,qs[0]),reference(z,qs[1]),mult,shift,outzp) for x,z in zip(a,b)]);y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,input_scale=inscale,input_zero_point=inzp,output_scale=outscale,output_zero_point=outzp,multiplier=mult&0xffff,shift=shift,input_shape=[ih,iw,3]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
