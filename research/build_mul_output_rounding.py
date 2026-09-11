"""MIT. Distinguish Mul output-zero-point placement at exact halfway products."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

root=Path(__file__).resolve().parent/'mul_output_rounding_suite';root.mkdir(exist_ok=True)
model=h.make_model(h.make_graph([h.make_node('Mul',['a','b'],['output'])],'mul_round',
    [h.make_tensor_value_info(v,1,[1,3,5,5]) for v in ('a','b')],[h.make_tensor_value_info('output',1,[1,3,5,5])]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
path=root/'model000.onnx';onnx.save(model,path);_,first=compile_sequence(path,1,0)
q0=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in z['quantization'].items()}) for z in first['branches']]
outscale=2*q0[0].output_scale*q0[1].output_scale;out=dict(scale=outscale,zero_point=1)
binary,meta=compile_sequence(path,1,0,out);path.with_suffix('.bin').write_bytes(binary)
qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in z['quantization'].items()}) for z in meta['branches']]
_,_,mult,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale,out)
rng=np.random.default_rng(110336);a=rng.integers(0,256,(8,5,5,3),dtype=np.uint8);b=rng.integers(0,256,a.shape,dtype=np.uint8)
# Identity branches map these odd source codes to odd signed values, making
# product/2 an exact half. An odd output zp reverses the ties-even parity choice.
a[:,0,0]=[3,5,7];b[:,0,0]=[3,5,7]
y=np.stack([mul_requant_reference(reference(x,qs[0]),reference(z,qs[1]),mult,shift,1) for x,z in zip(a,b)])
np.concatenate([a,b],axis=1).tofile(root/'input000.u8');y.tofile(root/'expected000.i8')
(root/'manifest.json').write_text(json.dumps([dict(index=0,input_scale=1.,input_zero_point=0,output_scale=outscale,output_zero_point=1,multiplier=mult&65535,shift=shift,compile_output_range=out)],indent=2)+'\n')
