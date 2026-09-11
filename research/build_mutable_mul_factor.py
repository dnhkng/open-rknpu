"""MIT. Build a v4 Mul whose packed per-channel factor changes at runtime."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence,CONSTANT
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_reference

root=Path(__file__).resolve().parent/'mutable_mul_factor_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110373)
factors=[np.array([[[[1.]],[[.5]],[[-1.]]]],np.float32),np.array([[[[-1.]],[[1.]],[[-.5]]]],np.float32)]
compiled=[]
for i,factor in enumerate(factors):
    m=h.make_model(h.make_graph([h.make_node('Mul',['input','factor'],['output'])],'mutable_factor',
        [h.make_tensor_value_info('input',1,[1,3,5,7])],[h.make_tensor_value_info('output',1,[1,3,5,7])],[nh.from_array(factor,'factor')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/('model000.onnx' if i==0 else 'donor.onnx');onnx.save(m,p);compiled.append(compile_sequence(p,mutable_constants=True))
binary,meta=compiled[0];(root/'model000.bin').write_bytes(binary);donor,dmeta=compiled[1];info=decode_sequence(donor);entry=info['constants'][0]
payload=96+16*info['task_count']+CONSTANT.size*info['constant_count'];(root/'parameters.bin').write_bytes(donor[payload+entry['offset']:payload+entry['offset']+entry['size']])
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in dmeta['branches'][0]['quantization'].items()});factor_q=np.clip(np.rint(factors[1]/dmeta['constant_scale']),-128,127).astype(np.int8)[0].transpose(1,2,0)
x=rng.integers(0,256,(8,5,7,3),dtype=np.uint8);x.tofile(root/'input000.u8');x[0].tofile(root/'input_single.u8')
y=np.stack([mul_reference(reference(v,q),factor_q) for v in x]);y.tofile(root/'expected000.i8');y[0].tofile(root/'expected_single.i8')
(root/'manifest.json').write_text(json.dumps([dict(meta,index=0,input_scale=1.,input_zero_point=0,parameter_bytes=entry['size'])],indent=2)+'\n')
