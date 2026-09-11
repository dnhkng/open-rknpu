"""MIT. Per-batch scalar/channel immutable Mul broadcasts."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_requant_reference

root=Path(__file__).resolve().parent/'mul_batch_broadcast_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110361);manifest=[]
cases=[(2,3,5,7,'batch'),(3,5,4,6,'batch_channel'),(8,1,3,5,'batch'),(16,16,2,3,'batch_channel')]
for i,(batch,c,height,width,kind) in enumerate(cases):
    factor=rng.uniform(-2,2,(batch,1,1,1) if kind=='batch' else (batch,c,1,1)).astype(np.float32)
    model=h.make_model(h.make_graph([h.make_node('Mul',['input','factor'],['output'])],'batch_broadcast',
        [h.make_tensor_value_info('input',1,[batch,c,height,width])],[h.make_tensor_value_info('output',1,[batch,c,height,width])],[nh.from_array(factor,'factor')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);requested={'scale':(.25,.75,.03125,2.)[i],'zero_point':(-128,127,-37,83)[i]}
    binary,meta=compile_sequence(path,output_range=requested);path.with_suffix('.bin').write_bytes(binary)
    q0=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['branches'][0]['quantization'].items()})
    fq=np.clip(np.rint(np.broadcast_to(factor,(batch,c,height,width))/meta['constant_scale']),-128,127).astype(np.int8)
    x=rng.integers(0,256,(8,batch,height,width,c),dtype=np.uint8);x.tofile(root/f'input{i:03}.u8')
    mult=meta['output_scale'];from open_rknpu.elementwise import mul_output_conversion
    _,_,conversion,shift=mul_output_conversion(meta['branches'][0]['output_scale']*meta['constant_scale'],requested)
    ys=[]
    for sample in x:
        a=np.stack([reference(sample[b],q0) for b in range(batch)])
        b=fq.transpose(0,2,3,1);b=np.broadcast_to(b,(batch,height,width,c))
        ys.append(mul_requant_reference(a,b,conversion,shift,requested['zero_point']))
    np.stack(ys).tofile(root/f'expected{i:03}.i8');manifest.append(dict(index=i,input_scale=1.0,input_zero_point=0,compile_output_range=requested,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
