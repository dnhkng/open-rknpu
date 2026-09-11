"""MIT. Verify independent output conversion for immutable broadcast Mul."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

root=Path(__file__).resolve().parent/'mul_broadcast_output_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110332);manifest=[]
cases=[('scalar',5,5,.5,-80),('channel',6,7,2.,31),('spatial',7,6,.25,100),('full',8,8,8.,-17)]
for i,(kind,ih,iw,outscale,outzp) in enumerate(cases):
    shapes={'scalar':(),'channel':(1,3,1,1),'spatial':(ih,iw),'full':(1,3,ih,iw)}
    factor=rng.uniform(-1.5,1.5,shapes[kind]).astype(np.float32) if shapes[kind] else np.array(-.75,np.float32)
    model=h.make_model(h.make_graph([h.make_node('Mul',['factor','input'] if i%2 else ['input','factor'],['output'])],
        'broadcast_output',[h.make_tensor_value_info('input',1,[1,3,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,3,ih,iw])],[nh.from_array(factor,'factor')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);inscale,inzp=((1.,0),(.25,128),(.5,255))[i%3]
    output_range=dict(scale=outscale,zero_point=outzp);binary,meta=compile_sequence(path,inscale,inzp,output_range);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['branches'][0]['quantization'].items()})
    qb=np.clip(np.rint(np.broadcast_to(factor,(1,3,ih,iw))/meta['constant_scale']),-128,127).astype(np.int8)[0].transpose(1,2,0)
    _,_,mult,shift=mul_output_conversion(q.output_scale*meta['constant_scale'],output_range)
    x=rng.integers(0,256,(12,ih,iw,3),dtype=np.uint8);x[0]=0;x[1]=255;x[2]=inzp
    y=np.stack([mul_requant_reference(reference(v,q),qb,mult,shift,outzp) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(meta,index=i,kind=kind,input_shape=[ih,iw,3],input_scale=inscale,input_zero_point=inzp,compile_output_range=output_range))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
