"""MIT. Standalone constant broadcasting, including actual per-channel DMA mode."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_reference
root=Path(__file__).resolve().parent/'mul_broadcast_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110327);manifest=[]
for i,kind in enumerate(('scalar','channel','row','column','spatial','full','zero','one','negative','channel','full','spatial')):
    height=5+i%4;width=5+(i//4)
    shapes={'scalar':(),'channel':(1,3,1,1),'row':(1,1,height,1),'column':(width,),'spatial':(height,width),'full':(1,3,height,width),'zero':(),'one':(),'negative':()}
    factor=np.asarray(rng.uniform(-2,2,shapes[kind]),np.float32)
    if kind in ('zero','one','negative'):factor=np.array({'zero':0.,'one':1.,'negative':-1.}[kind],np.float32)
    m=h.make_model(h.make_graph([h.make_node('Mul',['input','factor'] if i%2 else ['factor','input'],['output'])],'broadcast',
        [h.make_tensor_value_info('input',1,[1,3,height,width])],[h.make_tensor_value_info('output',1,[1,3,height,width])],[nh.from_array(factor,'factor')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    data,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['branches'][0]['quantization'].items()})
    b=np.clip(np.rint(np.broadcast_to(factor,(1,3,height,width))/meta['constant_scale']),-128,127).astype(np.int8)[0].transpose(1,2,0)
    x=rng.integers(0,256,(32,height,width,3),dtype=np.uint8);x[0]=0;x[1]=255;x[2]=zp
    x.tofile(root/f'input{i:03}.u8');np.stack([mul_reference(reference(v,q),b) for v in x]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_scale=scale,input_zero_point=zp,kind=kind,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
