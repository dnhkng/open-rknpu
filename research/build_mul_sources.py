"""MIT. Verify same-input and operand-order standalone Mul sources."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_reference

root=Path(__file__).resolve().parent/'mul_sources_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110329);manifest=[]
cases=[('square',1,1,7),('square',5,9,6),('square',16,32,17),
       ('reverse',3,5,8),('reverse',16,11,13)]
for i,(kind,c,ih,iw) in enumerate(cases):
    names=['a'] if kind=='square' else ['a','b'];operands=['a','a'] if kind=='square' else ['b','a']
    m=h.make_model(h.make_graph([h.make_node('Mul',operands,['output'])],'mul_sources',
        [h.make_tensor_value_info(v,1,[1,c,ih,iw]) for v in names],
        [h.make_tensor_value_info('output',1,[1,c,ih,iw])]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(binary)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()}) for b in meta['branches']]
    a=rng.integers(0,256,(16,ih,iw,c),dtype=np.uint8);a[0]=0;a[1]=255;a[2]=zp
    if kind=='square':
        packed=a;y=np.stack([mul_reference(reference(x,qs[0]),reference(x,qs[1])) for x in a])
    else:
        b=rng.integers(0,256,a.shape,dtype=np.uint8);b[0]=255;b[1]=0;b[2]=zp
        packed=np.concatenate([a,b],axis=1)
        y=np.stack([mul_reference(reference(yv,qs[0]),reference(x,qs[1])) for x,yv in zip(a,b)])
    packed.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kind=kind,input_scale=scale,input_zero_point=zp,input_shape=[ih,iw,c],**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
