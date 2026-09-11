"""MIT. Fresh fused LeakyRelu integer tests."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.activation import leaky_reference
root=Path(__file__).resolve().parent/'leaky_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110324);manifest=[]
for i in range(24):
    k=(1,3,5)[i%3];height=5+i%4;width=5+(i//4)%4;ic=1 if i%2 else 3;oc=(1,3,5,16)[i%4];alpha=(0.,.1,.2,.5,1.,.01)[i%6]
    w=rng.uniform(-.3,.3,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-2,2,oc).astype(np.float32)
    m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['conv'],kernel_shape=[k,k],pads=[k//2]*4),h.make_node('LeakyRelu',['conv'],['output'],alpha=alpha)],'leaky',
        [h.make_tensor_value_info('input',1,[1,ic,height,width])],[h.make_tensor_value_info('output',1,[1,oc,height,width])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    data,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=rng.integers(0,256,(32,height,width,ic),dtype=np.uint8);x[0]=0;x[1]=255;x[2]=zp
    x.tofile(root/f'input{i:03}.u8');np.stack([leaky_reference(v,q,meta['leaky_alpha']) for v in x]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
