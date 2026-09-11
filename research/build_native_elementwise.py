"""MIT. Two logical RGB inputs concatenated in one runtime input byte buffer."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_reference,add_reference,sub_reference,max_reference
root=Path(__file__).resolve().parent/'native_elementwise_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110326);manifest=[]
for i,(kind,height,width,channels) in enumerate((kind,ht,wd,c) for kind in ('Add','Mul','Sub','Max') for ht,wd in ((1,1),(5,5),(9,9),(32,32)) for c in (1,2,8,16)):
    oc=(1,4,9,16)[i%4]
    constants=[];nodes=[]
    for j,v in enumerate(('a','b')):
        w=rng.uniform(-.3,.3,(oc,channels,1,1)).astype(np.float32);bias=rng.uniform(-1,1,oc).astype(np.float32)
        constants.extend([nh.from_array(w,f'w{j}'),nh.from_array(bias,f'b{j}')]);nodes.append(h.make_node('Conv',[v,f'w{j}',f'b{j}'],[f'c{j}'],kernel_shape=[1,1]))
    nodes.append(h.make_node(kind,['c0','c1'],['output']))
    m=h.make_model(h.make_graph(nodes,'two_inputs',[h.make_tensor_value_info(v,1,[1,channels,height,width]) for v in ('a','b')],[h.make_tensor_value_info('output',1,[1,oc,height,width])],constants),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    data,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(data)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()}) for b in meta['branches']]
    a=rng.integers(0,256,(8,height,width,channels),dtype=np.uint8);b=rng.integers(0,256,a.shape,dtype=np.uint8)
    a[0]=0;b[0]=255;a[1]=255;b[1]=0
    np.concatenate([a,b],axis=1).tofile(root/f'input{i:03}.u8')
    np.stack([{'Mul':mul_reference,'Add':add_reference,'Sub':sub_reference,'Max':max_reference}[kind](reference(x,qs[0]),reference(y,qs[1])) for x,y in zip(a,b)]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_scale=scale,input_zero_point=zp,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
