"""MIT. Verify immutable Conv weight/bias edge cases and quantizer modes."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'weight_edge_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110331);manifest=[]
cases=[(1,4,3,'zero-no-bias'),(17,5,1,'bias-only'),(3,17,5,'zero-channels'),(32,16,3,'mixed-ranges')]
for i,(ic,oc,k,kind) in enumerate(cases):
    ih=7+i;iw=9+i;p=k//2;w=np.zeros((oc,ic,k,k),np.float32);b=np.zeros(oc,np.float32)
    if kind=='bias-only':b=np.linspace(-2,2,oc,dtype=np.float32)
    elif kind=='zero-channels':w[1::2]=rng.uniform(-.3,.2,w[1::2].shape).astype(np.float32);b=rng.uniform(-.2,.2,oc).astype(np.float32)
    elif kind=='mixed-ranges':
        for o in range(oc):w[o]=rng.uniform(-.001*(o+1),.003*(o+1),w[o].shape)
        b=rng.uniform(-1,1,oc).astype(np.float32)
    inputs=['input','w'] if kind=='zero-no-bias' else ['input','w','b']
    initial=[nh.from_array(w,'w')]+([] if len(inputs)==2 else [nh.from_array(b,'b')])
    model=h.make_model(h.make_graph([h.make_node('Conv',inputs,['output'],kernel_shape=[k,k],pads=[p]*4)],
        'weight_edges',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,ih,iw])],initial),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    output_range=None
    binary,meta=compile_sequence(path,scale,zp,output_range);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(v,q,zp,[p]*4) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(meta,index=i,kind=kind,input_shape=[ih,iw,ic],output_shape=[ih,iw,oc],
                         compile_output_range=output_range))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
