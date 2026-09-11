"""MIT. Independent Conv VALID/stride geometry and signed integer references."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.native import compile_native_input
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference
root=Path(__file__).resolve().parent/'native_input_blocks_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110321);manifest=[]
for i,(k,ic) in enumerate((k,ic) for k in (1,3,5,7) for ic in (17,18,23,24,31,32)):
    oc=(1,3,8,17,32,64)[i%6]
    sy=sx=1;padded=True
    ih=5+i%4;iw=5+(i//4)%4;pad=k//2 if padded else 0;relu=False
    oh=(ih+2*pad-k)//sy+1;ow=(iw+2*pad-k)//sx+1
    w=rng.uniform(-.3,.3,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-1,1,oc).astype(np.float32)
    nodes=[h.make_node('Conv',['input','w','b'],['conv' if relu else 'output'],kernel_shape=[k,k],pads=[pad]*4,strides=[sy,sx])]
    if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
    m=h.make_model(h.make_graph(nodes,'geometry',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(m,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    data,meta=compile_native_input(m,scale,zp);path.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=rng.integers(0,256,(16,ih,iw,ic),dtype=np.uint8);x[0]=0;x[1]=255;x[2]=zp
    margin=k//2-pad
    y=np.stack([native_input_reference(v,q,zp)[margin:ih-margin:sy,margin:iw-margin:sx] for v in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kernel=k,strides=[sy,sx],pad=pad,input_shape=[ih,iw,ic],output_channels=oc,relu=relu,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
