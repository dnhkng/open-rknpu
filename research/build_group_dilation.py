"""MIT. Independent Conv VALID/stride geometry and signed integer references."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference
from open_rknpu.normalize import normalize_model
from onnx.reference import ReferenceEvaluator
root=Path(__file__).resolve().parent/'group_dilation_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110321);manifest=[]
cases=[(g,m,k,d) for g in (2,3,4,5,7,8,16) for m in (1,2) if g*m<=16 for k,d in ((1,(1,1)),(2,(1,2)),(3,(2,1)),(3,(2,2)))]
for i,(group,mult,k,dilation) in enumerate(cases):
    ic=group;sy,sx=((1,1),(1,2),(2,1),(2,2))[i%4];padded=True
    ih=5+i%4;iw=5+(i//4)%4;oc=group*mult;relu=bool(i%2)
    kh,kw=[(k-1)*d+1 for d in dilation];pads=[kh//2,kw//2,kh//2,kw//2]
    oh=(ih+pads[0]+pads[2]-kh)//sy+1;ow=(iw+pads[1]+pads[3]-kw)//sx+1
    w=rng.uniform(-.3,.3,(oc,1,k,k)).astype(np.float32);b=rng.uniform(-1,1,oc).astype(np.float32)
    nodes=[h.make_node('Conv',['input','w','b'],['conv' if relu else 'output'],kernel_shape=[k,k],pads=pads,strides=[sy,sx],group=group,dilations=dilation)]
    if relu:nodes.append(h.make_node('Relu',['conv'],['output']))
    m=h.make_model(h.make_graph(nodes,'geometry',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(m,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    data,meta=compile_sequence(path,scale,zp);path.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=rng.integers(0,256,(16,ih,iw,ic),dtype=np.uint8);x[0]=0;x[1]=255;x[2]=zp
    normal=normalize_model(m);test=x[3].transpose(2,0,1)[None].astype(np.float32)
    np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':test})[0],ReferenceEvaluator(normal).run(None,{'input':test})[0],rtol=1e-5,atol=1e-4)
    actual={a.name:h.get_attribute_value(a) for a in normal.graph.node[0].attribute}
    y=np.stack([native_input_reference(v,q,zp,actual.get('pads',[0]*4),[sy,sx]) for v in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kernel=k,strides=[sy,sx],pads=pads,group=group,multiplier=mult,dilation=dilation,input_shape=[ih,iw,ic],output_channels=oc,relu=relu,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
