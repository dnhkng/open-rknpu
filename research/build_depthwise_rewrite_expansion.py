"""MIT. Verify direct depthwise Conv semantics through open dense rewriting."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.scheduler import compile_sequence
from open_rknpu.normalize import normalize_model
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'depthwise_rewrite_expansion_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110330);manifest=[]
cases=[(3,1,3,3,(2,2),(1,1)),(5,2,3,3,(3,2),(1,2)),
       (4,2,2,4,(1,1),(1,1)),(8,1,2,3,(1,1),(2,1)),
       (16,3,1,1,(1,1),(1,1)),(32,4,1,1,(1,1),(1,1))]
for i,(c,mul,kh,kw,dil,strides) in enumerate(cases):
    oc=c*mul;ih=9+i;iw=11+i;eh=(kh-1)*dil[0]+1;ew=(kw-1)*dil[1]+1
    pads=[eh//2,ew//2,(eh-1)//2,(ew-1)//2]
    oh=(ih+pads[0]+pads[2]-eh)//strides[0]+1;ow=(iw+pads[1]+pads[3]-ew)//strides[1]+1
    w=rng.uniform(-.15,.15,(oc,1,kh,kw)).astype(np.float32);b=rng.uniform(-.4,.4,oc).astype(np.float32)
    node=h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[kh,kw],group=c,
                     dilations=list(dil),pads=pads,strides=list(strides))
    model=h.make_model(h.make_graph([node],'depthwise_rewrite',[h.make_tensor_value_info('input',1,[1,c,ih,iw])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),
        opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    normal=normalize_model(model);sample=rng.uniform(-1,1,(1,c,ih,iw)).astype(np.float32)
    np.testing.assert_allclose(ReferenceEvaluator(model).run(None,{'input':sample})[0],ReferenceEvaluator(normal).run(None,{'input':sample})[0],rtol=1e-5,atol=1e-5)
    p=root/f'model{i:03}.onnx';onnx.save(model,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    binary,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    attrs={a.name:h.get_attribute_value(a) for a in normal.graph.node[0].attribute}
    x=rng.integers(0,256,(3,ih,iw,c),dtype=np.uint8);x[0]=zp
    y=np.stack([native_input_reference(v,q,zp,attrs['pads'],attrs.get('strides',(1,1)),attrs.get('dilations',(1,1))) for v in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_shape=[ih,iw,c],output_shape=[oh,ow,oc],multiplier=mul,
        kernel=[kh,kw],dilations=list(dil),strides=list(strides),pads=pads,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
