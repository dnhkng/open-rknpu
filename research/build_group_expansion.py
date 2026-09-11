"""MIT. Verify broad grouped Conv by exact zero-filled dense lowering."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.scheduler import compile_sequence
from open_rknpu.normalize import normalize_model
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization
root=Path(__file__).resolve().parent/'group_expansion_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110332128);manifest=[]
cases=[(2,16,64,3,(1,1)),(3,10,33,3,(2,3)),(4,8,17,5,(1,2)),(16,2,4,3,(3,2)),(32,1,1,1,(1,1)),(16,1,8,3,(2,3))]
for i,(group,per_in,per_out,k,dil) in enumerate(cases):
 ic=group*per_in;oc=group*per_out;ih=9+i;iw=10+i;eh=(k-1)*dil[0]+1;ew=(k-1)*dil[1]+1;pads=[min(15,eh//2),min(15,ew//2)]*2;strides=((1,1),(1,2),(2,1))[i%3]
 oh=(ih+pads[0]+pads[2]-eh)//strides[0]+1;ow=(iw+pads[1]+pads[3]-ew)//strides[1]+1;w=rng.uniform(-.12,.12,(oc,per_in,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32)
 m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],group=group,dilations=list(dil),pads=pads,strides=list(strides))],'group_expansion',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
 normal=normalize_model(m);sample=rng.uniform(-1,1,(1,ic,ih,iw)).astype(np.float32);np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':sample})[0],ReferenceEvaluator(normal).run(None,{'input':sample})[0],rtol=1e-5,atol=1e-5)
 p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3];binary,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(binary);q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
 attrs={a.name:h.get_attribute_value(a) for a in normal.graph.node[0].attribute};x=rng.integers(0,256,(3,ih,iw,ic),dtype=np.uint8);x[0]=zp;y=np.stack([native_input_reference(v,q,zp,attrs['pads'],attrs.get('strides',(1,1)),attrs.get('dilations',(1,1))) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],group=group,kernel=k,dilations=list(dil),strides=list(strides),pads=pads,input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
