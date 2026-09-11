"""MIT. Execute SAME_UPPER/SAME_LOWER/VALID normalization, including odd totals."""
from pathlib import Path
import copy,json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.scheduler import compile_sequence
from open_rknpu.normalize import normalize_model
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization
root=Path(__file__).resolve().parent/'auto_padding_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110300);manifest=[]
cases=[(5,7,3,5,2,(2,2),(1,1),'SAME_UPPER'),(5,7,3,5,2,(2,2),(1,1),'SAME_LOWER'),
 (7,9,17,3,4,(2,2),(1,1),'SAME_UPPER'),(7,9,17,3,4,(2,2),(1,1),'SAME_LOWER'),
 (8,9,3,1,3,(2,2),(3,2),'SAME_UPPER'),(8,9,3,1,3,(2,2),(3,2),'SAME_LOWER'),(8,9,32,3,3,(2,3),(1,1),'VALID')]
for i,(ih,iw,ic,oc,k,strides,dil,auto) in enumerate(cases):
 eh=(k-1)*dil[0]+1;ew=(k-1)*dil[1]+1;oh=(ih+strides[0]-1)//strides[0] if auto!='VALID' else (ih-eh)//strides[0]+1;ow=(iw+strides[1]-1)//strides[1] if auto!='VALID' else (iw-ew)//strides[1]+1
 w=rng.uniform(-.13,.13,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32)
 m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],strides=list(strides),dilations=list(dil),auto_pad=auto) if auto!='VALID' else h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],strides=list(strides),auto_pad='VALID')],'auto_padding',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
 normal=normalize_model(m);sample=rng.uniform(-1,1,(1,ic,ih,iw)).astype(np.float32)
 explicit=copy.deepcopy(m);node=explicit.graph.node[0];del node.attribute[:]
 totals=[0,0] if auto=='VALID' else [max(0,((size+stride-1)//stride-1)*stride+effective-size) for size,stride,effective in ((ih,strides[0],eh),(iw,strides[1],ew))]
 begin=[(v+(auto=='SAME_LOWER'))//2 for v in totals];epads=begin+[totals[j]-begin[j] for j in range(2)]
 node.attribute.extend([h.make_attribute('kernel_shape',[k,k]),h.make_attribute('strides',list(strides)),h.make_attribute('dilations',list(dil)),h.make_attribute('pads',epads)])
 np.testing.assert_allclose(ReferenceEvaluator(explicit).run(None,{'input':sample})[0],ReferenceEvaluator(normal).run(None,{'input':sample})[0],rtol=1e-5,atol=1e-5)
 p=root/f'model{i:03}.onnx';onnx.save(m,p);scale,zp=((1.,0),(.25,128),(.5,255))[i%3];binary,meta=compile_sequence(p,scale,zp);p.with_suffix('.bin').write_bytes(binary);q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()});attrs={a.name:h.get_attribute_value(a) for a in normal.graph.node[0].attribute}
 x=rng.integers(0,256,(3,ih,iw,ic),dtype=np.uint8);x[0]=zp;y=np.stack([native_input_reference(v,q,zp,attrs['pads'],attrs.get('strides',(1,1)),attrs.get('dilations',(1,1))) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
 manifest.append(dict(index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],kernel=k,auto_pad=auto,resolved_pads=attrs['pads'],strides=list(strides),dilations=list(dil),input_scale=scale,input_zero_point=zp))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
