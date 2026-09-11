"""MIT. Verify terminal spatial Reshape consuming a Mul result."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

root=Path(__file__).resolve().parent/'mul_reshape_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110340);manifest=[]
for i,(c,ih,iw,oh,ow,same) in enumerate(((3,5,6,3,10,False),(5,6,8,4,12,True),(16,8,6,3,16,True))):
    names=['a'] if same else ['a','b'];operands=['a','a'] if same else ['a','b'];shape_name='shape'
    nodes=[h.make_node('Mul',operands,['product']),h.make_node('Reshape',['product',shape_name],['output'])]
    model=h.make_model(h.make_graph(nodes,'mul_reshape',[h.make_tensor_value_info(v,1,[1,c,ih,iw]) for v in names],[h.make_tensor_value_info('output',1,[1,c,oh,ow])],[nh.from_array(np.array([1,c,oh,ow],np.int64),shape_name)]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i]
    binary,meta=compile_sequence(path,scale,zp);path.with_suffix('.bin').write_bytes(binary);qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in meta['branches']]
    _,outzp,mult,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale);a=rng.integers(0,256,(6,ih,iw,c),dtype=np.uint8)
    if same:b=a;packed=a
    else:b=rng.integers(0,256,a.shape,dtype=np.uint8);packed=np.concatenate([a,b],axis=1)
    y=np.stack([mul_requant_reference(reference(x,qs[0]),reference(v,qs[1]),mult,shift,outzp) for x,v in zip(a,b)]).reshape(6,oh,ow,c)
    packed.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_scale=scale,input_zero_point=zp,input_shape=[ih,iw,c],output_shape=[oh,ow,c]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
