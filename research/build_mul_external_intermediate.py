"""MIT. Verify Conv result multiplied by a second external tensor."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

base=Path(__file__).resolve().parent;root=base/'mul_external_intermediate_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110359);manifest=[]
cases=((5,8,(0,0),{'scale':.5,'zero_point':-37}),(8,5,(-128,127),{'scale':2.,'zero_point':91}),(7,7,(31,-64),None),(6,8,(127,-128),{'scale':1.25,'zero_point':0}))
for index,(height,width,zps,out_range) in enumerate(cases):
 w=rng.uniform(-.3,.3,(3,3,1,1)).astype(np.float32);b=rng.uniform(-.5,.5,3).astype(np.float32);inputs=[h.make_tensor_value_info(name,1,[1,3,height,width]) for name in ('x','y')]
 model=h.make_model(h.make_graph([h.make_node('Conv',['x','w','b'],['learned'],kernel_shape=[1,1]),h.make_node('Mul',['learned','y'],['output'])],'mul_external_intermediate',inputs,[h.make_tensor_value_info('output',1,[1,3,height,width])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
 path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,output_range=out_range,mul_operand_zero_points=zps);path.with_suffix('.bin').write_bytes(binary);qs=[Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in branch['quantization'].items()}) for branch in meta['branches']];x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);y=rng.integers(0,256,x.shape,dtype=np.uint8);packed=np.concatenate([x,y],axis=1);output_scale,output_zp,conversion,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale,out_range);expected=np.stack([mul_requant_reference(reference(a,qs[0]),reference(c,qs[1]),conversion,shift,output_zp,zps) for a,c in zip(x,y)])
 packed.tofile(root/f'input{index:03}.u8');expected.tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,input_scale=1.0,input_zero_point=0,compile_output_range=out_range,mul_operand_zero_points=list(zps),shape=[height,width,3],rewrite=meta['mul_graph_rewrite']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
