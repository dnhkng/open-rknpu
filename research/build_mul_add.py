"""MIT. Verify representable scalar Add folded into Mul output conversion."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_requant_reference,mul_output_conversion

root=Path(__file__).resolve().parent/'mul_add_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110381);manifest=[]
cases=[(3,.125,-64,2),(5,.25,31,-3),(9,.5,-17,5),(16,.03125,63,-7)]
for i,(c,scale,zp,steps) in enumerate(cases):
    height=5+i;width=8-i;initial=[];nodes=[];outs=[]
    for j in range(2):
        w=rng.uniform(-.4,.4,(c,3,1,1)).astype(np.float32);b=rng.uniform(-.5,.5,c).astype(np.float32);out=f'branch{j}'
        initial.extend([nh.from_array(w,f'w{j}'),nh.from_array(b,f'b{j}')]);nodes.append(h.make_node('Conv',['input',f'w{j}',f'b{j}'],[out],kernel_shape=[1,1]));outs.append(out)
    constant=np.array(steps*scale,np.float32);initial.append(nh.from_array(constant,'offset'));nodes.extend([h.make_node('Mul',outs,['product']),h.make_node('Add',['offset','product'],['output'])])
    model=h.make_model(h.make_graph(nodes,'mul_add',[h.make_tensor_value_info('input',1,[1,3,height,width])],[h.make_tensor_value_info('output',1,[1,c,height,width])],initial),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);requested={'scale':scale,'zero_point':zp};binary,meta=compile_sequence(path,output_range=requested);path.with_suffix('.bin').write_bytes(binary)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in meta['branches']]
    internal={'scale':scale,'zero_point':zp+steps};_,_,conversion,shift=mul_output_conversion(np.prod([b['output_scale'] for b in meta['branches']]),internal)
    x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);x.tofile(root/f'input{i:03}.u8')
    np.stack([mul_requant_reference(reference(v,qs[0]),reference(v,qs[1]),conversion,shift,zp+steps) for v in x]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_scale=1.,input_zero_point=0,compile_output_range=requested,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
