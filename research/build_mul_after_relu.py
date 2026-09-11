"""MIT. Verify Mul operands produced by fused branch Relu activations."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_requant_reference,mul_output_conversion

root=Path(__file__).resolve().parent/'mul_after_relu_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110367);manifest=[]
for i,(c,modes) in enumerate(((3,(1,0)),(5,(0,1)),(9,(1,1)),(16,(1,0)))):
    height=5+i;width=8-i;nodes=[];initial=[];outs=[]
    for j,active in enumerate(modes):
        w=rng.uniform(-.3,.3,(c,3,1,1)).astype(np.float32);b=rng.uniform(-.2,.2,c).astype(np.float32);raw=f'raw{j}'
        initial.extend([nh.from_array(w,f'w{j}'),nh.from_array(b,f'b{j}')]);nodes.append(h.make_node('Conv',['input',f'w{j}',f'b{j}'],[raw],kernel_shape=[1,1]))
        if active:nodes.append(h.make_node('Relu',[raw],[f'branch{j}']));outs.append(f'branch{j}')
        else:outs.append(raw)
    nodes.append(h.make_node('Mul',outs,['output']))
    model=h.make_model(h.make_graph(nodes,'mul_after_relu',[h.make_tensor_value_info('input',1,[1,3,height,width])],[h.make_tensor_value_info('output',1,[1,c,height,width])],initial),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);requested={'scale':(.5,.125,1.,.03125)[i],'zero_point':(-128,127,-41,77)[i]}
    binary,meta=compile_sequence(path,output_range=requested);path.with_suffix('.bin').write_bytes(binary)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in meta['branches']]
    _,_,conversion,shift=mul_output_conversion(np.prod([b['output_scale'] for b in meta['branches']]),requested)
    x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);x.tofile(root/f'input{i:03}.u8')
    np.stack([mul_requant_reference(reference(v,qs[0]),reference(v,qs[1]),conversion,shift,requested['zero_point']) for v in x]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_scale=1.,input_zero_point=0,compile_output_range=requested,branch_relu=list(modes),**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
