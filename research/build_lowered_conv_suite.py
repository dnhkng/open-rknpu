"""MIT. Group/dilation algebraic lowering, checked against ONNX float reference."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.normalize import normalize_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
root=Path(__file__).resolve().parent/'lowered_conv_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110319)
for i,(group,dilation) in enumerate([(3,1),(1,2),(3,2)]):
    k=3;c=3;w=rng.uniform(-.3,.3,(c,3//group,k,k)).astype(np.float32);b=rng.uniform(-1,1,c).astype(np.float32)
    m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[3,3],group=group,dilations=[dilation]*2,pads=[dilation]*4)],'lowered',
        [h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,8,8])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/f'model{i:03}.onnx';onnx.save(m,p);normal=normalize_model(m)
    x=rng.integers(0,256,(16,8,8,3),dtype=np.uint8);x[0]=0;x[1]=255
    xf=x[2:3].transpose(0,3,1,2).astype(np.float32)
    np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':xf})[0],ReferenceEvaluator(normal).run(None,{'input':xf})[0],atol=.0001,rtol=.00001)
    data,meta=compile_sequence(p);p.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x.tofile(root/f'input{i:03}.u8');np.stack([reference(v,q) for v in x]).tofile(root/f'expected{i:03}.i8')
