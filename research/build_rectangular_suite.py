"""MIT. Even and rectangular Conv rewritten into zero-filled square kernels."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.scheduler import compile_sequence
from open_rknpu.normalize import normalize_model
from open_rknpu.quantization import Quantization,reference
root=Path(__file__).resolve().parent/'rectangular_conv_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110325);manifest=[]
for i,(kh,kw) in enumerate([(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,5),(5,3),(4,4),(4,5),(5,4)]):
    stride=1+i%2;oh=(8-kh)//stride+1;ow=(8-kw)//stride+1
    w=rng.uniform(-.3,.3,(3,3,kh,kw)).astype(np.float32);b=rng.uniform(-1,1,3).astype(np.float32)
    m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[kh,kw],strides=[stride]*2)],'rect',
        [h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
    p=root/f'model{i:03}.onnx';onnx.save(m,p);n=normalize_model(m)
    data,meta=compile_sequence(p);p.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=rng.integers(0,256,(16,8,8,3),dtype=np.uint8);x[0]=0;x[1]=255
    xf=x[2:3].transpose(0,3,1,2).astype(np.float32)
    np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':xf})[0],ReferenceEvaluator(n).run(None,{'input':xf})[0],atol=.0002,rtol=1e-5)
    pad=meta['conv_pads'];margin=q.kernel_size//2
    y=np.stack([reference(np.pad(v,((pad[0],pad[2]),(pad[1],pad[3]),(0,0))),q)[margin:8+pad[0]+pad[2]-margin:stride,margin:8+pad[1]+pad[3]-margin:stride] for v in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(index=i,kernel=[kh,kw],stride=stride))
(root/'manifest.json').write_text(json.dumps(manifest))
