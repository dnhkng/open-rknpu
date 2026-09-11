"""MIT. Fresh depthwise shape/kernel hypotheses, no vendor inputs."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.depthwise import depthwise_reference
root=Path(__file__).resolve().parent/'depthwise_combined_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110317)
cases=[(5+(c%4),5+((c+k)%4),k,s,c) for c in range(3,17) for k in (1,3,5) for s in (1,2)]
for i,(height,width,k,stride,channels) in enumerate(cases):
    constants=[nh.from_array(rng.uniform(-.3,.3,shape).astype(np.float32),name) for name,shape in [('w1',(channels,3,1,1)),('b1',(channels,)),('w2',(channels,1,k,k)),('b2',(channels,))]]
    oh=(height+stride-1)//stride;ow=(width+stride-1)//stride
    m=h.make_model(h.make_graph([h.make_node('Conv',['input','w1','b1'],['stem'],kernel_shape=[1,1]),h.make_node('Conv',['stem','w2','b2'],['output'],kernel_shape=[k,k],pads=[k//2]*4,strides=[stride]*2,group=channels)],'dw',
        [h.make_tensor_value_info('input',1,[1,3,height,width])],[h.make_tensor_value_info('output',1,[1,channels,oh,ow])],constants),opset_imports=[h.make_opsetid('',13)])
    m.ir_version=8;p=root/f'model{i:03}.onnx';onnx.save(m,p)
    data,meta=compile_sequence(p);p.with_suffix('.bin').write_bytes(data)
    def q(d):return Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in d.items()})
    a,b=q(meta['first']['quantization']),q(meta['depthwise'])
    x=rng.integers(0,256,(16,height,width,3),dtype=np.uint8);x[0]=0;x[1]=255
    y=np.stack([depthwise_reference(reference(v,a),b,a.output_zero_point)[::stride,::stride] for v in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
(root/'cases.json').write_text(json.dumps(cases))
