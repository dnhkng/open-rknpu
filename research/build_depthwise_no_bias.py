"""MIT. Verify omitted bias in the chained depthwise profile."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.depthwise import depthwise_reference

root=Path(__file__).resolve().parent/'depthwise_no_bias_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110357);manifest=[]
for i,(channels,kernel,stride) in enumerate(((1,1,1),(4,3,1),(9,3,2),(16,5,2))):
    height=5+i;width=8-i;pad=kernel//2;oh=(height+stride-1)//stride;ow=(width+stride-1)//stride
    w1=rng.uniform(-.3,.3,(channels,3,1,1)).astype(np.float32)
    w2=rng.uniform(-.25,.25,(channels,1,kernel,kernel)).astype(np.float32)
    nodes=[h.make_node('Conv',['input','w1'],['stem'],kernel_shape=[1,1]),
           h.make_node('Conv',['stem','w2'],['output'],kernel_shape=[kernel,kernel],pads=[pad]*4,strides=[stride]*2,group=channels)]
    model=h.make_model(h.make_graph(nodes,'depthwise_no_bias',[h.make_tensor_value_info('input',1,[1,3,height,width])],
        [h.make_tensor_value_info('output',1,[1,channels,oh,ow])],[nh.from_array(w1,'w1'),nh.from_array(w2,'w2')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary)
    q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()})
    q2=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['depthwise'].items()})
    x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);x.tofile(root/f'input{i:03}.u8')
    np.stack([depthwise_reference(reference(v,q1),q2,q1.output_zero_point)[::stride,::stride] for v in x]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
