"""MIT. Verify Conv -> depthwise -> dense pointwise composition."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.chain import native_reference

root=Path(__file__).resolve().parent/'depthwise_pointwise_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110363);manifest=[]
for i,(c,oc,k,stride) in enumerate(((3,1,1,1),(5,7,3,1),(9,16,3,2),(16,32,5,2))):
    height=5+i;width=8-i;pad=k//2;oh=(height+stride-1)//stride;ow=(width+stride-1)//stride
    arrays=[rng.uniform(-.3,.3,(c,3,1,1)).astype(np.float32),rng.uniform(-.2,.2,c).astype(np.float32),
            rng.uniform(-.25,.25,(c,1,k,k)).astype(np.float32),rng.uniform(-.2,.2,c).astype(np.float32),
            rng.uniform(-.3,.3,(oc,c,1,1)).astype(np.float32),rng.uniform(-.2,.2,oc).astype(np.float32)]
    names=['w1','b1','wd','bd','wp','bp'];initial=[nh.from_array(v,n) for v,n in zip(arrays,names)]
    nodes=[h.make_node('Conv',['input','w1','b1'],['stem'],kernel_shape=[1,1]),
           h.make_node('Conv',['stem','wd','bd'],['depthwise'],kernel_shape=[k,k],pads=[pad]*4,strides=[stride]*2,group=c),
           h.make_node('Conv',['depthwise','wp','bp'],['output'],kernel_shape=[1,1])]
    model=h.make_model(h.make_graph(nodes,'depthwise_pointwise',[h.make_tensor_value_info('input',1,[1,3,height,width])],
        [h.make_tensor_value_info('output',1,[1,oc,oh,ow])],initial),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);requested={'scale':(.5,.125,1.,.03125)[i],'zero_point':(-128,127,-29,61)[i]}
    binary,meta=compile_sequence(path,output_range=requested);path.with_suffix('.bin').write_bytes(binary)
    get=lambda name:Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta[name].items()})
    q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()});qd=get('depthwise');qp=get('pointwise')
    x=rng.integers(0,256,(8,height,width,3),dtype=np.uint8);x.tofile(root/f'input{i:03}.u8');ys=[]
    for sample in x:
        d=depthwise_reference(reference(sample,q1),qd,q1.output_zero_point)[::stride,::stride]
        ys.append(native_reference(d,qp,qd.output_zero_point))
    np.stack(ys).tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_scale=1.,input_zero_point=0,compile_output_range=requested))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
