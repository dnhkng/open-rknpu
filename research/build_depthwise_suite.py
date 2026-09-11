"""MIT. Fresh independent depthwise graphs: no RKNN or captured command input."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.depthwise import depthwise_reference

ROOT=Path(__file__).resolve().parents[1]/'research/depthwise_suite'
ROOT.mkdir(exist_ok=True)
rng=np.random.default_rng(110311)
manifest=[]
for i in range(12):
    k=1 if i<6 else 3;relu=bool(i%2)
    w1=rng.uniform(-.2,.2,(3,3,k,k)).astype(np.float32)
    b1=rng.uniform(-2,2,3).astype(np.float32)
    w2=rng.uniform(-.4,.4,(3,1,3,3)).astype(np.float32)
    if i%6==2:w2=abs(w2)
    if i%6==3:w2=-abs(w2)
    if i%6==4:w2[1]=0
    b2=rng.uniform(-3,3,3).astype(np.float32)
    nodes=[h.make_node('Conv',['input','w1','b1'],['conv'],kernel_shape=[k,k],pads=[k//2]*4)]
    if relu:nodes.append(h.make_node('Relu',['conv'],['relu']))
    nodes.append(h.make_node('Conv',[nodes[-1].output[0],'w2','b2'],['output'],kernel_shape=[3,3],pads=[1]*4,group=3))
    model=h.make_model(h.make_graph(nodes,'depthwise',
        [h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,8,8])],
        [nh.from_array(v,n) for n,v in [('w1',w1),('b1',b1),('w2',w2),('b2',b2)]]),opset_imports=[h.make_opsetid('',13)])
    model.ir_version=8
    path=ROOT/f'model{i:03}.onnx';onnx.save(model,path)
    scale,zp=((1.,0),(.25,128),(.5,255))[i%3]
    data,meta=compile_sequence(path,scale,zp);path.with_suffix('.bin').write_bytes(data)
    def quant(d):return Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in d.items()})
    q1,q2=quant(meta['first']['quantization']),quant(meta['depthwise'])
    inputs=rng.integers(0,256,(16,8,8,3),dtype=np.uint8)
    inputs[0]=zp;inputs[1]=0;inputs[2]=255
    inputs[3]=zp;inputs[3,0,0,0]=255 if zp!=255 else 0
    inputs[4]=np.arange(192,dtype=np.uint8).reshape(8,8,3)
    expected=np.stack([depthwise_reference(reference(x,q1),q2,q1.output_zero_point) for x in inputs])
    inputs.tofile(ROOT/f'input{i:03}.u8');expected.tofile(ROOT/f'expected{i:03}.i8')
    manifest.append(dict(index=i,stem_kernel=k,stem_relu=relu,input_scale=scale,input_zero_point=zp,
        first=q1.metadata(),depthwise=q2.metadata()))
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built 12 independent depthwise models, 192 cases')
