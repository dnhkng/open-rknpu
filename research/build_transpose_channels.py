"""MIT. Verify every depthwise ConvTranspose channel count C1..16."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'transpose_channels_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110353);manifest=[]
stride_cases=([1,1],[2,2],[1,2],[2,1])
for i,channels in enumerate(range(1,17)):
    k=3 if channels%2 else 2;strides=stride_cases[i%4];pads=[1,1,1,1] if k==3 else [0,0,0,0]
    oh=7*strides[0]+k-pads[0]-pads[2];ow=7*strides[1]+k-pads[1]-pads[3]
    stem_w=rng.uniform(-.3,.3,(channels,3,1,1)).astype(np.float32);stem_b=rng.uniform(-.5,.5,channels).astype(np.float32);w=rng.uniform(-.3,.3,(channels,1,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,channels).astype(np.float32)
    nodes=[h.make_node('Conv',['input','stem_w','stem_b'],['stem'],kernel_shape=[1,1]),h.make_node('ConvTranspose',['stem','w','b'],['output'],group=channels,kernel_shape=[k,k],strides=strides,pads=pads)]
    model=h.make_model(h.make_graph(nodes,f'transpose_c{channels}',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,channels,oh,ow])],[nh.from_array(stem_w,'stem_w'),nh.from_array(stem_b,'stem_b'),nh.from_array(w,'w'),nh.from_array(b,'b')],value_info=[h.make_tensor_value_info('stem',1,[1,channels,8,8])]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['transposed_quantization'].items()});q1=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['first']['quantization'].items()});x=rng.integers(0,256,(4,8,8,3),dtype=np.uint8);qw=q.weights.reshape(channels,k,k);bias_units=q.biases+q1.output_zero_point*q.weights.sum(axis=1);ys=[]
    for sample in x:
        a=reference(sample,q1).astype(np.int64)-q1.output_zero_point;acc=np.broadcast_to(bias_units,(oh,ow,channels)).copy()
        for iy in range(8):
          for ix in range(8):
           for ky in range(k):
            for kx in range(k):
             oy=iy*strides[0]+ky-pads[0];ox=ix*strides[1]+kx-pads[1]
             if 0<=oy<oh and 0<=ox<ow:acc[oy,ox]+=a[iy,ix]*qw[:,ky,kx]
        product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;product=scaled*q.multiplier
        if q.shift:product+=q.output_zero_point<<q.shift;result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
        else:result=product+q.output_zero_point
        ys.append(np.clip(result,-128,127).astype(np.int8))
    x.tofile(root/f'input{i:03}.u8');np.stack(ys).tofile(root/f'expected{i:03}.i8');manifest.append(dict(index=i,channels=channels,kernel=k,strides=strides,pads=pads,output_shape=[oh,ow,channels]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
