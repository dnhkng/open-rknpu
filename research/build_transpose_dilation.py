"""MIT. Verify K2 dilation2 ConvTranspose through a sparse K3 rewrite."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'transpose_dilation_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110351);manifest=[]
cases=[([2,2],{'pads':[0,0,0,0]},[0,0,0,0]),([1,1],{'pads':[1,1,1,1]},[1,1,1,1]),([1,2],{'pads':[2,0,0,2],'output_padding':[0,1]},[2,0,0,2]),([2,1],{'auto_pad':'SAME_UPPER','output_padding':[1,0]},[1,1,1,1])]
for i,(strides,extra,pads) in enumerate(cases):
    model=onnx.load(base/'depthwise_suite'/f'model{i:03}.onnx');node=model.graph.node[-1];node.op_type='ConvTranspose';del node.attribute[:];attrs=dict(group=3,kernel_shape=[2,2],strides=strides,dilations=[2,2],**extra);node.attribute.extend(h.make_attribute(name,value) for name,value in attrs.items())
    w=rng.uniform(-.3,.3,(3,1,2,2)).astype(np.float32);b=rng.uniform(-.5,.5,3).astype(np.float32)
    for tensor in model.graph.initializer:
        if tensor.name==node.input[1]:tensor.CopyFrom(nh.from_array(w,tensor.name))
        elif tensor.name==node.input[2]:tensor.CopyFrom(nh.from_array(b,tensor.name))
    opad=extra.get('output_padding',[0,0]);oh=7*strides[0]+3-pads[0]-pads[2]+opad[0];ow=7*strides[1]+3-pads[1]-pads[3]+opad[1]
    for dim,value in zip(model.graph.output[0].type.tensor_type.shape.dim[2:],[oh,ow]):dim.dim_value=value
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['transposed_quantization'].items()});q1=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['first']['quantization'].items()});x=np.fromfile(base/'depthwise_suite'/f'input{i:03}.u8',np.uint8).reshape(16,8,8,3);qw=q.weights.reshape(3,3,3);bias_units=q.biases+q1.output_zero_point*q.weights.sum(axis=1);ys=[]
    for sample in x:
        a=reference(sample,q1).astype(np.int64)-q1.output_zero_point;acc=np.broadcast_to(bias_units,(oh,ow,3)).copy()
        for iy in range(8):
          for ix in range(8):
           for ky in range(3):
            for kx in range(3):
             oy=iy*strides[0]+ky-pads[0];ox=ix*strides[1]+kx-pads[1]
             if 0<=oy<oh and 0<=ox<ow:acc[oy,ox]+=a[iy,ix]*qw[:,ky,kx]
        product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;product=scaled*q.multiplier
        if q.shift:product+=q.output_zero_point<<q.shift;result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
        else:result=product+q.output_zero_point
        ys.append(np.clip(result,-128,127).astype(np.int8))
    x.tofile(root/f'input{i:03}.u8');np.stack(ys).tofile(root/f'expected{i:03}.i8');manifest.append(dict(index=i,kernel=2,dilation=2,strides=strides,attributes=extra,resolved_pads=pads,output_shape=[oh,ow,3],rewrite=meta['transposed_rewrite']))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
