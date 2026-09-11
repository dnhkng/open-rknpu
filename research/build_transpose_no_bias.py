"""MIT. Verify optional ConvTranspose bias by synthesized zero bias storage."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'transpose_no_bias_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110333);manifest=[]
for i,(k,pads) in enumerate(((2,[0,1,1,0]),(3,[1,0,2,1]))):
    model=onnx.load(base/'depthwise_suite'/f'model{i:03}.onnx');node=model.graph.node[-1]
    node.op_type='ConvTranspose';del node.attribute[:];node.attribute.extend([h.make_attribute('group',3),h.make_attribute('kernel_shape',[k,k]),h.make_attribute('strides',[2,2]),h.make_attribute('pads',pads)])
    w=rng.uniform(-.3,.3,(3,1,k,k)).astype(np.float32);bias_name=node.input[2];del node.input[2]
    kept=[]
    for t in model.graph.initializer:
        if t.name==node.input[1]:t.CopyFrom(nh.from_array(w,t.name))
        if t.name!=bias_name:kept.append(t)
    del model.graph.initializer[:];model.graph.initializer.extend(kept)
    oh=14+k-pads[0]-pads[2];ow=14+k-pads[1]-pads[3]
    for d,v in zip(model.graph.output[0].type.tensor_type.shape.dim[2:],[oh,ow]):d.dim_value=v
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['transposed_quantization'].items()})
    q1=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['first']['quantization'].items()})
    x=np.fromfile(base/'depthwise_suite'/f'input{i:03}.u8',np.uint8).reshape(16,8,8,3);qw=q.weights.reshape(3,k,k);ys=[]
    for sample in x:
        a=reference(sample,q1).astype(np.int64)-q1.output_zero_point
        acc=np.zeros((oh,ow,3),np.int64)
        for iy in range(8):
          for ix in range(8):
           for ky in range(k):
            for kx in range(k):
             oy=iy*2+ky-pads[0];ox=ix*2+kx-pads[1]
             if 0<=oy<oh and 0<=ox<ow:acc[oy,ox]+=a[iy,ix]*qw[:,ky,kx]
        product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14
        ys.append(np.clip(((scaled*q.multiplier+(1<<(q.shift-1)))>>q.shift)+q.output_zero_point,-128,127).astype(np.int8))
    x.tofile(root/f'input{i:03}.u8');np.stack(ys).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,kernel=k,pads=pads,output_shape=[oh,ow,3]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
