"""MIT. Verify independent output conversion for depthwise and dense ConvTranspose."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'transpose_output_quantization_suite';root.mkdir(exist_ok=True);manifest=[]
cases=[('transpose_stride1_suite',2,{'scale':0.25,'zero_point':-128}),('transpose_padding_suite',0,{'scale':1.5,'zero_point':127}),('transpose_dense_suite',0,{'scale':0.75,'zero_point':-37}),('transpose_dense_suite',3,{'scale':2.0,'zero_point':91})]
for index,(suite,source_index,out_range) in enumerate(cases):
    source=base/suite/f'model{source_index:03}.onnx';model=onnx.load(source);path=root/f'model{index:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,output_range=out_range);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['transposed_quantization'].items()});q1=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['first']['quantization'].items()});node=model.graph.node[-1];attrs={a.name:h.get_attribute_value(a) for a in node.attribute};strides=attrs.get('strides',[1,1]);pads=attrs.get('pads',[0,0,0,0]);oh,ow,oc=meta['output_shape_nhwc'][1:];ic=meta['first']['output_shape_nhwc'][3];k=q.kernel_size
    input_path=base/suite/f'input{source_index:03}.u8';x=np.fromfile(input_path,np.uint8).reshape(-1,8,8,3)[:4];ys=[]
    depthwise=attrs.get('group',1)==ic and oc==ic
    if depthwise:
        qw=q.weights.reshape(oc,k,k);bias_units=q.biases+q1.output_zero_point*q.weights.sum(axis=1)
    else:
        qw=q.weights.reshape(oc,ic,k,k)-q.weight_zero_points[:,None,None,None];bias_units=q.biases+q1.output_zero_point*qw.reshape(oc,-1).sum(axis=1)
    for sample in x:
        a=reference(sample,q1).astype(np.int64)-q1.output_zero_point;acc=np.broadcast_to(bias_units,(oh,ow,oc)).copy()
        for iy in range(8):
          for ix in range(8):
           for ky in range(k):
            for kx in range(k):
             oy=iy*strides[0]+ky-pads[0];ox=ix*strides[1]+kx-pads[1]
             if 0<=oy<oh and 0<=ox<ow:
              if depthwise:acc[oy,ox]+=a[iy,ix]*qw[:,ky,kx]
              else:acc[oy,ox]+=a[iy,ix]@qw[:,:,ky,kx].T
        product=acc*q.channel_multipliers;scaled=(product+8191+((product>>14)&1))>>14;product=scaled*q.multiplier
        if q.shift:product+=q.output_zero_point<<q.shift;result=(product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift
        else:result=product+q.output_zero_point
        ys.append(np.clip(result,-128,127).astype(np.int8))
    x.tofile(root/f'input{index:03}.u8');np.stack(ys).tofile(root/f'expected{index:03}.i8');manifest.append(dict(index=index,source=f'{suite}/model{source_index:03}',input_scale=1.0,input_zero_point=0,compile_output_range=out_range,mode='depthwise' if depthwise else 'dense',output_shape=[oh,ow,oc]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
