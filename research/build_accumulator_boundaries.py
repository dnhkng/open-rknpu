"""MIT. Exercise native Conv accumulators near, but within, signed INT32 bounds."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference

root=Path(__file__).resolve().parent/'accumulator_boundary_suite';root.mkdir(exist_ok=True);manifest=[]
for i,(sign,bias,zp) in enumerate(((1,300000.,0),(1,-300000.,255),(-1,300000.,0),(-1,-300000.,255))):
    c=32;k=31;w=np.full((1,c,k,k),float(sign),np.float32);b=np.array([bias],np.float32)
    model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k])],'accumulator_boundary',
        [h.make_tensor_value_info('input',1,[1,c,k,k])],[h.make_tensor_value_info('output',1,[1,1,1,1])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);requested={'scale':50000.,'zero_point':(-64,63,-31,29)[i]}
    binary,meta=compile_sequence(path,1.,zp,requested);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=np.empty((4,k,k,c),np.uint8);x[0]=0;x[1]=255;x[2]=zp;x[3]=np.indices((k,k,c)).sum(axis=0)%2*255
    x.tofile(root/f'input{i:03}.u8');np.stack([native_input_reference(v,q,zp,meta['conv_pads'],meta['conv_strides'],meta['conv_dilations']) for v in x]).tofile(root/f'expected{i:03}.i8')
    centered=q.weights-q.weight_zero_points[:,None];lo=int((q.biases+np.minimum((-128)*centered,127*centered).sum(axis=1))[0]);hi=int((q.biases+np.maximum((-128)*centered,127*centered).sum(axis=1))[0])
    manifest.append(dict(meta,index=i,input_scale=1.,input_zero_point=zp,compile_output_range=requested,accumulator_interval=[lo,hi]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
