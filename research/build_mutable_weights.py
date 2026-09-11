"""MIT. Build a v4 Conv whose packed parameter region is replaced at runtime."""
from pathlib import Path
import json
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence,CONSTANT
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference

root=Path(__file__).resolve().parent/'mutable_weights_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110371);wa=np.array([[[-1,0,1]],[[0,1,-1]],[[1,-1,0]]],np.float32).reshape(3,3,1,1)
wb=wa[:,:,0,0][:,[2,0,1]].reshape(3,3,1,1);bias=np.array([.25,-.5,.75],np.float32)
def model(weights,path):
    m=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[1,1])],'mutable_weights',
        [h.make_tensor_value_info('input',1,[1,3,7,6])],[h.make_tensor_value_info('output',1,[1,3,7,6])],[nh.from_array(weights,'w'),nh.from_array(bias,'b')]),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8;onnx.save(m,path)
paths=[root/'model000.onnx',root/'donor.onnx'];model(wa,paths[0]);model(wb,paths[1]);requested={'scale':.125,'zero_point':-7}
compiled=[]
for p in paths:compiled.append(compile_sequence(p,1.,128,requested,mutable_weights=True))
binary,meta=compiled[0];paths[0].with_suffix('.bin').write_bytes(binary);donor,dmeta=compiled[1]
info=decode_sequence(donor);entry=info['constants'][0];payload=96+16*info['task_count']+CONSTANT.size*info['constant_count']
(root/'parameters.bin').write_bytes(donor[payload+entry['offset']:payload+entry['offset']+entry['size']])
q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in dmeta['quantization'].items()})
x=rng.integers(0,256,(8,7,6,3),dtype=np.uint8);x.tofile(root/'input000.u8');x[0].tofile(root/'input_single.u8')
y=np.stack([native_input_reference(v,q,128,dmeta['conv_pads'],dmeta['conv_strides'],dmeta['conv_dilations']) for v in x]);y.tofile(root/'expected000.i8');y[0].tofile(root/'expected_single.i8')
(root/'manifest.json').write_text(json.dumps([dict(meta,index=0,input_scale=1.,input_zero_point=128,compile_output_range=requested,parameter_bytes=entry['size'])],indent=2)+'\n')
