"""MIT. Verify bounded Q/DQ Conv reconstruction/requantization import."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'qdq_conv_import_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110338);manifest=[]
for i,(ic,oc,k,per_channel,bias_on) in enumerate(((3,4,1,False,False),(17,17,3,True,True),(32,64,5,True,True))):
    ih=7+i;iw=9+i;p=k//2;xs=np.array((.25,.5,.125)[i],np.float32);xz=np.array((0,128,255)[i],np.uint8);ys=np.array((.5,2.,8.)[i],np.float32);yz=np.array((-128,-17,127)[i],np.int8)
    qw=rng.integers(-128,128,(oc,ic,k,k),dtype=np.int16).astype(np.int8);ws=rng.uniform(.001,.02,oc).astype(np.float32) if per_channel else np.array(.01,np.float32);wz=rng.integers(-8,9,oc,dtype=np.int16).astype(np.int8) if per_channel else np.array(-2,np.int8)
    initial=[nh.from_array(v,n) for v,n in ((xs,'xs'),(xz,'xz'),(qw,'qw'),(ws,'ws'),(wz,'wz'),(ys,'ys'),(yz,'yz'))]
    conv_inputs=['xf','wf']
    if bias_on:
        bias=rng.uniform(-.5,.5,oc).astype(np.float32);initial.append(nh.from_array(bias,'bias'));conv_inputs.append('bias')
    wdq=h.make_node('DequantizeLinear',['qw','ws','wz'],['wf'],axis=0) if per_channel else h.make_node('DequantizeLinear',['qw','ws','wz'],['wf'])
    nodes=[h.make_node('DequantizeLinear',['input','xs','xz'],['xf']),wdq,h.make_node('Conv',conv_inputs,['yf'],kernel_shape=[k,k],pads=[p]*4),h.make_node('QuantizeLinear',['yf','ys','yz'],['output'])]
    model=h.make_model(h.make_graph(nodes,'qdq_conv',[h.make_tensor_value_info('input',onnx.TensorProto.UINT8,[1,ic,ih,iw])],[h.make_tensor_value_info('output',onnx.TensorProto.INT8,[1,oc,ih,iw])],initial),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()});x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=int(xz)
    y=np.stack([native_input_reference(v,q,int(xz),[p]*4) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_shape=[ih,iw,ic],output_shape=[ih,iw,oc]))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
