"""MIT. Verify bounded QLinearConv reconstruction/requantization import."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'qlinearconv_import_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110335);manifest=[]
cases=[(1,1,1,False,False),(3,5,3,True,True),(17,17,3,True,False),(32,64,5,True,True)]
for i,(ic,oc,k,per_channel,with_bias) in enumerate(cases):
    ih=6+i;iw=8+i;p=k//2;xscale=np.array((.25,.5,1.,.125)[i],np.float32);xzp=np.array((0,128,255,17)[i],np.uint8)
    yscale=np.array((.5,2.,4.,8.)[i],np.float32);yzp=np.array((-128,-17,37,127)[i],np.int8)
    qw=rng.integers(-128,128,(oc,ic,k,k),dtype=np.int16).astype(np.int8)
    ws=(rng.uniform(.001,.02,oc).astype(np.float32) if per_channel else np.array(.01,np.float32))
    wz=(rng.integers(-12,13,oc,dtype=np.int16).astype(np.int8) if per_channel else np.array(-3,np.int8))
    names=['input','xs','xz','qw','ws','wz','ys','yz'];initial=[nh.from_array(v,n) for v,n in zip((xscale,xzp,qw,ws,wz,yscale,yzp),names[1:])]
    if with_bias:
        qb=rng.integers(-2000,2001,oc,dtype=np.int32);names.append('qb');initial.append(nh.from_array(qb,'qb'))
    node=h.make_node('QLinearConv',names,['output'],kernel_shape=[k,k],pads=[p]*4)
    model=h.make_model(h.make_graph([node],'qlinearconv',[h.make_tensor_value_info('input',onnx.TensorProto.UINT8,[1,ic,ih,iw])],
        [h.make_tensor_value_info('output',onnx.TensorProto.INT8,[1,oc,ih,iw])],initial),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()})
    x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=int(xzp)
    y=np.stack([native_input_reference(v,q,int(xzp),[p]*4) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(meta,index=i,input_shape=[ih,iw,ic],output_shape=[ih,iw,oc],source_input_scale=float(xscale),source_input_zero_point=int(xzp)))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
