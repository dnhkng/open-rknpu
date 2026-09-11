"""MIT. Public sequence verification for bounded Sigmoid/Tanh LUT profiles."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence

root=Path(__file__).resolve().parent/'lut_public_suite';root.mkdir(exist_ok=True)
for pattern in ('model*.onnx','model*.bin','input*.u8','expected*.i8'):
    for stale in root.glob(pattern):stale.unlink()
rng=np.random.default_rng(110334);manifest=[]
for i,(kind,factor) in enumerate((k,1/32) for k in ('Sigmoid','Tanh')):
    w=np.eye(3,dtype=np.float32).reshape(3,3,1,1)*factor;b=np.zeros(3,np.float32)
    model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['conv'],kernel_shape=[1,1]),h.make_node(kind,['conv'],['output'])],
        'lut_public',[h.make_tensor_value_info('input',1,[1,3,8,8])],[h.make_tensor_value_info('output',1,[1,3,8,8])],
        [nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);binary,meta=compile_sequence(path,1,128);path.with_suffix('.bin').write_bytes(binary)
    x=rng.integers(0,256,(16,8,8,3),dtype=np.uint8);x.reshape(-1)[:256]=np.arange(256,dtype=np.uint8)
    real=(x.astype(np.float64)-128)*factor;fn=(lambda z:1/(1+np.exp(-z))) if kind=='Sigmoid' else np.tanh
    q15=np.clip(np.rint(fn(real)*32768),-32768,32767)
    y=(np.rint(q15*(255 if kind=='Sigmoid' else 127)/32768)-(128 if kind=='Sigmoid' else 0)).astype(np.int8)
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,kind=kind,input_scale=1.,input_zero_point=128))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
