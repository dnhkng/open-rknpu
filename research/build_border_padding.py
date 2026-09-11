"""MIT. Verify convolution windows touching multiple padded edges."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'border_padding_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110343);manifest=[]
for i,(ic,oc,ih,iw,k,pads) in enumerate(((1,3,1,1,3,[2,2,2,2]),(3,5,1,2,5,[4,3,2,4]),(17,9,2,1,7,[6,5,4,6]))):
    oh=ih+pads[0]+pads[2]-k+1;ow=iw+pads[1]+pads[3]-k+1;w=rng.uniform(-.2,.2,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.3,.3,oc).astype(np.float32)
    model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],pads=pads)],'borders',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i];binary,meta=compile_sequence(path,scale,zp);path.with_suffix('.bin').write_bytes(binary);q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()});x=rng.integers(0,256,(4,ih,iw,ic),dtype=np.uint8);x[0]=zp;y=np.stack([native_input_reference(v,q,zp,pads) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],pads=pads))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
