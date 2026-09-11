"""MIT. Verify every dense output channel count C2 through C16."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization

root=Path(__file__).resolve().parent/'output_channels_complete_suite';root.mkdir(exist_ok=True);rng=np.random.default_rng(110342);manifest=[]
for i,oc in enumerate(range(2,17)):
    ic=(1,3,17,32)[i%4];k=(1,3,5)[i%3];ih=6+i%5;iw=7+(i*2)%5
    pads=([0,0,0,0] if k==1 else [i%k,(i*2)%k,(i*3)%k,(i*4)%k]);oh=ih+pads[0]+pads[2]-k+1;ow=iw+pads[1]+pads[3]-k+1
    w=rng.uniform(-.2,.2,(oc,ic,k,k)).astype(np.float32);b=rng.uniform(-.5,.5,oc).astype(np.float32)
    model=h.make_model(h.make_graph([h.make_node('Conv',['input','w','b'],['output'],kernel_shape=[k,k],pads=pads)],'output_channels',[h.make_tensor_value_info('input',1,[1,ic,ih,iw])],[h.make_tensor_value_info('output',1,[1,oc,oh,ow])],[nh.from_array(w,'w'),nh.from_array(b,'b')]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,zp=((1.,0),(.25,128),(.5,255))[i%3];binary,meta=compile_sequence(path,scale,zp);path.with_suffix('.bin').write_bytes(binary)
    q=Quantization(**{key:np.array(value) if isinstance(value,list) else value for key,value in meta['quantization'].items()});x=rng.integers(0,256,(2,ih,iw,ic),dtype=np.uint8);x[0]=zp;y=np.stack([native_input_reference(v,q,zp,pads) for v in x]);x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8');manifest.append(dict(meta,index=i,input_shape=[ih,iw,ic],output_shape=[oh,ow,oc],pads=pads))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
