"""MIT. Verify independent nonzero zero points on both Mul operands."""
from pathlib import Path
import json,numpy as np,onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import mul_output_conversion,mul_requant_reference

root=Path(__file__).resolve().parent/'mul_boundary_zero_points_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110337);manifest=[]
for i,zps in enumerate(((17,-23),(-128,127),(1,-1),(63,-64))):
    ih=5+i;iw=6+i;model=h.make_model(h.make_graph([h.make_node('Mul',['a','b'],['output'])],'mul_zp',
        [h.make_tensor_value_info(v,1,[1,3,ih,iw]) for v in ('a','b')],[h.make_tensor_value_info('output',1,[1,3,ih,iw])]),opset_imports=[h.make_opsetid('',13)]);model.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(model,path);scale,inzp=((1.,0),(.25,128),(.5,255))[i%3]
    _,initial=compile_sequence(path,scale,inzp,mul_operand_zero_points=zps)
    initial_q=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in initial['branches']]
    product_scale=initial_q[0].output_scale*initial_q[1].output_scale
    output_range=None if i==0 else dict(scale=product_scale*(2,.5,4)[i-1],zero_point=(37,-128,127)[i-1])
    binary,meta=compile_sequence(path,scale,inzp,output_range,mul_operand_zero_points=zps);path.with_suffix('.bin').write_bytes(binary)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in branch['quantization'].items()}) for branch in meta['branches']]
    _,outzp,mult,shift=mul_output_conversion(qs[0].output_scale*qs[1].output_scale,output_range)
    a=rng.integers(0,256,(12,ih,iw,3),dtype=np.uint8);b=rng.integers(0,256,a.shape,dtype=np.uint8);a[0]=0;b[0]=255;a[1]=255;b[1]=0;a[2]=inzp;b[2]=inzp
    y=np.stack([mul_requant_reference(reference(x,qs[0]),reference(v,qs[1]),mult,shift,outzp,zps) for x,v in zip(a,b)])
    np.concatenate([a,b],axis=1).tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(meta,index=i,input_scale=scale,input_zero_point=inzp,mul_operand_zero_points=list(zps),input_shape=[ih,iw,3],compile_output_range=output_range))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
