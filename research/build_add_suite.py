"""MIT. Fresh Add graphs and integer references; no vendor dependencies."""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.elementwise import add_reference,mul_reference,sub_reference,max_reference

parser=argparse.ArgumentParser();parser.add_argument('--op',choices=['Add','Mul','Sub','Max'],default='Add');args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/('research/'+args.op.lower()+'_suite');root.mkdir(exist_ok=True)
rng=np.random.default_rng({'Add':110312,'Mul':110313,'Sub':110314,'Max':110315}[args.op]);manifest=[]
for i in range(12):
    nodes=[];constants=[]
    for branch in ['a','b']:
        w=rng.uniform(-.3,.3,(3,3,1,1)).astype(np.float32)
        bias=rng.uniform(-3,3,3).astype(np.float32)
        if i%4==1:w=abs(w)
        if i%4==2:w=-abs(w)
        if i%4==3 and branch=='b':w*=.05
        constants.extend([nh.from_array(w,branch+'w'),nh.from_array(bias,branch+'b')])
        nodes.append(h.make_node('Conv',['input',branch+'w',branch+'b'],[branch],kernel_shape=[1,1]))
    nodes.append(h.make_node(args.op,['a','b'],['output']))
    m=h.make_model(h.make_graph(nodes,'add',[h.make_tensor_value_info('input',1,[1,3,8,8])],
        [h.make_tensor_value_info('output',1,[1,3,8,8])],constants),opset_imports=[h.make_opsetid('',13)])
    m.ir_version=8
    path=root/f'model{i:03}.onnx';onnx.save(m,path)
    scale,zp=[(1.,0),(.25,128),(.5,255)][i%3]
    data,meta=compile_sequence(path,scale,zp);path.with_suffix('.bin').write_bytes(data)
    qs=[Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in b['quantization'].items()}) for b in meta['branches']]
    x=rng.integers(0,256,(32,8,8,3),dtype=np.uint8)
    x[0]=zp;x[1]=0;x[2]=255;x[3]=np.arange(192,dtype=np.uint8).reshape(8,8,3)
    y=np.stack([{'Add':add_reference,'Mul':mul_reference,'Sub':sub_reference,'Max':max_reference}[args.op](reference(v,qs[0]),reference(v,qs[1])) for v in x])
    x.tofile(root/f'input{i:03}.u8');y.tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_scale=scale,input_zero_point=zp,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Built 12 independent {args.op} graphs, 384 inferences')
