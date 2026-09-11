"""MIT. Development-only RKNN primitive survey; vendor compilation is an oracle.
Each probe has the same small signed Conv stem and nonconstant test inputs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
VENDOR = os.environ.get('VENDOR_PYTHON', 'python')
NAMES = ['base','depthwise','relu','clip','leakyrelu','prelu','sigmoid','tanh',
         'hardsigmoid','hardswish','softplus','elu','swish','mish',
         'add','mul','sub','div','max','min','matmul','softmax',
         'concat','transpose','reshape','resize','globalavg','batchnorm']


def build(name):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh
    from rknn.api import RKNN
    out=ROOT/'fixtures'/('primitive_'+name);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(110310)
    constants=[];nodes=[]
    def const(name,value):
        constants.append(nh.from_array(np.asarray(value,np.float32),name));return name
    def node(op,inputs,output,**attrs):
        nodes.append(h.make_node(op,inputs,[output],**attrs));return output
    def conv(output):
        w=const(output+'_w',rng.uniform(-.025,.025,(3,3,1,1)))
        b=const(output+'_b',[-1,.5,1])
        return node('Conv',['input',w,b],output,kernel_shape=[1,1])
    x=conv('stem');result='output'
    if name=='base':node('Identity',[x],result)
    elif name=='depthwise':
        w=const('dw_w',rng.uniform(-.3,.3,(3,1,3,3)))
        node('Conv',[x,w,const('dw_b',[-.5,0,.5])],result,group=3,kernel_shape=[3,3],pads=[1]*4)
    elif name=='clip':node('Clip',[x,const('lo',0),const('hi',6)],result)
    elif name=='prelu':node('PRelu',[x,const('slope',np.array([.1,.2,.3]).reshape(3,1,1))],result)
    elif name in ['relu','leakyrelu','sigmoid','tanh','hardsigmoid','softplus','elu']:
        op={'relu':'Relu','leakyrelu':'LeakyRelu','sigmoid':'Sigmoid','tanh':'Tanh','hardsigmoid':'HardSigmoid','softplus':'Softplus','elu':'Elu'}[name]
        node(op,[x],result,**({'alpha':.1} if name=='leakyrelu' else {}))
    elif name in ['swish','mish','hardswish']:
        a=node('Softplus',[x],'sp') if name=='mish' else x
        a=node('Tanh' if name=='mish' else 'HardSigmoid' if name=='hardswish' else 'Sigmoid',[a],'act',**({'alpha':1/6,'beta':.5} if name=='hardswish' else {}))
        node('Mul',[x,a],result)
    elif name in ['add','mul','sub','div','max','min','concat']:
        # Nonlinear branches prevent constant folding / simple Conv weight folding.
        a=node('Sigmoid',[x],'a');b=node('Sigmoid',[conv('stem2')],'b')
        node({'add':'Add','mul':'Mul','sub':'Sub','div':'Div','max':'Max','min':'Min','concat':'Concat'}[name],[a,b],result,**({'axis':1} if name=='concat' else {}))
    elif name=='matmul':
        shape=nh.from_array(np.array([1,192],np.int64),'flatshape');constants.append(shape)
        a=node('Reshape',[x,'flatshape'],'flat')
        node('MatMul',[a,const('matrix',rng.uniform(-.2,.2,(192,8)))],result)
    elif name=='softmax':node('Softmax',[x],result,axis=1)
    elif name=='transpose':node('Transpose',[x],result,perm=[0,1,3,2])
    elif name=='reshape':
        constants.append(nh.from_array(np.array([1,3,4,16],np.int64),'shape'))
        node('Reshape',[x,'shape'],result)
    elif name=='resize':node('Resize',[x,'',const('scales',[1,1,2,2])],result,mode='nearest',coordinate_transformation_mode='asymmetric')
    elif name=='globalavg':node('GlobalAveragePool',[x],result)
    elif name=='batchnorm':node('BatchNormalization',[x,const('scale',[.8,1.2,.5]),const('bias',[.1,.2,.3]),const('mean',[.3,.5,.7]),const('var',[1,2,3])],result)
    shapes={'matmul':[1,8],'concat':[1,6,8,8],'reshape':[1,3,4,16],'resize':[1,3,16,16],'globalavg':[1,3,1,1]}
    shape=shapes.get(name,[1,3,8,8])
    m=h.make_model(h.make_graph(nodes,name,[h.make_tensor_value_info('input',1,[1,3,8,8])],
        [h.make_tensor_value_info(result,1,shape)],constants),opset_imports=[h.make_opsetid('',13)])
    m.ir_version=8;onnx.checker.check_model(m);onnx.save(m,out/'model.onnx')
    paths=[]
    for i in range(8):
        v=rng.integers(0,256,(1,3,8,8)).astype(np.float32);v[:,:,0,0]=0;v[:,:,-1,-1]=255
        p=out/f'calibration_{i}.npy';np.save(p,v);paths.append(str(p))
    (out/'dataset.txt').write_text('\n'.join(paths)+'\n')
    r=RKNN(verbose=True)
    try:
        assert r.config(target_platform='rv1103',mean_values=[[0]*3],std_values=[[1]*3])==0
        assert r.load_onnx(model=str(out/'model.onnx'))==0
        assert r.build(do_quantization=True,dataset=str(out/'dataset.txt'))==0
        assert r.export_rknn(str(out/'model.rknn'))==0
    finally:r.release()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--one',choices=NAMES);p.add_argument('--names',nargs='+',choices=NAMES,default=NAMES);a=p.parse_args()
    if a.one:build(a.one)
    else:
        out=ROOT/'primitive_survey';out.mkdir(exist_ok=True);results={}
        for name in a.names:
            with (out/(name+'_build.log')).open('w') as log:
                try:
                    r=subprocess.run([VENDOR,__file__,'--one',name],stdout=log,stderr=subprocess.STDOUT,timeout=120)
                    results[name]={'build_exit':r.returncode}
                except subprocess.TimeoutExpired:results[name]={'build_exit':'timeout'}
            print(name,results[name],flush=True)
            (out/'build_results.json').write_text(json.dumps(results,indent=2)+'\n')
