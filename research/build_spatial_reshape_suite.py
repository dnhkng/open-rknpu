"""MIT. Terminal spatial reshapes preserve NHWC pixel order."""
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
root=Path(__file__).resolve().parent;out=root/'spatial_reshape_suite';out.mkdir(exist_ok=True)
for i,(height,width) in enumerate([(4,16),(16,4),(2,32)]):
    m=onnx.load(root/'stride2_suite/model000.onnx');m.graph.node[0].output[0]='conv'
    m.graph.initializer.append(nh.from_array(np.array([1,3,height,width],np.int64),'shape'))
    m.graph.node.append(h.make_node('Reshape',['conv','shape'],['output']))
    dims=m.graph.output[0].type.tensor_type.shape.dim;dims[2].dim_value=height;dims[3].dim_value=width
    p=out/f'model{i:03}.onnx';onnx.save(m,p);data,meta=compile_sequence(p);p.with_suffix('.bin').write_bytes(data)
    q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
    x=np.fromfile(root/'stride2_suite/input000.u8',np.uint8).reshape(32,8,8,3);x.tofile(out/f'input{i:03}.u8')
    np.stack([reference(v,q).reshape(height,width,3) for v in x]).tofile(out/f'expected{i:03}.i8')
