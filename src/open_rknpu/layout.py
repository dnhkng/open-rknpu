"""MIT. Spatial-only terminal Reshape; preserves pixel order and channel count."""
from pathlib import Path
import copy,struct,tempfile
import onnx
from .model import checksum
from .sequence import decode_sequence


def compile_spatial_reshape(model,input_scale=1.0,input_zero_point=0):
    from .scheduler import compile_sequence
    g=model.graph;node=g.node[-1]
    values={v.name:v for v in [*g.input,*g.value_info,*g.output]}
    inputs={v.name for v in g.input};constants={v.name for v in g.initializer}-inputs
    if (node.domain not in ('','ai.onnx') or len(node.input)!=2 or node.input[1] not in constants
        or len(g.output)!=1 or list(node.output)!=[g.output[0].name] or node.input[0] not in values):
        raise ValueError('terminal Reshape requires an immutable shape and one output')
    before=values[node.input[0]];after=g.output[0]
    a=[d.dim_value for d in before.type.tensor_type.shape.dim];b=[d.dim_value for d in after.type.tensor_type.shape.dim]
    if (len(a)!=4 or len(b)!=4 or a[:2]!=b[:2] or a[0]!=1 or any(v<=0 for v in a+b)
        or a[2]*a[3]!=b[2]*b[3] or before.type.tensor_type.elem_type!=1 or after.type.tensor_type.elem_type!=1):
        raise ValueError('Reshape must preserve batch, channels and spatial pixel count')
    prefix=copy.deepcopy(model);del prefix.graph.node[-1];del prefix.graph.output[:];prefix.graph.output.append(before)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'prefix.onnx';onnx.save(prefix,p);data,meta=compile_sequence(p,input_scale,input_zero_point)
    data=bytearray(data);struct.pack_into('<II',data,28,b[2],b[3]);data[80:84]=bytes(4)
    struct.pack_into('<I',data,80,checksum(data));decode_sequence(data)
    return bytes(data),dict(meta,output_shape_nhwc=[1,b[2],b[3],b[1]],spatial_reshape=True)
