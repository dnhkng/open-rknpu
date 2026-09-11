"""MIT. Bounded ONNX QLinearConv import with explicit requantization semantics."""
import math
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from .native import compile_native_input
from .quantization import Quantization


def _scalar(constants,name,dtype):
    value=constants.get(name)
    if value is None or value.dtype!=dtype or value.size!=1:raise ValueError('QLinearConv requires scalar activation quantization constants')
    return value.reshape(()).item()


def _preserved_quantization(qw,wscale,wzp,qbias,xscale,xzp,yscale,yzp):
    """Build the native arithmetic contract without requantizing source weights."""
    oc,_,kernel,_=qw.shape
    weights=qw.astype(np.int64).reshape(oc,-1);zps=wzp.astype(np.int64)
    centered=weights-zps[:,None]
    hardware_bias=qbias.astype(np.int64)-(xzp-128)*centered.sum(axis=1)
    lo=-xzp;hi=255-xzp
    acc_lo=hardware_bias+np.minimum(lo*centered,hi*centered).sum(axis=1)
    acc_hi=hardware_bias+np.maximum(lo*centered,hi*centered).sum(axis=1)
    if np.any(acc_lo < -(1<<31)) or np.any(acc_hi >= 1<<31):
        raise ValueError('QLinearConv INT32 accumulator overflow')
    maximum=float(wscale.max())
    channel=np.rint(wscale/np.float32(maximum)*16384).astype(np.int64)
    if np.any(channel<1) or np.any(channel>16384):
        raise ValueError('source weight scales exceed native per-channel conversion range')
    factor=maximum*xscale/yscale
    shift=min(31,math.floor(math.log2(32767/factor)))
    multiplier=round(factor*2**shift)
    if shift<0 or not 1<=multiplier<=32767:
        raise ValueError('source output scale exceeds native conversion range')
    return Quantization(weights,zps,wscale.astype(np.float32),hardware_bias,channel,
                        multiplier,shift,yscale,yzp,kernel,input_scale=xscale,input_zero_point=xzp)


def compile_qlinearconv(model):
    g=model.graph
    if len(g.node)!=1 or g.node[0].op_type!='QLinearConv' or g.node[0].domain not in ('','ai.onnx'):
        raise ValueError('quantized import requires one standard QLinearConv')
    node=g.node[0];constants={v.name:nh.to_array(v) for v in g.initializer}
    if len(g.input)!=1 or len(g.output)!=1 or len(node.input) not in (8,9) or node.input[0]!=g.input[0].name or list(node.output)!=[g.output[0].name]:
        raise ValueError('QLinearConv requires one activation input/output and constant parameters')
    xscale=float(_scalar(constants,node.input[1],np.dtype('float32')))
    xzp=int(_scalar(constants,node.input[2],np.dtype('uint8')))
    yscale=float(_scalar(constants,node.input[6],np.dtype('float32')))
    yzp=int(_scalar(constants,node.input[7],np.dtype('int8')))
    qw=constants.get(node.input[3]);wscale=constants.get(node.input[4]);wzp=constants.get(node.input[5])
    if qw is None or qw.dtype!=np.int8 or qw.ndim!=4 or wscale is None or wscale.dtype!=np.float32 or wzp is None or wzp.dtype!=np.int8:
        raise ValueError('QLinearConv weights must be constant INT8 with float32 scale and INT8 zero point')
    oc=qw.shape[0]
    if wscale.size not in (1,oc) or wzp.size not in (1,oc):raise ValueError('weight quantization must be scalar or per output channel')
    ws=np.broadcast_to(wscale.reshape(-1),(oc,)).astype(np.float32)
    wz=np.broadcast_to(wzp.reshape(-1),(oc,)).astype(np.int32)
    if not np.isfinite(ws).all() or np.any(ws<=0) or not np.isfinite([xscale,yscale]).all() or xscale<=0 or yscale<=0:
        raise ValueError('QLinearConv scales must be finite and positive')
    weights=((qw.astype(np.int32)-wz[:,None,None,None])*ws[:,None,None,None]).astype(np.float32)
    bias=np.zeros(oc,np.float32);qb=np.zeros(oc,np.int32)
    if len(node.input)==9:
        qb=constants.get(node.input[8])
        if qb is None or qb.dtype!=np.int32 or qb.shape!=(oc,):raise ValueError('QLinearConv bias must be constant INT32 [output_channels]')
        bias=(qb.astype(np.float64)*xscale*ws).astype(np.float32)
    ishape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    oshape=[d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    conv=h.make_node('Conv',['input','weights','bias'],['output'],**{a.name:h.get_attribute_value(a) for a in node.attribute})
    lower=h.make_model(h.make_graph([conv],'qlinearconv_import',
        [h.make_tensor_value_info('input',onnx.TensorProto.FLOAT,ishape)],
        [h.make_tensor_value_info('output',onnx.TensorProto.FLOAT,oshape)],
        [nh.from_array(weights,'weights'),nh.from_array(bias,'bias')]),opset_imports=[h.make_opsetid('',13)]);lower.ir_version=min(model.ir_version,8)
    preserved=_preserved_quantization(qw,ws,wz,qb,xscale,xzp,yscale,yzp)
    binary,meta=compile_native_input(lower,xscale,xzp,quantization=preserved)
    meta.update(quantized_import='QLinearConv weights preserved bit-for-bit',
                source_weight_scales=ws.tolist(),source_weight_zero_points=wz.tolist(),source_output_scale=yscale,source_output_zero_point=yzp)
    return binary,meta


def compile_qdq_conv(model):
    g=model.graph;nodes=list(g.node)
    if len(nodes)!=4 or [n.op_type for n in nodes]!=['DequantizeLinear','DequantizeLinear','Conv','QuantizeLinear']:
        raise ValueError('Q/DQ import requires input DQ, constant-weight DQ, Conv, output Q')
    xdq,wdq,conv,yq=nodes;constants={v.name:nh.to_array(v) for v in g.initializer}
    if (len(g.input)!=1 or len(g.output)!=1 or any(n.domain not in ('','ai.onnx') for n in nodes)
        or xdq.input[0]!=g.input[0].name or conv.input[0]!=xdq.output[0] or conv.input[1]!=wdq.output[0]
        or yq.input[0]!=conv.output[0] or list(yq.output)!=[g.output[0].name]):
        raise ValueError('invalid Q/DQ Conv graph connections')
    xattrs={a.name:h.get_attribute_value(a) for a in xdq.attribute};wattrs={a.name:h.get_attribute_value(a) for a in wdq.attribute};yattrs={a.name:h.get_attribute_value(a) for a in yq.attribute}
    if xattrs or yattrs or any(k!='axis' for k in wattrs):raise ValueError('unsupported Q/DQ attributes')
    xscale=float(_scalar(constants,xdq.input[1],np.dtype('float32')));xzp=int(_scalar(constants,xdq.input[2],np.dtype('uint8')))
    yscale=float(_scalar(constants,yq.input[1],np.dtype('float32')));yzp=int(_scalar(constants,yq.input[2],np.dtype('int8')))
    qw=constants.get(wdq.input[0]);ws=constants.get(wdq.input[1]);wz=constants.get(wdq.input[2])
    if qw is None or qw.dtype!=np.int8 or qw.ndim!=4 or ws is None or ws.dtype!=np.float32 or wz is None or wz.dtype!=np.int8:
        raise ValueError('Q/DQ weights must be constant INT8/float32/INT8')
    oc=qw.shape[0]
    if ws.size not in (1,oc) or wz.size not in (1,oc) or (ws.size==oc and wattrs.get('axis',1)!=0):
        raise ValueError('per-channel weight Q/DQ requires axis0')
    scales=np.broadcast_to(ws.reshape(-1),(oc,)).astype(np.float32);zps=np.broadcast_to(wz.reshape(-1),(oc,)).astype(np.int32)
    if not np.isfinite(scales).all() or np.any(scales<=0) or not np.isfinite([xscale,yscale]).all() or xscale<=0 or yscale<=0:
        raise ValueError('Q/DQ scales must be finite and positive')
    weights=((qw.astype(np.int32)-zps[:,None,None,None])*scales[:,None,None,None]).astype(np.float32)
    bias=np.zeros(oc,np.float32) if len(conv.input)==2 else constants.get(conv.input[2])
    if bias is None or bias.dtype!=np.float32 or bias.shape!=(oc,):raise ValueError('Q/DQ Conv bias must be constant float32 [output_channels]')
    ishape=[d.dim_value for d in g.input[0].type.tensor_type.shape.dim];oshape=[d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    lower_node=h.make_node('Conv',['input','weights','bias'],['output'],**{a.name:h.get_attribute_value(a) for a in conv.attribute})
    lower=h.make_model(h.make_graph([lower_node],'qdq_conv_import',[h.make_tensor_value_info('input',onnx.TensorProto.FLOAT,ishape)],
        [h.make_tensor_value_info('output',onnx.TensorProto.FLOAT,oshape)],[nh.from_array(weights,'weights'),nh.from_array(bias,'bias')]),opset_imports=[h.make_opsetid('',13)]);lower.ir_version=min(model.ir_version,8)
    # Q/DQ carries a float bias, so convert only that bias to exact accumulator
    # units while retaining the supplied INT8 weight tensor unchanged.
    qb=np.rint(bias.astype(np.float64)/(xscale*scales)).astype(np.int64)
    if not np.isfinite(qb).all() or np.any(qb < -(1<<31)) or np.any(qb >= 1<<31):
        raise ValueError('Q/DQ Conv bias is outside INT32 accumulator range')
    preserved=_preserved_quantization(qw,scales,zps,qb,xscale,xzp,yscale,yzp)
    binary,meta=compile_native_input(lower,xscale,xzp,quantization=preserved)
    meta.update(quantized_import='Q/DQ Conv INT8 weights preserved bit-for-bit; float bias rounded once to INT32',source_weight_scales=scales.tolist(),source_weight_zero_points=zps.tolist())
    return binary,meta
