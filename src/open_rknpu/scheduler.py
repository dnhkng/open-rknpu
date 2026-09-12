"""SPDX-License-Identifier: MIT

Linear ONNX lowering with explicit task and activation allocations.
Supports Conv[/Relu] with pooling, the fixed depthwise profile, or two-branch Add/Mul/Sub/Max.
"""
from pathlib import Path
import struct
import tempfile
import numpy as np
import onnx
from onnx import helper as h
from .compiler import compile_model
from .normalize import normalize_model
from .pooling import POOL
from .register_profile import REGISTERS
from .model import decode

# Every profile `_compile_sequence` can dispatch, in dispatch order. Each key names one
# branch below; adding or removing a branch means adding or removing its key here and its
# row in `docs/calibration-cookbook.md` ("Which profiles accept calibration").
# `tests/test_calibration_parity.py` fails when the two sets disagree.
DISPATCH_PROFILES = (
    "qlinearconv",
    "qdq-conv",
    "join-chain",
    "join-dag",
    "depthwise-join",
    "pool-join",
    "pooled-branches",
    "lut",
    "mul-relu",
    "mul-clip",
    "mul-add",
    "leaky-relu",
    "prelu",
    "transposed-conv",
    "depthwise-pointwise",
    "reshape",
    "standalone-mul",
    "constant-mul",
    "per-channel-constant-mul",
    "runtime-scale-mul",
    "two-head",
    "join-walk",
    "diamond",
    "native-chain",
    "legacy-conv-chain",
    "multi-input-elementwise-dag",
    "elementwise-dag",
    "elementwise-join",
    "chain-walk",
    "native-input",
    "strided",
    "depthwise",
    "pooling-sequence",
)

def batched_layout(data,info):
    from .compose import batched_layout as check
    return check(data,info)


def batched_supported(data,info):
    """Raise ValueError unless the container is a valid batched job.

    Rule (measured, docs/plans/pipelining-plan.md S1/S2): one ioctl may carry a list of tasks
    only when every task links to the next program (register 0x10, control 0x40) and
    all tasks use one engine. Batched submission is slower than serial at 8x8 sizes,
    so serial stays the default.
    """
    reason=batched_layout(data,info)
    if reason:
        raise ValueError('batched submission rejected: '+reason)
    return True
from .sequence import encode_sequence


def align(n,a=64):
    return (n+a-1)//a*a


def _fold_leading_pad(loaded):
    """Fold a leading Pad on the graph input into the input shape.

    The CNA border path injects only the activation zero point, so reflection,
    edge/replication and circular borders have no verified NPU producer. The
    compiler records the mode/amounts and the caller pads the packed input with
    `open_rknpu.padding.pad_input` before calling the runtime.
    """
    graph=loaded.graph;nodes=list(graph.node)
    if not nodes or nodes[0].op_type!='Pad' or nodes[0].domain not in ('','ai.onnx'):return loaded,None
    pad=nodes[0]
    if len(graph.input)!=1 or pad.input[0]!=graph.input[0].name:return loaded,None
    from onnx import numpy_helper as nh
    constants={t.name:nh.to_array(t) for t in graph.initializer}
    if len(pad.input)<2 or pad.input[1] not in constants:raise ValueError('leading Pad requires constant pads')
    attrs={a.name:onnx.helper.get_attribute_value(a) for a in pad.attribute}
    mode=attrs.get('mode',b'constant')
    if isinstance(mode,bytes):mode=mode.decode()
    if mode not in ('constant','reflect','edge','wrap'):raise ValueError('unsupported Pad mode: '+str(mode))
    if len(pad.input)>3 and pad.input[3] in constants:
        raise ValueError('leading Pad with explicit axes is not supported')
    values=np.asarray(constants[pad.input[1]]).reshape(-1).astype(int)
    if values.size!=8:raise ValueError('leading Pad must pad a rank-four NCHW input')
    top,left,bottom,right=int(values[2]),int(values[3]),int(values[6]),int(values[7])
    if any(v<0 for v in (top,left,bottom,right)):raise ValueError('negative Pad amounts are not supported')
    shape=[d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    if len(shape)!=4 or any(v<=0 for v in shape):raise ValueError('leading Pad requires a static NCHW input')
    padded=[shape[0],shape[1],shape[2]+top+bottom,shape[3]+left+right]
    stripped=onnx.ModelProto();stripped.CopyFrom(loaded)
    del stripped.graph.node[0]
    del stripped.graph.input[:]
    stripped.graph.input.append(h.make_tensor_value_info(pad.output[0],stripped.graph.input[0].type.tensor_type.elem_type if False else 1,padded))
    info=dict(mode=mode,pads=[top,left,bottom,right],unpadded_shape=shape,padded_shape=padded,
              requires_host_preprocessing=True,
              helper='open_rknpu.padding.pad_input(array, pads, mode)')
    return stripped,info


def compile_sequence(path,input_scale=1.0,input_zero_point=0,output_range=None,mul_operand_zero_points=(0,0),mutable_weights=False,mutable_constants=False,calibration_ranges=None,expose_intermediates=False,reuse_intermediates=False,asymmetric_depthwise=False,per_channel_mul=False,submission=None,tiles=None):
    loaded=path if isinstance(path,onnx.ModelProto) else onnx.load(path)
    loaded,padding=_fold_leading_pad(loaded)
    binary,meta=_compile_sequence(loaded,input_scale,input_zero_point,output_range,mul_operand_zero_points,
                                  mutable_weights,mutable_constants,calibration_ranges,expose_intermediates,
                                  reuse_intermediates,asymmetric_depthwise,per_channel_mul,submission,tiles)
    if submission=='batched':
        # Every container can be submitted as one linked job: the programs are untouched
        # and only the task tails (next-program link + the successor's fetch amount) and
        # the header flag change (docs/plans/pipelining-plan.md S8/S10). A legacy single-program
        # container has no task table, so its submission shape is already the only one.
        from .sequence import relink_for_batched
        binary=relink_for_batched(binary)
        if binary[:8]==b"ORNPUSEQ":
            info=decode(binary)
            meta['engine_runs']=[info['task_count']]
            batched_supported(binary,info)
        else:
            meta['engine_runs']=[1]
    meta['submission']='batched' if submission=='batched' else 'serial'
    if padding is not None:
        meta['input_padding']=padding
        meta['shape_nhwc']=[padding['padded_shape'][0],padding['padded_shape'][2],padding['padded_shape'][3],padding['padded_shape'][1]]
    return binary,meta


def _compile_sequence(loaded,input_scale=1.0,input_zero_point=0,output_range=None,mul_operand_zero_points=(0,0),mutable_weights=False,mutable_constants=False,calibration_ranges=None,expose_intermediates=False,reuse_intermediates=False,asymmetric_depthwise=False,per_channel_mul=False,submission=None,tiles=None):
    if submission not in (None,'serial','batched'):
        raise ValueError('submission must be None, serial or batched')
    batched=submission=='batched'
    if tiles is not None and (not isinstance(tiles,int) or tiles<2):
        raise ValueError('tiles must be an integer of at least 2')
    if calibration_ranges is not None and output_range is not None:
        raise ValueError('calibration and output quantization overrides cannot be combined')
    def measured(name):
        from .calibration import measured_range
        return measured_range(calibration_ranges,name)
    if mutable_weights and not (len(loaded.graph.node) in (1,2) and loaded.graph.node[0].op_type=='Conv' and loaded.graph.node[-1].op_type in ('Conv','Relu')):
        raise ValueError('mutable weights currently require one native Conv[/Relu]')
    if mutable_constants and not (len(loaded.graph.node)==1 and loaded.graph.node[0].op_type=='Mul'):
        raise ValueError('mutable constants currently require one constant Mul')
    if tuple(mul_operand_zero_points)!=(0,0) and not any(n.op_type=='Mul' for n in loaded.graph.node):
        raise ValueError('Mul operand zero points require a Mul profile')
    if len(loaded.graph.node)==1 and loaded.graph.node[0].op_type=='QLinearConv':
        if (input_scale,input_zero_point)!=(1.0,0) or output_range is not None or calibration_ranges is not None:
            raise ValueError('QLinearConv carries its own quantization parameters')
        from .quantized_import import compile_qlinearconv
        return compile_qlinearconv(loaded)
    if [n.op_type for n in loaded.graph.node]==['DequantizeLinear','DequantizeLinear','Conv','QuantizeLinear']:
        if (input_scale,input_zero_point)!=(1.0,0) or output_range is not None or calibration_ranges is not None:
            raise ValueError('Q/DQ Conv carries its own quantization parameters')
        from .quantized_import import compile_qdq_conv
        return compile_qdq_conv(loaded)
    model=normalize_model(loaded)
    graph=model.graph
    nodes=list(graph.node)
    # The join chain reads the shared stem through three or more heads, so it must
    # be matched before the terminal elementwise profiles (`Mul`+`Add` and the
    # like) that only look at the last two nodes.
    if nodes and nodes[0].op_type=='Conv':
        from .graph import parse_join_chain
        chain=parse_join_chain(nodes,{v.name for v in graph.input})
        if chain is not None:
            if (input_scale,input_zero_point)!=(1.0,0):
                raise ValueError('join chain requires the established UINT8 scale1/zero-point0 boundary')
            if tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            from .graph import compile_join_chain
            return compile_join_chain(model,output_range,mul_operand_zero_points,asymmetric_depthwise,
                                      serial=not batched,calibration_ranges=calibration_ranges)
    if nodes and nodes[0].op_type=='Conv':
        from .join_dag import parse_join_dag
        if parse_join_dag(nodes):
            if (input_scale,input_zero_point)!=(1.0,0):
                raise ValueError('join DAG requires the established UINT8 scale1/zero-point0 boundary')
            if tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            from .join_dag import compile_join_dag
            return compile_join_dag(model,output_range,serial=not batched,calibration_ranges=calibration_ranges)
    if nodes and nodes[0].op_type=='Conv':
        from .depthwise_join import parse_depthwise_join
        if parse_depthwise_join(nodes):
            if (input_scale,input_zero_point)!=(1.0,0):
                raise ValueError('depthwise join requires the established UINT8 scale1/zero-point0 boundary')
            if tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            from .depthwise_join import compile_depthwise_join
            return compile_depthwise_join(model,output_range,mul_operand_zero_points,asymmetric_depthwise,
                                          serial=not batched,calibration_ranges=calibration_ranges)
    if nodes and nodes[0].op_type=='Conv':
        from .pool_join import parse_pool_join
        if parse_pool_join(nodes):
            if (input_scale,input_zero_point)!=(1.0,0):
                raise ValueError('pool join requires the established UINT8 scale1/zero-point0 boundary')
            if tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            from .pool_join import compile_pool_join
            return compile_pool_join(model,output_range,serial=not batched,calibration_ranges=calibration_ranges)
    if nodes and nodes[0].op_type=='Conv':
        from .pooled_branches import parse_pooled_branches
        if parse_pooled_branches(nodes):
            if (input_scale,input_zero_point)!=(1.0,0):
                raise ValueError('pooled branches require the established UINT8 scale1/zero-point0 boundary')
            if calibration_ranges is not None:
                raise ValueError('calibration is unsupported for pooled branches')
            if tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            from .pooled_branches import compile_pooled_branches
            return compile_pooled_branches(model,output_range,serial=not batched)
    def resolved_range():
        """The output band, resolved on demand from `output_range` or the report.

        Deferring the `measured()` lookup means a profile only asks for the graph
        output tensor when it consumes the band. The join walks and the diamond
        profile derive their bands from the Conv tensors instead, so a report measured
        on a join graph does not need an output entry the measurer cannot produce.
        """
        if output_range is not None:return output_range
        if calibration_ranges is not None and graph.output:return measured(graph.output[0].name)
        return None
    if nodes and nodes[-1].op_type in ('Sigmoid','Tanh'):
        if output_range is not None or calibration_ranges is not None:raise ValueError('output override unsupported for LUT profile')
        from .lut import compile_lut
        return compile_lut(model,input_scale,input_zero_point)
    if len(nodes)>=2 and nodes[-2].op_type=='Mul' and nodes[-1].op_type=='Relu':
        from .elementwise import compile_mul_relu
        return compile_mul_relu(model,input_scale,input_zero_point,resolved_range(),mul_operand_zero_points)
    if len(nodes)>=2 and nodes[-2].op_type=='Mul' and nodes[-1].op_type=='Clip':
        from .elementwise import compile_mul_clip
        return compile_mul_clip(model,input_scale,input_zero_point,resolved_range(),mul_operand_zero_points)
    if len(nodes)>=2 and nodes[-2].op_type=='Mul' and nodes[-1].op_type=='Add':
        from .elementwise import compile_mul_add
        return compile_mul_add(model,input_scale,input_zero_point,resolved_range(),mul_operand_zero_points)
    if nodes and nodes[-1].op_type=='LeakyRelu':
        if output_range is not None or calibration_ranges is not None:raise ValueError('output override unsupported for LeakyRelu profile')
        from .activation import compile_leaky
        return compile_leaky(model,input_scale,input_zero_point)
    if nodes and nodes[-1].op_type=='PRelu':
        if output_range is not None or calibration_ranges is not None:raise ValueError('output override unsupported for PRelu profile')
        from .activation import compile_prelu
        return compile_prelu(model,input_scale,input_zero_point)
    if nodes and nodes[-1].op_type=='ConvTranspose':
        from .transposed import compile_transposed
        return compile_transposed(model,input_scale,input_zero_point,resolved_range())
    if len(nodes)>=3 and nodes[-2].op_type=='Conv' and nodes[-1].op_type=='Conv':
        previous_attrs={a.name:h.get_attribute_value(a) for a in nodes[-2].attribute}
        if previous_attrs.get('group',1)>1:
            from .depthwise import compile_depthwise_pointwise
            return compile_depthwise_pointwise(model,input_scale,input_zero_point,resolved_range())
    if nodes and nodes[-1].op_type=='Reshape':
        if output_range is not None or calibration_ranges is not None:raise ValueError('output override unsupported for Reshape profile')
        from .layout import compile_spatial_reshape
        return compile_spatial_reshape(model,input_scale,input_zero_point)
    if len(nodes)==1 and nodes[0].op_type=='Mul':
        from .elementwise import compile_standalone_mul,compile_constant_mul
        shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.input}
        initializers={t.name for t in graph.initializer}
        if (len(graph.input)==2
            and any(shapes[v.name]==[1,3,1,1] and v.name not in initializers for v in graph.input)
            and any(len(shapes[v.name])==4 and shapes[v.name][:2]==[1,3] and shapes[v.name][2:]!=[1,1]
                    for v in graph.input)):
            from .elementwise import compile_runtime_scale_mul
            return compile_runtime_scale_mul(model,input_scale=input_scale,input_zero_point=input_zero_point,
                                             output_range=resolved_range())
        if any(v.name in nodes[0].input for v in graph.initializer if v.name not in {x.name for x in graph.input}):
            if per_channel_mul:
                from .elementwise import compile_per_channel_constant_mul
                return compile_per_channel_constant_mul(model,input_scale,input_zero_point,resolved_range())
            return compile_constant_mul(model,input_scale,input_zero_point,resolved_range(),mul_operand_zero_points,mutable_constants)
        return compile_standalone_mul(model,input_scale,input_zero_point,resolved_range(),mul_operand_zero_points)
    if [n.op_type for n in nodes]==['Conv','Relu','Conv','Conv'] and len(graph.output)==2:
        if (input_scale,input_zero_point)!=(1.0,0):
            raise ValueError('two-head profile requires the established UINT8 scale1/zero-point0 boundary')
        if output_range is not None or calibration_ranges is not None:raise ValueError('output override unsupported for the two-head profile')
        if tuple(mul_operand_zero_points)!=(0,0):raise ValueError('Mul operand zero points require a Mul profile')
        from .graph import compile_two_head
        return compile_two_head(model)
    join_index=next((index for index,node in enumerate(nodes)
                     if node.op_type in ('Add','Mul','Sub','Max')),None)
    # Op-level fan-in join with a pool between a head and the join: no profile above
    # covers that shape, so the walk lowers it op by op. Graphs whose join has no pool
    # belong to the diamond branch below (which also prefers the walk, byte-identically).
    if (join_index is not None and len(graph.input)==1 and len(graph.output)==1
            and any(node.op_type in ('MaxPool','AveragePool') for node in nodes)
            and (input_scale,input_zero_point)==(1.0,0)
            and tuple(mul_operand_zero_points)==(0,0)):
        from .walk import compile_join_walk, parse_join_walk
        if parse_join_walk(graph) is not None:
            return compile_join_walk(model,input_scale,input_zero_point,output_range,
                                     serial=not batched,ranges=calibration_ranges)
    if (join_index is not None and join_index>=3
        and all(n.op_type in ('Conv','Relu') for index,n in enumerate(nodes) if index!=join_index)):
        join=nodes[join_index];head_a,head_b=nodes[join_index-2],nodes[join_index-1];stem_nodes=nodes[:join_index-2]
        tail_nodes=nodes[join_index+1:]
        tail_ok=(not tail_nodes) or (len(tail_nodes)%2==1 and
                                     all(n.op_type==('Conv' if i%2==0 else 'Relu') for i,n in enumerate(tail_nodes)))
        if (head_a.op_type=='Conv' and head_b.op_type=='Conv' and tail_ok
            and [n.op_type for n in stem_nodes] in (['Conv'],['Conv','Relu'])
            and head_a.input[0]==stem_nodes[-1].output[0]
            and head_b.input[0]==head_a.input[0]
            and list(join.input)==[head_a.output[0],head_b.output[0]]
            and ((not tail_nodes and join.output[0]==graph.output[0].name)
                 or (tail_nodes and tail_nodes[0].input[0]==join.output[0]
                     and tail_nodes[-1].output[0]==graph.output[0].name))
            and len(graph.input)==1 and len(graph.output)==1):
            if (input_scale,input_zero_point)!=(1.0,0):
                raise ValueError('diamond profile requires the established UINT8 scale1/zero-point0 boundary')
            if join.op_type!='Mul' and tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            # The op-level walk lowers this class op by op and is byte-identical to the
            # profile (`tests/test_walk.py`); the profile stays the fallback for graphs
            # the walk does not cover, such as Mul joins with operand zero points.
            if tuple(mul_operand_zero_points)==(0,0):
                from .walk import compile_join_walk, parse_join_walk
                if parse_join_walk(graph) is not None:
                    return compile_join_walk(model,input_scale,input_zero_point,output_range,
                                             serial=not batched,ranges=calibration_ranges)
            from .graph import compile_diamond
            return compile_diamond(model,output_range,mul_operand_zero_points,serial=not batched,
                                   calibration_ranges=calibration_ranges)
    if (len(nodes)>=5 and len(nodes)%2==1 and all(n.op_type=='Conv' for n in nodes[::2])
        and all(n.op_type=='Relu' for n in nodes[1::2])):
        if (input_scale,input_zero_point)!=(1.0,0):raise ValueError('native chain requires the established UINT8 scale1/zero-point0 boundary')
        if tiles is not None:
            if output_range is not None or expose_intermediates or reuse_intermediates:
                raise ValueError('height-strip tiling cannot be combined with output overrides, exposed intermediates or arena reuse')
            if calibration_ranges is not None:
                raise ValueError('calibration is unsupported for the height-strip tiled chain')
            from .tiled_chain import compile_tiled_chain
            return compile_tiled_chain(model,tiles=tiles,serial=not batched)
        from .chain_n import compile_chain_n
        return compile_chain_n(model,output_range,expose_intermediates,reuse_intermediates,
                               serial=not batched,calibration_ranges=calibration_ranges)
    if [n.op_type for n in nodes]==['Conv','Relu','Conv']:
        attrs={a.name:h.get_attribute_value(a) for a in nodes[-1].attribute}
        if attrs.get('group',1)==1:
            if (input_scale,input_zero_point)!=(1.0,0):raise ValueError('legacy Conv chain requires the established UINT8 scale1/zero-point0 boundary')
            from .chain import compile_chain
            with tempfile.TemporaryDirectory() as tmp:
                chain_path=Path(tmp)/'chain.onnx';onnx.save(model,chain_path)
                payload,meta=compile_chain(chain_path,output_range=output_range,calibration_ranges=calibration_ranges)
            result=encode_sequence(payload,input_shape=(8,8,3),output_shape=(8,8,3),input_stride=16,
                arena_bytes=16384,input_offset=0x2000,output_offset=0x3000,
                tasks=[(0,126,29,768),(0x440,126,29,768)],input_scale=1.0,input_zero_point=0,
                output_scale=meta['output_scale'],output_zero_point=meta['output_zero_point'])
            meta['sequence_profile']='Conv-Relu-Conv';return result,meta
    if (len(nodes)>=5 and len(nodes)%2==1 and [n.op_type for n in nodes[:2]]==['Conv','Conv']
        and nodes[2].op_type in ('Add','Mul','Sub','Max') and all(n.op_type=='Conv' for n in nodes[3::2])
        and all(n.op_type=='Mul' for n in nodes[4::2]) and len(graph.input)>=3):
        if (input_scale,input_zero_point)!=(1.0,0):
            raise ValueError('multi-input elementwise DAG requires the established UINT8 scale1/zero-point0 boundary')
        if output_range is not None or calibration_ranges is not None:
            raise ValueError('output override unsupported for the multi-input elementwise DAG')
        if tuple(mul_operand_zero_points)!=(0,0):
            raise ValueError('Mul operand zero points require a Mul profile')
        from .elementwise_multi import compile_multi_input_dag
        return compile_multi_input_dag(model)
    if (len(nodes)>=4 and [n.op_type for n in nodes][:2]==['Conv','Conv'] and nodes[2].op_type in ('Add','Mul','Sub','Max')
        and all(n.op_type=='Mul' for n in nodes[3:])):
        if (input_scale,input_zero_point)!=(1.0,0):
            raise ValueError('elementwise DAG requires the established UINT8 scale1/zero-point0 boundary')
        if output_range is not None or calibration_ranges is not None:
            raise ValueError('output override unsupported for the elementwise DAG')
        if tuple(mul_operand_zero_points)!=(0,0):
            raise ValueError('Mul operand zero points require a Mul profile')
        from .elementwise_chain import compile_elementwise_dag
        return compile_elementwise_dag(model)
    if nodes and nodes[-1].op_type in ('Add','Mul','Sub','Max'):
        from .elementwise import compile_elementwise
        return compile_elementwise(model,input_scale,input_zero_point,resolved_range(),mul_operand_zero_points)
    # Op-level chain walk: a linear Conv/Relu chain with a pool that is *not* the last
    # node. No profile above matches that shape (the pooling profiles end in a pool or
    # branch around one), so the walk cannot hijack existing evidence.
    if any(node.op_type in ('MaxPool','AveragePool') and index < len(nodes) - 1
           for index, node in enumerate(nodes)):
        from .walk import compile_chain_walk, parse_chain
        if parse_chain(graph) is not None:
            if tuple(mul_operand_zero_points)!=(0,0):
                raise ValueError('Mul operand zero points require a Mul profile')
            return compile_chain_walk(model, input_scale, input_zero_point, resolved_range(),
                                      serial=not batched, ranges=calibration_ranges)
    if len(graph.input)!=1 or len(graph.output)!=1 or not nodes or nodes[0].op_type!="Conv":
        raise ValueError("sequence lowering requires one input, one output, and an initial Conv")
    attrs={a.name:h.get_attribute_value(a) for a in nodes[0].attribute}
    kernel=next((t.dims[2] for t in graph.initializer if t.name==nodes[0].input[1] and len(t.dims)==4),1)
    input_shape=[d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    if len(input_shape)!=4:raise ValueError('static NCHW input required')
    _,ic,ih,iw=input_shape
    output_channels=next((t.dims[0] for t in graph.initializer if t.name==nodes[0].input[1] and len(t.dims)==4),0)
    if ((output_range is not None and not any(n.op_type=='Conv' for n in nodes[1:])) or (len(nodes)==2 and nodes[1].op_type=='Clip') or ic not in (1,3) or output_channels>16 or kernel>=7 or (len(nodes)<=2 and not all(5<=v<=(32 if ic==1 else 8) for v in (ih,iw)))):
        from .native import compile_native_input
        return compile_native_input(model,input_scale,input_zero_point,resolved_range(),expose_constants=mutable_weights)
    if attrs.get('strides',[1,1])!=[1,1] or (len(nodes)<=2 and attrs.get('pads',[0]*4)!=[kernel//2]*4):
        from .strided import compile_strided
        if output_range is not None or calibration_ranges is not None:raise ValueError('output override unsupported for this strided profile')
        return compile_strided(model,input_scale,input_zero_point)
    first_count=2 if len(nodes)>1 and nodes[1].op_type=="Relu" else 1
    if any(n.op_type=='Conv' for n in nodes[first_count:]):
        from .depthwise import compile_depthwise
        return compile_depthwise(model,input_scale,input_zero_point,resolved_range(),asymmetric_pair=asymmetric_depthwise)
    if output_range is not None:raise ValueError('output override unsupported for this scheduled profile')
    if any(n.op_type not in ("MaxPool","AveragePool") for n in nodes[first_count:]):
        raise ValueError("sequence lowering currently supports Conv[/Relu] followed by 2x2 pooling")
    from .pooling import pool_registers,pool_tag
    values={v.name:v for v in [*graph.input,*graph.value_info,*graph.output]}
    first_output=nodes[first_count-1].output[0]
    if first_output not in values: raise ValueError("missing static convolution output shape")
    sub=h.make_model(h.make_graph(nodes[:first_count],"initial_conv",list(graph.input),
        [values[first_output]],list(graph.initializer)),opset_imports=list(model.opset_import))
    sub.ir_version=model.ir_version
    with tempfile.TemporaryDirectory() as tmp:
        subpath=Path(tmp)/"conv.onnx";onnx.save(sub,subpath)
        source,meta=compile_model(subpath,input_scale=input_scale,input_zero_point=input_zero_point,calibration_ranges=calibration_ranges)
    _,height,width,channels=meta["output_shape_nhwc"]
    stages=[];previous=first_output
    for node in nodes[first_count:]:
        attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
        allowed={"kernel_shape":[2,2],"strides":[2,2],"pads":[0,0,0,0],
                 "auto_pad":b"NOTSET","ceil_mode":0,"count_include_pad":0,"storage_order":0,"dilations":[1,1]}
        if (node.domain not in ("","ai.onnx") or list(node.input)!=[previous]
            or len(node.output)!=1 or attrs.get("kernel_shape")!=[2,2]
            or attrs.get("strides")!=[2,2]
            or any(k not in allowed or v!=allowed[k] for k,v in attrs.items())
            or height<2 or width<2):
            raise ValueError("unsupported pooling attributes, shape, or graph connections")
        stages.append((node.op_type,height,width,height//2,width//2))
        height//=2;width//=2;previous=node.output[0]
    output=graph.output[0]
    if (output.name!=previous or output.type.tensor_type.elem_type!=1
        or [d.dim_value for d in output.type.tensor_type.shape.dim]!=[1,channels,height,width]):
        raise ValueError("sequence output shape does not match graph")
    if len(stages)+1>64: raise ValueError("too many NPU tasks")

    # Allocate programs, immutable constants, then external and intermediate tensors.
    programs=[0];cursor=align((126+4)*8)
    for _ in stages:
        programs.append(cursor);cursor+=align((37+4)*8)
    regs={w&65535:(w>>16)&0xffffffff for w in struct.unpack_from("<126Q",source)}
    weight_size=align(regs[0x1030]);bias_size=align(((channels+3)//4)*32)
    weights=cursor;bias=weights+weight_size;payload_size=align(bias+bias_size)
    input_offset=align(payload_size,4096)
    _,ih,iw,ic=meta["shape_nhwc"];stride=align(iw,16)
    cursor=align(input_offset+ih*stride*ic)
    activations=[cursor];cursor+=align(ih*iw*16)
    for _,_,_,oh,ow in stages:
        activations.append(cursor);cursor+=align(oh*ow*16)
    arena_size=align(cursor,4096)
    data=bytearray(payload_size)
    data[weights:weights+weight_size]=source[regs[0x1110]:regs[0x1110]+weight_size]
    data[bias:bias+bias_size]=source[regs[0x5020]:regs[0x5020]+bias_size]
    regs.update({0x1110:weights,0x5020:bias,0x1070:input_offset,0x4020:activations[0]})
    for i,(reg,default,tag) in enumerate(REGISTERS):
        struct.pack_into("<Q",data,i*8,tag<<48|regs.get(reg,default)<<16|reg)
    tasks=[(0,126,29,768)]
    for index,(kind,ph,pw,oh,ow) in enumerate(stages,1):
        fields=pool_registers(kind,ph,pw,oh,ow,activations[index-1],activations[index])
        for i,(reg,default) in enumerate(POOL):
            tag=pool_tag(reg)
            struct.pack_into("<Q",data,programs[index]+i*8,tag<<48|fields.get(reg,default)<<16|reg)
        tasks.append((programs[index],37,96,3072))
    for index,(offset,count,enable,_) in enumerate(tasks):
        nxt=0  # Separate submissions avoid unverified producer/consumer overlap.
        control=0x14 if nxt else 0x28
        for j,(reg,value,tag) in enumerate(((0x10,nxt,0x101),(0x14,control,0x101),(0,0,0x41),(8,enable,0x81))):
            struct.pack_into("<Q",data,offset+(count+j)*8,tag<<48|value<<16|reg)
    result=encode_sequence(data,input_shape=(ih,iw,ic),output_shape=(height,width,channels),
        input_stride=stride,arena_bytes=arena_size,input_offset=input_offset,output_offset=activations[-1],
        tasks=tasks,input_scale=input_scale,input_zero_point=input_zero_point,
        output_scale=meta["output_scale"],output_zero_point=meta["output_zero_point"],serial=True)
    meta.update(output_shape_nhwc=[1,height,width,channels],pool_stages=[s[0] for s in stages])
    return result,meta
