"""SPDX-License-Identifier: MIT

Conservative ONNX rewrites before hardware lowering. No runtime CPU fallback.
"""
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper


def normalize_model(model):
    """Return a checked copy with constant reshapes and Conv bias/padding folded.

    Only immutable initializers are constants: older ONNX graphs may expose
    initializers as overridable inputs. Branches and public outputs are retained.
    """
    onnx.checker.check_model(model)
    result = copy.deepcopy(model)
    graph = result.graph
    inputs = {v.name for v in graph.input}
    constants = {v.name: numpy_helper.to_array(v) for v in graph.initializer
                 if v.name not in inputs}
    nodes = []
    for node in graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (node.domain in ("", "ai.onnx") and node.op_type == "Reshape"
                and len(node.input) == 2 and all(n in constants for n in node.input)):
            data, shape = (constants[n] for n in node.input)
            shape = [int(n) for n in shape]
            if not attrs.get("allowzero", 0):
                shape = [data.shape[i] if n == 0 else n for i, n in enumerate(shape)]
            value = data.reshape(shape)
            graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
            constants[node.output[0]] = value
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)

    # Constant grouped/dilated kernels can use the verified dense kernel path.
    # This is algebraic lowering, not a claim about native group/dilation fields.
    used_names={v.name for v in graph.initializer}|inputs|{v for n in graph.node for v in n.output}
    for node in graph.node:
        if node.op_type!='Conv' or node.domain not in ('','ai.onnx'):continue
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        w=constants.get(node.input[1]);group=attrs.get('group',1);dilation=attrs.get('dilations',[1,1])
        if w is None or w.dtype!=np.float32 or w.ndim!=4:continue
        # Leave native depthwise chains available to their dedicated compiler.
        direct=node.input[0] in inputs
        grouped=direct and 1<group<=32 and w.shape[1]*group<=32 and w.shape[0]%group==0 and 1<=w.shape[0]<=128
        effective=[(size-1)*d+1 for size,d in zip(w.shape[2:],dilation)]
        dilated=direct and len(dilation)==2 and all(d>=1 for d in dilation) and dilation!=[1,1] and all(v<=5 for v in effective)
        if not grouped and not dilated:continue
        if group!=1 and not grouped:continue
        value=w.copy()
        if grouped:
            value=np.zeros((w.shape[0],w.shape[1]*group,*w.shape[2:]),np.float32)
            per_group=w.shape[0]//group
            for g in range(group):value[g*per_group:(g+1)*per_group,g*w.shape[1]:(g+1)*w.shape[1]]=w[g*per_group:(g+1)*per_group]
        if dilated:
            expanded=np.zeros((*value.shape[:2],*effective),np.float32);expanded[:,:,::dilation[0],::dilation[1]]=value;value=expanded
        name=node.input[1]+'_dense'
        while name in used_names:name+='_' 
        used_names.add(name);graph.initializer.append(numpy_helper.from_array(value,name));constants[name]=value;node.input[1]=name
        replaced={'group':1} if grouped else {}
        if dilated:replaced.update(dilations=[1,1],kernel_shape=effective)
        kept=[a for a in node.attribute if a.name not in replaced];del node.attribute[:];node.attribute.extend(kept)
        node.attribute.extend(helper.make_attribute(k,v) for k,v in replaced.items())

    inferred = onnx.shape_inference.infer_shapes(result)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*inferred.graph.input, *inferred.graph.value_info]}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    outputs = {v.name for v in graph.output}
    removed = set()
    names = {v.name for v in graph.initializer} | inputs
    names.update(n for node in graph.node for n in node.output)
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Conv":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weights = constants.get(node.input[1])
        shape = shapes.get(node.input[0], [])
        auto = attrs.get("auto_pad", b"NOTSET")
        if (weights is not None and weights.ndim == 4 and len(shape) == 4
                and all(n > 0 for n in shape[2:])
                and auto in (b"SAME_UPPER", b"SAME_LOWER", b"VALID")):
            pads = []
            for size, kernel, stride, dilation in zip(
                    shape[2:], weights.shape[2:], attrs.get("strides", [1, 1]),
                    attrs.get("dilations", [1, 1])):
                total = (0 if auto == b"VALID" else max(
                    0, ((size + stride - 1) // stride - 1) * stride
                    + dilation * (kernel - 1) + 1 - size))
                begin = (total + (auto == b"SAME_LOWER")) // 2
                pads.append((begin, total - begin))
            kept = [a for a in node.attribute if a.name not in ("auto_pad", "pads")]
            del node.attribute[:]
            node.attribute.extend(kept)
            node.attribute.append(helper.make_attribute("pads", [p[0] for p in pads] + [p[1] for p in pads]))
        uses = consumers.get(node.output[0], [])
        if weights is None or len(uses) != 1 or node.output[0] in outputs:
            continue
        mul=uses[0]
        if mul.op_type=='Mul' and mul.domain in ('','ai.onnx') and not mul.attribute and len(mul.input)==2:
            other=mul.input[1] if mul.input[0]==node.output[0] else mul.input[0]
            factor=constants.get(other)
            bias=constants.get(node.input[2]) if len(node.input)==3 else np.zeros(weights.shape[0],weights.dtype)
            if (factor is not None and factor.dtype==np.float32 and weights.dtype==np.float32
                and factor.shape in ((),(1,),(weights.shape[0],1,1),(1,weights.shape[0],1,1))
                and bias is not None and bias.shape==(weights.shape[0],)
                and np.isfinite(factor).all()):
                scales=np.broadcast_to(factor.reshape(-1),(weights.shape[0],))
                newweights=weights*scales[:,None,None,None];newbias=bias*scales
                if np.isfinite(newweights).all() and np.isfinite(newbias).all():
                    for suffix,value,index in [('weights',newweights,1),('bias',newbias,2)]:
                        name=mul.output[0]+'_mul_'+suffix
                        while name in names:name+='_' 
                        names.add(name);graph.initializer.append(numpy_helper.from_array(value,name));constants[name]=value
                        if index<len(node.input):node.input[index]=name
                        else:node.input.append(name)
                    node.output[0]=mul.output[0];removed.add(mul.output[0]);continue
        add = uses[0]
        if (add.domain not in ("", "ai.onnx") or add.op_type != "Add"
                or add.attribute or len(add.input) != 2):
            continue
        other = add.input[1] if add.input[0] == node.output[0] else add.input[0]
        bias = constants.get(other)
        # NCHW broadcast: a [C] vector biases width, not channels.
        if bias is None or bias.shape not in ((weights.shape[0], 1, 1), (1, weights.shape[0], 1, 1)):
            continue
        if bias.dtype != weights.dtype or len(node.input) != 2:
            continue  # Existing bias + Add can introduce different float rounding.
        name = add.output[0] + "_conv_bias"
        while name in names:
            name += "_"
        names.add(name)
        graph.initializer.append(numpy_helper.from_array(bias.reshape(-1), name))
        node.input.append(name)
        node.output[0] = add.output[0]
        removed.add(add.output[0])
    retained = [n for n in graph.node if not (n.op_type in ("Add","Mul") and n.output[0] in removed)]
    del graph.node[:]
    graph.node.extend(retained)
    # Embed small even/rectangular kernels into supported odd square kernels.
    # Extra bottom/right padding keeps the original output extent and anchor.
    for node in graph.node:
        if node.op_type!='Conv' or node.domain not in ('','ai.onnx'):continue
        w=constants.get(node.input[1]);attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        if w is None or w.ndim!=4 or w.dtype!=np.float32 or attrs.get('dilations',[1,1])!=[1,1]:continue
        kh,kw=w.shape[2:]
        if not (1<=kh<=5 and 1<=kw<=5) or (kh==kw and kh in (1,3,5)):continue
        if attrs.get('auto_pad',b'NOTSET')!=b'NOTSET':continue
        kernel=3 if max(kh,kw)<=3 else 5
        value=np.zeros((*w.shape[:2],kernel,kernel),np.float32);value[:,:,:kh,:kw]=w
        pads=list(attrs.get('pads',[0]*4));pads[2]+=kernel-kh;pads[3]+=kernel-kw
        name=node.input[1]+'_square'
        while name in names:name+='_' 
        names.add(name);constants[name]=value;graph.initializer.append(numpy_helper.from_array(value,name));node.input[1]=name
        kept=[a for a in node.attribute if a.name not in ('kernel_shape','pads')];del node.attribute[:];node.attribute.extend(kept)
        node.attribute.extend([helper.make_attribute('kernel_shape',[kernel,kernel]),helper.make_attribute('pads',pads)])
    # Discard stale internal shape annotations after renaming folded outputs.
    del graph.value_info[:]
    onnx.checker.check_model(result)
    return onnx.shape_inference.infer_shapes(result)
