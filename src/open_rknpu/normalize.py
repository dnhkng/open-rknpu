"""SPDX-License-Identifier: MIT

Conservative ONNX rewrites before hardware lowering. No runtime CPU fallback.
"""
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper

# --- front-end lowerings ---------------------------------------------------
# The NPU has no dense engine and no rank-3 input, so both features are pure ONNX
# rewrites into verified Conv shapes: a dense layer is a 1x1 Conv over the channel axis,
# and a 1-D [N, C, L] graph is its H=1 2-D form. Each rewrite is pinned by an exact
# container-equality test in `tests/test_dense_lowering.py`.

# Ops whose 1-D form is the same elementwise mapping as the promoted [N, C, 1, L] form.
# Anything else (Reshape, Transpose, a constant broadcast, ...) is rejected by name, so a
# rank-3 graph is never promoted halfway.
_PROMOTABLE_OPS = frozenset({"Conv", "MaxPool", "AveragePool", "GlobalAveragePool",
                             "Relu", "Clip", "LeakyRelu", "Sigmoid", "Tanh", "Identity"})
_RANK_ONE_OPS = frozenset({"Conv", "MaxPool", "AveragePool", "GlobalAveragePool"})
_SPATIAL_ATTRS = ("kernel_shape", "pads", "strides", "dilations")


def _label(node):
    """A node's own name, else its output name, for an error that names the offender."""
    return node.name or (node.output[0] if node.output else node.op_type)


def _insert_unit_height(shape):
    """Rewrite a rank-3 [N, C, L] TensorShapeProto to [N, C, 1, L]."""
    dims = [(dim.dim_value, dim.dim_param) for dim in shape.dim]
    del shape.dim[:]
    for index, (value, param) in enumerate(dims):
        if index == 2:
            shape.dim.add().dim_value = 1
        entry = shape.dim.add()
        if param:
            entry.dim_param = param
        else:
            entry.dim_value = value


def _promote_rank_three(model, constants):
    """Lower a rank-3 [N, C, L] convolution graph to the supported H=1 2-D form.

    A 1-D convolution is the same arithmetic as a 2-D convolution with H=1, and [N, C, L]
    and [N, C, 1, L] are the same memory layout. The whole graph is promoted in place: the
    input and output annotations gain the unit height axis, every Conv weight gains the
    unit kernel row, and every spatial attribute gains the unit height entry. The bytes the
    caller supplies and reads are unchanged; the container reports H=1, which the docs
    state.

    Only graphs with a Conv or pool node are considered, and only the ops listed in
    `_PROMOTABLE_OPS` are accepted - a rank-sensitive node is rejected by name rather than
    promoting half the graph.
    """
    graph = model.graph
    if len(graph.input) != 1:
        return
    dims = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    if len(dims) != 3 or not all(size > 0 for size in dims):
        return
    if not any(node.domain in ("", "ai.onnx") and node.op_type in _RANK_ONE_OPS
               for node in graph.node):
        return
    for value in graph.output:
        if len(value.type.tensor_type.shape.dim) != 3:
            raise ValueError("1-D rank promotion requires rank-3 outputs; '%s' is rank %d"
                             % (value.name, len(value.type.tensor_type.shape.dim)))
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type not in _PROMOTABLE_OPS:
            raise ValueError("1-D rank promotion cannot rewrite %s node '%s'"
                             % (node.op_type, _label(node)))
        if node.op_type == "Conv":
            weights = constants.get(node.input[1])
            if weights is None or weights.dtype != np.float32 or weights.ndim != 3:
                raise ValueError("1-D rank promotion requires constant rank-3 weights for "
                                 "Conv node '%s'" % _label(node))
            promoted = np.ascontiguousarray(
                weights.reshape(*weights.shape[:2], 1, weights.shape[2]))
            for tensor in graph.initializer:
                if tensor.name == node.input[1]:
                    tensor.CopyFrom(numpy_helper.from_array(promoted, node.input[1]))
                    break
            constants[node.input[1]] = promoted
        for attribute in node.attribute:
            if attribute.name not in _SPATIAL_ATTRS:
                continue
            values = list(attribute.ints)
            # kernel/strides/dilations [k] -> [1, k]; pads [a, b] -> [0, a, 0, b] because
            # ONNX orders pads as [begin1, begin2, end1, end2], so the 1-D ends land on
            # the width axis and the new height axis stays unpadded.
            if attribute.name == "pads" and len(values) == 2:
                values = [0, values[0], 0, values[1]]
            elif attribute.name != "pads" and len(values) == 1:
                values = [1, *values]
            else:
                continue
            del attribute.ints[:]
            attribute.ints.extend(values)
    _insert_unit_height(graph.input[0].type.tensor_type.shape)
    for value in graph.output:
        _insert_unit_height(value.type.tensor_type.shape)


def _static_shapes(model):
    """Fully static shapes from non-strict shape inference, keyed by tensor name.

    A symbolic or missing dimension drops the entry, so a rewrite only fires on a shape
    inference proved.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if dims and all(size > 0 for size in dims):
            shapes[value.name] = dims
    return shapes


def _dense_source(name, constants, shapes, producers, consumers, outputs):
    """The [N, C, 1, 1] tensor a dense layer reads, and the flatten node to drop.

    Returns `(tensor_name, shape, flatten)`. The activation may be the rank-four feature
    map itself, or a rank-two [N, C] produced by `Flatten(axis=1)` or a constant `Reshape`
    of that map. Only a proven [N, C, 1, 1] producer is accepted, and the flatten must have
    this single consumer and not be a graph output, because the lowering drops it.
    """
    shape = shapes.get(name)
    if shape is not None and len(shape) == 4 and shape[2:] == [1, 1]:
        return name, shape, None
    flatten = producers.get(name)
    if (flatten is None or flatten.domain not in ("", "ai.onnx")
            or flatten.op_type not in ("Flatten", "Reshape")
            or len(consumers.get(name, [])) != 1 or name in outputs):
        return None, None, None
    source = shapes.get(flatten.input[0])
    if source is None or len(source) != 4 or source[2:] != [1, 1]:
        return None, None, None
    if shape is not None and shape != source[:2]:
        return None, None, None
    attrs = {a.name: helper.get_attribute_value(a) for a in flatten.attribute}
    if flatten.op_type == "Flatten":
        if attrs.get("axis", 1) != 1 or set(attrs) - {"axis"}:
            return None, None, None
        return flatten.input[0], source, flatten
    target = constants.get(flatten.input[1]) if len(flatten.input) == 2 else None
    if (attrs.get("allowzero", 0) != 0 or set(attrs) - {"allowzero"}
            or target is None or target.ndim != 1):
        return None, None, None
    dims = [int(value) for value in target]
    if len(dims) != 2:
        return None, None, None
    # A constant Reshape must flatten exactly to the producer's [N, C]: 0 copies the input
    # extent at the same index, -1 is inferred from the element count.
    resolved = [source[index] if value == 0 else (None if value == -1 else value)
                for index, value in enumerate(dims)]
    if resolved.count(None) > 1:
        return None, None, None
    if None in resolved:
        known = 1
        for value in resolved:
            if value is not None:
                known *= value
        total = source[0] * source[1]
        if known <= 0 or total % known:
            return None, None, None
        resolved[resolved.index(None)] = total // known
    if resolved != source[:2]:
        return None, None, None
    return flatten.input[0], source, flatten


def _dense_plan(node, constants, shapes, producers, consumers, outputs):
    """Derive the 1x1 Conv rewrite for one MatMul/Gemm node, or None when unsupported.

    Pure: it only reads the graph and returns `(tensor_name, weight, bias, flatten,
    output_shape)`, so a node that is not an exact match is left untouched rather than half
    rewritten. The accepted algebra is the identity `MatMul(A, B) = Conv(A, B.T)` with `B`
    constant, or its `Gemm` form with `alpha=1`, `beta=1`, optional `transB` and optional
    [K] bias.
    """
    if node.op_type == "MatMul":
        attrs, bias_input = {}, None
        if len(node.input) != 2 or node.attribute:
            return None
    else:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (len(node.input) not in (2, 3) or set(attrs) - {"transA", "transB", "alpha", "beta"}
                or attrs.get("transA", 0) != 0 or attrs.get("alpha", 1.0) != 1.0
                or attrs.get("beta", 1.0) != 1.0 or attrs.get("transB", 0) not in (0, 1)):
            return None
        bias_input = node.input[2] if len(node.input) == 3 else None
    operand = constants.get(node.input[1])
    if operand is None or operand.dtype != np.float32 or operand.ndim != 2 or operand.size == 0:
        return None
    source_name, source, flatten = _dense_source(node.input[0], constants, shapes, producers,
                                                 consumers, set(outputs))
    if source is None:
        return None
    transpose = attrs.get("transB", 0) == 1
    rows, channels = operand.shape if transpose else operand.shape[::-1]
    if channels != source[1] or rows < 1:
        return None
    weight = np.ascontiguousarray((operand if transpose else operand.T)
                                  .reshape(rows, channels, 1, 1)).astype(np.float32)
    bias = None
    if bias_input is not None:
        value = constants.get(bias_input)
        if value is None or value.dtype != np.float32 or value.size != rows:
            return None
        bias = np.ascontiguousarray(value.reshape(-1)).astype(np.float32)
    output = outputs.get(node.output[0])
    if output is None:
        return None
    declared = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    if declared == [source[0], rows]:
        target = [source[0], rows, 1, 1]
    elif declared == [source[0], rows, 1, 1]:
        target = declared
    else:
        return None
    return source_name, weight, bias, flatten, target


def _lower_dense_layers(model, constants):
    """Algebraically rewrite a dense MatMul/Gemm into the verified 1x1 Conv path.

    `MatMul(A, B)` with a constant `B[C, K]` and `Gemm(A, B[, Cb])` with `alpha=1`,
    `beta=1` are one dense layer over `A`'s channel axis. Its exact Conv form is
    `Conv(A, B.T)` with kernel `[K, C, 1, 1]`, so the dense node becomes that Conv and the
    optional Gemm bias becomes the Conv bias. `A` may be the `[N, C, 1, 1]` feature map
    itself or a rank-2 `[N, C]` produced by flattening one, in which case the flatten node
    is dropped and the Conv reads the producer. Everything else is left untouched, so the
    scheduler's explicit rejection still fires for an external operand, a spatial `A`, a
    non-identity Gemm or an unproven shape.
    """
    graph = model.graph
    dense = [node for node in graph.node if node.domain in ("", "ai.onnx")
             and node.op_type in ("MatMul", "Gemm")]
    if not dense:
        return
    shapes = _static_shapes(model)
    producers, consumers = {}, {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
        for name in node.output:
            producers[name] = node
    outputs = {value.name: value for value in graph.output}
    names = ({tensor.name for tensor in graph.initializer} | set(outputs)
             | {value.name for value in graph.input}
             | {name for node in graph.node for name in node.output})
    dropped = set()
    for node in dense:
        plan = _dense_plan(node, constants, shapes, producers, consumers, outputs)
        if plan is None:
            continue
        source, weight, bias, flatten, target = plan
        weight_name = node.output[0] + "_dense_weight"
        while weight_name in names:
            weight_name += "_"
        names.add(weight_name)
        graph.initializer.append(numpy_helper.from_array(weight, weight_name))
        constants[weight_name] = weight
        promoted_inputs = [source, weight_name]
        if bias is not None:
            bias_name = node.output[0] + "_dense_bias"
            while bias_name in names:
                bias_name += "_"
            names.add(bias_name)
            graph.initializer.append(numpy_helper.from_array(bias, bias_name))
            constants[bias_name] = bias
            promoted_inputs.append(bias_name)
        node.op_type = "Conv"
        node.domain = ""
        del node.attribute[:]
        node.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
        del node.input[:]
        node.input.extend(promoted_inputs)
        if flatten is not None:
            dropped.add(flatten.output[0])
        shape = outputs[node.output[0]].type.tensor_type.shape
        del shape.dim[:]
        for size in target:
            shape.dim.add().dim_value = size
    if dropped:
        retained = [node for node in graph.node if node.output[0] not in dropped]
        del graph.node[:]
        graph.node.extend(retained)


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

    # A rank-3 [N, C, L] graph is the same computation as its H=1 2-D form, and a dense
    # MatMul/Gemm is the verified 1x1 Conv over the channel axis. Both are pure rewrites
    # into shapes the emitters already accept; anything outside their envelope is left
    # for the scheduler to reject explicitly.
    _promote_rank_three(result, constants)
    _lower_dense_layers(result, constants)

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
