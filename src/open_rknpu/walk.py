"""SPDX-License-Identifier: MIT

Op-level chain walk: one container from a linear graph of Conv/Relu/Clip and 2x2
stride-2 pooling, dispatched one op at a time onto `open_rknpu.compose`.

Every other scheduled profile matches a whole graph shape and emits it in one piece.
This module instead validates the graph, walks its nodes in order and builds one
`Stage` per node with the same verified per-op builders the branch profiles use
(`_layer_fields`/`_pack_layer_head` for a Conv, `pool_registers` for a pool), tracking
each tensor's quantization band as it goes. It is the first path where the *position*
of a pool in a Conv chain is free: `Conv -> pool -> Conv`, repeated pools and a pool
before the output all lower to the same composed container.

Supported class (one input, one output):

* float32 `[1,3,H,W]` input, even `H`/`W` before every pool;
* Conv: dense group1 K1/K3 with symmetric padding, stride 1, constant float32
  weights/bias, 1..16 output channels, with an optional `Relu` or `Clip[0,6]`
  directly after it (folded into the same task);
* `MaxPool`/`AveragePool`: `kernel_shape [2,2]`, `strides [2,2]`; the default pool
  attributes (`pads 0`, `ceil_mode 0`, `dilations 1`, ...) may be written out explicitly,
  as torch's exporter does;
* `Add`/`Sub`/`Max`/`Mul` between the running tensor and an immutable float32 constant
  that is a scalar, per-channel `[1,C,1,1]` or full spatial `[1,C,H,W]`; the task is the
  standalone elementwise emitter's register program with its second operand read from a
  payload constant grid, and it must be followed (through pools) by a Conv;
* the graph ends with a Conv (which may take the output-range override) or a pool.

Two refinements matter for trained models:

* **large images** - the legacy single-Conv emitter only accepts 5..8 pixels a side, so
  an image outside that range is staged as a **native16 surface** and the first Conv is
  emitted with `native.py`'s verified register/weight builders (single task, up to the
  6144-atom limit). `research/mel_kws_suite/` is the board evidence: a trained audio CNN
  at `3x32x32`.
* **calibration** - with `ranges` from `open_rknpu.calibration.measure`, every Conv whose
  measured tensor has an entry uses the measured band instead of the analytic one; a
  Conv that feeds a pool is still re-quantized onto a zero-point-0 grid. Analytic bands
  are unusable for a trained network (they can be orders of magnitude too wide), so this
  is what makes a real model compile at all.

The walk is deliberately a *chain*: fan-out and joins are the branch profiles' job.
No vendor artifact is read.
"""
import struct
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from .chain import native_quantize, native_reference
from .compiler import compile_model
from .compose import Binding, ConstantSpec, Stage, TensorSpec, compose
from .elementwise import (add_reference, max_reference, mul_output_conversion,
                          mul_reference, sub_reference)
from .graph import _align, _join_fields
from .join_dag import _pack_layer_head
from .native import native_fields, native_input_reference
from .pooling import pool_registers
from .quantization import Quantization, reference
from .sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT, ROLE_INTERNAL,
                       ROLE_OUTPUT)

POOLS = ("MaxPool", "AveragePool")

# The constant elementwise stage the walk lowers inside a chain: one operand is the
# chain tensor, the other an immutable initializer. The register program is the
# verified elementwise emitter's (`elementwise.py`), reused through `_join_fields`
# exactly as the join walk does.
EW_KINDS = ("Add", "Sub", "Max", "Mul")
EW_REFERENCE = {"Add": add_reference, "Sub": sub_reference,
                "Max": max_reference, "Mul": mul_reference}


def _surface(height, width, channels):
    return ((height * width + 3) // 4) * 64 * ((channels + 15) // 16)


def _adjusted_scale(scale, zero_point):
    """Widest scale that still covers a band after it is moved to zero point 0.

    A pooled tensor keeps the band of the Conv that feeds it, and the DPU pool
    programs the verified profiles use assume zero point 0, so a Conv feeding a pool
    is re-quantized onto a zero-point-0 grid (`pooled_branches` does the same).
    """
    return float(np.float32(scale * max(128 + zero_point, 127 - zero_point) / 127))


def _elementwise_error(index, kind, reason):
    """One rejected elementwise stage: the stage, its op and the bound it broke."""
    return "walk elementwise stage %d (%s) %s" % (index, kind, reason)


def _constant_mode(constant, channels, height, width):
    """The broadcast mode of an elementwise constant, or None when it does not fit."""
    shape = tuple(np.asarray(constant).shape)
    if shape in ((), (1,), (1, 1, 1)):
        return "scalar"
    if shape in ((channels, 1, 1), (1, channels, 1, 1)):
        return "per-channel"
    if shape in ((channels, height, width), (1, channels, height, width)):
        return "spatial"
    return None


def _constant_codes(op, scale):
    """The INT8 operand grid the emitter writes: `clip(rint(constant / scale))`.

    The expression is the constant-Mul emitter's `expanded / scale` on the NCHW
    broadcast, so the bytes here and in the payload constant block are identical.
    """
    expanded = np.broadcast_to(np.asarray(op["constant"], np.float32),
                               (1, op["channels"], op["height"], op["width"]))[0]
    return np.clip(np.rint(expanded / scale), -128, 127).astype(np.int8).transpose(1, 2, 0)


def _elementwise_band(op, predecessor_scale):
    """Bands for one constant elementwise stage, from the emitter's own rules.

    Add/Sub/Max keep both operands on one shared zero-point-0 scale - the wider of the
    predecessor's adjusted band and the constant's symmetric bound, exactly like the
    standalone two-branch emitter takes the widest branch - and the output scale is
    twice it. Mul keeps the two operands on their own scales (the constant on
    `max|constant|/127`, the predecessor's adjusted band on the other) and folds them
    into its own output conversion.
    """
    constant = np.asarray(op["constant"], np.float32)
    bound = float(np.float32(np.max(np.abs(constant)) / 127)) if constant.size else 0.0
    if op["op"] == "Mul":
        constant_scale = bound or 1.0
        # The emitter folds the two operand scales exactly as they are (no intermediate
        # float32 rounding), so the walk's band must do the same.
        product = predecessor_scale * constant_scale
        try:
            output_scale, output_zero_point = mul_output_conversion(product)[:2]
        except ValueError:
            raise ValueError(_elementwise_error(
                op["index"], op["op"],
                "Mul output scale is outside the verified conversion range")) from None
        return dict(kind="ew", op=op["op"], scale=predecessor_scale,
                    constant_scale=constant_scale, output_scale=output_scale,
                    output_zero_point=output_zero_point)
    scale = float(np.float32(max(predecessor_scale, bound)))
    return dict(kind="ew", op=op["op"], scale=scale, constant_scale=scale,
                output_scale=float(np.float32(2 * scale)), output_zero_point=0)


def _input_bytes(height, width, channels):
    return height * ((width + 15) // 16 * 16) * channels


def _fused_tensor_names(graph):
    """Map each Conv output to its measured tensor (the fused activation output).

    The same naming `open_rknpu.calibration.measured_tensor_names` uses: when a `Relu`
    or `Clip` directly follows the Conv and consumes its only output, the measured
    tensor is the activation's output.
    """
    nodes = list(graph.node)
    names = {}
    for index, node in enumerate(nodes):
        if node.op_type != "Conv":
            continue
        tensor = node.output[0]
        if (index + 1 < len(nodes) and nodes[index + 1].op_type in ("Relu", "Clip")
                and list(nodes[index + 1].input) == [node.output[0]]):
            tensor = nodes[index + 1].output[0]
        names[node.output[0]] = tensor
    return names


def _conv_attributes(node, weight_shape):
    """Validate one dense Conv node and return its geometry."""
    if node.domain not in ("", "ai.onnx") or len(node.input) != 3:
        raise ValueError("walk Conv requires default domain with constant weights and bias")
    attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
    kernel = weight_shape[2]
    supported = {"kernel_shape": [kernel, kernel], "pads": [kernel // 2] * 4,
                 "strides": [1, 1], "dilations": [1, 1], "group": 1}
    if kernel not in (1, 3) or weight_shape[2] != weight_shape[3]:
        raise ValueError("walk Conv supports square K1/K3 kernels")
    if any(key not in supported or value != supported[key] for key, value in attrs.items()):
        raise ValueError("unsupported walk Conv attributes")
    return kernel


POOL_DEFAULTS = {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0],
                 "dilations": [1, 1], "ceil_mode": 0, "auto_pad": b"NOTSET",
                 "storage_order": 0, "count_include_pad": 0}


def _pool_attributes(node):
    if node.domain not in ("", "ai.onnx"):
        raise ValueError("walk pool requires the default domain")
    attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
    # Exporters are free to write defaults explicitly (torch does for MaxPool:
    # ceil_mode/dilations/pads), so the check is on values, not on absence. Only the
    # 2x2 stride-2 shape survives; anything else is rejected.
    if attrs.get("kernel_shape") != [2, 2] or attrs.get("strides") != [2, 2]:
        raise ValueError("walk pool supports 2x2 stride-2 MaxPool/AveragePool only")
    for name, value in attrs.items():
        if name not in POOL_DEFAULTS or POOL_DEFAULTS[name] != value:
            raise ValueError("walk pool supports 2x2 stride-2 MaxPool/AveragePool only")
    return node.op_type


def _elementwise_successor(nodes, position):
    """Why the node after an elementwise stage is outside the walk envelope, or None.

    The envelope is `elementwise -> (2x2 pool ->)* Conv[/Relu] ...`: the constant
    stage exists to change the band a later Conv reads, so it must reach a Conv. One
    elementwise stage per chain is the bound the register/band bookkeeping is written
    for.
    """
    for node in nodes[position + 1:]:
        if node.op_type in POOLS:
            continue
        if node.op_type == "Conv":
            return None
        if node.op_type in EW_KINDS:
            return "supports one elementwise stage per chain"
        return "must be followed (through 2x2 pools) by a Conv[/Relu]"
    return "must be followed (through 2x2 pools) by a Conv[/Relu]"


def _elementwise_stage_error(node, ops, previous, constants, index, channels, height, width,
                             nodes, position):
    """Why one elementwise node is outside the walk envelope, or None when it fits."""
    op = node.op_type
    if (node.domain not in ("", "ai.onnx") or node.attribute or len(node.input) != 2
            or len(node.output) != 1):
        return _elementwise_error(index, op,
            "must be a plain default-domain Add|Sub|Max|Mul with two inputs and one output")
    if index > 1:
        return _elementwise_error(index, op, "supports one elementwise stage per chain")
    if not ops or ops[-1]["kind"] != "conv":
        return _elementwise_error(index, op,
            "must directly follow the chain Conv[/Relu]")
    if previous is None or previous not in node.input:
        return _elementwise_error(index, op,
            "requires one operand to be the chain tensor (two external operands are unsupported)")
    other = node.input[0] if node.input[1] == previous else node.input[1]
    if other not in constants:
        return _elementwise_error(index, op,
            "requires the non-chain operand to be an immutable initializer constant")
    constant = constants[other]
    if constant.dtype != np.float32 or not np.isfinite(constant).all():
        return _elementwise_error(index, op, "requires a finite float32 constant")
    if _constant_mode(constant, channels, height, width) is None:
        return _elementwise_error(index, op,
            "constant shape %s is outside the broadcast bound (scalar, [1,C,1,1] or [1,C,H,W])"
            % (tuple(int(value) for value in constant.shape),))
    if op == "Sub" and node.input[0] != previous:
        return _elementwise_error(index, op,
            "requires the chain tensor minus the constant (chain tensor first)")
    successor = _elementwise_successor(nodes, position)
    if successor is not None:
        return _elementwise_error(index, op, successor)
    return None


def parse_chain(graph):
    """Describe a supported Conv/pool/elementwise chain, or return None.

    The description is a list of ops in execution order. Each Conv op carries its
    constants and folded activation; each pool op is only its kind; each elementwise
    op carries its constant and the geometry it broadcasts to.

    A graph that is shaped like a walk chain but breaks the *elementwise* envelope
    (a second elementwise stage, a non-constant operand, a constant shape outside the
    broadcast set, or a successor that never reaches a Conv) returns a description
    whose `error` key names the stage and the bound, so `compile_chain_walk` can raise
    the specific message instead of the fallback "unsupported chain walk graph".
    """
    if len(graph.input) != 1 or len(graph.output) != 1:
        return None
    entry = graph.input[0]
    shape = [d.dim_value for d in entry.type.tensor_type.shape.dim]
    if (entry.type.tensor_type.elem_type != 1 or len(shape) != 4 or shape[0] != 1
            or shape[1] != 3 or shape[2] < 2 or shape[3] < 2):
        return None
    constants = {t.name: nh.to_array(t) for t in graph.initializer}
    nodes = list(graph.node)
    if not nodes:
        return None
    ops = []
    channels, height, width = shape[1], shape[2], shape[3]
    position = 0
    previous = None
    while position < len(nodes):
        node = nodes[position]
        last = position == len(nodes) - 1
        if node.op_type == "Conv":
            if any(name not in constants for name in node.input[1:]):
                return None
            weight = constants[node.input[1]]
            bias = constants[node.input[2]]
            if (weight.dtype != np.float32 or bias.dtype != np.float32 or weight.ndim != 4
                    or weight.shape[0] < 1 or weight.shape[0] > 16 or weight.shape[1] != channels
                    or bias.shape != (weight.shape[0],)):
                return None
            kernel = _conv_attributes(node, weight.shape)
            output_channels = int(weight.shape[0])
            activation = None
            measured = node.output[0]
            if not last and nodes[position + 1].op_type in ("Relu", "Clip"):
                follower = nodes[position + 1]
                if follower.attribute or list(follower.input[:1]) != list(node.output):
                    return None
                if follower.op_type == "Relu":
                    if len(follower.input) != 1:
                        return None
                    activation = "Relu"
                else:
                    # Clip[0,6] is accepted only where the fixed native emitter uses it
                    # (the graph's last Conv); the walk reference has no clip path yet.
                    return None
                measured = follower.output[0]
                position += 1
            ops.append(dict(kind="conv", kernel=kernel, weights=weight, bias=bias,
                            activation=activation, input_channels=channels,
                            output_channels=output_channels, height=height, width=width,
                            output_name=node.output[0], tensor=measured))
            channels = output_channels
            previous = measured
        elif node.op_type in POOLS:
            kind = _pool_attributes(node)
            if height % 2 or width % 2 or height < 2 or width < 2:
                return None
            ops.append(dict(kind=kind, height=height, width=width,
                            output_height=height // 2, output_width=width // 2))
            height, width = height // 2, width // 2
            previous = node.output[0]
        elif node.op_type in EW_KINDS:
            index = 1 + sum(op["kind"] == "ew" for op in ops)
            reason = _elementwise_stage_error(node, ops, previous, constants, index,
                                              channels, height, width, nodes, position)
            if reason is not None:
                return dict(error=reason, ops=ops)
            other = node.input[0] if node.input[1] == previous else node.input[1]
            ops.append(dict(kind="ew", op=node.op_type, constant=constants[other],
                            channels=channels, height=height, width=width, index=index,
                            output_name=node.output[0]))
            previous = node.output[0]
        else:
            return None
        position += 1
    if not ops or ops[0]["kind"] != "conv":
        return None
    # The legacy single-Conv image emitter stops at 5..8 pixels a side; a larger (or
    # grayscale) image is staged as a native16 surface instead, so mark the first Conv
    # for the compiler and for `chain_walk_reference`.
    input_channels, input_height, input_width = shape[1], shape[2], shape[3]
    ops[0]["native_input"] = not (5 <= input_height <= 8 and 5 <= input_width <= 8)
    ops[0]["input_pads"] = [ops[0]["kernel"] // 2] * 4
    return dict(ops=ops, input_shape=(1, input_channels, input_height, input_width),
                output_shape=(1, channels, height, width),
                output_channels=channels, channels=channels)


def load_quantizations(meta):
    """Live `Quantization` objects for `chain_walk_reference` from composed meta.

    An elementwise stage has no Conv `Quantization`: its entry is the band dict the
    emitter wrote (`scale`, `constant_scale`, the output band) and is passed through
    unchanged so the reference can rebuild the constant operand.
    """
    loaded = []
    for params in meta["quantizations"]:
        if params is None:
            loaded.append(None)
            continue
        if params.get("kind") == "ew":
            loaded.append(dict(params))
            continue
        values = dict(params)
        for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                    "channel_multipliers"):
            values[key] = np.array(values[key])
        loaded.append(Quantization(**values))
    return loaded


def chain_walk_reference(inputs, quantizations, ops, input_zero_point=0):
    """Integer reference for the walked chain, op by op.

    The first Conv reads the packed UINT8 image through the established single-Conv
    profile (`reference`), or - when the image is larger than that profile accepts -
    through the native16 input profile (`native_input_reference`). Every later Conv
    reads the previous INT8 grid (`native_reference`), and a pool takes the 2x2 block
    maximum (MaxPool) or the rounded mean (AveragePool), which preserves the band.

    An elementwise stage rebuilds the constant operand on the exact scale the emitter
    wrote and applies the verified `add_reference`/`sub_reference`/`max_reference`/
    `mul_reference` for its op; its output is zero point 0 on the emitter's output
    scale, so the next Conv reads the band the container declares.
    """
    grid = None
    zero_point = input_zero_point - 128
    for op, quantization in zip(ops, quantizations):
        if op["kind"] == "conv":
            if grid is None:
                if op.get("native_input"):
                    grid = native_input_reference(inputs, quantization, input_zero_point,
                                                  op.get("input_pads"))
                else:
                    grid = reference(inputs, quantization)
            else:
                grid = native_reference(grid, quantization, zero_point)
            zero_point = quantization.output_zero_point
        elif op["kind"] == "ew":
            codes = _constant_codes(op, quantization["constant_scale"])
            grid = EW_REFERENCE[op["op"]](grid, codes)
            zero_point = quantization["output_zero_point"]
        else:
            height, width, channels = grid.shape
            blocked = grid.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
            if op["kind"] == "MaxPool":
                grid = blocked.max(axis=(1, 3))
            else:
                grid = np.rint(blocked.sum(axis=(1, 3)) / 4)
            grid = np.clip(grid, -128, 127).astype(np.int8)
    return grid


def compile_chain_walk(model, input_scale=1.0, input_zero_point=0, output_range=None,
                       serial=True, ranges=None):
    """Compile a supported Conv/pool chain through the composer.

    `ranges` is an optional `open_rknpu.calibration.measure` report: every Conv whose
    measured tensor has an entry uses that measured band instead of the analytic one,
    which is what a *trained* network needs - its activations are far from the
    analytic bound, so the analytic band can be orders of magnitude too wide.
    """
    onnx.checker.check_model(model)
    spec = parse_chain(model.graph)
    if spec is None:
        raise ValueError("unsupported chain walk graph")
    if spec.get("error"):
        # A chain-shaped graph whose elementwise stage is outside the envelope: the
        # message names the stage and the bound it broke.
        raise ValueError(spec["error"])
    ops = spec["ops"]
    last_conv = max(index for index, op in enumerate(ops) if op["kind"] == "conv")
    # The band of the elementwise stage that follows a Conv. It is computed while the
    # feeding Conv is quantized (the shared operand scale depends on that Conv's band)
    # and consumed when the stage itself is reached.
    pending_elementwise = None

    def measured(op):
        if ranges is None:
            return None
        entry = ranges.get(op.get("tensor"))
        if entry is None:
            raise ValueError("calibration ranges lack tensor " + str(op.get("tensor")))
        return dict(scale=float(entry["scale"]), zero_point=int(entry["zero_point"]))

    # The image-input Conv reuses the established single-Conv emitter - the same
    # layer the chain family copies for its first stage - so its verified fields,
    # weight layout and quantization band all come from that profile. Later Convs
    # read a native16 grid and use the chain family's `native_quantize` convention.
    if ops[0]["kind"] != "conv":
        raise ValueError("walk chain must start with a Conv")
    first = ops[0]
    image_channels = spec["input_shape"][1]
    image_height, image_width = spec["input_shape"][2], spec["input_shape"][3]
    native_input = bool(first.get("native_input"))
    if native_input:
        # The legacy single-Conv emitter only accepts 5..8 pixels a side, so a larger
        # (or grayscale) image uses the verified native16 input profile: the same
        # `native_fields` register builder and weight/bias layout as `native.py`, with
        # the image staged as a native16 surface. Height tiling is not emitted here.
        if image_channels > 16:
            raise ValueError("walk native input supports up to 16 channels")
        lanes = _align(image_channels, 16)
        tiles = lanes // 16
        if image_height * image_width * tiles > 6144:
            raise ValueError("walk native input geometry requires height tiling")
        first_quantization = native_quantize(first["weights"], first["bias"], input_scale,
                                             input_zero_point - 128, measured(first))
        if len(ops) > 1 and ops[1]["kind"] in POOLS:
            first_quantization = native_quantize(
                first["weights"], first["bias"], input_scale, input_zero_point - 128,
                dict(scale=_adjusted_scale(first_quantization.output_scale,
                                           first_quantization.output_zero_point), zero_point=0))
        elif len(ops) > 1 and ops[1]["kind"] == "ew":
            # The elementwise stage's operands are zero-point-zero, so the layer is
            # re-quantized onto the shared operand scale (the same adjustment the pool
            # successor uses, widened by the constant's own bound).
            pending_elementwise = _elementwise_band(ops[1], _adjusted_scale(
                first_quantization.output_scale, first_quantization.output_zero_point))
            first_quantization = native_quantize(
                first["weights"], first["bias"], input_scale, input_zero_point - 128,
                dict(scale=pending_elementwise["scale"], zero_point=0))
        first_quantization.relu = first["activation"] == "Relu"
        first_quantization.input_scale = input_scale
        first_quantization.input_zero_point = input_zero_point
        first_weight_size = _align(first["output_channels"] * lanes * first["kernel"] ** 2)
        first_bias_size = ((first["output_channels"] + 3) // 4) * 32
        data_entries = (image_width * tiles + 1) // 2
        plane_grains = 16 // (1 << (tiles.bit_length() - 1))
        feature_grains = max(min(plane_grains, (64 + data_entries - 1) // data_entries),
                             (first["kernel"] + 1) // 2 + 1)
        scan_flags = ((4 if first["kernel"] == 1 else 8)
                      if image_height * image_width * tiles <= 2048 else 0)
        native_lanes, native_tiles = lanes, tiles
        native_data_entries, native_feature_grains = data_entries, feature_grains
        native_scan_flags = scan_flags
        first_metadata = first_quantization.metadata()
        first_source = None
    else:
        first_nodes = [h.make_node("Conv", ["input", "layer_weights", "layer_bias"], ["layer_conv"],
                                   kernel_shape=[first["kernel"]] * 2,
                                   pads=[first["kernel"] // 2] * 4)]
        first_output = "layer_conv"
        if first["activation"] == "Relu":
            first_nodes.append(h.make_node("Relu", ["layer_conv"], ["layer_relu"]))
            first_output = "layer_relu"
        first_graph = h.make_graph(first_nodes, "layer1",
                                   [h.make_tensor_value_info("input", 1, [1, image_channels,
                                                                          image_height, image_width])],
                                   [h.make_tensor_value_info(first_output, 1,
                                                             [1, first["output_channels"],
                                                              image_height, image_width])],
                                   [nh.from_array(first["weights"], "layer_weights"),
                                    nh.from_array(first["bias"], "layer_bias")])
        first_model = h.make_model(first_graph, opset_imports=list(model.opset_import))
        first_model.ir_version = model.ir_version
        with tempfile.TemporaryDirectory() as folder:
            layer_path = Path(folder) / "layer1.onnx"
            onnx.save(first_model, layer_path)
            selected = output_range if len(ops) == 1 else measured(first)
            first_data, first_meta = compile_model(
                layer_path, output_scale=selected["scale"] if selected else None,
                output_zero_point=selected["zero_point"] if selected else None,
                input_scale=input_scale, input_zero_point=input_zero_point)
            if len(ops) > 1 and ops[1]["kind"] in POOLS:
                # The layer feeds a pool, so re-quantize it onto a zero-point-0 grid.
                first_data, first_meta = compile_model(
                    layer_path,
                    output_scale=_adjusted_scale(first_meta["output_scale"],
                                                 first_meta["output_zero_point"]),
                    output_zero_point=0,
                    input_scale=input_scale, input_zero_point=input_zero_point)
            elif len(ops) > 1 and ops[1]["kind"] == "ew":
                # The layer feeds an elementwise stage: both operands are zero-point
                # zero on one shared scale, chosen from this layer's adjusted band and
                # the constant's own bound (the emitter's rule).
                pending_elementwise = _elementwise_band(ops[1], _adjusted_scale(
                    first_meta["output_scale"], first_meta["output_zero_point"]))
                first_data, first_meta = compile_model(
                    layer_path, output_scale=pending_elementwise["scale"], output_zero_point=0,
                    input_scale=input_scale, input_zero_point=input_zero_point)
        first_source = {word & 0xFFFF: (word >> 16) & 0xFFFFFFFF
                        for word in struct.unpack_from("<126Q", first_data)}
        first_weight_size = _align(first["kernel"] * first["kernel"]
                                   * _align(first["output_channels"], 4) * 4)
        first_bias_size = ((first["output_channels"] + 3) // 4) * 32
        first_quantization = dict(first_meta["quantization"])
        first_metadata = dict(first_meta)

    quantizations = []
    scale, zero_point = first_metadata["output_scale"], first_metadata["output_zero_point"]
    for index, op in enumerate(ops):
        if index == 0:
            quantizations.append(first_quantization)
            continue
        if op["kind"] == "ew":
            # Written by the feeding Conv: the shared/independent operand scales and
            # the band the stage's own output carries.
            quantizations.append(pending_elementwise)
            scale, zero_point = pending_elementwise["output_scale"], \
                pending_elementwise["output_zero_point"]
            continue
        if op["kind"] != "conv":
            quantizations.append(None)
            continue
        selected = (output_range if output_range is not None else measured(op)) \
            if index == last_conv else measured(op)
        if index + 1 < len(ops) and ops[index + 1]["kind"] in POOLS:
            natural = native_quantize(op["weights"], op["bias"], scale, zero_point, None)
            selected = dict(scale=_adjusted_scale(natural.output_scale, natural.output_zero_point),
                            zero_point=0)
        elif index + 1 < len(ops) and ops[index + 1]["kind"] == "ew":
            natural = native_quantize(op["weights"], op["bias"], scale, zero_point, None)
            pending_elementwise = _elementwise_band(ops[index + 1], _adjusted_scale(
                natural.output_scale, natural.output_zero_point))
            selected = dict(scale=pending_elementwise["scale"], zero_point=0)
        quantization = native_quantize(op["weights"], op["bias"], scale, zero_point, selected)
        quantization.relu = op["activation"] == "Relu"
        quantizations.append(quantization)
        scale, zero_point = quantization.output_scale, quantization.output_zero_point
    output_scale = scale
    output_zero_point = zero_point

    stages = []
    # Tensor-table shapes are (batch, height, width, channels); `spec["input_shape"]`
    # is the internal NCHW view. A native16 image is staged as a lane-plane surface;
    # the legacy profile stages a packed U8 image (rows padded to 16).
    if native_input:
        tensors = [TensorSpec("input0", ROLE_INPUT, LAYOUT_NATIVE16,
                              (1, image_height, image_width, image_channels),
                              _surface(image_height, image_width, image_channels), 0)]
    else:
        tensors = [TensorSpec("input0", ROLE_INPUT, LAYOUT_PACKED_U8,
                              (1, spec["input_shape"][2], spec["input_shape"][3], image_channels),
                              _input_bytes(spec["input_shape"][2], spec["input_shape"][3],
                                           image_channels), 0)]
    source = "input0"
    source_zero_point = input_zero_point - 128
    previous_height, previous_width = spec["input_shape"][2], spec["input_shape"][3]
    previous_scale = input_scale
    for index, (op, quantization) in enumerate(zip(ops, quantizations)):
        target = "output" if index == len(ops) - 1 else f"{op['kind'].lower()}{index}"
        if op["kind"] == "conv":
            name = target
            kernel = op["kernel"]
            input_channels = op["input_channels"]
            output_channels = op["output_channels"]
            previous_zero_point = source_zero_point
            if index == 0 and native_input:
                # Native16 image input: the fields and constants come from the
                # `native.py` builders, so the walk inherits that profile's verified
                # registers instead of the legacy single-Conv layout.
                weight_size, bias_size = first_weight_size, first_bias_size

                def fill(payload, offset, kernel=kernel, input_channels=input_channels,
                         output_channels=output_channels, quantization=quantization,
                         weight_size=weight_size):
                    _pack_layer_head(payload, kernel, input_channels, output_channels,
                                     offset, offset + weight_size, quantization)

                def fields(addresses, constant_offsets, kernel=kernel,
                           input_channels=input_channels, output_channels=output_channels,
                           quantization=quantization, weight_size=weight_size, name=name,
                           source=source, lanes=native_lanes, tiles=native_tiles,
                           data_entries=native_data_entries, feature_grains=native_feature_grains,
                           scan_flags=native_scan_flags, op=op,
                           previous_height=previous_height, previous_width=previous_width):
                    return native_fields(previous_width, previous_height, input_channels, lanes,
                                         tiles, op["width"], op["height"], output_channels, kernel,
                                         kernel // 2, kernel // 2, 1, 1, 1, 1,
                                         addresses[source], addresses[name],
                                         constant_offsets[name], constant_offsets[name] + weight_size,
                                         _surface(previous_height, previous_width, output_channels),
                                         data_entries, feature_grains, scan_flags, input_zero_point,
                                         quantization, op["activation"] == "Relu", False,
                                         previous_height, input_scale)
            elif index == 0:
                # Copy the established single-Conv layer: its whole register set and
                # its packed weight/bias blocks, with only the four addresses moved.
                weight_size, bias_size = first_weight_size, first_bias_size

                def fill(payload, offset, data=first_data, source_regs=first_source,
                         weight_size=weight_size, bias_size=bias_size):
                    payload[offset:offset + weight_size] = data[
                        source_regs[0x1110]:source_regs[0x1110] + weight_size]
                    payload[offset + weight_size:offset + weight_size + bias_size] = data[
                        source_regs[0x5020]:source_regs[0x5020] + bias_size]

                def fields(addresses, constant_offsets, source_regs=first_source,
                           weight_size=weight_size, name=name, source=source):
                    values = dict(source_regs)
                    values.update({0x1110: constant_offsets[name],
                                   0x5020: constant_offsets[name] + weight_size,
                                   0x4020: addresses[name], 0x1070: addresses[source]})
                    return values
            else:
                weight_size = _align(output_channels * 16 * kernel * kernel)
                bias_size = ((output_channels + 3) // 4) * 32

                def fill(payload, offset, kernel=kernel, input_channels=input_channels,
                         output_channels=output_channels, quantization=quantization,
                         weight_size=weight_size):
                    _pack_layer_head(payload, kernel, input_channels, output_channels,
                                     offset, offset + weight_size, quantization)

                def fields(addresses, constant_offsets, kernel=kernel, input_channels=input_channels,
                           output_channels=output_channels, quantization=quantization,
                           weight_size=weight_size, name=name, source=source,
                           previous_zero_point=previous_zero_point, op=op,
                           previous_height=previous_height, previous_width=previous_width,
                           previous_scale=previous_scale):
                    # A Conv that reads a native16 grid uses the geometry-aware builder:
                    # the pooled grid is not 8x8, so the 8x8-derived defaults are wrong
                    # (registers 0x107c/0x1080/0x118c/0x3014/0x4030/0x4034/0x405c/0x500c/
                    # 0x5010 all describe the surface this task reads and writes).
                    lanes = _align(input_channels, 16)
                    tiles = lanes // 16
                    data_entries = (previous_width * tiles + 1) // 2
                    plane_grains = 16 // (1 << (tiles.bit_length() - 1))
                    feature_grains = max(min(plane_grains,
                                             (64 + data_entries - 1) // data_entries),
                                         (kernel + 1) // 2 + 1)
                    scan_flags = ((4 if kernel == 1 else 8)
                                  if previous_height * previous_width * tiles <= 2048 else 0)
                    values = native_fields(previous_width, previous_height, input_channels,
                                           lanes, tiles, op["width"], op["height"],
                                           output_channels, kernel, kernel // 2, kernel // 2,
                                           1, 1, 1, 1, addresses[source], addresses[name],
                                           constant_offsets[name],
                                           constant_offsets[name] + weight_size,
                                           _surface(op["height"], op["width"], output_channels),
                                           data_entries, feature_grains, scan_flags,
                                           previous_zero_point + 128, quantization,
                                           op["activation"] == "Relu", False,
                                           previous_height, previous_scale)
                    return values

            stages.append(Stage(name=name, family="native-conv", reads=(source,), writes=(target,),
                                fields=fields,
                                constants=(ConstantSpec(name, weight_size + bias_size, fill),),
                                bindings=(Binding(0x1070, source, "read"),
                                          Binding(0x4020, target, "write"))))
            tensors.append(TensorSpec(target, ROLE_INTERNAL if index < len(ops) - 1 else ROLE_OUTPUT,
                                      LAYOUT_NATIVE16,
                                      (1, op["height"], op["width"], output_channels),
                                      _surface(op["height"], op["width"], output_channels), 0))
            # The first layer's quantization is the compiled layer's metadata dict.
            source_zero_point = (quantization["output_zero_point"] if isinstance(quantization, dict)
                                 else quantization.output_zero_point)
            previous_scale = (quantization["output_scale"] if isinstance(quantization, dict)
                              else quantization.output_scale)
        elif op["kind"] == "ew":
            # The standalone elementwise emitter's 78-word DPU program, reused through
            # `_join_fields`: the primary operand is the feeding Conv's native16 grid
            # (0x5018) and the secondary is the payload constant grid (0x5038), laid out
            # as a native16 surface with the first C lanes of each pixel meaningful -
            # exactly the constant-Mul emitter's per-pixel operand layout.
            name = target
            height, width, channels = op["height"], op["width"], op["channels"]
            surface = _surface(height, width, channels)
            constant_key = f"{name}_constant"
            codes = _constant_codes(op, quantization["constant_scale"])

            def fill(payload, offset, codes=codes, channels=channels, height=height,
                     width=width):
                packed = np.zeros((height, width, 16), np.int8)
                packed[:, :, :channels] = codes
                payload[offset:offset + packed.size] = packed.tobytes()

            def fields(addresses, constant_offsets, kind=op["op"], height=height, width=width,
                       channels=channels, surface=surface, name=name, source=source,
                       constant_key=constant_key, quantization=quantization):
                values, output_scale, output_zero_point = _join_fields(
                    kind, height, width, channels, addresses[source],
                    constant_offsets[constant_key], addresses[name], surface,
                    [quantization["scale"], quantization["constant_scale"]], (0, 0), None)
                if (output_scale, output_zero_point) != (quantization["output_scale"],
                                                         quantization["output_zero_point"]):
                    raise ValueError("walk elementwise stage band disagrees with the emitter")
                return values

            stages.append(Stage(name=name, family="elementwise", reads=(source,),
                                writes=(target,), fields=fields,
                                constants=(ConstantSpec(constant_key, surface, fill),),
                                bindings=(Binding(0x5018, source, "read"),
                                          Binding(0x4020, target, "write"))))
            tensors.append(TensorSpec(target, ROLE_INTERNAL if index < len(ops) - 1 else ROLE_OUTPUT,
                                      LAYOUT_NATIVE16, (1, height, width, channels), surface, 0))
            source_zero_point = 0
            previous_height, previous_width = height, width
            previous_scale = quantization["output_scale"]
        else:
            name = target
            height, width = op["height"], op["width"]
            output_height, output_width = op["output_height"], op["output_width"]

            def fields(addresses, constant_offsets, kind=op["kind"], height=height, width=width,
                       output_height=output_height, output_width=output_width, name=name,
                       source=source):
                return pool_registers(kind, height, width, output_height, output_width,
                                      addresses[source], addresses[name])

            stages.append(Stage(name=name, family="pool", reads=(source,), writes=(target,),
                                fields=fields,
                                bindings=(Binding(0x701C, source, "read"),
                                          Binding(0x6070, target, "write"))))
            channels = spec["channels"]
            tensors.append(TensorSpec(target, ROLE_INTERNAL if index < len(ops) - 1 else ROLE_OUTPUT,
                                      LAYOUT_NATIVE16,
                                      (1, output_height, output_width, channels),
                                      _surface(output_height, output_width, channels), 0))
            previous_height, previous_width = output_height, output_width
        source = target
    # A chain mixes task families (CNA Conv and DPU pool), and a slot written by two
    # different families returned stale data on the board (see the investigation log),
    # so every internal gets a fresh slot - the same policy as `pooled_branches`.
    binary, composed = compose(stages, tensors, input_scale=input_scale,
                               input_zero_point=input_zero_point,
                               output_scale=output_scale, output_zero_point=output_zero_point,
                               serial=serial, reuse=False)
    meta = dict(profile="chain-walk", walk_ops=[op["kind"] for op in ops],
                walk_activations=[op.get("activation") for op in ops if op["kind"] == "conv"],
                walk_elementwise=[op["op"] for op in ops if op["kind"] == "ew"],
                input_shape=list(spec["input_shape"]), output_shape=list(spec["output_shape"]),
                output_scale=output_scale, output_zero_point=output_zero_point,
                quantizations=[(q if isinstance(q, dict) else q.metadata()) if q is not None else None
                               for q in quantizations])
    meta.update({key: value for key, value in composed.items() if key != "profile"})
    return binary, meta



# --- fan-in joins ----------------------------------------------------------
JOIN_KINDS = ("Add", "Mul", "Sub", "Max")


def _value_shape(graph):
    """The graph input's `(height, width)`, or None when it is not an RGB image."""
    entry = graph.input[0]
    shape = [d.dim_value for d in entry.type.tensor_type.shape.dim]
    if (entry.type.tensor_type.elem_type != 1 or len(shape) != 4 or shape[0] != 1
            or shape[1] != 3 or shape[2] < 2 or shape[3] < 2):
        return None
    return shape[2], shape[3]


def _as_conv(node, constants):
    """Validate one dense Conv node; return its description or None."""
    if node.domain not in ("", "ai.onnx") or len(node.input) != 3:
        return None
    if any(name not in constants for name in node.input[1:]):
        return None
    weight = constants[node.input[1]]
    bias = constants[node.input[2]]
    if (weight.dtype != np.float32 or bias.dtype != np.float32 or weight.ndim != 4
            or not 1 <= weight.shape[0] <= 16 or bias.shape != (weight.shape[0],)):
        return None
    try:
        kernel = _conv_attributes(node, weight.shape)
    except ValueError:
        return None
    return dict(weights=weight, bias=bias, kernel=kernel, channels=int(weight.shape[0]),
                input_channels=int(weight.shape[1]))


def _peel_operand(nodes, producer, name, join_index, constants):
    """Walk a join operand back through Relu/Pool to its head Conv.

    Returns `(conv, node, pooled)` where `pooled` is the pool kind applied to the head
    output, or None when the head is the operand itself.
    """
    index = producer.get(name)
    if index is None or index >= join_index:
        return None
    node = nodes[index]
    pooled = None
    for _ in range(2):
        if node.op_type == "Relu":
            if node.attribute or len(node.input) != 1:
                return None
            index = producer.get(node.input[0])
            if index is None or index >= join_index:
                return None
            node = nodes[index]
        elif node.op_type in POOLS:
            try:
                pooled = _pool_attributes(node)
            except ValueError:
                return None
            index = producer.get(node.input[0])
            if index is None or index >= join_index:
                return None
            node = nodes[index]
        else:
            break
    conv = _as_conv(node, constants)
    if conv is None:
        return None
    return conv, node, pooled


def parse_join_walk(graph):
    """Describe a fan-in join graph the walk can lower, or return None.

    The class is the diamond the DAG profiles cover - one `Conv[,Relu]` stem, two head
    `Conv`s, one elementwise join, an optional `[Conv,Relu]* Conv` tail - plus an
    optional 2x2 stride-2 pool after each head. The join is lowered from the graph, not
    from a profile pattern.
    """
    nodes = list(graph.node)
    if not nodes or len(graph.input) != 1 or len(graph.output) != 1:
        return None
    sizes = _value_shape(graph)
    if sizes is None:
        return None
    constants = {t.name: nh.to_array(t) for t in graph.initializer}
    joins = [index for index, node in enumerate(nodes) if node.op_type in JOIN_KINDS]
    if len(joins) != 1:
        return None
    join_index = joins[0]
    join = nodes[join_index]
    if join.attribute or len(join.input) != 2 or len(join.output) != 1:
        return None
    producer = {}
    for index, node in enumerate(nodes):
        for name in node.output:
            producer[name] = index
    operands = [_peel_operand(nodes, producer, name, join_index, constants) for name in join.input]
    if any(operand is None for operand in operands):
        return None
    if operands[0][1].input[0] != operands[1][1].input[0]:
        return None
    traced = {id(operands[0][1]), id(operands[1][1])}
    for name in join.input:
        index = producer[name]
        while nodes[index].op_type in ("Relu",) + tuple(POOLS):
            traced.add(id(nodes[index]))
            name = nodes[index].input[0]
            index = producer[name]
    stem_index = producer[operands[0][1].input[0]]
    stem_node = nodes[stem_index]
    stem_activation = None
    if stem_node.op_type == "Relu":
        if stem_node.attribute or len(stem_node.input) != 1:
            return None
        stem_activation = "Relu"
        traced.add(id(stem_node))
        stem_index = producer[stem_node.input[0]]
        stem_node = nodes[stem_index]
    stem = _as_conv(stem_node, constants)
    if stem is None or stem["kernel"] != 1 or stem_node.input[0] != graph.input[0].name:
        return None
    traced.add(id(stem_node))
    if len(traced) != join_index:
        return None
    hidden = stem["channels"]
    height, width = sizes
    heads = []
    for conv, node, pooled in operands:
        if conv["input_channels"] != hidden:
            return None
        head_height, head_width = height, width
        if pooled is not None:
            if head_height % 2 or head_width % 2:
                return None
            head_height, head_width = head_height // 2, head_width // 2
        heads.append(dict(conv=conv, node=node, pooled=pooled, height=head_height,
                          width=head_width, channels=conv["channels"]))
    shape = (heads[0]["height"], heads[0]["width"], heads[0]["channels"])
    if shape != (heads[1]["height"], heads[1]["width"], heads[1]["channels"]):
        return None
    tails = []
    channels = shape[2]
    index = join_index + 1
    while index < len(nodes):
        node = nodes[index]
        if node.op_type == "Conv":
            conv = _as_conv(node, constants)
            if conv is None or conv["input_channels"] != channels:
                return None
            channels = conv["channels"]
            tails.append(dict(conv=conv, node=node, activation=None))
        elif node.op_type == "Relu":
            if not tails or tails[-1]["activation"] or node.attribute or len(node.input) != 1:
                return None
            tails[-1]["activation"] = "Relu"
        else:
            return None
        index += 1
    if tails and tails[-1]["activation"] == "Relu":
        return None
    return dict(stem=stem, stem_activation=stem_activation, heads=heads, join_kind=join.op_type,
                join_shape=shape, tails=tails, input_shape=sizes, hidden=hidden)


def join_walk_reference(inputs, quantizations, plan, join_zero_point=0):
    """Integer reference for a walked fan-in graph, in execution order."""
    heads, tails = plan["heads"], plan["tails"]
    grid = reference(inputs, quantizations["stem"])
    zero_point = quantizations["stem"].output_zero_point
    for index, head in enumerate(heads):
        head_grid = native_reference(grid, quantizations["head%d" % index], zero_point)
        if head["pooled"] is not None:
            height, width, channels = head_grid.shape
            blocked = head_grid.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
            if head["pooled"] == "MaxPool":
                head_grid = blocked.max(axis=(1, 3))
            else:
                head_grid = np.rint(blocked.sum(axis=(1, 3)) / 4)
            head_grid = np.clip(head_grid, -128, 127).astype(np.int8)
        plan.setdefault("_grids", {})[index] = head_grid
    from .graph import join_reference
    joined = join_reference(plan["join_kind"], plan["_grids"][0], plan["_grids"][1])
    previous = join_zero_point
    for index, tail in enumerate(tails):
        joined = native_reference(joined, quantizations["tail%d" % index], previous)
        previous = quantizations["tail%d" % index].output_zero_point
    plan.pop("_grids", None)
    return joined


def compile_join_walk(model, input_scale=1.0, input_zero_point=0, output_range=None,
                      serial=True, ranges=None):
    """Compile a supported fan-in join graph through the composer.

    The band rules are the diamond emitter's: each head is quantized on its natural
    band, a Mul join folds the two head scales, an Add/Sub/Max join re-quantizes both
    heads onto one shared scale (zero point 0), and the tail chain continues from the
    join band. With `ranges` (an `open_rknpu.calibration.measure` report) the stem,
    each head Conv and each tail Conv take their measured band as the natural band
    through the shared `measured_range` helper; a join operand that a shared-scale
    Add/Sub/Max captures is still re-centered onto zero point 0.
    """
    from .elementwise import mul_output_conversion
    from .graph import _join_fields, join_reference  # noqa: F401  (reference helper)
    onnx.checker.check_model(model)
    plan = parse_join_walk(model.graph)
    if plan is None:
        raise ValueError("unsupported join walk graph")
    if ranges is not None and output_range is not None:
        raise ValueError("calibration and output quantization overrides cannot be combined")
    from .calibration import measured_range
    fused = _fused_tensor_names(model.graph)
    stem, heads, tails = plan["stem"], plan["heads"], plan["tails"]
    kind = plan["join_kind"]
    if output_range is not None and kind != "Mul" and not tails:
        # Same boundary as the diamond profile: an Add/Sub/Max join derives its output
        # band from the shared operand scale, so only a Mul join or a Conv tail can
        # carry a requested output range.
        raise ValueError("the join walk output override requires a Mul join or a Conv tail")
    height, width = plan["input_shape"]
    hidden = plan["hidden"]
    head_height, head_width, head_channels = plan["join_shape"]

    def measured(node):
        return measured_range(ranges, fused[node.output[0]]) if ranges is not None else None

    # 1. The stem is the established single-Conv layer, copied like the chain family's
    #    first stage so its verified fields and constants come from that emitter.
    stem_nodes = [h.make_node("Conv", ["input", "stem_weights", "stem_bias"], ["stem_conv"],
                              kernel_shape=[1, 1])]
    stem_output = "stem_conv"
    if plan["stem_activation"] == "Relu":
        stem_nodes.append(h.make_node("Relu", ["stem_conv"], ["stem_relu"]))
        stem_output = "stem_relu"
    stem_graph = h.make_graph(stem_nodes, "stem",
                              [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
                              [h.make_tensor_value_info(stem_output, 1, [1, hidden, height, width])],
                              [nh.from_array(stem["weights"], "stem_weights"),
                               nh.from_array(stem["bias"], "stem_bias")])
    stem_model = h.make_model(stem_graph, opset_imports=list(model.opset_import))
    stem_model.ir_version = model.ir_version
    with tempfile.TemporaryDirectory() as folder:
        stem_path = Path(folder) / "stem.onnx"
        onnx.save(stem_model, stem_path)
        # The reconstructed stem renames its tensors, so the report's entry for the
        # original stem tensor is re-keyed onto this graph's output name.
        stem_node = model.graph.node[0]
        stem_selected = measured_range(ranges, fused[stem_node.output[0]]) if ranges is not None else None
        stem_ranges = None if stem_selected is None else {stem_output: stem_selected}
        stem_data, stem_meta = compile_model(stem_path, input_scale=input_scale,
                                             input_zero_point=input_zero_point,
                                             calibration_ranges=stem_ranges)
    stem_source = {word & 0xFFFF: (word >> 16) & 0xFFFFFFFF
                   for word in struct.unpack_from("<126Q", stem_data)}
    stem_weight_size = _align(1 * 1 * _align(hidden, 4) * 4)
    stem_bias_size = ((hidden + 3) // 4) * 32
    stem_scale, stem_zero_point = stem_meta["output_scale"], stem_meta["output_zero_point"]

    # 2. Bands.
    natural = [native_quantize(head["conv"]["weights"], head["conv"]["bias"],
                               stem_scale, stem_zero_point, measured(head["node"]))
               for head in heads]
    adjusted = [_adjusted_scale(q.output_scale, q.output_zero_point) for q in natural]
    if kind == "Mul":
        scales = [float(np.float32(value)) for value in adjusted]
    else:
        scales = [float(np.float32(max(adjusted)))] * 2
    q_heads = [native_quantize(head["conv"]["weights"], head["conv"]["bias"],
                               stem_scale, stem_zero_point,
                               dict(scale=scale, zero_point=0))
               for head, scale in zip(heads, scales)]
    if kind == "Mul":
        join_scale, join_zero = mul_output_conversion(scales[0] * scales[1], output_range)[:2]
    else:
        join_scale, join_zero = float(np.float32(2 * scales[0])), 0
    q_tails = []
    previous = (join_scale, join_zero)
    for index, tail in enumerate(tails):
        if ranges is not None:
            selected = measured(tail["node"])
        else:
            selected = output_range if index == len(tails) - 1 else None
        quantization = native_quantize(tail["conv"]["weights"], tail["conv"]["bias"],
                                       previous[0], previous[1], selected)
        quantization.relu = tail["activation"] == "Relu"
        q_tails.append(quantization)
        previous = (quantization.output_scale, quantization.output_zero_point)
    if tails:
        output_scale, output_zero_point = q_tails[-1].output_scale, q_tails[-1].output_zero_point
    else:
        output_scale, output_zero_point = join_scale, join_zero

    def stem_fill(payload, offset):
        payload[offset:offset + stem_weight_size] = stem_data[
            stem_source[0x1110]:stem_source[0x1110] + stem_weight_size]
        payload[offset + stem_weight_size:offset + stem_weight_size + stem_bias_size] = stem_data[
            stem_source[0x5020]:stem_source[0x5020] + stem_bias_size]

    def stem_fields(addresses, constant_offsets):
        values = dict(stem_source)
        values.update({0x1110: constant_offsets["stem"],
                       0x5020: constant_offsets["stem"] + stem_weight_size,
                       0x4020: addresses["stem"], 0x1070: addresses["input0"]})
        return values

    stages = [Stage(name="stem", family="native-conv", reads=("input0",), writes=("stem",),
                    fields=stem_fields,
                    constants=(ConstantSpec("stem", stem_weight_size + stem_bias_size, stem_fill),),
                    bindings=(Binding(0x1070, "input0", "read"),
                              Binding(0x4020, "stem", "write")))]
    tensors = [TensorSpec("input0", ROLE_INPUT, LAYOUT_PACKED_U8, (1, height, width, 3),
                          _input_bytes(height, width, 3), 0),
               TensorSpec("stem", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, height, width, hidden),
                          _surface(height, width, hidden), 0)]

    def conv_stage(name, source, conv, quantization, activation, previous_zero_point,
                   out_height, out_width, out_channels):
        kernel = conv["kernel"]
        input_channels = conv["input_channels"]
        weight_size = _align(out_channels * 16 * kernel * kernel)
        bias_size = ((out_channels + 3) // 4) * 32
        lanes = _align(input_channels, 16)
        tiles = lanes // 16
        in_height, in_width = (height, width) if source == "stem" else (out_height, out_width)

        def fill(payload, offset, kernel=kernel, input_channels=input_channels,
                 out_channels=out_channels, quantization=quantization, weight_size=weight_size):
            _pack_layer_head(payload, kernel, input_channels, out_channels,
                             offset, offset + weight_size, quantization)

        def fields(addresses, constant_offsets, kernel=kernel, input_channels=input_channels,
                   out_channels=out_channels, quantization=quantization, weight_size=weight_size,
                   name=name, source=source, activation=activation,
                   previous_zero_point=previous_zero_point, in_height=in_height, in_width=in_width,
                   lanes=lanes, tiles=tiles):
            data_entries = (in_width * tiles + 1) // 2
            plane_grains = 16 // (1 << (tiles.bit_length() - 1))
            feature_grains = max(min(plane_grains, (64 + data_entries - 1) // data_entries),
                                 (kernel + 1) // 2 + 1)
            scan_flags = ((4 if kernel == 1 else 8)
                          if in_height * in_width * tiles <= 2048 else 0)
            return native_fields(in_width, in_height, input_channels, lanes, tiles,
                                 out_width, out_height, out_channels, kernel, kernel // 2,
                                 kernel // 2, 1, 1, 1, 1, addresses[source], addresses[name],
                                 constant_offsets[name], constant_offsets[name] + weight_size,
                                 _surface(out_height, out_width, out_channels), data_entries,
                                 feature_grains, scan_flags, previous_zero_point + 128,
                                 quantization, activation == "Relu", False, in_height, 1.0)

        return Stage(name=name, family="native-conv", reads=(source,), writes=(name,),
                     fields=fields,
                     constants=(ConstantSpec(name, weight_size + bias_size, fill),),
                     bindings=(Binding(0x1070, source, "read"),
                               Binding(0x4020, name, "write")))

    for index, (head, quantization) in enumerate(zip(heads, q_heads)):
        name = "head_%s" % ("ab"[index])
        stages.append(conv_stage(name, "stem", head["conv"], quantization, None,
                                 stem_zero_point, height, width, head["channels"]))
        # The head tensors keep the diamond profile's internal index (head_a 0,
        # head_b 1), which is part of the container bytes.
        tensors.append(TensorSpec(name, ROLE_INTERNAL, LAYOUT_NATIVE16,
                                  (1, height, width, head["channels"]),
                                  _surface(height, width, head["channels"]), index))
        if head["pooled"] is not None:
            pooled_name = "pool_%s" % ("ab"[index])

            def pool_fields(addresses, constant_offsets, kind=head["pooled"], name=pooled_name,
                            source=name, in_height=height, in_width=width,
                            out_height=head["height"], out_width=head["width"]):
                return pool_registers(kind, in_height, in_width, out_height, out_width,
                                      addresses[source], addresses[name])

            stages.append(Stage(name=pooled_name, family="pool", reads=(name,),
                                writes=(pooled_name,), fields=pool_fields,
                                bindings=(Binding(0x701C, name, "read"),
                                          Binding(0x6070, pooled_name, "write"))))
            tensors.append(TensorSpec(pooled_name, ROLE_INTERNAL, LAYOUT_NATIVE16,
                                      (1, head["height"], head["width"], head["channels"]),
                                      _surface(head["height"], head["width"], head["channels"]), 0))
    operand_names = [("pool_%s" % "ab"[index]) if head["pooled"] is not None else ("head_%s" % "ab"[index])
                     for index, head in enumerate(heads)]
    join_target = "join" if tails else "output"
    join_override = output_range if not tails else None

    def join_fields(addresses, constant_offsets, first=operand_names[0], second=operand_names[1],
                    target=join_target, kind=kind, scales=scales, override=join_override):
        values, _, _ = _join_fields(kind, head_height, head_width, head_channels,
                                    addresses[first], addresses[second], addresses[target],
                                    _surface(head_height, head_width, head_channels),
                                    scales, (0, 0), override)
        return values

    stages.append(Stage(name="join", family="elementwise",
                        reads=(operand_names[0], operand_names[1]), writes=(join_target,),
                        fields=join_fields,
                        bindings=(Binding(0x5018, operand_names[0], "read"),
                                  Binding(0x5038, operand_names[1], "read"),
                                  Binding(0x4020, join_target, "write"))))
    if tails:
        tensors.append(TensorSpec("join", ROLE_INTERNAL, LAYOUT_NATIVE16,
                                  (1, head_height, head_width, head_channels),
                                  _surface(head_height, head_width, head_channels), 0))
    source = "join" if tails else operand_names[0]
    source_zero_point = join_zero
    in_height, in_width = head_height, head_width
    for index, (tail, quantization) in enumerate(zip(tails, q_tails)):
        last = index == len(tails) - 1
        name = "output" if last else "tail%d" % index
        stages.append(conv_stage(name, source, tail["conv"], quantization, tail["activation"],
                                 source_zero_point, in_height, in_width, tail["conv"]["channels"]))
        tensors.append(TensorSpec(name, ROLE_INTERNAL if not last else ROLE_OUTPUT,
                                  LAYOUT_NATIVE16,
                                  (1, in_height, in_width, tail["conv"]["channels"]),
                                  _surface(in_height, in_width, tail["conv"]["channels"]), 0))
        source = name
        source_zero_point = quantization.output_zero_point
    if not tails:
        # With a tail the last tail stage already writes the external output tensor.
        tensors.append(TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16,
                                  (1, head_height, head_width, head_channels),
                                  _surface(head_height, head_width, head_channels), 0))
    binary, composed = compose(stages, tensors, input_scale=input_scale,
                               input_zero_point=input_zero_point, output_scale=output_scale,
                               output_zero_point=output_zero_point, serial=serial, reuse=True)
    # The walk is the dispatch for this class, so its meta also carries the key names
    # the diamond profile's tests and callers read.
    meta = dict(profile="walk-join", join=kind, join_kinds=[kind], hidden_channels=hidden,
                stem_kernel=1, head_count=len(heads),
                head_pools=[head["pooled"] for head in heads],
                join_shape=list(plan["join_shape"]),
                head_kernels=[head["conv"]["kernel"] for head in heads],
                tail_kernels=[tail["conv"]["kernel"] for tail in tails],
                operand_zero_points=[0, 0],
                join_scale=float(join_scale), join_zero_point=int(join_zero),
                input_scale=input_scale, input_zero_point=input_zero_point,
                output_scale=output_scale, output_zero_point=output_zero_point,
                head_scales=[q.output_scale for q in q_heads],
                head_zero_points=[q.output_zero_point for q in q_heads],
                shape_nhwc=[1, height, width, 3],
                output_shape_nhwc=[1, head_height, head_width, head_channels],
                output_tensors=["output"],
                stem_quantization=stem_meta["quantization"],
                head_quantization=[q.metadata() for q in q_heads],
                tail_quantization=[q.metadata() for q in q_tails],
                tail_weights=[tail["conv"]["weights"].tolist() for tail in tails],
                tail_bias=[tail["conv"]["bias"].tolist() for tail in tails],
                weights=[head["conv"]["weights"].tolist() for head in heads],
                bias=[head["conv"]["bias"].tolist() for head in heads],
                stem_weights=stem["weights"].tolist(), stem_bias=stem["bias"].tolist(),
                quantizations=dict(
                    stem=stem_meta["quantization"],
                    **{"head%d" % index: q.metadata() for index, q in enumerate(q_heads)},
                    join=None,
                    **{"tail%d" % index: q.metadata() for index, q in enumerate(q_tails)}),
                lifetime_bytes=composed["live_bytes"])
    meta.update({key: value for key, value in composed.items() if key != "profile"})
    return binary, meta
