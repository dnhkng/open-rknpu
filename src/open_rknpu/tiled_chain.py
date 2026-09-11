"""SPDX-License-Identifier: MIT

Height-strip tiled dense Conv chain for RV1103.

The untiled N-layer chain (`chain_n.py`) emits one 8x8 program per layer. This module
emits the same graph as `tiles` horizontal strips: for every (layer, strip) pair a
native16 program whose geometry is the strip (`tih`/`toh` rows), reading the strip's
rows of the previous surface and writing its rows of the next one. Intermediates are
double-buffered, so the arena holds two surfaces instead of one per layer - the memory
half of tiling, which is the half this IP can actually use (a job may not mix engines,
so there is nothing to overlap with; see docs/plans/pipelining-plan.md S3).

Two layouts, one per kernel:

* **1x1 (`K = 1`, the S3 layout).** A strip needs no halo, so each strip owns an
  isolated double-buffered strip surface: `2 * tiles` surfaces of `strip * 8 * 16 B`.
* **3x3 (`K = 3`, S7).** A strip's output rows need one input row from each neighbour,
  and those halo rows belong to the neighbouring strips, so all strips of a layer share
  one full double-buffered 8x8 native16 surface and each writes only its own rows. The
  halo is read straight out of that surface; `tpt`/`pl` supply the zero-point rows at
  the image edge, exactly as `native.compile_native_input` does for its 6144-atom height
  tiles. Surface bytes are the same as the 1x1 layout at 8x8 (two full surfaces either
  way); the value is the bounded per-task input fetch and a general emitter.

Bounds: `[Conv, Relu]*(N-1) + [Conv]` at fixed 8x8 C3, `chain_n`'s padded 1x1/3x3
kernels, hidden channels 3..16, tiles dividing 8, at most the loader's 64 tasks.
Quantization and activation come from the untiled emitter, so the two containers share
one integer reference and one expected-bytes file: a strip task applies the same
activation its `chain_n` layer does (measured 2026-09-11: the legacy first layer applies
the graph's leading Relu, the native hidden layers currently do not - the activation is
taken from each layer's quantization metadata, so a future fix propagates here).
"""
import struct

import numpy as np

from .chain_n import _validate, compile_chain_n, _as_quantization
from .native import native_fields
from .register_profile import REGISTERS
from .sequence import encode_sequence


def _align(n, a=64):
    return (n + a - 1) // a * a


def compile_tiled_chain(model, tiles=2, serial=True):
    """Compile a 1x1/3x3 dense Conv chain as `tiles` height strips."""
    if tiles not in (1, 2, 4, 8):
        raise ValueError("tiles must divide the 8-row chain height (1, 2, 4 or 8)")
    graph, nodes, convs, layers, constants = _validate(model)
    if any(kernel not in (1, 3) for _, _, kernel in layers):
        raise ValueError("tiled chain supports the chain family's 1x1 and 3x3 kernels")
    if [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim] != [1, 3, 8, 8]:
        raise ValueError("tiled chain requires the 8x8 C3 input")
    if any(w.shape[0] > 16 or w.shape[1] > 16 for w, _, _ in layers):
        raise ValueError("tiled chain supports up to 16 channels per layer")
    _, meta = compile_chain_n(model)          # same quantization chain and reference
    quantizations = [_as_quantization(entry) for entry in meta["quantizations"]]
    count = len(layers)
    if count * tiles > 64:
        raise ValueError("tiled chain would need %d tasks; the loader table holds 64"
                         % (count * tiles))
    strip = 8 // tiles
    plane = _align(8 * 8 * 16)                # a full native16 8x8 surface
    shared = any(kernel > 1 for _, _, kernel in layers)
    if shared:
        surface = output_surface = plane
        input_storage = plane
    else:
        surface = _align(strip * 8 * 16)      # one 16-lane plane per strip
        output_surface = plane
        input_storage = plane
    program_bytes = _align(130 * 8)

    # Programs in layer-major order, one per (layer, strip).
    programs = [index * program_bytes for index in range(count * tiles)]
    cursor = _align(programs[-1] + program_bytes)
    weights_offsets = []
    for w, _, kernel in layers:
        oc, ic = w.shape[0], w.shape[1]
        weight_size = _align(oc * 16 * kernel * kernel)
        bias_size = ((oc + 3) // 4) * 32
        weights_offsets.append((cursor, cursor + weight_size))
        cursor = _align(cursor + weight_size + bias_size)
    payload_size = _align(cursor)
    input_offset = _align(payload_size, 4096)
    buffer_base = input_offset + input_storage * 2
    if shared:
        # One full surface per ping-pong phase, shared by every strip of the layer.
        buffers = [[buffer_base + phase * plane for phase in range(2)]]
    else:
        buffers = [[buffer_base + (s * 2 + p) * surface for p in range(2)]
                   for s in range(tiles)]
    output_offset = _align(buffer_base + (2 * plane if shared else tiles * 2 * surface))
    arena = _align(output_offset + output_surface, 4096)

    data = bytearray(payload_size)
    for layer in range(count):
        w, b, kernel = layers[layer]
        oc, ic = w.shape[0], w.shape[1]
        halo = kernel // 2
        q = quantizations[layer]
        weight_offset, bias_offset = weights_offsets[layer]
        for strip_index in range(tiles):
            if shared:
                # The strip's window: `tih` existing rows from `top`, with `tpt` rows
                # of zero-point padding above the image. The bottom side pads itself
                # because the window runs past the 8-row surface (`ih`).
                out_row = strip_index * strip
                top = max(0, out_row - halo)
                bottom = min(8, out_row + strip + halo)
                tih = bottom - top
                tpt = max(0, halo - out_row)
                source = input_offset if layer == 0 else buffers[0][(layer - 1) % 2]
                target = output_offset if layer == count - 1 else buffers[0][layer % 2]
                inp = source + top * 8 * 16
                out = target + out_row * 8 * 16
                ih = 8
            else:
                tih, tpt = strip, 0
                inp = input_offset + strip_index * strip * 8 * 16 if layer == 0 \
                    else buffers[strip_index][(layer - 1) % 2]
                out = output_offset + strip_index * strip * 8 * 16 if layer == count - 1 \
                    else buffers[strip_index][layer % 2]
                ih = 8 if layer == 0 else strip
            fields = native_fields(
                8, tih, ic, 16, 1, 8, strip, oc, kernel, halo, tpt, 1, 1, 1, 1,
                inp, out, weight_offset, bias_offset,
                output_surface if layer == count - 1 and not shared else surface,
                4, 16, 4 if kernel == 1 else 8,
                # Native surfaces store the 128-shifted INT8 value, so the engine's
                # input zero point is 0 in this domain (register 0x1184 = 0xff80) - the
                # value the untiled chain's native layers use. The first layer's input
                # is packed the same way.
                0,
                q, bool(getattr(q, "relu", False)), False, ih, 1.0)
            position = layer * tiles + strip_index
            link = programs[position + 1] if not serial and position + 1 < count * tiles else 0
            base = programs[position]
            for i, (reg, default, tag) in enumerate(REGISTERS):
                struct.pack_into("<Q", data, base + i * 8,
                                 tag << 48 | fields.get(reg, default) << 16 | reg)
            for i, (reg, value, tag) in enumerate(((0x10, link, 0x101),
                                                   (0x14, 0x40 if link else 0x28, 0x101),
                                                   (0, 0, 0x41), (8, 29, 0x81))):
                struct.pack_into("<Q", data, base + (126 + i) * 8,
                                 tag << 48 | value << 16 | reg)
        # Pack this layer's weights and bias once (every strip reads the same block).
        zero_points = np.asarray(q.weight_zero_points).reshape(-1)
        for o in range(oc):
            taps = q.weights[o].reshape(ic, kernel, kernel)
            for tap in range(kernel * kernel):
                y, x = divmod(tap, kernel)
                values = [int(v) for v in taps[:, y, x][:ic]]
                values += [int(zero_points[o])] * (16 - len(values))
                struct.pack_into("<16b", data, weight_offset + (tap * oc + o) * 16, *values)
            block = bias_offset + (o // 4) * 32
            struct.pack_into("<i", data, block + (o % 4) * 4, int(q.biases[o]))
            struct.pack_into("<h", data, block + 16 + (o % 4) * 2, -int(zero_points[o]))
            struct.pack_into("<H", data, block + 24 + (o % 4) * 2,
                             int(q.channel_multipliers[o]))
    final = quantizations[-1]
    binary = encode_sequence(
        data, input_shape=(8, 8, 3), output_shape=(8, 8, 3), input_stride=8,
        arena_bytes=arena, input_offset=input_offset, output_offset=output_offset,
        tasks=[(offset, 126, 29, 768) for offset in sorted(programs)],
        input_scale=1.0, input_zero_point=0, output_scale=final.output_scale,
        output_zero_point=final.output_zero_point, serial=serial, input_layout="native16")
    return binary, dict(profile="tiled-native-chain", tiles=tiles, layers=count,
                        strip_rows=strip, tasks=count * tiles,
                        kernels=[kernel for _, _, kernel in layers],
                        layout="shared-surfaces" if shared else "strip-surfaces",
                        halo_rows=max((kernel // 2) for _, _, kernel in layers),
                        hidden_channels=[w.shape[0] for w, _, _ in layers],
                        input_offset=input_offset, output_offset=output_offset,
                        strip_surface=surface, output_surface=output_surface,
                        arena_bytes=arena, payload_bytes=payload_size,
                        quantizations=meta["quantizations"],
                        output_scale=final.output_scale,
                        output_zero_point=final.output_zero_point,
                        submission="batched" if not serial else "serial")
