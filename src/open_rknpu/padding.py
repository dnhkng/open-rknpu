"""SPDX-License-Identifier: MIT

Input padding for graphs whose border mode the CNA cannot produce.

The CNA border path injects only the activation zero point, so constant padding
is native. Reflection, edge/replication and circular padding have no verified
NPU producer on this profile: the compiler folds a leading `Pad` node into the
graph input shape and records the required preprocessing, and callers pad the
input array with `pad_input` before calling the runtime.
"""
import numpy as np

# ONNX Pad mode -> numpy pad mode.
MODES = {"constant": "constant", "reflect": "reflect", "edge": "edge", "wrap": "wrap"}


def pad_input(array, pads, mode="constant", constant_values=0):
    """Pad a packed NHWC array.

    `pads` is (top, left, bottom, right); it applies to the H and W axes of a
    rank-3 HWC array or a rank-4 NHWC batch. Constant mode uses the activation
    zero point for the graph's UINT8 input.
    """
    if mode not in MODES:
        raise ValueError("unsupported Pad mode: " + str(mode))
    values = np.asarray(array)
    if values.ndim == 3:
        spec = ((pads[0], pads[2]), (pads[1], pads[3]), (0, 0))
    elif values.ndim == 4:
        spec = ((0, 0), (pads[0], pads[2]), (pads[1], pads[3]), (0, 0))
    else:
        raise ValueError("pad_input expects HWC or NHWC data")
    if any(p < 0 for p in pads):
        raise ValueError("negative padding is not supported")
    kwargs = {"constant_values": constant_values} if mode == "constant" else {}
    return np.pad(values, spec, mode=MODES[mode], **kwargs).astype(values.dtype, copy=False)
