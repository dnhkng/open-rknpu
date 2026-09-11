"""SPDX-License-Identifier: MIT

Experimental RV1103 1x1 affine weight quantization, UINT8 input scale 1.
Randomized hardware tests establish separate rounding after channel scaling
and global scaling for this profile. Other operator profiles remain untested.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class Quantization:
    weights: np.ndarray
    weight_zero_points: np.ndarray
    weight_scales: np.ndarray
    biases: np.ndarray
    channel_multipliers: np.ndarray
    multiplier: int
    shift: int
    output_scale: float
    output_zero_point: int
    kernel_size: int = 1
    relu: bool = False
    input_scale: float = 1.0
    input_zero_point: int = 0

    def metadata(self):
        return {name: value.tolist() if isinstance(value,np.ndarray) else value
                for name,value in vars(self).items()}

def quantize(weights, bias, output_scale=None, output_zero_point=None, relu=False, input_scale=1.0, input_zero_point=0):
    if not np.isfinite(input_scale) or input_scale<=0 or not isinstance(input_zero_point,(int,np.integer)) or not 0<=input_zero_point<=255:
        raise ValueError("invalid UINT8 input quantization")
    if (output_scale is None)!=(output_zero_point is None):
        raise ValueError("specify output scale and zero point together")
    weights=np.asarray(weights,dtype=np.float32)
    kernel=weights.shape[2] if weights.ndim==4 else 1
    channels=weights.shape[0]
    if (weights.ndim not in (2,4) or not 1<=channels<=16 or
        weights.shape[1:] not in ((1,), (1,1,1), (1,3,3), (1,5,5), (3,), (3,1,1), (3,3,3), (3,5,5))):
        raise ValueError("expected [O,I,K,K], I=1 or 3, 1<=O<=16, K=1, 3 or 5 weights")
    weights=weights.reshape(channels,-1)
    bias=np.asarray(bias,dtype=np.float32).reshape(channels)
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("finite weights and bias required")
    minimum=np.minimum(weights.min(axis=1),0)
    maximum=np.maximum(weights.max(axis=1),0)
    scales=((maximum-minimum)/np.float32(255)).astype(np.float32)
    max_scale=float(scales.max())
    if max_scale==0:
        max_scale=1.0/32768  # Finite bias-only accumulator unit; no weight contribution.
    # A zero channel may carry a constant bias; use the common accumulator scale.
    scales=np.where(scales==0,np.float32(max_scale),scales)
    zero_points=np.clip(np.rint(-128-minimum/scales),-128,127).astype(np.int64)
    qweights=np.clip(np.rint(weights/scales[:,None])+zero_points[:,None],-128,127).astype(np.int64)
    centered=qweights-zero_points[:,None]
    qbias=np.rint(bias/(scales*input_scale)).astype(np.int64)+(128-input_zero_point)*centered.sum(axis=1)
    if np.any(qbias < -(1<<31)) or np.any(qbias >= (1<<31)):
        raise ValueError("INT32 bias overflow")
    acc_min=qbias+np.minimum(-128*centered,127*centered).sum(axis=1)
    acc_max=qbias+np.maximum(-128*centered,127*centered).sum(axis=1)
    if np.any(acc_min < -(1<<31)) or np.any(acc_max >= (1<<31)):
        raise ValueError("INT32 accumulator overflow")
    if output_scale is None:
        # Exact floating-point interval for independently bounded inputs [0,255].
        lo=-input_zero_point*input_scale;hi=(255-input_zero_point)*input_scale
        lower=min(0,float((np.minimum(weights*lo,weights*hi).sum(axis=1)+bias).min()))
        upper=max(0,float((np.maximum(weights*lo,weights*hi).sum(axis=1)+bias).max()))
        if relu: lower=0
        output_scale=float(np.float32((upper-lower)/255)) or 1.0
        output_zero_point=int(np.clip(np.rint(-128-lower/output_scale),-128,127))
    if not output_scale or not np.isfinite(output_scale) or output_scale<=0 or output_zero_point is None or not -128<=output_zero_point<=127:
        raise ValueError("invalid output quantization")
    factor=max_scale*input_scale/output_scale
    shift=math.floor(math.log2(32767/factor))
    if not 0<=shift<=31:
        raise ValueError("output scale outside validated multiplier/shift range")
    multiplier=round(factor*(1<<shift))
    channel_multipliers=np.rint(scales/np.float32(max_scale)*16384).astype(np.int64)
    return Quantization(qweights,zero_points,scales,qbias,channel_multipliers,
                        multiplier,shift,float(output_scale),int(output_zero_point),kernel,relu,float(input_scale),int(input_zero_point))

def receptive_fields(inputs,kernel,padding_value=0):
    inputs=np.asarray(inputs)
    if kernel==1: return inputs
    pad=kernel//2
    padded=np.pad(inputs,((pad,pad),(pad,pad),(0,0)),constant_values=padding_value)
    patches=np.lib.stride_tricks.sliding_window_view(padded,(kernel,kernel),axis=(0,1))
    return patches.reshape(inputs.shape[0],inputs.shape[1],-1)

def reference(inputs,quantization,rounding="separate"):
    q=quantization
    centered=q.weights-q.weight_zero_points[:,None]
    patches=receptive_fields(np.asarray(inputs,dtype=np.int64)-128,q.kernel_size,q.input_zero_point-128)
    acc=np.einsum("...c,oc->...o",patches,centered)+q.biases
    if q.relu: acc=np.maximum(acc,0)
    if rounding=="separate":
        # Channel conversion rounds exact halves to even (out2/out4 probes).
        product=acc*q.channel_multipliers
        scaled=(product+8191+((product>>14)&1))>>14
        product=scaled*q.multiplier
        outputs=((product+(1<<(q.shift-1))-1+((product>>q.shift)&1))>>q.shift) if q.shift else product
    elif rounding=="combined":
        outputs=(acc*q.channel_multipliers*q.multiplier+(1<<(q.shift+13)))>>(q.shift+14)
    else:
        raise ValueError("unknown rounding candidate")
    return np.clip(outputs+q.output_zero_point,-128,127).astype(np.int8)
