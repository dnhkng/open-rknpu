"""SPDX-License-Identifier: MIT

Accuracy reporting that keeps two claims separate:
  * integer exactness of the quantized computation (board vs integer reference);
  * float-model error of the quantized/dequantized result (vs the ONNX model).

`layerwise_quantization_error` measures the round-trip error introduced by the
compiled per-tensor range alone, without hardware, which is the useful signal
when choosing or rejecting a calibration set.
"""
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from .calibration import measured_tensor_names


def dequantize(values,scale,zero_point):
    return (np.asarray(values,np.float64)-zero_point)*scale


def quantize_roundtrip(values,scale,zero_point):
    codes=np.clip(np.rint(np.asarray(values,np.float64)/scale)+zero_point,-128,127)
    return dequantize(codes,scale,zero_point)


def error_metrics(integer,float_reference,scale,zero_point):
    """Compare INT8 output bytes against a float reference after dequantization."""
    integer=np.asarray(integer,np.int64);reference=np.asarray(float_reference,np.float64)
    if integer.shape!=reference.shape:
        raise ValueError('integer and reference outputs must share a shape')
    decoded=dequantize(integer,scale,zero_point)
    error=np.abs(decoded-reference)
    denominator=np.maximum(np.abs(reference),1e-9)
    return {"samples":int(integer.shape[0]) if integer.ndim else 1,
            "values":int(integer.size),
            "mae":float(error.mean()),"rmse":float(np.sqrt((error**2).mean())),
            "max_error":float(error.max()) if error.size else 0.0,
            "mean_relative":float((error/denominator).mean()) if error.size else 0.0,
            "output_scale":float(scale),"output_zero_point":int(zero_point)}


def classification_accuracy(logits,labels):
    """Top-1 accuracy for a [N,C] logit array against integer labels [N]."""
    logits=np.asarray(logits);labels=np.asarray(labels)
    if logits.ndim!=2 or labels.shape!=(logits.shape[0],):
        raise ValueError('expected [N,C] logits and [N] labels')
    predictions=logits.argmax(axis=1)
    return {"samples":int(labels.size),"correct":int((predictions==labels).sum()),
            "accuracy":float((predictions==labels).mean()),
            "predictions":predictions.tolist()}


def layerwise_quantization_error(model_path,ranges,inputs):
    """Round-trip error per measured Conv/activation tensor for a range set.

    `ranges` maps tensor name to a dict with `scale` and `zero_point`, as
    produced by `open_rknpu.calibration.measure`. Inputs are packed NHWC uint8
    (or float) samples; they are only used to evaluate the float reference.
    """
    model=model_path if isinstance(model_path,onnx.ModelProto) else onnx.load(model_path)
    names=measured_tensor_names(model)
    missing=[name for name in names if name not in ranges]
    if missing:raise ValueError('ranges lack measured tensors: '+', '.join(missing))
    evaluator=ReferenceEvaluator(model)
    input_name=model.graph.input[0].name
    accumulators={name:{"abs":0.0,"max":0.0,"values":0} for name in names}
    samples=0
    for sample in np.asarray(inputs):
        values=evaluator.run(names,{input_name:np.asarray(sample,np.float32)[None].transpose(0,3,1,2)})
        for name,value in zip(names,values):
            entry=ranges[name]
            reference=np.asarray(value,np.float64)
            error=np.abs(quantize_roundtrip(reference,float(entry["scale"]),int(entry["zero_point"]))-reference)
            accumulators[name]["abs"]+=float(error.sum())
            accumulators[name]["max"]=max(accumulators[name]["max"],float(error.max()))
            accumulators[name]["values"]+=int(error.size)
        samples+=1
    return {"samples":samples,"samples_per_layer":samples,
            "layers":{name:{"mae":acc["abs"]/acc["values"] if acc["values"] else 0.0,
                            "max_error":acc["max"],"values":acc["values"]}
                      for name,acc in accumulators.items()}}
