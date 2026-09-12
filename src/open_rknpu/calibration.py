"""SPDX-License-Identifier: MIT

Streaming activation calibration for supported static ONNX graphs.

Calibration only measures activation ranges with ONNX's host reference; it does
not require the legacy compiler to accept the graph, so the same ranges can feed
sequence profiles. Producing those ranges is separate from using them: the
compiler and its emitters decide which profiles accept a measured range.

Three range selectors share one measurement pass:

* `minmax` - the observed minimum and maximum (exact for the samples seen);
* `percentile` - the central `percentile`% of the distribution, so a handful of
  outliers cannot stretch the scale over the whole int8 range;
* `kl` - a TensorRT-style KL-divergence saturation search: build a fine histogram,
  and for every candidate threshold quantize it to 128 levels and keep the
  threshold whose quantized distribution is closest (minimum KL divergence) to the
  original, which trades a little clipping for much better resolution.
"""
from pathlib import Path
import math
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

METHODS = ("minmax", "percentile", "kl")


def measured_tensor_names(model):
    """Conv outputs, using the fused activation output when one directly follows."""
    graph=model.graph
    names=[]
    for i,node in enumerate(graph.node):
        if node.op_type!="Conv":continue
        following=graph.node[i+1] if i+1<len(graph.node) else None
        tensor=node.output[0]
        if (following is not None and following.op_type in ("Relu","Clip")
            and list(following.input)==[node.output[0]] and len(following.output)==1):
            tensor=following.output[0]
        names.append(tensor)
    return names


def range_to_quantization(entry):
    """Convert a measured min/max entry into (scale, zero_point)."""
    scale=float(np.float32((entry["max"]-entry["min"])/255)) or 1.0
    if not math.isfinite(scale) or scale<=0:raise ValueError("invalid calibration scale")
    zero_point=int(np.clip(np.rint(-128-entry["min"]/scale),-128,127))
    return scale,zero_point


def measured_range(ranges,name):
    """The measured `(scale, zero_point)` band for `name`, or `None` without a report.

    This is the one band-selection contract every profile shares: `None` means "no
    calibration" (the emitter keeps its analytic band), and a missing tensor fails
    closed with the scheduler's documented message rather than a `KeyError`.
    """
    if ranges is None:return None
    entry=ranges.get(name)
    if entry is None:raise ValueError("calibration ranges lack tensor "+str(name))
    return dict(scale=float(entry["scale"]),zero_point=int(entry["zero_point"]))


def _quantile(hist,edges,q):
    """The `q`-quantile of a histogram as an interval `(lo, hi)`."""
    total=float(hist.sum())
    if total<=0:return float(edges[0]),float(edges[-1])
    cumulative=np.cumsum(hist)
    index=int(np.searchsorted(cumulative,max(q,0.0)*total,side="left"))
    index=min(max(index,0),len(hist)-1)
    return float(edges[index]),float(edges[index+1])


def percentile_range(hist,edges,percentile):
    """The observed range with the upper `100-percentile`% tail clipped.

    The lower bound stays the observed minimum (the same asymmetry as `kl_range`), so
    a few extreme positive activations cannot stretch the scale - the usual reason to
    prefer percentile calibration over min/max.
    """
    if not 0<percentile<=100:raise ValueError("percentile must be in (0,100]")
    low=float(edges[0])
    high=_quantile(hist,edges,percentile/100.0)[1]
    return low,high


def kl_range(hist,edges,levels=128):
    """TensorRT-style saturation: the threshold minimizing the KL divergence.

    The reference distribution `P` is the full histogram. For a candidate upper bound
    `i` the model `Q` quantizes the range `[min, edges[i]]` into `levels` equal-value
    levels - each level's mass is shared evenly by the histogram bins it covers - and
    every sample above `edges[i]` is saturated into the top level. `KL(P||Q)` therefore
    penalizes *both* saturation (mass landing in the wrong bins) and a coarse bulk
    (wide levels), and the minimizing `i` is the chosen upper bound. The lower bound
    stays the observed minimum, matching the min/max path's asymmetric handling.

    Evaluating `Q` bin-count (rather than value-width) levels would make the divergence
    zero at `i == levels`; the value-width formulation above does not have that
    degeneracy, which is why a threshold of ~10 is selected for a bulk in [0,10] with
    outliers at 100 (`tests/test_calibration.py`).
    """
    counts=np.asarray(hist,dtype=np.float64)
    total=counts.sum()
    if total<=0:return float(edges[0]),float(edges[-1])
    bins=len(counts)
    if bins<=levels:return float(edges[0]),float(edges[-1])
    reference=counts/total
    positive=reference>0
    best_index,best_divergence=bins-1,None
    values=np.arange(bins)
    for i in range(levels,bins+1):
        cuts=(values[:i]*levels//i).astype(int)
        widths=np.bincount(cuts,minlength=levels)
        mass=np.bincount(cuts,weights=counts[:i],minlength=levels)
        mass[-1]+=counts[i:].sum()
        model=np.zeros(bins)
        model[:i]=mass[cuts]/np.maximum(widths[cuts],1)
        model/=model.sum()
        divergence=float(np.sum(reference[positive]
                                *np.log(reference[positive]
                                        /np.maximum(model[positive],1e-12))))
        if best_divergence is None or divergence<best_divergence:
            best_index,best_divergence=i,divergence
    return float(edges[0]),float(edges[best_index])


def _histogram(values,bins,low,high):
    if high<=low:return np.zeros(bins,dtype=np.int64),np.linspace(low,low+1.0,bins+1)
    edges=np.linspace(low,high,bins+1)
    counts,_=np.histogram(values,bins=edges)
    return counts.astype(np.int64),edges


def measure(model_path,directory,method="minmax",percentile=99.99,bins=2048):
    """Measure activation ranges with min/max, percentile or KL selection.

    `minmax` needs one pass. `percentile` and `kl` need the observed range first and
    then a second pass that fills a `bins`-wide histogram per measured tensor, so the
    selected range is data-driven rather than assumed.
    """
    if method not in METHODS:
        raise ValueError("calibration method must be one of %s" % (METHODS,))
    model=onnx.load(model_path)
    onnx.checker.check_model(model)
    graph=model.graph
    if len(graph.input)!=1:
        raise ValueError("calibration requires exactly one graph input")
    shape=[d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    if len(shape)!=4 or any(v<=0 for v in shape):
        raise ValueError("calibration requires a static rank-four NCHW input")
    names=measured_tensor_names(model)
    if not names:raise ValueError("calibration requires at least one Conv")
    evaluator=ReferenceEvaluator(model)
    ranges={name:{"min":0.0,"max":0.0} for name in names}
    files=sorted(Path(directory).glob("*.npy"))
    if not files:raise ValueError("calibration directory must contain .npy arrays")
    samples=[]
    for path in files:
        data=np.load(path,allow_pickle=False,mmap_mode="r")
        if data.ndim!=4 or list(data.shape[1:])!=shape[1:] or data.shape[0]<1 or data.dtype not in (np.dtype("uint8"),np.dtype("float32")):
            raise ValueError(f"{path}: expected uint8/float32 [N,{','.join(map(str,shape[1:]))}]")
        for sample in data:
            if not np.isfinite(sample).all() or np.any(sample<0) or np.any(sample>255):
                raise ValueError(f"{path}: calibration inputs must be finite values in [0,255]")
            samples.append(sample.astype(np.float32))
    if not samples:raise ValueError("calibration directory produced no samples")
    histograms={name:None for name in names}
    for sample in samples:
        values=evaluator.run(names,{graph.input[0].name:sample[None]})
        for name,value in zip(names,values):
            if not np.isfinite(value).all():raise ValueError("nonfinite calibration activation: "+name)
            ranges[name]["min"]=min(ranges[name]["min"],float(value.min()))
            ranges[name]["max"]=max(ranges[name]["max"],float(value.max()))
    if method!="minmax":
        histograms={name:np.zeros(bins,dtype=np.int64) for name in names}
        edges={name:np.linspace(ranges[name]["min"],ranges[name]["max"],bins+1)
               for name in names}
        for sample in samples:
            values=evaluator.run(names,{graph.input[0].name:sample[None]})
            for name,value in zip(names,values):
                counts,_=_histogram(np.asarray(value,dtype=np.float64).reshape(-1),
                                    bins,ranges[name]["min"],ranges[name]["max"])
                histograms[name]+=counts
        for name in names:
            if method=="percentile":
                low,high=percentile_range(histograms[name],edges[name],percentile)
            else:
                low,high=kl_range(histograms[name],edges[name])
            ranges[name]["min"],ranges[name]["max"]=low,high
    for entry in ranges.values():
        entry["scale"],entry["zero_point"]=range_to_quantization(entry)
    report={"method":method,"input_layout":"NCHW","input_scale":1,"input_zero_point":0,
            "samples":len(samples),"files":[str(p) for p in files],"tensors":names,
            "ranges":ranges}
    if method!="minmax":
        report["bins"]=bins
    if method=="percentile":
        report["percentile"]=percentile
    return report
