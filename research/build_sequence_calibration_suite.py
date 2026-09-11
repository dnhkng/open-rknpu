"""Sequence-calibration suite: analytic versus measured output ranges.

Each graph is compiled twice with `compile --sequence` semantics (through
open_rknpu.scheduler): once with analytic ranges and once with ranges measured
from a separate calibration set. Expected bytes come from the documented integer
references, so the board check compares exact INT8 execution, not accuracy.
Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.calibration import measure,measured_tensor_names
from open_rknpu.accuracy import error_metrics,layerwise_quantization_error
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference
from open_rknpu.chain import native_reference
from open_rknpu.native import native_input_reference
from onnx.reference import ReferenceEvaluator

parser=argparse.ArgumentParser()
parser.add_argument("--directory",default="research/sequence_calibration_suite")
parser.add_argument("--cases",type=int,default=16)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/args.directory
root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110330)

def qfrom(params):
    params=dict(params)
    for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
        params[key]=np.array(params[key])
    return Quantization(**params)

def conv_graph(kind,hidden=5):
    if kind=="rgb":
        w=rng.uniform(-.7,.8,(3,3,3,3)).astype(np.float32);b=rng.uniform(-2,2,(3,)).astype(np.float32)
        nodes=[h.make_node("Conv",["input","w","b"],["output"],kernel_shape=[3,3],pads=[1,1,1,1])]
        tensors=[nh.from_array(w,"w"),nh.from_array(b,"b")];outs=[("output",3)]
    elif kind=="native":
        w=rng.uniform(-.7,.8,(32,3,3,3)).astype(np.float32);b=rng.uniform(-2,2,(32,)).astype(np.float32)
        nodes=[h.make_node("Conv",["input","w","b"],["output"],kernel_shape=[3,3],pads=[1,1,1,1])]
        tensors=[nh.from_array(w,"w"),nh.from_array(b,"b")];outs=[("output",32)]
    else:
        w1=rng.uniform(-.7,.8,(hidden,3,1,1)).astype(np.float32);b1=rng.uniform(-2,2,(hidden,)).astype(np.float32)
        w2=rng.uniform(-.7,.8,(3,hidden,1,1)).astype(np.float32);b2=rng.uniform(-2,2,(3,)).astype(np.float32)
        nodes=[h.make_node("Conv",["input","w1","b1"],["c1"],kernel_shape=[1,1]),
               h.make_node("Relu",["c1"],["r1"]),
               h.make_node("Conv",["r1","w2","b2"],["output"],kernel_shape=[1,1])]
        tensors=[nh.from_array(w1,"w1"),nh.from_array(b1,"b1"),nh.from_array(w2,"w2"),nh.from_array(b2,"b2")]
        outs=[("output",3)]
    graph=h.make_graph(nodes,f"seq_{kind}",[h.make_tensor_value_info("input",1,[1,3,8,8])],
        [h.make_tensor_value_info(name,1,[1,channels,8,8]) for name,channels in outs],tensors)
    if kind=="chain":graph.value_info.append(h.make_tensor_value_info("r1",1,[1,hidden,8,8]))
    model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)]);model.ir_version=8
    return model

def expected_for(kind,x,meta):
    if kind=="chain":
        first=qfrom(meta["first"]["quantization"]);second=qfrom(meta["second"])
        return native_reference(reference(x,first),second)
    if kind=="native":
        q=qfrom(meta["quantization"])
        return native_input_reference(x,q,meta["input_zero_point"],pads=meta["conv_pads"],
                                      strides=tuple(meta["conv_strides"]),dilations=tuple(meta["conv_dilations"]))
    return reference(x,qfrom(meta["quantization"]))

def analytic_ranges(model,meta):
    """The per-tensor scales the analytic compile chose, keyed by measured name."""
    names=measured_tensor_names(model)
    if "first" in meta:
        return {names[0]:{"scale":meta["first"]["quantization"]["output_scale"],
                          "zero_point":meta["first"]["quantization"]["output_zero_point"]},
                names[1]:{"scale":meta["second"]["output_scale"],"zero_point":meta["second"]["output_zero_point"]}}
    return {names[0]:{"scale":meta["quantization"]["output_scale"],
                      "zero_point":meta["quantization"]["output_zero_point"]}}

manifest=[];report_records=[]
for kind in ("rgb","native","chain"):
    model=conv_graph(kind)
    path=root/f"{kind}.onnx";onnx.save(model,path)
    calibration_dir=root/f"{kind}_calibration";calibration_dir.mkdir(exist_ok=True)
    calibration=rng.integers(0,256,(24,3,8,8),dtype=np.uint8)
    np.save(calibration_dir/"cal.npy",calibration)
    report=measure(path,calibration_dir)
    (calibration_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    # Evaluation inputs are drawn after the ranges are fixed.
    cases=rng.integers(0,256,(args.cases,8,8,3),dtype=np.uint8)
    cases[0]=0;cases[1]=255;cases[2]=128
    evaluator=ReferenceEvaluator(model)
    input_name=model.graph.input[0].name
    targets=np.stack([evaluator.run(None,{input_name:x.transpose(2,0,1)[None].astype(np.float32)})[0][0].transpose(1,2,0)
                      for x in cases])
    variants={}
    for calibrated in (False,True):
        index=len(manifest)
        binary,meta=compile_sequence(path,calibration_ranges=report["ranges"] if calibrated else None)
        (root/f"model{index:03}.bin").write_bytes(binary)
        cases.tofile(root/f"input{index:03}.u8")
        outputs=np.stack([expected_for(kind,x,meta) for x in cases])
        outputs.tofile(root/f"expected{index:03}.i8")
        metrics=error_metrics(outputs,targets,meta["output_scale"],meta["output_zero_point"])
        record={"index":index,"graph":kind,"calibrated":calibrated,"cases":args.cases,
                "output_scale":meta["output_scale"],"output_zero_point":meta["output_zero_point"],
                "float_mae":metrics["mae"],"float_max_error":metrics["max_error"]}
        manifest.append(record);variants[calibrated]=(meta,outputs)
        print(record,flush=True)
    layer_analytic=layerwise_quantization_error(model,analytic_ranges(model,variants[False][0]),cases)
    layer_calibrated=layerwise_quantization_error(model,report["ranges"],cases)
    report_records.append({"graph":kind,
        "analytic":{"metrics":error_metrics(variants[False][1],targets,variants[False][0]["output_scale"],variants[False][0]["output_zero_point"]),
                    "layers":layer_analytic["layers"]},
        "calibrated":{"metrics":error_metrics(variants[True][1],targets,variants[True][0]["output_scale"],variants[True][0]["output_zero_point"]),
                      "layers":layer_calibrated["layers"]}})
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
(root/"accuracy_report.json").write_text(json.dumps(report_records,indent=2)+"\n")
print(f"Built {len(manifest)} sequence-calibration models in {root}")
