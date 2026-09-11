"""Compare analytic and calibrated models on separate deterministic inputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import json
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from open_rknpu.calibration import measure
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.quantization import Quantization,reference
from open_rknpu.chain import native_reference

root=Path(__file__).resolve().parent
out=root/"calibration_suite";out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110316)
def qfrom(params):
    params=dict(params)
    for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
        params[key]=np.array(params[key])
    return Quantization(**params)

def run_integer(x,meta):
    profile=meta.get("profile",1)
    if profile==2:return native_reference(reference(x,qfrom(meta["first"]["quantization"])),qfrom(meta["second"]))
    y=reference(x,qfrom(meta["quantization"]))
    if profile>=3:
        for stage in range(3 if profile>=5 else 1):
            height,width=y.shape[:2];x=y.astype(np.int64).reshape(height//2,2,width//2,2,3)
            y=(x.max(axis=(1,3)) if profile in (3,5) else np.rint(x.sum(axis=(1,3))/4)).astype(np.int8)
    return y

manifest=[]
for name in ("dense_heldout","chain_heldout16","pool_max_r1","pool_average_r0","reduce_max_r1","reduce_average_r0"):
    path=root/f"generated/{name}.onnx";model=onnx.load(path)
    shape=[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    calibration=rng.integers(0,256,(32,*shape[1:]),dtype=np.uint8)
    np.save(folder/"calibration.npy",calibration)
    report=measure(path,folder);(folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    # Evaluation inputs are drawn afterward and are not used to measure ranges.
    inputs=rng.integers(0,256,(64,shape[2],shape[3],3),dtype=np.uint8)
    evaluator=ReferenceEvaluator(model)
    targets=np.stack([evaluator.run(None,{model.graph.input[0].name:x.transpose(2,0,1)[None].astype(np.float32)})[0][0].transpose(1,2,0) for x in inputs])
    for calibrated in (False,True):
        index=len(manifest)
        payload,meta=compile_model(path,calibration_ranges=report["ranges"] if calibrated else None)
        (out/f"model{index:03}.bin").write_bytes(encode(payload,meta))
        inputs.tofile(out/f"input{index:03}.u8")
        outputs=np.stack([run_integer(x,meta) for x in inputs]);outputs.tofile(out/f"expected{index:03}.i8")
        decoded=(outputs.astype(float)-meta["output_zero_point"])*meta["output_scale"]
        error=np.abs(decoded-targets)
        record={"index":index,"model":name,"calibrated":calibrated,"samples":64,
                "mae":float(error.mean()),"max_error":float(error.max()),"output_scale":meta["output_scale"]}
        manifest.append(record);print(record,flush=True)
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
