"""Generate independent graphs/inputs and integer references for public C API."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.quantization import Quantization,reference

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/application_suite"
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(11030602)
manifest=[]
for height in range(5,9):
    for width in range(5,9):
        for kernel in (1,3):
            for relu in (False,True):
                index=len(manifest)
                weights=rng.uniform(-.25,.25,(3,3,kernel,kernel)).astype(np.float32)
                bias=rng.uniform(-5,5,(3,)).astype(np.float32)
                intermediate="conv" if relu else "output"
                nodes=[helper.make_node("Conv",["input","weights","bias"],[intermediate],kernel_shape=[kernel,kernel],pads=[kernel//2]*4)]
                if relu: nodes.append(helper.make_node("Relu",[intermediate],["output"]))
                shape=[1,3,height,width]
                graph=helper.make_graph(nodes,"application_suite",
                    [helper.make_tensor_value_info("input",TensorProto.FLOAT,shape)],
                    [helper.make_tensor_value_info("output",TensorProto.FLOAT,shape)],
                    [numpy_helper.from_array(weights,"weights"),numpy_helper.from_array(bias,"bias")])
                model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",13)])
                model.ir_version=8
                path=OUT/("model%03d.onnx"%index)
                onnx.save(model,path)
                payload,metadata=compile_model(path)
                (OUT/("model%03d.bin"%index)).write_bytes(encode(payload,metadata))
                params=metadata["quantization"]
                for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
                    params[key]=np.array(params[key])
                q=Quantization(**params)
                inputs=rng.integers(0,256,(16,height,width,3),dtype=np.uint8)
                inputs[0]=0; inputs[1]=255; inputs[2]=128
                expected=np.stack([reference(x,q) for x in inputs])
                inputs.tofile(OUT/("input%03d.u8"%index))
                expected.tofile(OUT/("expected%03d.i8"%index))
                manifest.append({"index":index,"shape":shape,"kernel":kernel,"relu":relu,"inputs":16})
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("Built",len(manifest),"independent models and",len(manifest)*16,"input cases")
