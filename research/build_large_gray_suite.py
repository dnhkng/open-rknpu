"""Independent grayscale tests across row-stride boundaries and MNIST size."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode
from open_rknpu.quantization import Quantization, reference

root=Path(__file__).resolve().parents[1]/"research/gray_large_suite"
root.mkdir(exist_ok=True)
rng=np.random.default_rng(110320)
manifest=[]
for height,width in ((8,9),(9,8),(15,16),(16,15),(16,17),(17,16),(28,28),(32,32)):
    for kernel in (1,3,5):
        index=len(manifest)
        channels=(2,8,16)[index%3]
        relu=bool(index%2)
        weights=rng.uniform(-.3,.3,(channels,1,kernel,kernel)).astype(np.float32)
        bias=rng.uniform(-3,3,channels).astype(np.float32)
        nodes=[h.make_node("Conv",["input","w","b"],["hidden" if relu else "output"],
                           kernel_shape=[kernel]*2,pads=[kernel//2]*4)]
        if relu:nodes.append(h.make_node("Relu",["hidden"],["output"]))
        model=h.make_model(h.make_graph(nodes,"large_gray",
            [h.make_tensor_value_info("input",1,[1,1,height,width])],
            [h.make_tensor_value_info("output",1,[1,channels,height,width])],
            [nh.from_array(weights,"w"),nh.from_array(bias,"b")]),opset_imports=[h.make_opsetid("",13)])
        model.ir_version=8
        path=root/f"model{index:03}.onnx";onnx.save(model,path)
        payload,meta=compile_model(path);path.with_suffix(".bin").write_bytes(encode(payload,meta))
        q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta["quantization"].items()})
        inputs=rng.integers(0,256,(8,height,width,1),dtype=np.uint8)
        inputs[0]=0;inputs[1]=255;inputs[2]=128
        inputs.tofile(root/f"input{index:03}.u8")
        np.stack([reference(x,q) for x in inputs]).tofile(root/f"expected{index:03}.i8")
        manifest.append(dict(index=index,height=height,width=width,kernel=kernel,channels=channels,relu=relu,inputs=8))
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Built {len(manifest)} models")
