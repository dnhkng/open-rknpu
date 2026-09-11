"""Independent rectangular/multistage pool coverage for sequence lowering."""
from pathlib import Path
import copy
import numpy as np
import onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization,reference

root=Path(__file__).resolve().parents[1]/"research/scheduled_pool_suite";root.mkdir(exist_ok=True)
rng=np.random.default_rng(110323);index=0
for source_index in (2,14,20):
    base=onnx.load(root.parent/f"gray_large_suite/model{source_index:03}.onnx")
    shape=[d.dim_value for d in base.graph.output[0].type.tensor_type.shape.dim]
    _,channels,ih,iw=shape
    for kind in ("MaxPool","AveragePool"):
        for levels in (1,2):
            model=copy.deepcopy(base);g=model.graph;previous="features"
            g.node[-1].output[0]=previous
            oh,ow=ih,iw
            for stage in range(levels):
                output=f"pool{stage}"
                g.node.append(h.make_node(kind,[previous],[output],kernel_shape=[2,2],strides=[2,2]))
                previous=output;oh//=2;ow//=2
            del g.output[:];g.output.append(h.make_tensor_value_info(previous,1,[1,channels,oh,ow]))
            path=root/f"model{index:03}.onnx";onnx.save(model,path)
            data,meta=compile_sequence(path,.5,128);path.with_suffix(".bin").write_bytes(data)
            q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta["quantization"].items()})
            inputs=rng.integers(0,256,(8,ih,iw,1),dtype=np.uint8);inputs[0]=128;inputs[1]=0;inputs[2]=255
            expected=[]
            for x in inputs:
                y=reference(x,q)
                for _ in range(levels):
                    ph,pw=y.shape[0]//2,y.shape[1]//2
                    blocks=y[:ph*2,:pw*2].reshape(ph,2,pw,2,channels)
                    y=blocks.max(axis=(1,3)) if kind=="MaxPool" else np.rint(blocks.astype(np.int64).sum(axis=(1,3))/4).astype(np.int8)
                expected.append(y)
            inputs.tofile(root/f"input{index:03}.u8")
            np.stack(expected).tofile(root/f"expected{index:03}.i8")
            index+=1
print(f"Built {index} scheduled models")
