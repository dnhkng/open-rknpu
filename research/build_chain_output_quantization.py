"""MIT. Verify independent output conversion for the legacy Conv-Relu-Conv profile."""
from pathlib import Path
import json
import numpy as np,onnx
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.chain import chain_reference

base=Path(__file__).resolve().parent;root=base/'chain_output_quantization_suite';root.mkdir(exist_ok=True)
rng=np.random.default_rng(110359);manifest=[]
models=sorted((base/'generated').glob('chain_heldout*.onnx'))[:4]
for i,(source,zp,scale) in enumerate(zip(models,(-128,127,-43,79),(.5,1.25,.03125,3.0))):
    model=onnx.load(source);path=root/f'model{i:03}.onnx';onnx.save(model,path)
    requested={'scale':scale,'zero_point':zp};binary,meta=compile_sequence(path,output_range=requested);path.with_suffix('.bin').write_bytes(binary)
    q1=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()})
    q2=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['second'].items()})
    x=rng.integers(0,256,(8,8,8,3),dtype=np.uint8);x.tofile(root/f'input{i:03}.u8')
    np.stack([chain_reference(v,q1,q2) for v in x]).tofile(root/f'expected{i:03}.i8')
    manifest.append(dict(index=i,input_scale=1.0,input_zero_point=0,compile_output_range=requested,**meta))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
