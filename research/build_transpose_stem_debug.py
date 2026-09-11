"""MIT. Isolate the symmetrically requantized stem used by K5 investigation."""
from pathlib import Path
import json,numpy as np
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import encode_sequence
from open_rknpu.quantization import Quantization,reference

base=Path(__file__).resolve().parent;root=base/'transpose_stem_debug_suite';root.mkdir(exist_ok=True)
binary,meta=compile_sequence(base/'transpose_k5_suite/model000.onnx');q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['first']['quantization'].items()})
program=encode_sequence(binary[128:],input_shape=(8,8,3),output_shape=(8,8,3),input_stride=16,arena_bytes=24576,input_offset=0x1000,output_offset=0x2000,tasks=[(0,126,29,768)],input_scale=1.0,input_zero_point=0,output_scale=q.output_scale,output_zero_point=q.output_zero_point,serial=True)
(root/'model000.bin').write_bytes(program);x=np.fromfile(base/'transpose_k5_suite/input000.u8',np.uint8).reshape(16,8,8,3);x.tofile(root/'input000.u8');np.stack([reference(v,q) for v in x]).tofile(root/'expected000.i8');(root/'manifest.json').write_text(json.dumps([{'index':0,'purpose':'K5 symmetric stem isolation'}],indent=2)+'\n')
