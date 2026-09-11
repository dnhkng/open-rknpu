"""MIT. Materialize the passing native-dilation boundary cases from the retained exploration."""
from pathlib import Path
import json,shutil,onnx
from open_rknpu.native import compile_native_input
root=Path(__file__).resolve().parent;source=root/'native_dilation_suite';out=root/'native_dilation_verified_suite';out.mkdir(exist_ok=True)
entries=json.loads((source/'manifest.json').read_text());selected=[0,1,2,3,5,8,9];manifest=[]
for index,old in enumerate(selected):
    entry=dict(entries[old]);entry['index']=index;entry['exploration_index']=old
    src=source/f'model{old:03}.onnx';dst=out/f'model{index:03}.onnx';shutil.copyfile(src,dst)
    binary,_=compile_native_input(onnx.load(dst),entry['input_scale'],entry['input_zero_point']);dst.with_suffix('.bin').write_bytes(binary)
    shutil.copyfile(source/f'input{old:03}.u8',out/f'input{index:03}.u8')
    shutil.copyfile(source/f'expected{old:03}.i8',out/f'expected{index:03}.i8');manifest.append(entry)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
