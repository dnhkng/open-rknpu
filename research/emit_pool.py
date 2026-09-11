"""Compatibility wrapper for the open pooling compiler."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from open_rknpu.pooling import *
if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument("model",type=Path);p.add_argument("-o",type=Path,required=True)
    a=p.parse_args();data,meta=compile_pool(a.model);a.o.write_bytes(data);a.o.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
