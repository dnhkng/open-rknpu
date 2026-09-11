"""Run an independently compiled small program using the open harness."""
import argparse
import json
from pathlib import Path
from run_oracle import adb,BOARD,ROOT

def run(name,random_cases):
    if not name.replace("_", "").isalnum(): raise ValueError("invalid name")
    artifact=ROOT/"generated"/(name+".bin")
    meta=json.loads(artifact.with_suffix(".json").read_text())
    _,height,width,_=meta["shape_nhwc"]
    assert artifact.stat().st_size==8192 and 5<=height<=8 and 5<=width<=8
    assert 0<=random_cases<=256
    print(adb("push",str(artifact),BOARD+"/"+name+".bin").decode(),flush=True)
    cmd="OPEN_NPU_RANDOM_CASES={n} {b}/raw_replay {b}/{name}.bin --relocate {h} {w}".format(n=random_cases,b=BOARD,name=name,h=height,w=width)
    log=adb("shell",cmd)
    target=ROOT/("generated_"+name+".log")
    target.write_bytes(log)
    if "OUTPUT impulse " not in log.decode(errors="replace"):
        raise RuntimeError("inference did not produce all base outputs: "+str(target))
    print("Saved",target,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--random-cases",type=int,default=16)
    args=parser.parse_args()
    run(args.name,args.random_cases)
