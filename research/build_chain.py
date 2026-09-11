"""Development-only two-convolution oracle for intermediate tensor discovery."""
import argparse
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from rknn.api import RKNN

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--channels",type=int,choices=(3,4,8,16),default=4)
parser.add_argument("--second-kernel",type=int,choices=(1,3,5),default=1)
parser.add_argument("--size",type=int,default=8,choices=range(5,33))
parser.add_argument("--input-channels",type=int,default=3,choices=(1,3))
parser.add_argument("--output-channels",type=int,default=3,choices=range(2,17))
parser.add_argument("--name")
args=parser.parse_args()
size=args.size;ic=args.input_channels;oc=args.output_channels
c=args.channels
kernel=args.second_kernel
name=args.name or f"chain{c}"+(f"k{kernel}" if kernel!=1 else "")
out=Path(__file__).resolve().parent/"fixtures"/name
out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(110311)
w1=rng.uniform(-.25,.75,(c,ic,1,1)).astype(np.float32)
w2=rng.uniform(-.5,.5,(oc,c,kernel,kernel)).astype(np.float32)
b1=np.linspace(-2,2,c,dtype=np.float32)
b2=np.array([1,-2,3],np.float32) if oc==3 else np.linspace(-2,2,oc,dtype=np.float32)
nodes=[h.make_node("Conv",["input","w1","b1"],["conv1"],kernel_shape=[1,1]),
       h.make_node("Relu",["conv1"],["hidden"]),
       h.make_node("Conv",["hidden","w2","b2"],["output"],kernel_shape=[kernel,kernel],pads=[kernel//2]*4)]
g=h.make_graph(nodes,"chain_probe",[h.make_tensor_value_info("input",1,[1,ic,size,size])],
               [h.make_tensor_value_info("output",1,[1,oc,size,size])],
               [nh.from_array(a,n) for a,n in ((w1,"w1"),(w2,"w2"),(b1,"b1"),(b2,"b2"))])
m=h.make_model(g,opset_imports=[h.make_opsetid("",13)])
m.ir_version=8
onnx.checker.check_model(m)
onnx.save(m,out/"model.onnx")
paths=[]
for i in range(8):
    a=rng.integers(0,256,(1,ic,size,size)).astype(np.float32)
    a[:,:,0,0]=255; a[:,:,-1,-1]=0
    p=out/f"calibration_{i}.npy"
    np.save(p,a); paths.append(str(p))
(out/"dataset.txt").write_text("\n".join(paths)+"\n")
rknn=RKNN(verbose=True)
try:
    assert rknn.config(target_platform="rv1103",mean_values=[[0]*ic],std_values=[[1]*ic])==0
    assert rknn.load_onnx(model=str(out/"model.onnx"))==0
    assert rknn.build(do_quantization=True,dataset=str(out/"dataset.txt"))==0
    assert rknn.export_rknn(str(out/"model.rknn"))==0
finally:
    rknn.release()
(out/"reference.json").write_text(json.dumps({"hidden_channels":c,"size":size,"input_channels":ic,"output_channels":oc,"weights1":w1.tolist(),
    "weights2":w2.tolist(),"bias1":b1.tolist(),"bias2":b2.tolist()},indent=2)+"\n")
