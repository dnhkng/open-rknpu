"""Development-only vendor oracle: controlled three-input-channel convolution."""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from rknn.api import RKNN

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="identity")
parser.add_argument("--permutation", default="0,1,2")
parser.add_argument("--height", type=int, default=8)
parser.add_argument("--width", type=int, default=8)
parser.add_argument("--output-channels", type=int, default=3, choices=range(1,17))
parser.add_argument("--input-channels", type=int, default=3, choices=(1,3))
parser.add_argument("--weights", help="JSON weights with output_channels * 3 * kernel_size**2 values")
parser.add_argument("--bias", help="JSON bias vector with output_channels values")
parser.add_argument("--kernel-size", type=int, choices=(1,3,5), default=1)
parser.add_argument("--tap", help="row,column for sparse spatial weight experiments")
parser.add_argument("--relu", action="store_true")
parser.add_argument("--pool", choices=("max","average"), help="append 2x2 stride-2 pooling")
parser.add_argument("--pool-size", type=int, choices=(2,4,8), default=2)
args = parser.parse_args()
permutation = [int(x) for x in args.permutation.split(",")]
assert sorted(permutation) == [0, 1, 2]
height, width = args.height, args.width
assert 1 <= height <= 32 and 1 <= width <= 32
out = Path(__file__).resolve().parent / "fixtures" / args.name
out.mkdir(parents=True, exist_ok=True)
kernel=args.kernel_size
channels=args.output_channels
tap=[int(x) for x in args.tap.split(",")] if args.tap else [kernel//2,kernel//2]
assert len(tap)==2 and all(0<=v<kernel for v in tap)
w = np.zeros((channels, args.input_channels, kernel, kernel), dtype=np.float32)
for c in range(channels):
    w[c, permutation[c % 3] if args.input_channels==3 else 0, tap[0], tap[1]] = (c + 1) / 4
if args.weights:
    w = np.asarray(json.loads(args.weights), dtype=np.float32).reshape(channels, args.input_channels, kernel, kernel)
initializers = [numpy_helper.from_array(w, "weights")]
conv_inputs = ["input", "weights"]
bias = np.zeros(channels, dtype=np.float32)
if args.bias:
    bias = np.asarray(json.loads(args.bias), dtype=np.float32).reshape(channels)
    initializers.append(numpy_helper.from_array(bias, "bias"))
    conv_inputs.append("bias")
conv_output="conv_output" if args.relu else ("pool_input" if args.pool else "output")
nodes=[helper.make_node("Conv",conv_inputs,[conv_output],kernel_shape=[kernel,kernel],pads=[kernel//2]*4)]
if args.relu:
    nodes.append(helper.make_node("Relu",[conv_output],["pool_input" if args.pool else "output"]))
if args.pool:
    nodes.append(helper.make_node("MaxPool" if args.pool=="max" else "AveragePool",
                                  ["pool_input"],["output"],kernel_shape=[args.pool_size]*2,strides=[args.pool_size]*2))
graph = helper.make_graph(
    nodes,
    "identity_probe",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, args.input_channels, height, width])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, height//args.pool_size if args.pool else height, width//args.pool_size if args.pool else width])],
    initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out / "model.onnx")
rng = np.random.default_rng(1103)
paths = []
for i in range(8):
    path = out / ("calibration_%d.npy" % i)
    calibration = rng.integers(0, 256, (1, args.input_channels, height, width)).astype(np.float32)
    calibration[:, :, 0, 0] = 255  # fix extrema across shape probes
    if height * width > 1:
        calibration[:, :, -1, -1] = 0
    np.save(path, calibration)
    paths.append(str(path))
(out / "dataset.txt").write_text("\n".join(paths) + "\n")
rknn = RKNN(verbose=True)
try:
    assert rknn.config(target_platform="rv1103", mean_values=[[0]*args.input_channels], std_values=[[1]*args.input_channels]) == 0
    assert rknn.load_onnx(model=str(out / "model.onnx")) == 0
    assert rknn.build(do_quantization=True, dataset=str(out / "dataset.txt")) == 0
    assert rknn.export_rknn(str(out / "model.rknn")) == 0
finally:
    rknn.release()
(out / "reference.json").write_text(json.dumps({"shape_nhwc": [1, height, width, args.input_channels], "output_shape_nhwc": [1,height//args.pool_size if args.pool else height,width//args.pool_size if args.pool else width,channels], "pool": args.pool, "pool_size": args.pool_size, "kernel_size":kernel,"tap":tap,"relu":args.relu,"channel_multipliers": [(c+1)/4 for c in range(channels)], "input_channel_for_output": [permutation[c%3] if args.input_channels==3 else 0 for c in range(channels)], "weights": w.tolist(), "bias": bias.tolist()}, indent=2) + "\n")
