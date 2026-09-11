"""Train the Fashion-MNIST companion model with the pinned example architecture.

Same graph as the pinned MNIST example so `examples/fashion/main.c` and the
build/evaluation harnesses apply unchanged:

    Conv(1->8, 5x5, pad2) -> Relu -> MaxPool(2x2/2)
    -> Conv(8->16, 5x5, pad2) -> Relu -> MaxPool(3x3/3)
    -> Reshape[1,256] -> MatMul(256->10) -> Add

Writes `research/pretrained/fashion-mnist/fashion.onnx`, a `normalized.onnx`
copy and a training report. MIT-licensed code; the Fashion-MNIST dataset is
distributed under the MIT license by Zalando Research.
"""
from pathlib import Path
import argparse
import gzip
import json
import struct
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research" / "pretrained" / "fashion-mnist"
DATA = SOURCE / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seed", type=int, default=110800)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def load(name):
    raw = gzip.decompress((DATA / name).read_bytes())
    return raw


images = load("train-images-idx3-ubyte.gz")
labels = load("train-labels-idx1-ubyte.gz")
test_images = load("t10k-images-idx3-ubyte.gz")
test_labels = load("t10k-labels-idx1-ubyte.gz")
assert struct.unpack_from(">4I", images) == (2051, 60000, 28, 28)
assert struct.unpack_from(">2I", labels) == (2049, 60000)
assert struct.unpack_from(">4I", test_images) == (2051, 10000, 28, 28)
assert struct.unpack_from(">2I", test_labels) == (2049, 10000)

x = np.frombuffer(images, np.uint8, offset=16).reshape(60000, 1, 28, 28).astype(np.float32) / 255
y = np.frombuffer(labels, np.uint8, offset=8).astype(np.int64)
tx = np.frombuffer(test_images, np.uint8, offset=16).reshape(10000, 1, 28, 28).astype(np.float32) / 255
ty = np.frombuffer(test_labels, np.uint8, offset=8).astype(np.int64)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.fc = nn.Linear(256, 10)

    def forward(self, value):
        value = torch.relu(self.conv1(value))
        value = torch.max_pool2d(value, 2, 2)
        value = torch.relu(self.conv2(value))
        value = torch.max_pool2d(value, 3, 3)
        return self.fc(value.flatten(1))


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.4)
loss_fn = nn.CrossEntropyLoss()
report = {"epochs": []}
for epoch in range(args.epochs):
    model.train()
    order = np.random.permutation(len(x))
    total = 0.0
    for start in range(0, len(x), args.batch):
        index = order[start:start + args.batch]
        batch = torch.from_numpy(x[index])
        target = torch.from_numpy(y[index])
        optimizer.zero_grad()
        loss = loss_fn(model(batch), target)
        loss.backward()
        optimizer.step()
        total += float(loss) * len(index)
    schedule.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(tx), 500):
            logits = model(torch.from_numpy(tx[start:start + 500]))
            correct += int((logits.argmax(1).numpy() == ty[start:start + 500]).sum())
    entry = {"epoch": epoch + 1, "train_loss": total / len(x), "test_accuracy": correct / len(tx)}
    report["epochs"].append(entry)
    print(json.dumps(entry), flush=True)

model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(tx))
    final = float((logits.argmax(1).numpy() == ty).mean())
report["test_accuracy"] = final
print("final float test accuracy", final, flush=True)

# Export the pinned structure. Names follow the MNIST fixture so the shared C
# suffix and build harness keep working; only the weights differ.
import onnx
from onnx import helper as h, numpy_helper as nh

def initializer(tensor):
    return nh.from_array(tensor.detach().numpy().astype(np.float32), name=None) if False else tensor.detach().numpy().astype(np.float32)

graph = h.make_graph(
    [h.make_node("Conv", ["Input3", "Parameter5", "Plus30_Output_0_conv_bias"], ["Plus30_Output_0"],
                 kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], dilations=[1, 1], group=1),
     h.make_node("Relu", ["Plus30_Output_0"], ["ReLU32_Output_0"]),
     h.make_node("MaxPool", ["ReLU32_Output_0"], ["Pooling66_Output_0"],
                 kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0], auto_pad="NOTSET"),
     h.make_node("Conv", ["Pooling66_Output_0", "Parameter87", "Plus112_Output_0_conv_bias"], ["Plus112_Output_0"],
                 kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], dilations=[1, 1], group=1),
     h.make_node("Relu", ["Plus112_Output_0"], ["ReLU114_Output_0"]),
     h.make_node("MaxPool", ["ReLU114_Output_0"], ["Pooling160_Output_0"],
                 kernel_shape=[3, 3], strides=[3, 3], pads=[0, 0, 0, 0], auto_pad="NOTSET"),
     h.make_node("Reshape", ["Pooling160_Output_0", "Pooling160_Output_0_reshape0_shape"],
                 ["Pooling160_Output_0_reshape0"]),
     h.make_node("MatMul", ["Pooling160_Output_0_reshape0", "Parameter193_reshape1"], ["Times212_Output_0"]),
     h.make_node("Add", ["Times212_Output_0", "Parameter194"], ["Plus214_Output_0"])],
    "fashion_mnist",
    [h.make_tensor_value_info("Input3", 1, [1, 1, 28, 28])],
    [h.make_tensor_value_info("Plus214_Output_0", 1, [1, 10])],
    [nh.from_array(model.conv1.weight.detach().numpy().astype(np.float32), "Parameter5"),
     nh.from_array(model.conv1.bias.detach().numpy().astype(np.float32), "Plus30_Output_0_conv_bias"),
     nh.from_array(model.conv2.weight.detach().numpy().astype(np.float32), "Parameter87"),
     nh.from_array(model.conv2.bias.detach().numpy().astype(np.float32), "Plus112_Output_0_conv_bias"),
     nh.from_array(np.array([1, 256], np.int64), "Pooling160_Output_0_reshape0_shape"),
     # MatMul operand is [256,10] like the MNIST fixture, i.e. the transposed
     # Linear weight that the shared C suffix indexes as dense_w[k*10+j].
     nh.from_array(model.fc.weight.detach().numpy().astype(np.float32).T, "Parameter193_reshape1"),
     nh.from_array(model.fc.bias.detach().numpy().astype(np.float32), "Parameter194")])
graph.value_info.append(h.make_tensor_value_info("ReLU32_Output_0", 1, [1, 8, 28, 28]))
graph.value_info.append(h.make_tensor_value_info("Pooling66_Output_0", 1, [1, 8, 14, 14]))
graph.value_info.append(h.make_tensor_value_info("ReLU114_Output_0", 1, [1, 16, 14, 14]))
graph.value_info.append(h.make_tensor_value_info("Pooling160_Output_0", 1, [1, 16, 4, 4]))
graph.value_info.append(h.make_tensor_value_info("Pooling160_Output_0_reshape0", 1, [1, 256]))
graph.value_info.append(h.make_tensor_value_info("Times212_Output_0", 1, [1, 10]))
onnx_model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
onnx_model.ir_version = 8
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, SOURCE / "fashion.onnx")
onnx.save(onnx_model, SOURCE / "normalized.onnx")
report["parameters"] = int(sum(p.numel() for p in model.parameters()))
report["architecture"] = "Conv(1-8,5x5,pad2)-Relu-MaxPool(2/2)-Conv(8-16,5x5,pad2)-Relu-MaxPool(3/3)-Reshape[1,256]-MatMul(256-10)-Add"
(SOURCE / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
