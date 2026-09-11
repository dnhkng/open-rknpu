# Pretrained MNIST integration target

Upstream ONNX Model Zoo `mnist-12`, pinned and checksummed in `source.json`.
The original archive, supplied test case, upstream README, and Apache-2.0
license are retained here. This is an existing trained model, not a newly
trained model designed around our compiler restrictions.

Run `PYTHONPATH=src python research/verify_pretrained_mnist.py`.
The host reference predicts digit 3 on the supplied fixture, matching upstream;
the largest logit difference is 0.00003052. This single fixture does not establish
dataset accuracy. The normalization pass reduces 12 nodes to 9 and preserves
outputs exactly on the fixture and 32 deterministic random inputs.

## Required hardware work

| Stage | Actual model requirement | Current gap |
| --- | --- | --- |
| Input | float32 1×1×28×28; fixture range −36.56 to 32.57 | explicit affine UINT8 encoding now hardware-verified; dataset calibration remains |
| Conv + bias + ReLU | 1→8 channels, 5×5, pad 2, 28×28 | trained weights and affine input now hardware-verified through Conv/Relu |
| MaxPool | 2×2 stride 2, 28→14, 8 channels | trained Conv/Relu/MaxPool prefix now compiler/runtime verified |
| Conv + bias + ReLU | 8→16 channels, 5×5, pad 2, 14×14 | native 5×5 convolution, 16 outputs, second activation |
| MaxPool | 3×3 stride 3, 14→4, 16 channels | 3×3 pooling |
| Reshape | NCHW 16×4×4→256 | preserve flatten order across native channel tiling |
| MatMul + bias | 256→10 | dense layer lowering and output quantization |
| Runtime | general sequence and activation allocations | replace fixed profiles with a task table and memory planner |

The separate `open-rknpu normalize input.onnx -o normalized.onnx` command
folds constant reshapes and eligible channel biases and makes Conv padding
explicit. It does not compile these missing operations or provide CPU fallback.
MNIST is not yet executable on the board. A MobileNet-class classifier or tiny
detector remains part of the broader model-coverage goal after this integration.
