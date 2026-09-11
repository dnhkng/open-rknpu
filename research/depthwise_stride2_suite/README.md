# Independent depthwise stride 2

12 graphs, 192 board inferences, 9,216 exact output bytes passed. Public
compile --sequence accepts Conv[/Relu] -> depthwise 3x3, group3, pad1, stride2:
8x8/C3 external and intermediate tensors, 4x4/C3 final output. Stem kernels1/3.
All twelve public binaries match the independently tested programs exactly.

Reproduction: research/probe_depthwise_stride2.py derives register hypotheses
from our previously compiled depthwise suite; expected outputs subsample its
independent stride1 integer reference. research/run_elementwise_suite.py
depthwise_stride2 streams the suite to the board. See ../depthwise_stride2_suite.log.
The public-ONNX equivalence test is tests/test_depthwise_stride.py.

All 41 host tests pass. The stride and reduced-output controls follow the dense
stride2 experiment; native depthwise output storage still reserves two lane planes.
Wider shapes/channels, other kernels and channel multipliers remain pending.
