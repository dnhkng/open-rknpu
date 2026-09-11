# Stride-2 dense Conv

Independent 1x1, 3x3 and 5x5 Conv at 8x8/C3 -> 4x4/C3 passed 96 board
inferences, 4,608 exact output bytes. Public compile --sequence emits identical
programs (host regression test). One Conv only; symmetric padding K//2, stride2,
dilation1, group1. Other shapes/channels and fused activation remain unverified.

research/probe_stride2.py constructs the independent tests. Its ONNX files are
stride-1 reference graphs; it patches independently emitted commands to stride2
and subsamples the integer reference. tests/test_strided.py constructs actual
stride-2 ONNX graphs and verifies public compiler byte equality.

The first attempt timed out: stride and downstream output dimensions alone were
insufficient. Preserved log: ../stride2_initial_timeout.log. The vendor oracle
../capture_stride2 revealed 0x1028=output width and 0x102c=output pixel count also
need updating. No capture is consumed by the public emitter. Final evidence:
../stride2_suite.log. Mesa/TRM stride fields at 0x1014 use literal 2 per axis.

Board staging uses the reusable mul_suite scratch slot. Old depthwise-suite board
copies were removed to make capture space; all source artifacts remain on host.
