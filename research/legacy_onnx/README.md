# Legacy ONNX leftovers

Three unreferenced ONNX files (`check0_base_optimize.onnx`, `check2_correct_ops.onnx`,
`check3_fuse_ops.onnx`, ~1.8 KB total) that sat at the repository root from an early
operator-fusion experiment. Nothing in the compiler, the tests or the docs references
them; they were moved here during the 2026-09-11 cleanup instead of being deleted, so the
experiment stays recoverable. They are **not** evidence for any supported profile.
