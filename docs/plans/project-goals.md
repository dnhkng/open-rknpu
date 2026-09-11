# Goal: Open hybrid inference on RV1103/RV1106

## Status — triage pass complete, framework goal open

The ordered **triage pass** over [coverage-matrix](coverage-matrix.md) is
complete: every inventory entry now has a bounded public implementation with exact
RV1103 verification, or a specific retained failure/prerequisite. This does **not**
mean the framework goal is complete; many coverage boxes remain unchecked and the
remaining work is planned in [completion-plan](completion-plan.md). Earlier
dated milestone descriptions below are historical.

The pass covered the remaining geometry/tiling and channel modes, depthwise and
ConvTranspose options, weights and quantized-import contracts, Mul operands and
broadcasting, general graph scheduling and tensor/buffer APIs, numerical
conversion modes, fusions, and public LUT integration. Validate combinations
as each bounded profile is added.

Each item requires either a public implementation with independently generated
commands and exact board-reference verification, or a specific evidenced
limitation/blocker recorded before moving to the next item. An untested mode
or implementation prerequisite is not proof of a hardware limitation. Partial
passes and capture replay do not complete the goal.

Keep the checklist, ledger and investigation log current. Preserve the camera
service and board resources. Vendor tools, Ghidra and related-hardware references
may guide development; delivered compilation/runtime remain open and independent.
Quantization accuracy tuning stays deferred. The attached board's NPU is the shared
`rockchip,rv1106-rknpu` IP, so NPU-level RV1106 results are the ledger results
(`research/hardware_identity.md`); a distinct RV1106 SoC remains unclaimed.


## Latest status — 2026-09-11

Two utilisation items landed after the triage pass. **Mixed-engine DAGs now fit one
job:** the task tail's control word names the engine hand-off (`0x40` inside a CNA or
DPU run - but the board then showed that this word is really the **successor program's
fetch amount** (`PC_DATA_AMOUNT`), not an engine code: it is `0x40` before a 126-word
Conv, `0x14` before a 37-word pool and `0x28` before a 78-word elementwise task. With the
amount written at every transition a whole DAG is one job, so all the DAG emitters
(`diamond_tail`, `depthwise_join`, `pooled_dag`, `join_dag`, the composer and join-chain
ones) emit single-job containers for 4-9-task graphs, exact at about half the serial
minimum (`research/grouped_probe/`, `research/mixed_batched_probe/`). The driver's
`core_mask`/`subcore_task[]` fields are inert on RV1106 (one IRQ). **Height-strip tiling
now covers 3x3 chains:** the `K > 1` path shares one double-buffered surface per layer so
a strip can read its halo rows, and a 16-layer 3x3 chain plus a mixed 1x1/3x3 chain are
exact at tiles 1/2/4, serially and as one job (`research/tiled_k3_probe/`). That work
exposed `docs/plans/pipelining-plan.md` S9 - the chain family emitted its native hidden layers with
the activation off, so a hidden `Relu` after a native layer was dropped by both the
container and its reference. It is **fixed**: the layer carries its Relu, the composed
reference clamps the accumulator, and the four affected suites plus both tiled probes
were rebuilt and re-run exact. A container then also declares its engine runs in the
tails, so the runtime submits one ioctl per run (S10). The control word turned out to
be the successor program's `PC_DATA_AMOUNT` - a fetch size, not an engine code - so every
transition links and every DAG emitter threaded in this pass (`diamond_tail`,
`depthwise_join`, `pooled_dag`, `join_dag`, plus the composer profiles) emits a **single
job** for 4-9-task graphs, exact and at roughly half the serial minimum latency
(`research/grouped_probe/`).

## Latest status — 2026-09-10

The ordered triage/expansion pass is complete: every inventory entry now has either a
bounded public implementation with exact RV1103 execution or a specific retained
failure/prerequisite. The ledger contains 1,696 passing models, 28,266 inferences
and 10,216,467 exact output bytes (including the v5 two-head fan-out, sequence-calibration, native C33..128 (a vendor C128 Conv is one CNA task), N-layer chain, per-channel Mul, runtime-scale, join-chain fan-out, depthwise-branch, pooled-branch, mixed-head fan-out, dilated-transpose, runtime-scale, runtime-residual, general-join-DAG and multi-layer-branch, depthwise-chain, pooled-DAG, pooled-branch and deep-chain suites). New results include v4 runtime parameter updates,
bit-preserving quantized imports, depthwise-to-pointwise composition, per-batch
Mul constants, ordered activation/Mul chains, Mul Clip/Add successors and near-INT32
accumulator tests, plus `--per-channel-mul` per-channel constant quantization on the
1x1 depthwise profile. [Current results and blockers](../../research/COVERAGE_EXPANSION_RESULTS.md)
separate future DAG/ABI work and failed hardware hypotheses from supported modes;
[the completion plan](completion-plan.md) sequences the remaining work. The trained
model milestone now has two datapoints: MNIST (98.67% both-Conv NPU over 10,000
images) and Fashion-MNIST (88.15% over 10,000 images, `examples/fashion/`).


## Current objective (2026-09-08)

Deliver complete pretrained MNIST inference on the connected RV1103. Use the
existing open compiler and small C runtime for supported NPU operations, and
explicit C CPU fallbacks for the remaining operations. No proprietary compiler
or runtime is required to build or run the delivered pipeline. Vendor tools may
remain development references only.

The first complete path is Conv/Relu/MaxPool on the NPU, followed by the second
Conv/Relu/MaxPool and dense classifier on the CPU. Moving more work onto the NPU
comes after this baseline works. A model-specific generated example is acceptable
for this milestone; it must not be advertised as general ONNX support.

## Acceptance criteria

- Compile the trained model prefix and export fallback weights on the host using
  open tools; run the entire inference on the board with its camera service alive.
- Verify NPU intermediates against the integer reference and final logits against
  an independent ONNX evaluation of the suffix fed the same dequantized activation.
- Separately report quantization error and prediction versus the original float
  model. Fixture agreement does not establish dataset accuracy.
- Report CPU/NPU placement, measured inference latency, and memory requirements.
- Provide reproducible commands and fail clearly on unsupported model structures.

## Roadmap

1. **Current:** complete and verify the small hybrid MNIST example.
2. Integrate additional proven native Conv shapes; investigate pooling/dense
   offload only where useful. Keep CPU fallback visible in conversion reports.
3. Generalize the graph plan and buffer lifetimes from this working example.
   Keep ONNX parsing and conversion host-side; keep the board runtime small C.
4. Choose a useful classifier or detector from the vendor model zoo after checking
   its operators and actual memory requirements on this board.
5. Broaden coverage, quantify accuracy/performance, package the compiler/runtime,
   and validate a distinct RV1106 SoC separately before claiming SoC-level support
(the attached board already exercises the rv1106-rknpu NPU IP). Publication is later.

## Existing evidence and constraints

Independent int8 Conv and pooling command generation, a libc-only direct-ioctl
runtime, and explicit serial task sequences already exist. The trained first
Conv/Relu/MaxPool prefix has hardware verification. Complete hybrid inference is now verified on eight cases; see
examples/mnist/README.md for output comparisons, latency and memory results. The board has roughly 33 MB RAM shared
with its camera application; do not stop that service or fill RAM-backed /tmp.
Use the existing GPL kernel driver. Userspace source remains MIT licensed.

## OpenNPU review and reuse decision

Reviewed https://github.com/poad42/opennpu_rk3588 at
983282e3ebadad80c9a3c6062d7541c2c20bc112. Its C matmul register generation is a
useful reference. Its Python ONNX path uses fixed-shape captured templates,
hardcoded paths and subprocess/file exchange per NPU operation. ARM64 assembly
and the RK3588 DRM interface require porting. Reuse ideas and individually
validated formulas, not the whole runtime or assumptions about ISA compatibility.

## Non-goals for this milestone

A full ONNX operator library, ONNX Runtime on the board, full-model NPU placement,
LLMs, a new kernel driver, and performance parity with Rockchip. CPU fallback is
part of the design, not a failed conversion. Do not generalize RV1103 results to
RK3588 or a distinct untested RV1106 SoC.

## Current milestone result

Completed on 2026-09-08: full hybrid MNIST inference, exact NPU prefix outputs,
independent final-logit verification, runtime/memory measurements and reproducible
commands in examples/mnist/README.md. Broader roadmap items remain future work.

Second Conv offload is now verified in the opt-in MNIST native variant. It reduced
observed average inference time from 20.589 to 4.490 ms in interleaved tests, but
increased quantization error. Next priority is representative calibration and
held-out accuracy measurement before making it the default; see the example README.

Rough 100-image held-out check completed without tuning: float and CPU-Conv2 hybrid
100/100; both-Conv NPU with the analytic Conv2 range 13/100 (always predicts 5),
reproduced by integer reference. Calibrating the Conv2 output range (2026-09-10)
took the both-Conv NPU variant to 100/100 on the same 100 images at ~2.45 ms/image, and 98.67% on the full
10,000-image test set versus 98.90% for the float model, so full-NPU placement is
now accurate.

Primitive survey completed: 26/28 small graphs pass hardware-only captured replay;
Div uses CPU and Min is rejected by the available CPU fallback runtime. Evidence in
research/primitive_survey/README.md. Next engineering work is independent depthwise,
elementwise and activation-table generation, then public graph/runtime integration.
Do not equate this replay milestone with all-primitives compiler support.

Independent depthwise now implemented and verified for the fixed 8x8/C3 profile
through compile --sequence: 12 models, 192 cases, 36,864 exact bytes. See
research/depthwise_suite/README.md. Next primitive: elementwise Add/Mul/Sub/Max.

Independent Add completed for two 1x1 Conv branches at 8x8/C3 with shared branch
quantization: 12 graphs, 384 runs, 73,728 exact bytes. Runtime now accepts the
78-word elementwise descriptor. Next primitive: Mul; see research/add_suite/README.md.

Use Mesa Rocket register definitions and RK3588 TRM chapter 36 to guide remaining
primitives. Relevant chapter sections have now been inspected. Treat named fields
as hypotheses: compare RV1103 captures, then test independently generated commands.
Known layout differences must remain explicit; see research/hardware_refs/README.md.

## 2026-09-08: independent Mul milestone

The public `compile --sequence` path now supports two 1x1 Conv branches feeding
Mul at fixed `[1,3,8,8]`, without broadcasting. All three tasks execute on RV1103.
Both branches use symmetric INT8 activations with shared scale s; output scale
is 128*s*s and zero point 0. Integer arithmetic is nearest-even(A*B/128), clipped
to INT8. No calibration or float-model accuracy claim is made.

12 independently compiled models passed 384 board inferences and 73,728 exact
output bytes. All 37 host tests pass. Mesa/TRM EW_OP_TYPE and OD_BYPASS names guided
the profile; complete command behavior was checked on RV1103, without claiming
that each changed field has been isolated. See research/mul_suite/README.md.
Next primitive: Sub.


## Ordered remaining-mode work

User authorized sequential expansion of the missing Mul/Conv modes. The current
checklist and next task are in docs/plans/primitive-roadmap.md. Sub, Max, initial dense
stride2 and initial depthwise stride2 are now independently board-verified.
Broader modes remain explicitly pending; this does not complete the framework.
