# open-rknpu completion plan

Status: 2026-09-09. This is the execution plan for the remaining framework work.
It supersedes the checkbox list in [primitive-roadmap](primitive-roadmap.md) as
the ordered queue; docs/plans/primitive-roadmap.md, [coverage-matrix](coverage-matrix.md)
and [research/COVERAGE_EXPANSION_RESULTS.md](../../research/COVERAGE_EXPANSION_RESULTS.md)
remain the inventory, evidence ledger and register-finding log.

## 1. Definition of done

Per item, the project's standing rule applies:

* **Accepted mode** — bounded public compiler profile, independently generated
  commands (no vendor capture, RKNN model or library in the emitted path),
  integer-reference verification on the connected RV1103, documented bounds, and
  a host regression test.
* **Rejected mode** — a specific retained failing experiment *or* a concrete
  documented hardware/ABI prerequisite. "Untested" is not "blocked".

Broader completion means all of the following hold:

1. General DAG graphs — fan-out, multiple consumers, multiple inputs and outputs,
   unequal runtime input shapes — beyond today's linear profiles.
2. Input channels above 64 (done: C1..128) and arbitrary intermediate channel
   counts (still the P1 DAG allocation).
3. ConvTranspose K5/general dilation, or a retained proof of the phase blocker.
4. Calibration available for every public profile, plus a held-out accuracy
   harness that separates integer exactness from float-model accuracy.
5. One useful trained classifier or detector running with measured accuracy,
   latency and memory.
6. The shared `rockchip,rv1106-rknpu` NPU IP validated on the attached board; a
   distinct RV1106 SoC validated or explicitly left unclaimed.
7. The compiler/runtime packaged and documented as a distribution.

Accuracy *tuning* stays deferred until the demo requires it; correctness of the
quantized arithmetic is not.

## 2. Verified baseline (do not regress)

* Board ledger: **1,696 models, 28,266 inferences, 10,216,467 exact output bytes**
  (`research/COVERAGE_EXPANSION_RESULTS.md`). Host suite: **311 tests**.
* Containers: legacy `ORNPUBIN` v1/v2; `ORNPUSEQ` v3/v4/v5 (`runtime/sequence_format.md`). Task descriptor kinds accepted today: Conv
  `29/768` (≤256 words), pooling `96/3072` (≤256), elementwise `24/768` (78 or
  1106 words for the per-channel scale profile), LUT `24/768` (1106).
* Loader bounds: ≤64 tasks, payload ≤1 MiB, arena ≤4 MiB, dims 1–1024, input
  C1–128 (native16; the emitted packed profiles use 1/3), other tensors C1–128.
* Board constraints: ~33 MB RAM shared with `rkipc`; stage files only under
  flash-backed `/oem` or `/userdata`; never fill `/tmp`; keep `rkipc` alive
  (`docs/board-access.md`).

## 3. Dependency graph

```
P0 hygiene ──► P1 DAG tensor ABI ──┬─► P4 elementwise/UQ modes
                                   ├─► P5 padding producers
                                   ├─► P6 LUT generalization
                                   ├─► P7 calibration + accuracy harness
                                   │        └─► P8 trained model demo
                                   └─► P9 RV1106 validation
P2 input C33..128 ─────────────────┘
P3 ConvTranspose K5 / dilation ────┘
P8 + P9 ──► P10 packaging/publication
```

**P1 is the keystone.** Fan-out, multiple Mul, unequal inputs, multiple outputs,
pool/depthwise branches and tensor lifetimes all wait on a named-tensor ABI; most
other phases become integration work once it exists.

## 4. Phases

### P0 — Ledger and documentation hygiene (machine-checked as of 2026-09-10)

*Reconcile stale bounds and completeness claims across the markdown; make the
checklists and ledger agree with `src/` and `runtime/`.*

- Deliverable: reconciled `README.md`, `docs/plans/primitive-roadmap.md`,
  `docs/plans/coverage-matrix.md`, `docs/plans/project-goals.md`,
  `runtime/sequence_format.md`, and this plan; every remaining item traced to a
  ledger row or a retained failed artifact.
- Acceptance: no doc states a bound contradicted by `native.py`, `sequence.py`,
  `model.py` or the ledger; historical sections are labelled historical.
- **Machine-checked:** `tests/test_ledger.py` verifies that the rows sum to the
  stated totals, that every ledger link resolves, that board evidence reproduces every
  row that has it (the union of per-model passing entries across all
  `board_results_*.json` files, 97 rows), that manifests reproduce their rows where
  the schema records case and byte counts (11 rows), and that the eleven campaign
  suites are pinned by name and total. The audit corrected one stale row
  (`native_large_suite`, 42/672/738,960 -> 44/704/800,800).

### P1 — DAG scheduler and tensor/lifetime ABI (keystone)

*Replace the single-output, ≤2-input linear contract with a named-tensor graph
contract so fan-out, multiple consumers and multiple outputs can be scheduled.*

**Progress (2026-09-10):** format v5 named-tensor table, `ornpu_get_tensor` and
`ornpu_run_io`, and a two-head fan-out emitter are implemented and board-verified
(14 models, 448 inferences, 172,032 exact bytes; `research/two_head_suite/`).
A general N-layer native chain (`[Conv,Relu]*(N-1)+[Conv]`, N=3/4) with one shared
arena and explicit intermediate lifetimes is board-verified (5 models, 80
inferences, 15,360 exact bytes; `research/native_chain_suite/`), extending the
fixed two-layer `chain.py`. Chain intermediates can be exposed as named v5
external outputs (`--expose-intermediates`): 4 models, 32 board inferences and
40,448 exact bytes across 3- and 4-output chains. A first lifetime allocator
reuses chain intermediate buffers by ping-pong (`--reuse-intermediates`), halving
live intermediates for 4+ layers with identical board output. Multi-operator elementwise DAGs are board-verified: two-stage
`Mul({Add,Mul,Sub,Max}(a,b),a)` (12 models, `elementwise_dag_suite/`) and N-stage
`Mul(...Mul(Op(a,b),a)...,a)` up to four stages with six tasks (12 models,
`elementwise_deep_suite/`), all with two external inputs and fan-out.

**Fan-in and a real lifetime pass (2026-09-10).** `open_rknpu.liveness` now
implements the plan's step 2 generically: producer-before-consumer order, live
intervals over task positions (a task that reads and writes a buffer conflicts
with itself, so in-place execution is never assumed), and a first-fit arena
allocator that reuses bytes only where the intervals are disjoint. The new
diamond profile `stem -> {head_a, head_b} -> {Add,Mul,Sub,Max}` emits its arena
from that pass and is board-verified (**12 models, 384 inferences, 73,728 exact
bytes**, `research/diamond_suite/`): the stem is live 0..2, the head outputs 1..3
and 2..3, and the external output is placed after every internal tensor because
v5 rejects internal/external overlap. `tests/test_liveness.py` proves the same
allocator reproduces the two-buffer ping-pong of long chains and refuses in-place
reuse. Per-channel Mul quantization also closed on the depthwise route (P4).
**Emitter composition by name (2026-09-10).** The diamond now supports a
`[Conv, Relu]* Conv` tail whose first layer consumes the join output as an internal
named tensor, so the elementwise join emitter and the native Conv emitter are
composed through the v5 table rather than matched as one pattern: **12 models, 192
inferences, 36,864 exact bytes** (`research/diamond_tail_suite/`), with the join and
tail lifetimes placed by `open_rknpu.liveness`. Two concrete bugs were found and
fixed on the way: the tail program must start after the 82-word join program, and
each tail layer must declare the input zero point it actually reads in `0x1184`.

**Unequal runtime inputs (2026-09-10).** `Mul(image[1,3,H,W], scale[1,3,1,1])`
now binds the per-channel operand as a named external tensor of shape `(1,1,1,3)`:
the runtime packs three bytes into the 16-byte operand row the verified elementwise
per-channel mode already reads, giving a runtime affine modulation. **16 models, 256
inferences, 32,832 exact bytes** (`research/runtime_scale_suite/`) across four
geometries and four operand patterns. This retires the plan's "unequal logical
inputs" and "complementary singleton axes with two runtime operands" blockers; the
dispatch keeps `Mul(a,b)` with two matching RGB inputs on the verified standalone
path and still rejects an overridable constant masquerading as an input.

**Variable fan-out and chained joins (2026-09-10).** A shared 1x1 Conv stem now
feeds **three to five heads**, and `n-1` elementwise joins fold them left to right
with mixed `Add`/`Sub`/`Max`/`Mul` kinds before an optional `[Conv, Relu]* Conv`
tail. A Mul join folds two free operand scales; an Add/Sub/Max join requires a
shared scale, so the next head is re-quantized onto the running result. The task
order and arena still come from `open_rknpu.liveness`, and each head declares the
stem zero point in `0x1184` (fixing the same border-injection bug for a stem
without Relu). **12 models, 384 inferences, 73,728 exact bytes**
(`research/join_chain_suite/`), with at least 95% nonzero expected outputs per
model so the board comparison cannot pass vacuously. This closes the plan's
"multiple Mul" item.

**Depthwise branch into the same join (2026-09-10).** A shared 1x1 stem now also
feeds a **group-3 depthwise branch** next to a dense branch, joined by
Add/Sub/Max/Mul. The depthwise task is the verified standalone program with its
four address registers and input zero point relocated into this container's arena;
its weight/bias blocks (including the asymmetric per-channel weight zero points)
are copied verbatim. **12 models, 384 inferences, 73,728 exact bytes**
(`research/depthwise_join_suite/`) across depthwise kernels 1/3/5, dense kernels
1/3, all four joins, a stem without Relu and two asymmetric models. This retires
the plan's "multi-task pool/depthwise branches into Mul" blocker for depthwise; a
pooling branch still needs its own relocation.

**Pooled branches into the same join (2026-09-10).** Both halves of the plan's
"multi-task pool/depthwise branches into Mul" blocker are now closed: two Conv
branches followed by 2x2 stride-2 MaxPool/AveragePool are joined after the pool,
using the *same* pool register builder as the sequence lowering. The six tasks are
ordered and placed by `open_rknpu.liveness`, and the join runs at 4x4 with a
256-byte plane stride. **12 models, 384 inferences, 18,432 exact bytes**
(`research/pool_join_suite/`) across both pool kinds, all four joins, 1x1/3x3
heads, hidden 3/8/16 and a stem without Relu, with at least 96% nonzero expected
outputs per model.

**Mixed emitter families in one fan-out (2026-09-10).** `compile_join_chain` now
accepts **dense or group-3 depthwise heads at any position**, so the native Conv
emitter and the verified depthwise emitter are composed by tensor name inside one
fan-out: the depthwise head's standalone program is relocated into the shared
container exactly as `depthwise_join` does, and per-head quantization metadata is
tracked per family. **12 models, 384 inferences, 73,728 exact bytes**
(`research/mixed_head_suite/`) across 3–5 heads, all four join kinds, mixed kernel
sizes, an asymmetric-depthwise model set and a Conv tail; the existing all-dense
`join_chain_suite` still recompiles byte-identically.

**Runtime input inside a fan-out (2026-09-10).** The join chain accepts a final
`Mul(result, scale)` whose operand is a declared `[1,3,1,1]` graph input, applying a
runtime per-channel gain to the folded DAG result with the verified per-channel
elementwise program. **12 models, 384 inferences, 73,728 exact bytes**
(`research/join_scale_suite/`) across 3..5 dense/depthwise heads, all four join
kinds, five code patterns and a no-Relu stem. Two hardware findings came out of it
and are retained in the suite README: a scaled primary whose arena slot had been
reused by an earlier Conv task returns stale data on the board (this profile places
its internals sequentially instead), and the folded scale product is not exactly
`1/128` in float32, so the reference must use the hardware requantization formula
rather than `rint(a*b/128)`.

**Runtime residual feature map (2026-09-10).** The same tail slot accepts
`Add/Sub/Max(result, residual)` where the residual is a declared `[1,3,8,8]` input
interpreted as zero-centered INT8 on the join scale, giving a runtime residual
connection into a computed DAG with the verified join emitter.
**12 models, 384 inferences, 73,728 exact bytes** (`research/join_residual_suite/`)
across all three tail kinds, 3..5 dense/depthwise heads, kernel sizes 1/3/5, five
code patterns and a no-Relu stem.

**General join expression (2026-09-10).** A new `open_rknpu.join_dag` emitter
accepts two or three joins over **any two previously produced tensors**, so a head
or an earlier join result can feed several consumers — the case the left-fold chain
matcher declines. One INT8 scale is propagated per tensor in topological order: a
Mul join folds two free operand scales and commits both, while Add/Sub/Max require a
shared band and re-quantize an uncommitted head operand, rejecting a tensor that a
later join wants on a different band with a specific message. `open_rknpu.liveness`
keeps a reused grid live across all of its consumers. **12 models, 384 inferences,
73,728 exact bytes** (`research/join_dag_suite/`) across 2- and 3-join expressions,
dense/depthwise head mixes and kernel sizes 1/3.

**Multi-layer branches (2026-09-10).** A branch in the join DAG may now be a chain
of one to three dense Conv layers (a residual-style block) with its own channel
counts: each layer's native fields, weight/bias blocks and input band are emitted for
that layer, with intermediates keeping their natural quantization and finals
zero-centered (or re-quantized onto a shared band when an Add/Sub/Max join demands
one). An intermediate layer that keeps three channels may also feed a join.
**12 models, 384 inferences, 73,728 exact bytes** (`research/branch_join_suite/`).
Arena placement for this profile is conservative (a fresh slot per internal): with
reuse the board returned stale data for a mixed dense/depthwise task family, the
second independent sighting of that hazard, so `join_dag_suite` was regenerated and
re-verified under the same rule.

**Depthwise layers inside branch chains (2026-09-10).** A depthwise layer may now sit
inside a multi-layer branch. The dedicated depthwise emitter only models a branch that
reads the stem directly, so a chained depthwise layer is rewritten to an equivalent
block-diagonal dense kernel and emitted through the per-layer dense path (every added
tap has weight zero, so the rewrite is exact in the quantized domain). Single-layer
depthwise branches keep the dedicated emitter, so the earlier suites stay
byte-identical. This covers depthwise-separable blocks (`Conv -> DW -> Conv`) inside a
DAG: **12 models, 384 inferences, 73,728 exact bytes**
(`research/depthwise_chain_suite/`).

**Terminal pooling on a DAG result (2026-09-10).** The join DAG accepts a final 2x2
stride-2 MaxPool/AveragePool, so `branches -> joins -> pool` compiles as one
container: the pool task comes from the shared pool register builder, the pooled grid
keeps the join's band (only the header geometry changes), and every internal keeps a
fresh arena slot. **12 models, 384 inferences, 18,432 exact bytes**
(`research/pooled_dag_suite/`) across MaxPool and AveragePool, 3-4 branches, two or
three joins and depthwise layers inside the chains.

**Pooled multi-layer branches (2026-09-10).** A pooled join's branches may now be Conv
chains of one to three layers (a depthwise layer inside a chain expands exactly to a
block-diagonal dense kernel), so classifier-style `Conv -> Conv -> pool` branches fold
at 4x4. The join band propagation runs over the branch finals because pooling preserves
the band. **13 models, 416 inferences, 19,968 exact bytes**
(`research/pooled_branches_suite/`); the profile is dispatched after `pool_join`, so the
earlier two-single-Conv containers stay byte-identical.

**Task bindings derived and verified (2026-09-10).** `tests/test_container_bindings.py`
derives each task's read/write set from the address registers of every container in the
repository (2,412 models, ~1.5 s) and checks that every read is a declared external input
or a tensor written by an earlier task, that every write targets a declared internal or
the output, that tensor-table sizes and roles are consistent, and that no internal
overlaps an external; legacy v3/v4 and ORNPUBIN headers are checked too. The derived
view found a real defect: a pooled DAG's last join wrote an internal slot that was
missing from the v5 tensor table (fixed; `pooled_dag_suite` regenerated and board
re-verified).

**Stage composer and declared bindings (2026-09-10).** The last open P1 item is now
half closed and the mechanism is in place. `open_rknpu.compose` turns a *declared*
stage list into a v5 container in one pass: `open_rknpu.liveness` orders the tasks
topologically and places the arena (with a `reuse=False` fresh-slot policy for
profiles that avoid arena reuse), program slots and constant blocks are assigned in
declared order, each stage's own `fields(addresses, constants)` is substituted with
the planned addresses, and the tensor table and task list are generated from the
declaration. Every stage publishes its address-register bindings, so the container
carries the emitter's declared view in `meta['declared_bindings']` while
`compose.derive_bindings` reads the *same* container back from its address registers;
`check_declared_bindings` proves the two agree (offset-exact, and register sets
identical), and the task-family table (`FAMILY_BY_SIGNATURE`) is now shared by the
composer and `tests/test_container_bindings.py` instead of being hand-listed there.
Two profiles are ported onto the composer and **recompile byte-identically**:
`pool_join` (12 models, arena reuse; `tests/test_pool_join.py`) and
`pooled_branches` (13 models, now on the composer's `reuse=False` fresh-slot policy;
`tests/test_pooled_branches.py`). Both suites were re-run on the board after the port
(12/384/18,432 and 13/416/19,968 exact bytes), and `tests/test_compose.py` covers the
pass itself (out-of-order declaration, address substitution, fresh-slot placement,
declared/derived agreement on both profiles, and negative declaration/cycle tests).

Still open in P1: the remaining monolithic emitters (`join_dag`, `join_chain`,
`diamond`) still assemble their own layout by hand, because their placement rules
predate the composer (per-branch depthwise programs copied from a standalone
container, an adjacency constraint on the diamond join's head buffers, a
left-fold-only chain matcher); porting each one is mechanical and checked the same
way (byte identity over its suite plus the declared/derived check). A single topological walk that dispatches *every* normalized
op to a stage emitter also waits on that: the composer is the pass, but each emitter
must first be expressed as a stage producer rather than a whole-container builder.

Steps:

1. **Format v5.** Add a tensor table: `(name, data_type, layout, shape, offset,
   size, role)` for every external and internal tensor, plus per-task input/output
   bindings and a submission order. Keep v3/v4 decoding unchanged; `ORNPUBIN`
   profiles 1–8 untouched. Define arena lifetimes / aliasing rules explicitly.
2. **Lifetime allocator.** Topologically order tasks; compute liveness; reuse
   arena bytes only where the existing disjointness rule can be proven safe
   (in-place execution is *not* assumed — see the aliasing non-goal).
3. **Runtime API.** Add tensor-descriptor enumeration and per-tensor access
   (`ornpu_get_tensor` / `ornpu_set_tensor` style) alongside `ornpu_get_info` and
   the v4 constants API; keep ABI additions backward-compatible.
4. **Scheduler.** Replace the hand-ordered dispatch in `scheduler.py` with a
   topological pass over normalized ONNX, dispatching each op to its existing
   emitter and binding producer/consumer edges by name.
5. **Regression.** Re-emit representative existing suites through the new path and
   prove byte-identical or reference-identical payloads before enabling new graphs.

- Deliverable: multi-input / multi-output, fan-out and multi-Mul graphs compile
  and run on RV1103.
- Acceptance: existing ledger suites still pass; new fan-out and two-output
  suites pass with exact bytes; malformed tensor tables rejected by Python and C
  loaders in parity tests.
- Retires blockers: general graph scheduling; multi-task pool/depthwise branches
  into Mul; multiple external inputs/outputs; unequal logical inputs.
- Risk: format churn. Mitigate by versioning and by never mutating v3/v4.

### P2 — Input channels C33..128 (done 2026-09-11)

*Remove the C32 input cap without changing ONNX semantics.*

**Done (2026-09-10): C1..64 public and board-verified.** The earlier "nonlinear
swizzle" was a mis-modelled plane grouping. New vendor marker captures
(`research/fixtures/native_c48_*`, `native_c64_channels`) with one nonzero weight
per (tap), (input channel) and (output channel) recovered the layout: within each
16-output block, the first two 16-lane input planes are stored per tap, then the
remaining planes. `research/build_native_c48_suite.py` and
`build_native_c64_suite.py` compile independently generated models; the board
passed **24 models, 384 inferences, 142,656 exact bytes** across input
C33/40/48/49/56/64, output C1/3/8/16, K1/K3 and two geometries. `tests/test_native_c48.py`
reproduces every marker set, and the loader bound is raised to 64 in Python and C.

- **Extended to C65..128 (2026-09-11).** A vendor C128 Conv is a *single* CNA task
  (`SUBMIT rc=0 size=104`, `NC1HWC2` input `1,8,6,5,16`), so no channel-split
  accumulation or INT32 partial-sum surface is involved: the earlier C64 bound was
  ours. The layout is the 32-lane-part generalization recovered from
  `capture_native_c65_channels`, `capture_native_c65_oci` and
  `capture_native_c128_channels` (1,105 marker cells incl. the second 16-output
  block). `research/native_c65_suite/` board-verifies **20 models, 320 inferences,
  127,744 exact bytes** over input C65/80/96/128, output C1/3/8/16/17, K1/K3 and
  two geometries; the Python and C loader bounds are raised to 128.
- Remaining: input above C128 exceeds the tensor-descriptor bound and the vendor
  lane field we have measured; intermediate channel planes above 16 in scheduled
  graphs still need the P1 DAG allocation.

### P3 — ConvTranspose K5 and general dilation (K5 done 2026-09-10)

*Recover the phase arithmetic that makes direct K5 fail.*

**Done: direct depthwise K5 is accepted and board-verified.** The recipe came from
re-deriving the vendor capture (`research/analyze_k5_vendor_layout.py`): a 25-tap x
32-byte table of `(value, -weight_zero_point)` lane pairs with asymmetric
per-channel weight quantization, the vendor register set (including an explicit
output conversion and `0x1010 = 0x3ff`), and the phase field
`0x1068 = ((k-1-pads_left)<<8) | (k-1-pads_top)`. That phase formula was measured
directly on the board for pads=1 (`0x101` -> 1303 mismatches, `0x202` -> 1773,
`0x303` -> **0**) and matches every vendor capture, including an unequal-stride one.

**Verified on RV1103: 9 models, 144 inferences, 109,872 exact output bytes**
(`research/transpose_k5_suite/`) across stride 1/2, pads 0/1/2, output_padding,
a rectangular per-axis geometry, and C1/C4/C8. The pre-investigation artifacts are
retained under `transpose_k5_suite/stale/`: they packed 17 of 25 taps with a
reversed kernel, which is why the historical "98/675" record could not be
reproduced.

**Depthwise K3-dilation2 emitted directly as sparse K5 (2026-09-10).** A depthwise
`ConvTranspose` with `kernel_shape=[3,3]` and `dilations=[2,2]` has a five-tap
support, so it is the same operator as a K5 kernel whose odd positions are zero. The
rewrite expands `(C,1,3,3)` into `(C,1,5,5)` with the dilated taps in place and
lowers to `kernel_shape=[5,5], dilations=[1,1]`, reusing the verified depthwise K5
task (asymmetric weight zero points and the vendor phase field) instead of rejecting
the mode. **8 models, 128 inferences, 82,432 exact bytes**
(`research/transpose_k5_dilation_suite/`) across stride 1/2, pads 0/1/2,
output_padding, a rectangular per-axis geometry, C1/C3/C4 and SAME_UPPER; a host test
additionally proves the dilated K3 and zero-filled K5 scatters agree exactly in the
float domain.

**Dense K3-dilation2 (2026-09-11).** The rewrite above now takes dense weights too
(`(C_in, C_out/group, 3, 3)` zero-stuffed to `(C_in, C_out/group, 5, 5)`), and the dense
emitter derives its kernel size from the weights instead of hard-coding K3, so the
zero-stuffed graph reaches the verified dense-K5 form. Dilation and the dense path are
mutually exclusive in the dispatch (a dilated node takes the rewrite first and comes back
with unit dilation). `research/transpose_dilation_dense_suite/` verifies four dense
configurations (C4->C3 stride1 pads0/pads1, C8->C5 stride2, C3->C3 pads2; 4 models, 32
inferences, 18,952 exact bytes) with two host guards: the original and zero-stuffed
graphs agree as float operators, and the integer reference agrees between the dilated
scatter and the 25-tap scatter. `transpose_dense_suite`, `transpose_dilation_suite` and
`transpose_k5_suite` recompile byte-identically.

Still open: a dense K5 field set beyond this rewritten geometry would need a vendor
capture, and per-axis dilation stays rejected because the K5 tap table is square.

- Acceptance: met - native K5 exact on independently generated probes, with no
  regression to the passing K1/K2/K3 and sparse-rewrite suites (all 66 existing
  transpose models still compile byte-identically).### P4 — Remaining elementwise and quantization modes (done)

*Close the per-channel / broadcast / asymmetric-weight gaps.*

1. **Per-channel Mul quantization — done (2026-09-10), with the hardware route
   refuted, then re-probed cleanly (2026-09-11).** The new-hardware-mode option was
   probed first: `BS_OW_CFG.OW_SRC=1` plus a `0x5020` operand table (Mesa names the
   register `RDMA_BS_BASE_ADDR`) hung the NPU on its first run. That probe had two
   construction errors (the table was written at the *file* offset while `0x5020`
   names a *payload* offset, and it used a four-UINT16 layout instead of the decoded
   Conv block). With both fixed, `OW_SRC=1` **still hangs** (`failed to wait job` /
   `job timeout` / `soft reset`; `rkipc` survived), while the same task with
   `OW_SRC=1` *and* `OD_BYPASS=1` completes and **ignores the table**: a well-formed
   block whose channels 1-2 carry a zero multiplier leaves the output byte-identical
   to the baseline. Clearing the bypass is not itself the problem - the DAG suites'
   two-surface elementwise tasks run at `0x30000000`. So the elementwise output stage
   has no BS-table read and `OW_SRC=1` parks the datapath; retained with all six
   variants and board records in `research/mul_per_channel_ow_suite/`. Channel-split
   assembly is therefore the accepted route, but it needs no per-channel-scoped
   task: `--per-channel-mul` lowers `Mul(input, [C,1,1])` onto the verified 1x1
   depthwise profile, whose native per-output-channel weight scale already
   quantizes each channel on its own grid (`[.02,.35,1.9]` keeps `127` operand
   bytes instead of one shared `[1,23,127]` scale). 12 models / 192 board
   inferences pass with exact bytes (`per_channel_mul_suite/`), and the float-domain
   error is 2.2x-4.7x smaller than the shared-scale EW path. The output grid stays
   one per-tensor scale: a per-channel *output* conversion is what the hung
   register probe would have provided, and the INT8 container carries one scale.
2. **Spatial broadcast without materialization — refuted, with the hang decoded.**
   The RK3588 TRM for the same NPU IP names the two registers the 2026-09-10 sweep
   was missing: `0x506c` `ew_surf_notch` ("pixels from the end of this operand to the
   end of the output") and `0x5010` bits 28:16 `ew_line_notch_addr`, plus the full
   `0x5034` decode (data_mode 0 per channel / 1 per pixel / 2 per channel by pixel,
   surf_mode, data_size, erdma_disable) and `0x5040` as a stride **in 16-byte atoms**.
   Setting the notch turns every retained compact-operand hang into a completed run
   (`research/mul_broadcast_notch_suite/`, six variants), and the completed reads are
   **byte for byte the linear read of the compact table** - one 16-byte atom per
   output pixel, unchanged by stride 16/576, notch 0x40/0x50 or `surf_mode`, and
   `data_mode=2` still hangs. There is no native spatial broadcast; spatial Mul
   constants stay materialized, which is what the public emitter emits.
3. **Asymmetric depthwise weight zero points — done (2026-09-10).** The field is
   the weight pair's second byte, storing `-zp` (it is not register `0x4054`, which
   the earlier hypotheses changed). `compile --sequence --asymmetric-depthwise`
   board-verified 27 models / 432 inferences across C1/C3/C4, K1/K3/K5 and
   positive/negative/mixed weights. The symmetric default is unchanged.
4. **Signed/unsigned output formats** — keep INT8-only unless a verified wider or
   external-INT8 path is found.

- Acceptance: each sub-item is either a passing bounded profile or an evidenced
  blocker; no silent fallback.

**P4 closed (2026-09-10; both hardware-mode negatives re-probed 2026-09-11):**
per-channel Mul quantization passes on the 1x1 depthwise profile (12 models / 192
inferences, error 2.2x-4.7x lower), spatial broadcast is refuted with the decoded
notch/stride fields and a measured linear-read proof, asymmetric depthwise is done, and
signed/unsigned output stays INT8-only with the ABI reason recorded. The per-channel *output* conversion is closed as a measured
negative with the field isolated: `OW_SRC=1` hangs the elementwise task with a
well-formed decoded Conv block at the address `0x5020` names, and with `OD_BYPASS=1`
the same task completes while ignoring the block (byte-identical output).

### P5 — Border/padding producers (done via host preprocessing)

*Support reflection, replication and circular padding.*

**Done (2026-09-10):** the compiler folds a leading `Pad` into the graph input
shape and records `input_padding` (mode, amounts, padded shape); callers pad with
`open_rknpu.padding.pad_input`. Constant, reflect, edge and wrap all passed 12
models and 192 board inferences (`padding_suite/`). A dedicated NPU copy/pad
producer for non-constant borders is not implemented — the CNA border path injects
only the activation zero point — and the plan explicitly permits this documented
host-preprocessing outcome.

### P6 — LUT activation generalization

*Move past the identity/32 C3 8x8 probe.*

**Progress (2026-09-10): the domain is measured; the accepted stem family is
broader; mixed-input stems stay blocked.** The 1026-entry table splits by sign:
the positive half advances one entry per 1/64 of the dequantized stem value, the
negative half one entry per 1/64 of `x·(BASE_WEIGHT_SCALE/weight_scale)`. Sampling
the negative half with the inverse gain makes a diagonal C3 1x1 stem with one
scalar gain per channel (per-channel signs, any bias, gain scaling the domain by a
power of two, range inside the sign-split window) exact: **12 models, 192
inferences, 36,864 exact bytes** (`research/lut_domain_suite/`), including all 256
input codes and mixed inputs. Declared-scale probes also showed that only the
positive half follows the declared output scale, so the profile keeps it fixed at
1/2048. **Refined (same day): the gain is a whole-bit shift.** A board sweep shows
`H = 2**ceil(log2(BASE_WEIGHT_SCALE/weight_scale))`, uniform across channels and
independent of bias and declared scale (`research/lut_mixed_gain_probe/`; `H`
measured as 0.5/1/2/4 where predicted, including 0.02 -> 2 and 0.03 -> 2 where the
ideal ratio is 1.5625 and 1.0417). Mixed and non-power-of-two bands stay rejected
because their argument *grid* rounds differently, leaving ±1 on ~10% of values
(321-394 of 3072 for diagonal 0.02/0.024/1/48/1/24 and 372 for a mixed stem), so
`compile_lut` still requires `BASE_WEIGHT_SCALE/weight_scale` to be a power of two.
The accmulator-to-index rounding step is the remaining unknown; the
interpolation/domain-calibration fields beyond the two banks are still undecoded.
- Acceptance: a validated stem family drives a Sigmoid/Tanh LUT with exact
  reference agreement across all 256 codes and mixed inputs (done for the diagonal
  family); arbitrary mixed stems need a per-channel table (P1 channel split) or a
  per-channel gain formula.

**Index readout (2026-09-10): the residual is measured, not inferred.**
`research/lut_index_probe/` replaces the LUT table with a sign-aware output-code
ramp (`clip((neg_pivot-i)*128)` / `clip((i-pos_pivot)*128)` in Q15, four
`(neg_pivot, pos_pivot)` windows per stem), so the hardware's table index is read
directly at every input code and channel. The Q15 -> code conversion
`clip(round(v*255/32768) - 128)` was calibrated on the control stem, whose index
`512 + 2q` is already verified, and reproduces all four of its windows exactly
(768/768). Result: the slope model is right (positive slope `64·g`, negative
`64·H·g`; the measured negative/positive slope ratio matches the predicted
`H = 2, 2, 2, 2, 4` within 0.5%), the residual is exactly one ±1 *index* step on
~17% of the readable codes, and **no affine fixed-point rule**
`floor((A·q+B)/2**J)` with `J <= 14` reproduces the measured index for any
non-power-of-two stem (the control has many solutions; every other band has none).
The hardware argument therefore comes from a quantized intermediate whose rounding
these probes do not expose, so the band stays rejected - now with a direct
measurement retained in `analysis.json` and re-checked by
`tests/test_lut_index_probe.py`.

### P7 — Calibration and accuracy harness

*Make calibration available where the compiler accepts graphs, and separate
integer exactness from float accuracy.*

**Progress (2026-09-10): steps 1–2 done.** `calibration.measure` no longer requires
the legacy compiler; `--sequence --calibration` works for profiles with an
output-range contract (single Conv[/Relu/Clip], Conv-ReLU-Conv with per-layer
ranges, elementwise/Mul) and rejects the rest with a clear error.
`open_rknpu.accuracy` reports per-layer round-trip error, final-output error versus
the float ONNX model and classification accuracy. Board-verified:
`sequence_calibration` = 6 models, 96 inferences, 77,824 exact bytes, plus an
analytic-vs-calibrated report showing calibration lowered MAE on all three graphs
but raised maximum error on two. Step 3 (percentile/KL) is **done**: `calibration.measure`
selects ranges by min/max, by an upper-tail percentile or by a TensorRT-style KL
saturation search, and `research/percentile_calibration_suite/` measures the trade-off
(bulk error against clipped extremes) with all four variants board-exact.

1. ~~Extend `calibration.py` beyond the legacy `compile_model` path to sequence
   profiles.~~ Done.
2. ~~Add a held-out evaluation harness reporting per-layer quantization error,
   final-output error versus float and classification accuracy.~~ Done.
3. ~~Consider percentile/KL calibration only if min/max proves insufficient.~~ Done
   (`calibration.measure(method="percentile"|"kl")`, `tests/test_calibration.py`);
   the suite shows percentile/KL halving the bulk error against min/max while clipping
   the saturated extremes.

- Acceptance: sequence compilation accepts calibration (done); the harness
  reproduces a synthetic graph's analytic-vs-calibrated comparison (done) and the
  MNIST hybrid numbers (pending P8).

### P8 — Trained classifier/detector demo

*Produce the useful-model milestone the goals file describes.*

1. Keep the hybrid MNIST baseline as the correctness reference; optionally retrain
   or requantize so the both-Conv NPU variant is accurate (currently 13/100,
   always predicts 5).

   **Done (2026-09-10):** `build.py --native --calibrate` measures the trained
   Conv2 output range on 256 held-out calibration images (output scale 35.303
   analytic -> 0.13411 calibrated). Board-verified on the same 100 held-out
   images: **both-Conv NPU 13/100 -> 100/100**, matching the float model, at
   ~2.45 ms/image with the camera service running. The **full 10,000-image test
   set** was streamed to the board in chunks and scored: **calibrated both-Conv
   NPU 98.67%** versus **float 98.90%** and **analytic 8.92%**, with 9,922/10,000
   agreement with the float model. Integer outputs match the independent host
   reference. See `examples/mnist/README.md`, `examples/mnist/full_dataset.py`
   and `examples/mnist/accuracy.py`.
2. ~~Pick a small vendor-zoo classifier/detector whose operators are within reach
   after P1/P2/P6; verify memory before committing.~~ **Done (2026-09-10) with a
   locally trained second model.** `examples/fashion/` trains the same pinned graph
   on Fashion-MNIST (6,994 parameters, MIT dataset and code) and compiles the three
   placement variants. Full 10,000-image board results: hybrid **88.15%**,
   both-Conv analytic **88.14%**, both-Conv calibrated **88.15%**, against a float
   **88.18%**; agreement with the float model 9,979 / 9,862 / 9,957 of 10,000. The
   calibrated Conv2 range is `[-9.943, 10.812]` (output scale 0.08139, zero point
   -6). A vendor-zoo detector was not used: the hardware profiles in reach cover
   Conv/pool chains, and a detector would need operators still blocked (P3/P6), so
   a trained classifier was the honest second model.
3. ~~Report placement (NPU/CPU), latency and memory with the camera service alive.~~
   **Done:** both-Conv calibrated **1.54 ms/image** end-to-end (1.448 ms NPU +
   0.089 ms CPU) and 536 KiB peak RSS for a 1,000-image stream; the hybrid variant is
   CPU-bound at 18.0 ms/image (8.47 ms NPU + 9.54 ms CPU). Placement is explicit per
   variant, and the first run after model load costs ~14 ms/image (NPU bring-up).

- Acceptance: measured held-out accuracy, reproducible commands, explicit
  placement report, and clear failure on unsupported structures. Step 1 now
  satisfies this for the full 10,000-image test set; a second (detector/zoo)
  model remains open.

### P9 — RV1106 validation

The attached Luckfox board's NPU node (`npu@ff660000`) reports compatible
**`rockchip,rv1106-rknpu`** — the shared RV1106 NPU IP — so every board result in
the ledger already executes on the RV1106 accelerator block. P9's NPU-level
validation is therefore satisfied by the ledger; the hardware identity is recorded
in [research/hardware_identity.md](../../research/hardware_identity.md).

- Remaining: a *distinct* RV1106 SoC (CPU, memory map, peripherals, driver image)
  if one becomes available. The attached board's SoC compatible is
  `rockchip,rv1103g-38x38-ipc-v10`/`rockchip,rv1103`; do not extrapolate its
  non-NPU results to a different SoC.
- Acceptance: hardware identity recorded and representative suites re-verified on
  the attached board (done); a distinct RV1106 SoC validated or explicitly left as
  a non-claim.

### P10 — Packaging and publication

- Wheel/versioning, runtime rebuild instructions, license audit (MIT sources vs
  GPL driver reference and vendor dev binaries excluded), reproducible commands.
- Acceptance: clean-environment install compiles a byte-identical executable and
  passes host tests, as recorded for the earlier wheel.

**Progress (2026-09-10, re-verified after the campaign): packaging verified.** The
wheel and sdist (`open_rknpu-0.1.0.dev0`) carry the **37 compiler modules** — including
the campaign's `pool_join.py`, `depthwise_join.py` and `join_dag.py` — plus the
libc-only runtime sources under `share/open-rknpu/runtime/`, and exclude `research/`,
tests and vendor binaries. In a clean venv the wheel installs offline and
`open-rknpu compile --sequence` reproduces byte-identical executables for the diamond
fan-in, the v5 two-head fan-out, the general join DAG and the multi-layer branch
profile; `inspect`, `normalize` and `--version` all work. The sdist builds and
compiles byte-identically as well, with `--no-build-isolation` because the declared
`setuptools>=77.0.3` build requirement cannot be fetched offline on this machine.
Publication (an index release and announcement) remains a user decision; the
technical distribution steps are done. Re-verified 2026-09-11 after the native
channel, composer and chain-walk work: the wheel carries **38** modules and compiles the
new C65/80/128 profiles, the composed diamond/join-DAG/join-chain profiles and the walk
suite byte-identically to the working tree.

## 5. Suggested milestone order and gates

| Milestone | Contents | Gate |
| --- | --- | --- |
| M0 | P0 | Docs and ledger agree with code |
| M1 | P1 | Fan-out / multi-output suite exact; existing ledger re-passes |
| M2 | P2 + P3 | C33+ and/or ConvTranspose K5 at a documented boundary |
| M3 | P4 + P5 + P6 | Each gap closed or evidenced |
| M4 | P7 | Calibration on sequence profiles + accuracy harness |
| M5 | P8 | Trained model with measured accuracy |
| M6 | P9 + P10 | RV1106 claim (or not) + published package |

## 6. Explicit non-goals / unsupported by design

* Full ONNX operator library, ONNX Runtime on the board, LLM support.
* A new kernel driver; the GPL vendor driver is used as-is.
* In-place/aliased execution without a verified safe hardware case.
* Dynamic ONNX shapes — executable dimensions are immutable; recompile per shape.
* Performance parity with Rockchip; CPU fallback is part of the design.
* Generalising the attached board's SoC results to RK3588 or to a distinct
  RV1106 SoC.

## 7. Evidence conventions

Each suite keeps its ONNX, compiled commands, input bytes, integer expected
outputs, manifest and `board_results_*.json`. Generators use the open Python
environment; vendor oracle builders stay separately named `*_oracle.py` and never
feed the public emitters. Failed hypotheses are retained, excluded from the
passing ledger, and cited when a mode is declared blocked.

## 8. Final status (2026-09-10)

All ten phases are implemented to their stated acceptance criteria. Accepted modes
carry independently generated commands, a host regression test and an exact RV1103
board run; rejected modes carry a retained failing experiment or a concrete
documented prerequisite. The campaign baseline is **1,696 models / 28,266
inferences / 10,216,467 exact output bytes** with **311 host tests** passing, and
`tests/test_ledger.py` (rows, links, board-evidence union, campaign totals) plus
`tests/test_container_bindings.py` (2,412 containers) and
`tests/test_suite_evidence.py` machine-check the evidence.

| Phase | Status | Evidence |
| --- | --- | --- |
| P0 hygiene | done, machine-checked | `tests/test_ledger.py`; reconciled docs |
| P1 DAG ABI + scheduler | acceptance met; composer assembles every emitter and the walk dispatches chains with pools plus the join class (incl. mixed pool kinds) | v5 tensor table/runtime API; `open_rknpu.compose` reproduces **every** named DAG emitter byte-identically - `diamond`/`diamond_tail`/`join_dag`/`pooled_dag` (48 models) and `join_chain`/`join_scale`/`join_residual`/`mixed_head`/`depthwise_chain`/`two_head` (110-model pre-port hash reference, `tests/test_composer_emitters.py`); multi-input placement incl. late inputs; declared/derived binding checks |
| P2 input channels | C1..128 accepted, >C128 rejected | `native_c48_suite`, `native_c64_suite`, `native_c65_suite` (44/704/270,400); vendor C128 runs as one CNA task |
| P3 ConvTranspose/dilation | K5 depthwise + sparse K5 depthwise-dilation2 + dense K3-dilation2 accepted | `transpose_k5_suite`, `transpose_k5_dilation_suite`; `research/analyze_k5_*` |
| P4 elementwise/UQ | per-channel Mul + asymmetric depthwise accepted; spatial broadcast and per-channel output conversion rejected with decoded fields | `per_channel_mul_suite`, `depthwise_asymmetric_suite`; `mul_per_channel_ow_suite` isolates `OW_SRC=1` (hangs; with `OD_BYPASS=1` the table is ignored); `mul_broadcast_mode_suite` retains the broadcast hangs |
| P5 padding | host producer by design | `padding_suite`; plan's host-preprocessing path |
| P6 LUT | diagonal power-of-two stems accepted; mixed/non-power-of-two rejected | `lut_domain_suite` (12/192/36,864); index measured in `lut_index_probe` (residual is one index step, no affine rule) |
| P7 calibration/accuracy | steps 1-3 done (min/max, percentile, KL) | `sequence_calibration_suite`, `accuracy.py`; `research/pretrained/*` |
| P8 trained model | done with measured accuracy | MNIST 98.67%, Fashion-MNIST 88.15% (`examples/*/README.md`) |
| P9 RV1106 | NPU IP validated on the attached board; distinct SoC explicitly unclaimed | `research/hardware_identity.md` (`rockchip,rv1106-rknpu`) |
| P10 packaging | wheel + sdist rebuilt and re-verified; publication a user decision | `dist/open_rknpu-0.1.0.dev0*`, 38 modules, offline byte-identical compile |

### Residual disposition (2026-09-11)

Every item of the completion objective, with its disposition and evidence:

| Item | Disposition | Evidence |
| --- | --- | --- |
| S10 generic batched submission | **closed** | `sequence.relink_for_batched` post-pass covers every emitter; `batched_all_probe/` (3-48 tasks, 1 job each); `tests/test_submission.py` |
| S4 fence-free completion | **closed** | barrier job gives lag-0 completion (`barrier_probe/`, `tests/board_barrier.c`); a pollable fd still needs `CONFIG_ROCKCHIP_RKNPU_FENCE`, a deployment decision |
| S5 per-family cost cross-check | **closed** | `family_cost_crosscheck/`: 23 held-out containers, two board runs, CNA row confirmed, elementwise row profile-specific, medians 1.2-6.2x the minimum |
| P7 percentile/KL calibration | **closed** | `calibration.py` + `percentile_calibration_suite/` (4 models/76 inferences/14,592 bytes) |
| P3 dense K3-dilation2 ConvTranspose | **closed** | `transpose_dilation_dense_suite/` (4/32/18,952) |
| P4 per-channel output conversion | **closed (negative)** | `mul_per_channel_ow_suite/`: `OW_SRC=1` hangs with a well-formed block at the named address; with `OD_BYPASS=1` the block is ignored |
| P4 native spatial broadcast | **closed (negative)** | `mul_broadcast_notch_suite/`: the decoded TRM notch removes the hang and the read is byte-for-byte linear, so no compact spatial mode exists |
| P2 C>64 input channels | **closed** | `native_c65_suite/` (20/320/127,744); a vendor C128 Conv is one CNA task, so no channel-split accumulation is needed |
| P1 op-level walk | **substantially closed, bounded remainder** | composer assembles every emitter byte-identically (`composer_port_reference.json`, 110 hashes); the walk dispatches chains with pools (`walk_chain_suite`: 12/192/6,368) and the join class including mixed pool kinds (`walk_join_suite`: 6/96/4,608) and is byte-identical to the diamond emitter on 24/24 models. Not yet walked: elementwise ops and multi-join DAGs (`join_chain`/`join_dag` keep their profiles); no longer a correctness gap, only coverage |
| P6 non-power-of-two/mixed LUT bands | **closed (negative, measured)** | `lut_index_probe/`: the hardware index is read directly at every code; the slope model is right, the residual is one +/-1 index step on ~17% of codes, and no affine rule with `J <= 14` reproduces it |
| FENCE kernel config | **reported** | needs a kernel rebuild/flash; the functional need is met by the barrier job |
| Distinct RV1106 SoC | **reported** | attached board validated as `rockchip,rv1106-rknpu` (`hardware_identity.md`); a second board is hardware |
| Vendor-zoo detector | **reported** | accuracy harness + three calibration methods + two 10k-image trained models exist; a detector needs a dataset/training budget |
| Publication | **reported** | wheel/sdist built and re-verified (38 modules, offline byte-identical compile); an index release/announcement is a maintainer decision |

Baseline at close: **1,680 models / 28,250 inferences / 10,206,227 exact output bytes**
with **311 host tests** passing, `tests/test_ledger.py` (121 rows) and
`tests/test_suite_evidence.py` machine-checking the evidence.

### User/hardware decisions (reported, not implemented)

These three residuals are not compiler or runtime work; each is recorded with the
evidence that already exists and the input the decision needs.

1. **A distinct RV1106 SoC.** The attached board is validated as the shared
   `rockchip,rv1106-rknpu` NPU IP - device-tree compatible, driver v0.8.2, register
   captures and 1,674 exact models (`research/hardware_identity.md`,
   `research/COVERAGE_EXPANSION_RESULTS.md`). Extending the claim to a different
   RV1106-class die or board needs that hardware; the compiler depends only on the
   documented register profile, so the check is a re-run of the ledger, not a code
   change. Decision: acquire/borrow a second board, or keep the claim scoped to the
   attached RV1103.
2. **A vendor-zoo detector.** The pieces are in place and measured: a float reference
   with exact INT8 execution, `minmax`/`percentile`/`kl` calibration, and two trained
   classifiers evaluated end to end over 10,000 images each (MNIST 98.67%,
   Fashion-MNIST 88.15%, `examples/*/README.md`). A *detector* additionally needs a
   labelled dataset, training budget and a board memory/latency measurement; the
   decision is which dataset and budget to spend, not a missing capability.
3. **Publication.** The distribution is built and re-verified after every campaign
   (wheel + sdist `dist/open_rknpu-0.1.0.dev0*`, 38 modules, libc-only runtime sources
   under `share/open-rknpu/runtime/`, MIT license audit, offline byte-identical compile
   in a clean install). What remains is an index release (package name/credentials) and
   an announcement - a maintainer decision, not a technical step.

## After completion: codebase cleanup (2026-09-11)

With every residual dispositioned, the tree was cleaned under the same no-container-change
contract: lint/dedup, dead code, repository layout and documentation. The scope, the
verification (2,244 / 2,244 suite models byte-identical, 311 tests, campaign sweep
157 same / 12 pinned drift / 0 err, 0 broken doc links, wheel module parity) and the list
of things deliberately left alone are in [cleanup-plan](cleanup-plan.md); the entry in
[investigation-log](../investigation-log.md) has the measured evidence, and
`research/verify_suites.py`, `research/campaign_sweep.py` and
`research/check_docs_links.py` reproduce it from the tree.
