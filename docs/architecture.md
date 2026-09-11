# Architecture

## The pipeline

```text
ONNX file
  │  normalize_model()            constant folding, bias/padding folding, 1x1/even-kernel
  ▼                               rewrites, Q/DQ handling, NCHW validation
normalized graph
  │  scheduler._compile_sequence() dispatch in a fixed profile order (below);
  ▼                               the first profile that accepts the graph wins
emitter (one module per profile)
  │  compose()/Stage              tasks as (registers, constants, bindings); the composer
  ▼                               assigns arena offsets and rewrites addresses
container
  │  sequence.encode_sequence()   header + task descriptors + register words + constants
  ▼                               + tensor table (+ optional relink for batched submission)
.bin  ──► runtime/open_rknpu.c ──► /dev/rknpu ──► INT8 outputs
```

The compiler never reads a vendor artifact. Every register value comes from constants
recovered by experiment and recorded in `runtime/sequence_format.md`,
`src/open_rknpu/register_profile.py` and the investigation log.

## Module map

| Layer | Modules |
| --- | --- |
| Front end | `normalize.py`, `network.py`, `quantized_import.py`, `graph.py` (DAG analysis, join-chain and diamond emitters) |
| Dispatch | `scheduler.py` (profile order and fallbacks), `walk.py` (op-level walk for chains with interior pools and fan-in joins) |
| Profiles | `native.py`, `native_elementwise.py`, `strided.py`, `depthwise.py`, `pooling.py`, `reduction.py`, `chain.py`, `chain_n.py`, `tiled_chain.py`, `elementwise.py`, `elementwise_chain.py`, `elementwise_multi.py`, `join_dag.py`, `pool_join.py`, `depthwise_join.py`, `pooled_branches.py`, `transposed.py`, `lut.py`, `layout.py` |
| Composition | `compose.py` (Stage/Binding/ConstantSpec/TensorSpec, arena, liveness), `liveness.py`, `sequence.py` (task linking, tail control, encode/decode), `model.py` (task/register model, legacy container) |
| Numerics | `quantization.py`, `calibration.py`, `activation.py`, `padding.py`, `register_profile.py` |
| Tooling | `cli.py`, `compiler.py`, `accuracy.py` |

## The scheduler's profile order

`_compile_sequence` tests profiles in this order. The order matters: composite shapes are
matched before the simpler profiles that would otherwise capture part of them, and the
op-level walk sits where no existing profile overlaps.

| # | Graph class | Emitter |
| --- | --- | --- |
| 1 | `QLinearConv`, DQ→Conv→Q | `quantized_import.py` |
| 2 | stem + 3+ heads folded by joins | `graph.compile_join_chain` |
| 3 | stem + branches + 2+ joins (general DAG) | `join_dag.py` |
| 4 | dense + depthwise branch join | `depthwise_join.py` |
| 5 | pooled-branch join | `pool_join.py` |
| 6 | multi-layer pooled branches | `pooled_branches.py` |
| 7 | `Sigmoid`/`Tanh` tail | `lut.py` |
| 8 | `Mul`+`Relu`/`Clip`/`Add` tails | `elementwise.py` |
| 9 | `LeakyRelu` / `PRelu` tail | `activation.py` |
| 10 | `ConvTranspose` | `transposed.py` |
| 11 | depthwise → pointwise pair | `depthwise.py` |
| 12 | terminal spatial `Reshape` | `layout.py` |
| 13 | standalone `Mul` (external/runtime/constant/per-channel) | `elementwise.py` |
| 14 | two-head fan-out (`Conv,Relu,Conv,Conv`, two outputs) | `graph.compile_two_head` |
| 15 | join with a pool between head and join | `walk.compile_join_walk` |
| 16 | diamond: stem, two heads, one join, optional tail | `walk.compile_join_walk`, falling back to `graph.compile_diamond` |
| 17 | odd-length Conv/Relu chain (native chain) | `chain_n.py`, or `tiled_chain.py` with `tiles=` |
| 18 | `Conv,Relu,Conv` 8×8/C3 chain | `chain.compile_chain` |
| 19 | multi-input elementwise DAG | `elementwise_multi.py` |
| 20 | elementwise DAG `Conv,Conv,join,Mul…` | `elementwise_chain.py` |
| 21 | terminal Add/Mul/Sub/Max | `elementwise.py` |
| 22 | chain with an interior pool | `walk.compile_chain_walk` |
| 23 | single Conv / Conv+Relu / Conv+Clip (image input) | `native.compile_native_input` |
| 24 | strided / dilated single Conv | `strided.py` |
| 25 | Conv followed by 2×2 pools | `pooling.py` |
| 26 | 1×1 or 3×3 Conv, 8×8, small hidden counts | `native.py`, `depthwise.py` fallbacks |

`meta["profile"]` records which one ran. Profiles 22–24 are the fallbacks: they are
deliberately permissive on geometry but conservative about what they claim.

## The composer

Emitters do not lay out memory. They build `Stage` objects:

```python
Stage(name="conv1", family="native-conv", reads=("input0",), writes=("conv1",),
      fields=fields_fn,                      # register -> value (or a callable of addresses)
      constants=(ConstantSpec("conv1.parameters", size, fill_fn),),
      bindings=(Binding(0x1070, "input0", "read"), Binding(0x4020, "conv1", "write")))
```

`compose()` then:

1. assigns every `TensorSpec` an arena offset (64-byte aligned, with an optional lifetime
   allocator that reuses a dead tensor's space for the next one — `liveness.py`);
2. calls each stage's `fields(addresses, constant_offsets)` so register values referring to
   tensors become final addresses, and `fill(payload, offset)` to pack weights/biases;
3. emits one task descriptor per stage in dependency order and writes the payload;
4. computes the tail control word of each task. The tail is the *successor's fetch amount*
   (`PC_DATA_AMOUNT = (regconfig_words + 4 + 2 − 1)/2 − 1`), and `0x28` is terminal — which
   is why a whole DAG can be submitted as one job: every transition is just a fetch size,
   not an engine hand-off.

The composed container is a v5 program with a named-tensor table, so the runtime can bind
several inputs/outputs (`ornpu_run_io`) and expose intermediates.

## The op-level walk

`walk.py` validates a linear graph, walks the nodes in order and emits one stage per node
using the same verified per-op builders as the branch profiles (`native_fields` +
`_pack_layer_head` for a Conv, `pool_registers` for a pool). It is the only path where the
*position* of a pool in a chain is free, and it also lowers the diamond class. Two
refinements exist for trained models:

* **native16 image input** — an image outside the legacy 5..8-pixel range is staged as a
  native16 surface and the first Conv is emitted with `native.py`'s builders (single task,
  ≤6144 atoms). This is how `3x32x32` audio features reach the NPU.
* **calibrated bands** — `compile_chain_walk(..., ranges=report["ranges"])` uses a measured
  band per Conv instead of the analytic one. Analytic bands are unusable for a trained
  network (they can be orders of magnitude too wide), so this is what makes a real model
  compile at chance-plus accuracy instead of collapsing onto the zero point.

## Adding a primitive

1. Reproduce the operation with vendor-free commands and get *exact* board output for a
   small model (`research/` has the oracle captures and the probe recipes).
2. Decide the profile class and its bounds; write the emitter as a module that returns a
   composed container and a `meta` dict (`profile`, `quantization`, shapes, …).
3. Add the Python integer reference next to the emitter, using the same quantization
   parameters as the container.
4. Add the dispatch branch in `scheduler.py` (order matters — comment why it is where it
   is) and a generator under `research/build_<name>.py`.
5. Run the generator, add the suite's `board_results_*.json`, and add its row to
   `research/COVERAGE_EXPANSION_RESULTS.md` (the ledger test checks the row against the
   evidence).
6. Add host tests: parser, reference vs container equivalence, rejection cases, and a
   retained-board-run assertion. Regenerate `research/container_baseline.json` only when a
   container change is intended.
