# Python API

Everything the CLI does is available as a library. This page is the map; the docstrings in
`src/open_rknpu/` are the reference.

## Entry points

```python
from open_rknpu.scheduler import compile_sequence        # the normal entry point
from open_rknpu.compiler import compile_model            # legacy single-Conv container
from open_rknpu.model import encode, decode              # ORNPUBIN v1/v2
from open_rknpu.sequence import decode_sequence          # ORNPUSEQ v3/v4/v5
from open_rknpu.calibration import measure               # activation ranges
from open_rknpu.normalize import normalize_model         # ONNX front end only
```

`compile_sequence(path, input_scale=1.0, input_zero_point=0, output_range=None,
mul_operand_zero_points=(0, 0), mutable_weights=False, mutable_constants=False,
calibration_ranges=None, expose_intermediates=False, reuse_intermediates=False,
asymmetric_depthwise=False, per_channel_mul=False, submission=None, tiles=None)`
returns `(binary, meta)`.

The `meta` dict is the practical interface. Which keys are present depends on the profile
that accepted the graph: `submission`, `shape_nhwc`, `output_shape_nhwc` and `quantization`
(or `quantizations` for composed graphs) are always there, and the profile identifier key is
`profile`, `sequence_profile`, `lut_profile`, `transposed_profile`, `depthwise_profile` or
`elementwise_profile` - a composed path (the op-level walk in particular) deliberately
carries none of them, which is why `docs/api-stability.md` pins the keys it promises rather
than "`profile` always exists". Other keys you will see per family: `task_count`,
`engine_runs`, `arena_bytes`, `allocated_bytes`, `tensor_offsets`, `constant_offsets`,
`walk_ops`, `stages`, `pool_stages`, `limitations`. Print it while developing - every
example does.

## Per-profile references

A container is only trustworthy if an independent implementation reproduces it. The
references live next to their emitters:

| Profile family | Reference |
| --- | --- |
| image-input Conv | `native.native_input_reference` |
| pooling / reduction | `pooling.pool_reference` (1-3 2x2 levels), `reduction.reduction_reference` (3 levels), `network.network_reference` (profiles 7/8) |
| internal INT8 grids | `chain.native_reference` |
| generic quantized single Conv | `quantization.reference` |
| LUT activations | `lut.lut_reference`, `lut.stem_range` |
| activations | `activation.leaky_reference`, `activation.prelu_reference` |
| joins / DAGs | `graph.diamond_reference`, `graph.join_chain_scale_reference`, `join_dag.join_dag_reference` |
| walked chains and joins | `walk.chain_walk_reference`, `walk.join_walk_reference` with `walk.load_quantizations(meta)` |
| depthwise | `depthwise.depthwise_reference` |
| transposed Conv | `transposed.transposed_reference` (dense and depthwise, K2/K3/K5, off-centre taps) |
| elementwise | `elementwise.add_reference`, `sub_reference`, `max_reference`, `mul_reference`, `mul_requant_reference`, `runtime_scale_reference` |

## Composing your own container

`open_rknpu.compose` is a small library:

```python
from open_rknpu.compose import Binding, ConstantSpec, Stage, TensorSpec, compose

stage = Stage(name="conv0", family="native-conv", reads=("input0",), writes=("conv0",),
              fields=my_fields, constants=(ConstantSpec("conv0.parameters", size, my_fill),),
              bindings=(Binding(0x1070, "input0", "read"), Binding(0x4020, "conv0", "write")))
binary, meta = compose([stage], [TensorSpec("input0", ROLE_INPUT, LAYOUT_NATIVE16, ...)])
```

`compose` handles arena allocation (with optional lifetime reuse via `open_rknpu.liveness`),
tensor address rewriting, constant packing and task linking. `open_rknpu.sequence` turns the
result into bytes and validates it.

## Module inventory

| Layer | Modules |
| --- | --- |
| Front end | `normalize`, `network`, `quantized_import`, `graph` |
| Dispatch | `scheduler`, `walk` |
| Profiles | `native`, `native_elementwise`, `strided`, `depthwise`, `pooling`, `reduction`, `chain`, `chain_n`, `tiled_chain`, `elementwise`, `elementwise_chain`, `elementwise_multi`, `join_dag`, `pool_join`, `depthwise_join`, `pooled_branches`, `transposed`, `lut`, `layout` |
| Composition | `compose`, `liveness`, `sequence`, `model` |
| Mutable parameters | `mutable` (v4 constant regions: read, replace, band-checked graft) |
| Numerics | `quantization`, `calibration`, `activation`, `padding`, `register_profile` |
| Tooling | `cli`, `compiler`, `accuracy` |

## CLI

```text
open-rknpu compile  <model.onnx> -o <out.bin> [--sequence] [--input-scale S]
                    [--input-zero-point Z] [--output-scale S] [--output-zero-point Z]
                    [--calibration DIR] [--submission serial|batched] [--tiles N]
                    [--mutable-weights] [--mutable-constants] [--per-channel-mul]
                    [--asymmetric-depthwise] [--expose-intermediates] [--reuse-intermediates]
open-rknpu inspect  <container.bin>        # JSON header/tasks/tensors
open-rknpu normalize <model.onnx> -o <out.onnx>
```

`--calibration DIR` expects a directory of `uint8`/`float32` NCHW `.npy` batches, exactly
like `calibration.measure`. Note that the scheduler rejects calibration combined with an
explicit output scale/zero point.
