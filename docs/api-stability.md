# Public API stability

This page is the contract: it names what a 0.x user may depend on, what may change without
notice, and what the container writer promises. [`api.md`](api.md) is the tour;
this page is the guarantee. The code is the reference for prose that drifts.

The project is **0.x**. There is no frozen byte-level format and no 1.0 compatibility
promise yet, but there is a stable *Python* surface: the names below are frozen within
0.x, and a break in one of them is a deliberate, documented event in
[`CHANGELOG.md`](../CHANGELOG.md), announced at least one minor release ahead
([deprecation policy](#deprecation-policy)).

## Supported in 0.x

These names are supported at their documented module path. The exact signatures are frozen:
parameter names, parameter order and defaults are the contract. A new optional parameter
with a default is compatible; renaming, reordering or removing one is not.

```text
compile_sequence(path, input_scale=1.0, input_zero_point=0, output_range=None,
                 mul_operand_zero_points=(0, 0), mutable_weights=False,
                 mutable_constants=False, calibration_ranges=None,
                 expose_intermediates=False, reuse_intermediates=False,
                 asymmetric_depthwise=False, per_channel_mul=False, submission=None,
                 tiles=None) -> (bytes, dict)

measure(model_path, directory, method='minmax', percentile=99.99, bins=2048) -> dict

encode_sequence(payload, *, input_shape, output_shape, input_stride, arena_bytes,
                input_offset, output_offset, tasks, input_scale=1.0,
                input_zero_point=0, output_scale=1.0, output_zero_point=0,
                serial=False, input_layout='packed', batch=1,
                input_tensor_count=1, constants=()) -> bytes

decode_sequence(data) -> dict
encode(payload, metadata) -> bytes
decode(data) -> dict

Quantization(weights, weight_zero_points, weight_scales, biases, channel_multipliers,
             multiplier, shift, output_scale, output_zero_point, kernel_size=1,
             relu=False, input_scale=1.0, input_zero_point=0)
reference(inputs, quantization, rounding='separate') -> numpy.ndarray
```

| Name | Returns | What it promises |
| --- | --- | --- |
| `open_rknpu.scheduler.compile_sequence` | `(bytes, meta)` | the normal entry point: one normalized ONNX graph to a validated container plus its `meta` |
| `open_rknpu.calibration.measure` | `dict` | activation ranges per tensor: `minmax`, `percentile` or `kl`, ready to pass back as `calibration_ranges` |
| `open_rknpu.sequence.encode_sequence` | `bytes` | writes `ORNPUSEQ` v3, or v4 iff constant descriptors are given; the encoder validates itself through `decode_sequence` before returning |
| `open_rknpu.sequence.decode_sequence` | `dict` | parses `ORNPUSEQ` v3/v4/v5; raises `ValueError` on any non-sequence magic |
| `open_rknpu.model.encode` | `bytes` | writes legacy `ORNPUBIN` v1 (input band `1.0/0`) or v2 (any other band); self-validates through `decode` |
| `open_rknpu.model.decode` | `dict` | one parser for **both** families: `ORNPUBIN` v1/v2 and `ORNPUSEQ` v3/v4/v5 |
| `open_rknpu.quantization.Quantization` | instance | the quantization parameters a compiled Conv carries (`metadata()` exposes them as a plain dict) |
| `open_rknpu.quantization.reference` | `numpy.ndarray` | integer reference for one generic quantized Conv, with `rounding='separate'` or `'combined'` |
| `open_rknpu.mutable.compile_mutable` | `(bytes, meta, regions)` | compiles a v4 container that carries replaceable constant regions, or raises if the profile has none |
| `open_rknpu.mutable.constant_regions` / `constant_payload` | `tuple[ConstantRegion, ...]` / `bytes` | the named regions of a container and one region's bytes; raises the exact message for legacy and v5 containers |
| `open_rknpu.mutable.replace_constant` | `bytes` | a new container with one whole region replaced and the checksum recomputed (the host side of `ornpu_set_constant`) |
| `open_rknpu.mutable.graft_region` | `bytes` | the same replacement, but refuses a donor whose `program_bytes` differ (the band lives in the program) |
| `open_rknpu.mutable.program_bytes` | `bytes` | the header/tasks/descriptors/programs with the checksum field zeroed: the comparison that says two containers share a band |

The `open-rknpu` console script is supported: its entry point is `open_rknpu.cli:main`
([`pyproject.toml`](../pyproject.toml)), and the `compile`, `inspect` and `normalize`
subcommands and their flags are the stable CLI contract (pinned by the CLI matrix test).

### Profile references

Every profile ships an independent integer reference next to its emitter. These are
supported **at the canonical path below only**; the same function re-exported from another
emitter module is an implementation detail, not part of the surface.

| Profile family | Supported reference |
| --- | --- |
| image-input Conv, pooling | `open_rknpu.native.native_input_reference` |
| internal INT8 grids | `open_rknpu.chain.native_reference` |
| generic quantized single Conv | `open_rknpu.quantization.reference` |
| LUT activations | `open_rknpu.lut.lut_reference`, `open_rknpu.lut.stem_range` |
| activations | `open_rknpu.activation.leaky_reference`, `open_rknpu.activation.prelu_reference` |
| joins and DAGs | `open_rknpu.graph.diamond_reference`, `open_rknpu.join_dag.join_dag_reference`, `open_rknpu.graph.join_chain_scale_reference` |
| walked chains and joins | `open_rknpu.walk.chain_walk_reference`, `open_rknpu.walk.join_walk_reference`, `open_rknpu.walk.load_quantizations` |
| depthwise | `open_rknpu.depthwise.depthwise_reference` |
| pooling and reduction | `open_rknpu.pooling.pool_reference`, `open_rknpu.reduction.reduction_reference` |
| transposed Conv | `open_rknpu.transposed.transposed_reference` |
| legacy profiles 7/8 | `open_rknpu.network.network_reference` (Conv-Relu-Conv plus three 2x2 pools) |
| elementwise | `open_rknpu.elementwise.add_reference`, `.sub_reference`, `.max_reference`, `.mul_reference`, `.mul_requant_reference`, `.runtime_scale_reference` |

Two notes on this table. The reference is frozen at its canonical defining module: a
re-export from another module (for example `walk.native_reference`) is internal even though
the function is the same object. And the pooling, reduction, transposed and network
references were added in the same unreleased cycle as this page; they are part of the
surface from the release that first ships them.
The `native_elementwise` emitter has no reference of its own; it is covered by the native
and elementwise references above.

### `meta` keys `compile_sequence` promises

`meta` is a plain `dict`; keys beyond this table are informative and may appear or change.
The documented keys are:

| Key | Present on | Promise |
| --- | --- | --- |
| `submission` | every successful call | `"serial"` or `"batched"`, the shape the container was written for |
| `shape_nhwc` | Conv, pool, chain, depthwise and elementwise paths | `[N, H, W, C]` of the input the container consumes, after any leading `Pad` the scheduler folded |
| `output_shape_nhwc` | the same paths | `[N, H, W, C]` of the external output |
| `profile` | the legacy `Conv-Relu-Conv` chain and the two constant-Mul profiles | an integer legacy profile number (the chain reports `2`) or, on the Mul profiles, a profile-name string |
| `sequence_profile`, `lut_profile`, `transposed_profile`, `depthwise_profile`, `elementwise_profile` | the matching sequence-family profile | a string naming the selected profile |
| `quantization` | the single-Conv path | `Quantization.metadata()`, a dict with `weights`, `weight_zero_points`, `weight_scales`, `biases`, `channel_multipliers`, `multiplier`, `shift`, `output_scale`, `output_zero_point` |
| `quantizations` | composed and walked graphs | per-tensor quantization dicts keyed by tensor name |
| `input_scale`, `input_zero_point` | paths that carry an input band | the band the container was written with |
| `output_scale`, `output_zero_point` | every quantized path | the output band the container was written with |

One honesty note: the profile identifier is **profile-dependent by design**. A bare dense
Conv (the scheduler's generic Conv[/Relu]-plus-pool path) currently sets no profile
identifier at all; it reports `pool_stages` and `limitations` instead. Code that needs a
profile name should accept any of the six keys above, not test for `profile` alone.

## Internal surface

Everything else under `open_rknpu.*` is internal and may change in any release without a
deprecation cycle:

* the emitter modules — `native`, `native_elementwise`, `strided`, `depthwise`, `pooling`,
  `reduction`, `chain`, `chain_n`, `tiled_chain`, `elementwise`, `elementwise_chain`,
  `elementwise_multi`, `join_dag`, `pool_join`, `depthwise_join`, `pooled_branches`,
  `transposed`, `lut`, `layout`;
* the front end and composition internals — `graph.py`, `normalize.py`, `network.py`,
  `quantized_import.py`, `compose.py`, `liveness.py`, `padding.py`, `register_profile.py`
  (the six `open_rknpu.mutable` functions in the table above are the supported part of that
  module; `ConstantRegion`'s fields are part of the promise);
* the `walk` planner internals (everything except the three references above);
* `compiler.py` (`compile_model`, the legacy single-Conv entry point), `accuracy.py`, and
  the `cli.py` internals behind `main`.

`open_rknpu` declares **no `__all__`**; `open_rknpu.__version__` is a string and is not a
stability signal by itself. The supported names are therefore reached by their documented
module path, which is exactly what the pinned test checks.

## Container-format promise

Two families exist: legacy `ORNPUBIN` v1/v2 and task-table `ORNPUSEQ` v3/v4/v5. A 0.x
runtime keeps reading **all five**; a version is only ever *added*, never redefined.

| Loader | `ORNPUBIN` v1/v2 | `ORNPUSEQ` v3 | v4 | v5 |
| --- | --- | --- | --- | --- |
| C runtime `ornpu_inspect` / `ornpu_open` | yes | yes | yes | yes |
| Python `open_rknpu.model.decode` | yes | yes | yes | yes |
| Python `open_rknpu.sequence.decode_sequence` | no (non-sequence magic) | yes | yes | yes |
| `open-rknpu inspect` (calls `model.decode`) | yes | yes | yes | yes |
| `ornpu_run_io` | no | no | no | yes, multi-tensor |

The full matrix, the field-level rules and what each version adds are in
[`container-migration.md`](container-migration.md); the byte layout is in
[`runtime/sequence_format.md`](../runtime/sequence_format.md).

### "Never emit a container an older runtime rejects without you asking"

In practice this means **version selection is capability-driven and opt-in**:

* `model.encode` writes v1 when the input band is exactly `(1.0, 0)` and v2 otherwise.
* `encode_sequence` writes v4 only when you pass `constants=`; otherwise v3.
* `encode_sequence_v5` writes v5, and `compile_sequence` reaches it only for the profiles
  that need the named tensor table (fan-out, several external inputs/outputs, exposed
  intermediates, `--expose-intermediates`). Mutable weights/constants need the v4 table
  (`--mutable-weights`, `--mutable-constants`).

So a default compile of a given model emits the same version it always did, and a newer
version only appears when you asked for the capability that requires it. Because every 0.x
runtime reads v1–v5, an old runtime still opens containers the compiler writes without an
explicit opt-in.

### Changing a format

The rule is: **add a new version; never reuse a field silently.**

* New capabilities get a new version (v6, then v7, …). An existing version's bytes and its
  decoding do not change; v5 left v3/v4 untouched.
* A field is never re-purposed in place. If the meaning of a byte must change, the version
  changes with it — the v5 `batch − 1` byte is documented rather than patched into the
  v3/v4 table ([`container-example.md`](container-example.md)).
* Correction of *documentation* is allowed without a byte change; correction of a *byte
  layout* is a new version.
* The emitted bytes of the evidence suites are pinned in
  `research/container_baseline.json` and checked by `research/verify_suites.py`
  ([`verification.md`](verification.md)), so a container that changes is a visible,
  deliberate event rather than a silent drift.

Before 1.0 the format is **not byte-frozen**: the procedural rules above are the promise,
not "v5 never changes". [`container-migration.md`](container-migration.md) records this
plainly.

## Deprecation policy

For a supported name in 0.x:

1. **One minor release of notice.** The name keeps working for at least one minor release
   after deprecation; it is never removed in the release that announces it.
2. **A warning where Python is involved.** A deprecated Python API raises
   `DeprecationWarning` (or a subclass) on use, naming the replacement.
3. **A changelog entry.** [`CHANGELOG.md`](../CHANGELOG.md) gets a `### Deprecated` entry
   that names the old name, the replacement and the first version that may remove it.
4. **Removal only in a minor bump** — never in a patch release. The removal itself is a
   `### Removed` changelog entry.

Internal names get no notice. Container versions are not deprecated: an old version keeps
decoding, and a producer that wants a new capability moves to the new version.

## How to check your code before upgrading

The contract above is machine-checked by the pinned test
[`tests/test_public_api.py`](../tests/test_public_api.py): it imports every supported name,
compares each signature's parameter names and defaults against this page, checks the
console entry point, and asserts the documented `meta` keys for a tiny Conv. Run it against
the version you are upgrading to:

```sh
PYTHONPATH=src python -m pytest -q tests/test_public_api.py
```

A bare import check is not enough — it does not notice a renamed parameter, a changed
default or a missing `meta` key. For a fast smoke test that the surface is present:

```sh
PYTHONPATH=src python -c "from open_rknpu.scheduler import compile_sequence; from open_rknpu.calibration import measure; from open_rknpu.sequence import encode_sequence, decode_sequence; from open_rknpu.model import encode, decode; from open_rknpu.quantization import Quantization, reference; print('supported surface imports OK')"
```

If you ship containers rather than call Python, the compatibility check is
`research/verify_suites.py` ([`verification.md`](verification.md)); it confirms a build
still emits the pinned bytes, which is the strongest available signal that a runtime older
than your compiler still opens them.

## See also

* [`api.md`](api.md) — the entry-point tour and the module inventory.
* [`container-migration.md`](container-migration.md) — the version matrix and the field-level
  compatibility rules.
* [`container-format.md`](container-format.md) — the two families and the v5 anatomy.
* [`verification.md`](verification.md) — the container baseline and the board ledger.
* [`CHANGELOG.md`](../CHANGELOG.md) — the record of every documented break.
