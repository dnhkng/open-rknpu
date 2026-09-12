# Troubleshooting

Organised by symptom, in a user's words. Each entry gives the likely cause, the exact
strings to look for, and the fix or the document to read. Definitions of the terms
used here (band, zero point, profile, arena, …) are in [glossary.md](glossary.md);
the accepted bounds behind every rejection are in
[support-matrix.md](support-matrix.md).

## Start here

1. **Compile errors are `ValueError`s.** The CLI catches them and prints
   `open-rknpu: <message>` before exiting 1; the Python API raises the same text. The
   message names the profile and the bound that failed.
2. **Which profile ran** is `meta["profile"]`; on rejection there is no `meta`, so run
   `open-rknpu normalize model.onnx -o norm.onnx` to see whether the front end already
   rewrote the graph (auto-pad, group/dilation, even kernels, leading `Pad`, Conv+Mul).
3. **Correctness is byte equality** against the profile's Python integer reference for
   the same quantization parameters — not closeness to the float ONNX model. A small
   float gap can be normal; a byte mismatch is a finding.
4. **Capture the exact failure**: the compiler message, the `meta` dict, and the board
   runner's failing line (`model N run R output J: got X expected Y`).

---

## "My model is rejected"

**Likely cause.** The graph is outside the bounded static-CNN class this compiler
accepts: a rank or dtype the front end does not take, a non-constant weight/bias, an
attribute combination no verified profile covers, or an operator with no primitive at
all. There is no silent fallback — rejection is the designed outcome for an out-of-envelope
graph ([getting-started.md](getting-started.md), [roadmap.md](roadmap.md)).

**Error strings to look for** (the common ones, with the construct each names):

| Message | What it rejects |
| --- | --- |
| `sequence lowering requires one input, one output, and an initial Conv` | a graph the scheduler cannot start on: multiple inputs/outputs, or a first node that is not `Conv` |
| `sequence lowering currently supports Conv[/Relu] followed by 2x2 pooling` | a node after the first Conv that is neither a pool nor one of the matched profiles |
| `static NCHW input required` | rank-3 / 1-D convolution, or a non-static input rank |
| `native Conv requires static batch1..16, H/W1..128, input C1..16352, constant weights/bias` | batch/shape/channel outside the native range, a dynamic dimension, non-constant weights, or input channels above the measured 16352-channel (511-part) wall |
| `native Conv supports odd K1..31, explicit padding, stride 1..4, input C1..16352/output C1..8192` | even or >31 kernels that no rewrite covered, `group != 1`, wrong dtype, output channels above the measured 8192-channel (512-block) wall, or an output shape that disagrees with the graph |
| `native padding/stride/dilation unsupported` | stride outside 1..4, dilation outside 1..17, or more than four pads |
| `invalid native Conv output geometry` | padding so large the effective kernel does not fit the input |
| `elementwise profile requires two Conv branches feeding Add, Mul, Sub or Max` | a join whose operands are not two Conv[/Relu] branches |
| `depthwise profile requires supported group/kernel/padding/stride and constant weights/bias` | a group Conv outside the depthwise bounds, or non-symmetric padding |
| `ConvTranspose profile requires depthwise C1..16, K2/K3/K5, per-axis stride1/2 and matching output geometry` | unsupported ConvTranspose geometry, or pads/output_padding out of range |
| `LUT stem must be a diagonal 1x1 Conv with one scalar magnitude per channel` | a `Sigmoid`/`Tanh` tail on anything but the bounded diagonal C3 stem |
| `one Conv with optional following Relu is supported` | the legacy (non-`--sequence`) compiler path given a different graph |

**Fix.** Find the construct in [support-matrix.md](support-matrix.md): the accepted
bounds and the profile that accepts it are listed there, and the rejected-constructs
table names the roadmap item if the gap is deliberate. Common practical rewrites: add
`--sequence` so the modern profiles are reachable; reshape a 1-D conv to
`[1,C,1,W]` where the graph allows it; replace a `MatMul`/`Gemm` head with a 1×1 Conv;
split a kernel >31 into shorter taps; keep the recurrent or `Concat` part on the host.
The Silero VAD envelope ([plans/primitive-roadmap.md](plans/primitive-roadmap.md))
shows how one real model hits four independent limits at once.

## "The output is all the same value"

**Likely cause.** The classic failure of an uncalibrated trained model: analytic bands
(derived from weight magnitudes and the input range) are far too wide, so a later grid
collapses onto its zero point and every output code is the same. In `examples/mel-kws/`
the first Conv's analytic band came out ~20× too wide and the model scored 10% (chance)
until it was calibrated ([quantization.md](quantization.md)). A second cause is a
saturated output band, and a third is binding the wrong tensor when the container has
several inputs/outputs.

**Strings / observations to look for.** Output bytes are all equal, and usually equal
the container's `output_zero_point` (`open-rknpu inspect model.bin` prints it). On the
host, `open_rknpu.accuracy` reports per-layer quantization error and the analytic output
scale; a scale orders of magnitude away from the calibrated one is the signature
(analytic scale 35.303 versus measured 0.13411 on MNIST Conv2,
[plans/completion-plan.md](plans/completion-plan.md) P8).

**Fix.** Calibrate the trained model:
`open_rknpu.calibration.measure(..., method="minmax"|"percentile"|"kl")` and pass
`calibration_ranges=report["ranges"]` (see
[examples/primitives/10_calibration.py](../examples/primitives/10_calibration.py)). If
the graph has several outputs, bind each with `ornpu_run_io` and check the tensor table
rather than assuming one output ([container-format.md](container-format.md)). To localise
which layer collapsed, expose chain intermediates (`--expose-intermediates`) or replay
per-stage probes the way `examples/mel-kws/` does
([research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)).

## "Accuracy collapsed"

**Likely cause.** Same analytic-band problem as above, plus three specific traps:
(1) a calibration set whose values do not cover the real inputs — measured ranges clip
everything outside them, and on the `sequence_calibration` suite calibration lowered
output MAE on all three graphs but *raised* maximum error on two
([research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md));
(2) the chain family's hidden-border quirk; (3) an output override narrower than the real
logit range.

**Error strings / observations to look for.** `calibration ranges lack tensor <name>`
(a Conv the report does not cover); `calibration and output quantization overrides
cannot be combined`; `calibration is unsupported for the join chain` (or the join DAG,
pooled branches); `output override unsupported for LUT profile` / `… LeakyRelu profile` /
`… PRelu profile` / `… Reshape profile`; and the saturated-output failure mode in
[quantization.md](quantization.md#failure-modes-seen-in-practice).

**The chain-family border zero point (fixed 2026-09-12).** `chain.py` and `chain_n.py`
used to leave the internal border register `0x1184` at its `-128` reset instead of
programming the producer's zero point, so a hidden border read
`(-128 - zero_point) × scale` rather than the real zero ONNX pads with. Every chain layer
now declares the band it reads, and the references model it: deep chains improved by up to
seven orders of magnitude against the float ONNX model (e.g. `deep_chain_suite/model001`
from 1.57e10 to 2.2e3 mean error). If an *old* container shows a hidden-band shift, it was
compiled before the fix - recompile it ([roadmap.md](roadmap.md)). The board evidence that would
be invalidated by changing it is the chain family's
([research/native_chain_suite/](../research/native_chain_suite/)); if your accuracy gap is
a hidden-band shift on a chain, that is the cause to check first.

**Fix.** Calibrate; put the output band in the calibration report instead of an override;
check the measured ranges against real data; keep hidden chain weights and biases
non-negative so the hidden band stays at `-128` (what
[examples/primitives/03_conv_chain.py](../examples/primitives/03_conv_chain.py) does).

## "The board hangs"

**Likely cause.** A malformed or hand-built register program can wedge the NPU — the
repository records this as a known hazard, and a bad program can also hang the driver
(`failed to wait job` / `job timeout` / `soft reset`). Two specific measured hangs are
retained: compact spatial-broadcast Mul variants before the ERDMA notch was decoded
([research/mul_broadcast_mode_suite/](../research/mul_broadcast_mode_suite/)), and the
per-channel output-conversion probe with `OW_SRC=1`
([research/mul_per_channel_ow_suite/](../research/mul_per_channel_ow_suite/)). A related
non-hang hazard is *stale data* when one arena slot is written by two different task
families (two independent board sightings;
[plans/pipelining-plan.md](plans/pipelining-plan.md)).

**Strings / observations to look for.** `model N failed: -110` from the file runner
(timeout), `-22`/`-EINVAL` for a submit the driver refused, and the investigation log's
`failed to wait job` / `job timeout` / `soft reset` sequence. A program overlap usually
shows up first as `model 0 failed: -22` because a later program overwrote the earlier
one's fields.

**Fix.** The recovery is a reboot: `adb shell reboot` ([board.md](board.md)). Keep
`rkipc` alive, never issue `GET_VOLT` — it **oopses the caller** on this board because the
device tree has no rknpu regulator ([board.md](board.md)). Use compiler output rather than
patching register words by hand; the runtime validates magic, checksum, geometry and
tensor indices before submitting. If you are hand-assembling a container, check for
overlapping programs and give each internal a fresh arena slot when a profile does not
prove reuse safe.

## "The container will not open"

**Likely cause.** The file is truncated, edited, or is a different container version than
the loader expects; or a descriptor is internally inconsistent (size, arena bounds,
overlapping tensors, non-contiguous external indices). The Python decoder and the C loader
both fail closed.

**Error strings to look for.** Python (`open_rknpu.model.decode`,
`open_rknpu.sequence.decode_sequence`):

* legacy `ORNPUBIN`: `truncated model header`, `unsupported model format`,
  `unsupported NPU target/profile`, `unsupported pooling profile`,
  `unsupported two-layer profile`, `unsupported tensor shape`, `unsupported memory layout`,
  `unsupported model options`, `nonzero reserved fields`, `invalid input quantization`,
  `invalid output quantization`, `incorrect model length`, `model checksum mismatch`.
* `ORNPUSEQ` v3/v4/v5: `invalid sequence magic`, `truncated sequence header`,
  `invalid sequence header`, `invalid v5 sequence header`, `truncated v5 extension`,
  `invalid v5 extension`, `invalid task count`, `invalid task descriptor`,
  `invalid sequence allocation`, `invalid sequence quantization`,
  `incorrect sequence length`, `invalid tensor name`, `invalid tensor descriptor`,
  `tensor descriptor size mismatch`, `tensor outside arena`,
  `external tensor indices must be contiguous from zero`, `overlapping external tensors`,
  `internal tensor overlaps external tensor`, `v5 primary tensor mismatch`,
  `sequence checksum mismatch`.

**On the board** (`tests/board_io.c`): `model N failed: <rc>`,
`model N is not a named-tensor executable`, `tensor descriptor N failed`,
`model N accepted a missing output`, `model N accepted too many inputs`.

**Fix.** Recompile the model (`open-rknpu compile model.onnx -o model.bin --sequence`)
rather than repairing bytes; confirm with `open-rknpu inspect model.bin`. If you are
comparing against a published suite container, remember that 12 of 169 campaign models
are pinned *artifact* drifts — the published container predates a later serializer fix,
so a fresh compile cannot reproduce its bytes
([verification.md](verification.md#the-campaign-sweep),
[research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md)).
See [container-format.md](container-format.md) for the field-by-field layout.

## "I ran out of space on /userdata"

**Likely cause.** `/userdata` is a small flash-backed UBI volume (≈4.5 MB) and the
existing staging directory already uses part of it; a suite of containers, inputs and
expected outputs can fill it. Writing to `/tmp` is never a substitute for anything you
want to keep ([board.md](board.md), [plans/completion-plan.md](plans/completion-plan.md) §2).

**Strings / observations to look for.** `adb push` fails with `No space left on device`;
`adb shell 'df -h /userdata'` shows the volume full; the board's RAM is also only ~33 MB
shared with `rkipc`.

**Fix.** Check first, then clean up:

```sh
adb shell 'df -h /userdata'
adb shell 'ls -la /userdata/open-npu-research'
```

Stage under `/userdata/open-npu-research/<name>/` and delete suites you no longer need;
`research/run_v5_suite.py` stages one suite and its manifest rather than everything.
Never stop `rkipc` to free memory or space — the board is a camera appliance first.

## "My numbers differ from the reference"

**Likely cause.** Four distinct things, in decreasing order of how often they are the
answer:

1. **You are comparing to the float ONNX model, not the integer reference.** An INT8
   model is allowed to differ from its float parent; the correctness criterion is byte
   equality with the profile's integer reference *for the same quantization parameters*
   ([quantization.md](quantization.md)).
2. **A reference-vs-hardware rounding tie.** In the mel-CNN board run, 32 of 192,000
   output bytes differed: four utterances, each differing in one of 64 output cells by
   ≤4 LSB of the 0.23 output scale, with **no** classification change. Stage probes
   localised it to the third or fourth calibrated-band Conv with interior cells affected,
   so it is a rounding residual, not a geometry or addressing error
   ([research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)).
3. **The reference formula is not the naive one.** For the folded runtime scale product,
   `rint(a·b/128)` is not the hardware model — the folded scale product is not exactly
   `1/128` in float32, so the reference must use the hardware requantization formula
   (71 of 6,144 bytes differed by one at the boundary;
   [research/join_scale_suite/README.md](../research/join_scale_suite/README.md)).
4. **An artifact drift.** A published suite container captured before a serializer fix
   cannot be reproduced by a fresh compile; the 12 known drifts are pinned in
   `research/campaign_sweep.py` and in the ledger
   ([verification.md](verification.md#the-campaign-sweep)).

**Strings / observations to look for.** The board runner's per-byte line
`model N run R output J: got X expected Y`; the runner's
`FAIL: 4 samples differ from the reference (32 bytes)`; the campaign sweep's
`same=157 diff=12 err=0`.

**Fix.** Compare against the integer reference next to the emitter
([api.md](api.md#per-profile-references)), replay the per-stage probes for the layer where
the first difference appears, and re-check the reference's rounding order (output zero
point before the ties-even shift; half ties round to even). If a *published* artifact is
the mismatch, check the drift list before calling it a regression.

## "Calibration and output override conflict"

**Likely cause.** You passed both a calibration report and an explicit output band, or
you asked calibration for a profile that has no measured-band contract.

**Error strings to look for.**

* `calibration and output quantization overrides cannot be combined` — the scheduler and
  the CLI pre-check (the Python `compile_model` legacy path and `chain.py` raise their own
  wording: `calibration and chain output override cannot be combined`).
* `calibration with input quantization overrides is not supported yet` — the CLI when
  `--calibration` is combined with `--input-scale`/`--input-zero-point`.
* `calibration is unsupported for the join chain` (also the join DAG, pooled branches) —
  those profiles have no measured-band path.
* `output override unsupported for LUT profile` / `… LeakyRelu profile` /
  `… PRelu profile` / `… Reshape profile`.

**Fix.** Put the output band **inside** the calibration report as `ranges["output"]` and
pass only `calibration_ranges` — that is what `examples/mel-kws/build.py` does
([quantization.md](quantization.md), [examples/primitives/10_calibration.py](../examples/primitives/10_calibration.py)).
Drop one of the two if the profile has no band contract, or use a profile that does (the
walk accepts calibrated bands per Conv).

## "The CLI ignored my flag"

**Likely cause.** Most flags are profile-scoped, and a flag that does not apply to the
selected profile has no effect. `--sequence` changes which compiler path runs at all
(without it the modern profiles are unreachable), and `--tiles` is only consulted by the
height-strip chain profile, so it is silently ignored on any other graph;
`--per-channel-mul`, `--asymmetric-depthwise`, `--expose-intermediates` and
`--reuse-intermediates` are likewise only consulted by specific profiles.

**Error strings to look for** (these are *not* silent):

| Message | Meaning |
| --- | --- |
| `mutable parameters require --sequence` | `--mutable-weights`/`--mutable-constants` passed without `--sequence` |
| `Mul operand zero points require --sequence` | `--mul-a-zero-point`/`--mul-b-zero-point` without `--sequence` |
| `specify output scale and zero point together` | only one of `--output-scale`/`--output-zero-point` |
| `calibration with input quantization overrides is not supported yet` | `--calibration` plus `--input-scale`/`--input-zero-point` |
| `tiles must be an integer of at least 2` | `--tiles` below 2 |
| `height-strip tiling cannot be combined with output overrides, exposed intermediates or arena reuse` | `--tiles` plus `--output-scale`, `--expose-intermediates` or `--reuse-intermediates` |
| `output override unsupported for this strided profile` / `… scheduled profile` | an output band on a profile that does not carry one |
| argparse's `unrecognized arguments: <flag>` | the flag does not exist (check spelling; the CLI surface is in [api.md](api.md#cli)) |

**Fix.** Add `--sequence` when the flag requires it; check `meta["profile"]` and
`meta["submission"]` after compiling to confirm the flag you set is the one that was
used; remember `--submission batched` is honoured but slower than serial below roughly
eight same-engine tasks, so a latency *increase* is expected, not an ignored flag
([plans/pipelining-plan.md](plans/pipelining-plan.md), [api.md](api.md#cli)).

---

## See also

* [support-matrix.md](support-matrix.md) — exact bounds and board evidence per op.
* [glossary.md](glossary.md) — the vocabulary used above.
* [quantization.md](quantization.md) — bands, calibration and the accuracy failure modes.
* [board.md](board.md) — the board, `/dev/rknpu`, staging and safety notes.
* [verification.md](verification.md) — what "verified" and "exact" mean here.
