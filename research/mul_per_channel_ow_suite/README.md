# Retained failure: per-channel elementwise *output* conversion (OW_SRC)

This directory is the retained experiment for the P4 sub-item "per-channel elementwise
output conversion". It is **not** a supported profile and must not be staged as one.

## Hypothesis

Mesa's Rocket register map names `0x4050` `BS_OW_CFG` (`OW_SRC@0`, `OD_BYPASS@1`) and
`0x5020` `RDMA_BS_BASE_ADDR`, and Conv does read a per-channel block there. The verified
elementwise constant Mul runs with `0x4050=0x30000002` (`OW_SRC=0`, `OD_BYPASS=1`) and a
scalar `0x4084`/`0x4088` conversion. If the ew path honoured `OW_SRC=1`, a per-channel
multiplier table would replace the scalar conversion and a per-channel Mul output grid
would be possible.

## Files

Baseline `model000.bin` is `../mul_broadcast_suite/model007.bin` (8x6x3 scalar-constant
Mul over a 32-input batch: Conv task `0x30000001`, elementwise task `0x30000002`,
`0x5020=0`, `0x4084=0x14000`, `0x4088=0x15`). `check_mul_per_channel_ow.py` rebuilds
every variant and `run_mul_per_channel_ow.py` runs them on the board, comparing each
output with the suite's expected file and with the baseline per channel
(`ow_results.json`, `ow_summary.txt`, `output*.i8`).

| Variant | EW `0x4050` | Table at payload `0x5020=0x1800` | Result |
| --- | --- | --- | --- |
| `model000` | `0x30000002` (baseline) | absent | **exact**, 32/32 runs |
| `model001` | `0x30000001` | 4xUINT16 at *file* offset 0x1800 (0x80 past the named address) | hang |
| `model002` | `0x30000001` | decoded Conv block, unit multipliers | hang |
| `model003` | `0x30000001` | decoded Conv block, ch0 unit / ch1-2 zero | hang |
| `model004` | `0x30000003` (`OW_SRC=1`, `OD_BYPASS=1`) | decoded Conv block, ch0 unit / ch1-2 zero | **exact**, byte-identical to baseline |
| `model005` | `0x30000002` (baseline conversion) | decoded Conv block present | **exact**, byte-identical to baseline |

The decoded Conv block is the format every verified Conv/depthwise emitter uses: a
32-byte block per four output channels with INT32 bias at +0..15, `-weight_zero_point`
INT16 at +16..23 and the Q14 channel multiplier UINT16 at +24..31.

## Observed result (2026-09-11)

```
model000  runs=32 rc=0        (exact)
model001  runs=0 rc=1         RKNPU: failed to wait job / job timeout / soft reset
model002  runs=0 rc=1         RKNPU: failed to wait job / job timeout / soft reset
model003  runs=0 rc=1         RKNPU: failed to wait job / job timeout / soft reset
model004  runs=32 rc=0        (exact, identical to baseline)
model005  runs=32 rc=0        (exact, identical to baseline)
```

The NPU recovered after each soft reset and `rkipc` stayed alive.

## Conclusion

* The 2026-09-10 result was confounded by two construction errors: the table was written
  at the *file* offset `0x1800` while `0x5020` names a *payload* offset (0x80 bytes away),
  and it used a four-UINT16 layout rather than the decoded Conv block. With both fixed,
  `OW_SRC=1` **still hangs**, so the failure is a property of the field, not of the probe.
* Whether the output-conversion bypass is cleared or not is what decides it: with
  `OD_BYPASS=1` the elementwise task completes and **ignores `0x5020` entirely** - a
  well-formed table with two zero multipliers leaves the output byte-identical to the
  baseline (channels 1 and 2 are not scaled). With `OD_BYPASS=0` the same task never
  completes. Clearing the bypass alone is not the problem: the DAG suites' two-surface
  elementwise tasks run with `0x4050=0x30000000` (`OW_SRC=0`, `OD_BYPASS=0`).
* A per-channel elementwise **output** conversion is therefore not available through
  `BS_OW_CFG`/`0x5020`; the elementwise output stage has no BS-table read. The accepted
  route for per-channel Mul quantization stays the verified 1x1 depthwise lowering with
  native per-channel weight scales ([`../per_channel_mul_suite/`](../per_channel_mul_suite/)),
  and the INT8 container carries one output scale per tensor anyway.
* The baseline in this directory still reproduces the scalar-conversion profile, so the
  failures are attributable to the variant registers alone. `tests/test_mul_per_channel_ow.py`
  pins the built variants and the retained board records.
