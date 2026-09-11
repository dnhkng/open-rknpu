# ERDMA notch probe: spatial broadcast decoded and refuted (P4)

The earlier compact-operand sweep (`../mul_broadcast_mode_suite/`) hung on every
variant and left the field map incomplete. The RK3588 TRM for the same NPU IP
(`../hardware_refs/rk3588-trm-part1.txt`, "RKNN_dpu_rdma_*") decodes the two
registers that were missing:

| Register | Field | Meaning |
| --- | --- | --- |
| `0x5034` | `erdma_data_mode` 31:30 | 0 per channel, 1 per pixel, 2 **per channel by pixel**, 3 reserved |
| `0x5034` | `erdma_surf_mode` 29 | 0 one surface series, 1 two surface series |
| `0x5034` | `erdma_data_size` 3:2 | 1 = 8-bit (the value every verified profile stores) |
| `0x5034` | `erdma_disable` 0 | must stay 0 for a secondary operand |
| `0x5040` | `ew_surf_stride` 31:4 | operand surface stride, in 16-byte atoms (per-channel mode requires 1) |
| `0x5010` | `ew_line_notch_addr` 28:16 | line notch of EW |
| `0x506c` | `ew_surf_notch` 31:4 | "how many pixels from the end of this process feature map to the end of the shape feature map" |

`0x506c` **is already in the elementwise register list** (default 0), so the probe can
set it without changing the program's word count.

## Hypothesis

A compact `H`-row operand table (one 16-byte atom per row) plus `ew_surf_notch =
W-1` lets the ERDMA wrap at the end of each operand row and repeat it across the
output row - a native per-row broadcast. The earlier sweep never set the notch, so
the ERDMA walked off the compact table, which is why it hung instead of broadcasting.

## Result (RV1103, 2026-09-11)

`check_mul_broadcast_notch.py` builds six variants of the verified per-row model
(`mul_broadcast_suite/model002`, 7x5x3, 32 cases); `run_mul_broadcast_notch.py` runs
them with `board_run` against the same expected bytes (`notch_results.json`,
`notch_summary.txt`).

| Variant | `0x5034` | `0x5040` | `0x506c` | Result |
| --- | --- | --- | --- | --- |
| `baseline_materialized` | `0x40000004` | 0x240 (36 atoms) | 0 | **exact**, 32/32 |
| `compact_stride_plane_notch_w-1` | `0x40000004` | 0x240 | 0x40 | runs, **linear read** |
| `compact_stride_1_notch_w-1` | `0x40000004` | 0x10 | 0x40 | runs, linear read |
| `compact_stride_plane_notch_w` | `0x40000004` | 0x240 | 0x50 | runs, linear read |
| `compact_mode2_notch_w-1` | `0x80000004` | 0x240 | 0x40 | **hang** (`failed to wait job` / `job timeout` / `soft reset`) |
| `compact_mode1_surf1_notch_w-1` | `0x60000004` | 0x240 | 0x40 | runs, linear read |

Two findings:

1. **The notch removes the hang.** With `ew_surf_notch` set, every compact variant
   completes its 32 runs; only the undocumented `data_mode=2` still times out. The
   2026-09-10 hangs were a missing notch, not an out-of-range address.
2. **There is no broadcast.** All four completed compact variants produce *byte for
   byte the linear read of the compact table*: the ERDMA consumes one 16-byte atom per
   output pixel in output order, and `ew_surf_stride` (16 or 576 bytes),
   `ew_surf_notch` (0x40/0x50) and `surf_mode` (0/1) do not change the addressing.
   The test recomputes that prediction from the suite's own reference
   (`mul_reference(reference(v, q), linear_plane)`) and asserts equality with the
   board bytes, so the "linear, not broadcast" claim is measured, not inferred.

The secondary operand is therefore read strictly linearly; a spatial Mul constant
stays materialized in a full `[H,W,16]` plane, which is what the public emitter does.
`tests/test_mul_broadcast_notch.py` pins the variants, the board records and the
linear-read prediction.

## Reproduction

```sh
PYTHONPATH=src python3 research/check_mul_broadcast_notch.py
PYTHONPATH=src python3 research/run_mul_broadcast_notch.py
PYTHONPATH=src python3 -m pytest tests/test_mul_broadcast_notch.py -q
```
