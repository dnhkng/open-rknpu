# ERDMA data-mode probe — retained failed hypothesis

The EW secondary operand is read by the ERDMA unit. Mesa names its registers:

| Register | Field | Bits |
| --- | --- | --- |
| `0x5034` | `RDMA_ERDMA_CFG` | `ERDMA_DATA_MODE` 30..31, `ERDMA_SURF_MODE` 29, `ERDMA_NONALIGN` 28, `ERDMA_DATA_SIZE` 2..3 |
| `0x5038` | `RDMA_EW_BASE_ADDR` | operand base address |
| `0x5040` | `RDMA_EW_SURF_STRIDE` | stride bits 4..31 |

The two verified constant-Mul modes are:

* `DATA_MODE=0`: a single 16-byte per-channel vector, `EW_SURF_STRIDE=16`.
* `DATA_MODE=1`: a full per-pixel `[H,W,16]` plane, `EW_SURF_STRIDE=plane size`.

## Hypothesis tested

Store a compact per-row table (H rows of 16 bytes) with a small
`EW_SURF_STRIDE`, and let `DATA_MODE` 0/2/3 or `ERDMA_SURF_MODE` broadcast it, so a
`[1,1,H,1]` (or `[1,1,H,W]`) immutable Mul constant need not be materialized.

`check_mul_broadcast_modes.py` builds 12 variants from the board-verified
`mul_broadcast_suite` row-constant model, all scored against that suite's expected
bytes, so a correct broadcast must match exactly.

## Result

* Model 0 (unchanged, materialized) passed all 32 inputs, confirming the harness.
* Every compact variant (`DATA_MODE` 0/2/3, `SURF_MODE` 0/1, stride 16 or plane
  size) **hung the NPU** instead of returning a wrong value. The driver logged
  `failed to wait job`, `job timeout`, `soft reset` and `job abort ret: -22`; the
  board recovered by soft reset and `rkipc` stayed alive.

The compact modes therefore do not broadcast. **Decoded 2026-09-11:** the variants
hung because `0x506c` `ew_surf_notch` was left at zero (the RK3588 TRM names it, and
`0x5010` bits 28:16 as `ew_line_notch_addr`). Setting the notch makes every compact
variant complete, and the completed read is byte for byte the *linear* read of the
compact table, unchanged by stride, notch or `surf_mode` (`../mul_broadcast_notch_suite/`).
No native compact spatial-broadcast mode exists, and public spatial Mul constants
remain materialized (unchanged, still board verified).

Evidence: `manifest.json`, the 12 `modelNNN.bin` variants, and the driver log
above. `mul_broadcast_mode_suite` is excluded from the passing ledger.
