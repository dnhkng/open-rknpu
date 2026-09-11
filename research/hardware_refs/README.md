# Related-hardware references and RV1103 hypotheses

Read on 2026-09-08. These documents guide hypotheses; **RV1103 compatibility must
be established by captures and independent command-generation tests**. Compiler
and runtime execution do not load these documents or require Mesa/Rocket.

Sources:

- [Linux Rocket documentation](https://docs.kernel.org/accel/rocket/index.html)
  points to chapter 36 of the RK3588 TRM and distinguishes kernel submission from
  Mesa userspace work. Its supported hardware list is RK3588, not RV1103.
- [Mesa Rocket registers.xml](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/gallium/drivers/rocket/registers.xml)
  supplies register and field names. Local copy: `mesa_registers.xml`.
- [Mesa rkt_regcmd.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/gallium/drivers/rocket/rkt_regcmd.c)
  provides a useful example of composing fields and scaling an Add input.
- [RK3588 TRM Part 1](https://gitlab.com/rock-chips/rk3588/rk3588-doc/-/blob/master/Rockchip%20RK3588%20TRM%20V1.0-Part1-20220309.pdf),
  chapter 36: inspected architecture/address map and relevant DPU/CNA register
  descriptions. Printed pages 1984–1986, 2001–2006 and 2025–2027 are relevant.
- [Linux Rocket generated register header](https://github.com/torvalds/linux/blob/master/drivers/accel/rocket/rocket_registers.h)
  is a supplementary reference; its own header records Mesa provenance.

Downloaded file URLs and hashes are recorded in `sources.json`. Keep these
third-party reference materials separate from our MIT implementation; retain their
upstream notices. Do not package the TRM into compiler distributions.

## Cross-checks for current Add work

| Field/reference | RV1103 observation | Status |
|---|---|---|
| DPU EW config at 0x4070, ALU selector bits 19:16 | Captured Add uses 2 and Max uses 0. Changing only these bits in a fresh Add program produces independently reference-checked Max output. | Verified for 8x8/C3 equal-scale profile |
| Output conversion at 0x4088, rounding bit 30 | Baseline 0 gives ties-even. Setting 1 gives ties-away-from-zero. | Verified on signed inputs; not assumed for other pipelines |
| Output scale low 16 bits at 0x4084, shift at 0x4088 | Matches the Add implementation's scale/shift structure. | Consistent, not exhaustively isolated |
| CNA stride/dilation at 0x1014 | Names provide specific candidates for future stride/dilation probes. | Hypothesis; wider profile not implemented |
| CNA padding at 0x1068 | RK3588 places top/left padding in adjacent nibbles. RV1103's verified symmetric pad1 value is 0x0101, rather than the RK3588 layout's 0x0011. | Known incompatibility; do not import layout |
| EW config bits 15:11 | Mesa/TRM mark them reserved; RV1103 Add uses bits 15 and 14. | Preserve verified RV1103 values; meaning unresolved |
| Output scale bit 16 | Related docs name FP32-to-FP16 enable, while the RV1103 int8 Add profile uses 0x14000. | Do not infer RV1103 dtype behavior from this name |

The TRM describes an acceleration core, DPU, planar processing and configuration
fetch units. Its register map associates CNA/CORE/DPU/PPU with the same broad
address windows seen in our captures. This supports engine-family hypotheses,
not identical field encodings or capabilities.

The TRM's output-rounding description initially suggested testing a half-up
interpretation. Negative halfway cases rejected that interpretation; signed
ties-away-from-zero matched. Preserve that distinction in subsequent work.

## Discriminating hardware experiments

`../check_add_register_hypotheses.py` starts from our independently generated Add
model, never from a vendor command capture. It produces two modified programs:

1. Rounding bit 30: 1,438 output bytes are predicted to differ from baseline.
2. ALU selector Max: 5,966 output bytes are predicted to differ from baseline.

Both pass all 32 cases each: **64 inferences, 12,288 exact bytes**, through the
public C runtime. Evidence: `../add_register_suite/board.log`; the original
rejected half-up prediction is retained in `initial_half_up_hypothesis.log`.
The Max result is a field experiment, not yet a public Max graph compiler.

Run from repository root:

```bash
PYTHONPATH=src python research/check_add_register_hypotheses.py
```

Transfer the six `.bin`/`.u8`/`.i8` files to the matching board suite directory,
then run `board_api_test /userdata/open-npu-research/add_register_suite 2`.

For each remaining primitive: identify a named field, predict its effect, compare
the RV1103 vendor capture, change one field in independently generated commands,
and verify signed/boundary inputs. Only then promote it into production lowering.

## Mul follow-up

The independent Mul emitter uses the EW_OP_TYPE=MUL hypothesis (0x4070 bit2)
and OD_BYPASS (0x4050 bit1), preserving RV1103-specific packed values. The full
profile with output conversion A*B/128 passed 12 models, 384 runs and 73,728
exact bytes. This validates the combined profile, not every changed bit in
isolation. See [Mul evidence](../mul_suite/README.md).
