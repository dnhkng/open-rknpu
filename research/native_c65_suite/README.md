# Native input C65–128 (five to eight 16-lane planes)

Extends the native input layout to five through eight 16-lane planes, raising the
public dense native Conv input cap from C64 to **C128**. No channel-split
accumulation is involved: the vendor runs a C128 Conv as a *single* CNA task, so
the earlier "INT32 partial-sum surface" blocker was a cap of ours, not of the
hardware.

**Verified on RV1103: 20 models, 320 inferences, 127,744 exact output bytes**
(`board_api_test native_c65_suite 20`). Models cover input C65/80/96/128, output
C1/3/8/16/17, K1 and K3, 6x5 and 8x8, with independently generated commands and
the documented native integer reference as expected output.

## Layout

Input channels stay packed in 16-lane planes, and the planes are grouped in
**32-lane parts**: all taps of the first 32 lanes, then all taps of the next 32,
and a trailing part holding the last 16 or 32 lanes. Inside a part the layout is
`[tap][lane][member]`, inside each 16-output block:

```
byte(o, t, c) = block*16*k*k*lanes + sum over parts < p of k*k*part_size*OCb
              + t*part_size*OCb + lane*part_size + within
```

where `lanes = ceil(C/16)*16`, `p, within = divmod(c, 32)`,
`part_size = min(32, lanes - 32*p)`, `block = o//16`, `lane = o%16` and
`OCb = min(16, oc - block*16)`. This is the exact generalization of the C33..64
rule (`part_size = 16` for a C48 tail, `32` for C64); the emitted bytes for
C1..64 are unchanged.

## Evidence

Development-only vendor marker captures, one nonzero weight per position:

| Capture | Model | Result |
| --- | --- | --- |
| `capture_native_c65_channels` | C65, one tap, 65 channels | parts at 0, 288, 576 |
| `capture_native_c128_channels` | C128, one tap, 128 channels | parts at 0, 288, 576, 864 |
| `capture_native_c65_oci` | C65, tap 0 for every (o, c), C17 out | 1,105 cells incl. block base 11,520 |

Both channel captures submit **one** 104-byte CNA task (`SUBMIT n rc=0
size=104`) and allocate a five/eight-plane `NC1HWC2` input surface
(`ATTR 8 dims=1,5,6,5,16` / `1,8,6,5,16`). The only vendor register fields that
move with the plane count are the documented lane-derived ones
(`0x1024=(C-1)<<16|lanes`, `0x1030=0x1034=k*k*lanes`, `0x1088=lanes`,
`0x1188=k*k*lanes/2`, bias after the table); `0x1010` also changes, and since our
own C64 emitter already differs from the vendor there (0x48 vs 0x70) and is
board-verified, that field is a scheduling hint rather than a correctness gate.

`research/analyze_native_c48_layout.py` validates all seven marker captures and
`tests/test_native_c48.py` reproduces every marker set from the layout formula,
including the second 16-output block. The open emitter never reads these
captures; they are evidence for the formula.

## Reproduction

```sh
PYTHONPATH=src python research/build_native_c65_suite.py
PYTHONPATH=src python research/run_profile_suite.py native_c65_suite
```

The suite is regenerated independently (no vendor artifacts) and only the
captures above come from the vendor oracle
(`research/build_native_c128_oracle.py`, `research/run_oracle.py`).

## Scope

Dense native Conv input C1..128 is now public and board-verified. Intermediate
channel counts above 16 inside scheduled multi-operator graphs still need the
channel-plane DAG allocation from [completion-plan](../../docs/plans/completion-plan.md)
phase P1, and the vendor's `0x1024` lane field has only been measured up to 128
(8 planes), which is also the tensor-descriptor bound.
