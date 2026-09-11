# Native input C33–48 (three 16-lane planes)

Public `compile --sequence` now accepts dense native Conv input channels 33–48
in addition to C1–32. The three 16-lane input planes use a plane-pair weight
layout recovered from vendor marker captures.

**Verified on RV1103: 12 models, 192 inferences, 71,328 exact output bytes**
(`board_api_test native_c48_suite 12`). Models cover input C33/40/48, output
C1/3/8/16, K1 and K3, 6x5 and 8x8. Commands and weights are independently
generated; expected bytes come from the documented native input integer
reference. No RKNN model or library is used by the emitter.

## Layout

Within each 16-output-channel block the input lanes are grouped as two planes
(32 lanes) followed by the remaining planes:

```
byte(o, t, c) = block*16*k*k*lanes
              + (t*OCb + lane)*lanes + c                       # lanes <= 32
              + t*32*OCb + lane*32 + c                          # c < 32,  lanes > 32
              + k*k*32*OCb + t*part_b*OCb + lane*part_b + (c-32) # c >= 32, lanes > 32
```

where `lanes = ceil(C/16)*16`, `block = o//16`, `lane = o%16`,
`OCb = min(16, oc - block*16)` and `part_b = lanes - 32`.

## Evidence

Development-only vendor marker captures, one nonzero weight per position:

| Capture | Model | Result |
| --- | --- | --- |
| `capture_native_c48_taps` | C48, one input channel, nine taps | tap t at `32*t` |
| `capture_native_c48_channels` | C48, one tap, 48 channels | c<32 at `c`, c>=32 at `256+c` |
| `capture_native_c48_oci` | C48, identity markers, C16 out | output o at `33*o` |
| `capture_native_c64_channels` | C64, one tap, 64 channels | c<32 at `c`, else `256+c` |

`tests/test_native_c48.py` reproduces every marker set from the layout formula.
The open emitter never reads these captures; they are evidence for the formula.

**Generalized 2026-09-11:** the two-part rule above is the `part_size = 16` tail case
of the 32-lane-part rule verified for C65..128 in
[`native_c65_suite/`](../native_c65_suite/); emitted bytes for C1..64 are unchanged.
