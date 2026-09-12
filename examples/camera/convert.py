"""SPDX-License-Identifier: MIT

The reference capture-frame conversion for `examples/camera/` (checklist row E5).

A V4L2 capture node hands the caller one packed frame in a YUV format; the NPU wants a
packed NHWC UINT8 RGB buffer at the model's input geometry. This module is the single
definition of the conversion between those two, and `camera_npu.c` implements exactly the
same rule in C. `tests/test_example_camera.py` compiles the C harness on the host, runs its
`--emit-input` mode on the synthetic frames, and asserts the C bytes equal the bytes this
module produces - that byte equality *is* the conversion contract.

The rule, in order:

1. **Decode the capture frame to RGB888** at the frame geometry `(width, height)`, one byte
   per channel, row-major, three bytes per pixel, using full-range BT.601 with the integer
   (Q8) coefficients below. `>= 0` shifts are floor divisions by 256 in both languages, and
   the C helper is written to floor negative values too, so the result is bit-identical.
   * `yuyv` - YUYV 4:2:2: `Y0 U Y1 V` per two horizontal pixels; the pair shares `U`/`V`.
   * `nv12` - NV12 4:2:0: a `height x width` `Y` plane followed by an interleaved `U`/`V`
     plane at `height/2 x width/2`; each 2x2 luma block shares one `U`/`V` pair.
   * `rgb`  - already RGB888, used unchanged.
2. **Downsample to the model input** `(out_width, out_height)` by nearest neighbour:
   `src_x = x * width // out_width`, `src_y = y * height // out_height`. No filtering, no
   averaging, no rounding: the same integer ratio in both languages.
3. **Emit packed NHWC UINT8**: `out_height` rows of `out_width * 3` bytes, no row padding
   or alignment. That is exactly the API layout `ornpu_run` documents, and the runtime's
   own 16-byte row stride is applied later, inside the arena.

Why integer arithmetic and nearest-neighbour: a float `round()` differs between Python's
banker's rounding and C's `nearbyint`, and a box filter divides by a geometry-dependent
count. Neither is needed for the plumbing this example demonstrates, and the byte equality
between the two implementations is a much stronger statement when the rule is integer-only.

Run as a command to convert a recorded frame (host-only, no board):

    PYTHONPATH=src python examples/camera/convert.py --frame examples/camera/build/frame.nv12 \\
      --format nv12 --width 64 --height 48 --out-width 32 --out-height 32 \\
      --output /tmp/camera-input.u8
"""
import argparse
import hashlib
import sys
from pathlib import Path

# Full-range BT.601, Q8. Encode (used by build.py to synthesize frames from a source RGB
# scene): Y = (77R + 150G + 29B) / 256; U = (-43R - 85G + 128B) / 256 + 128;
# V = (128R - 107G - 21B) / 256 + 128. Decode (this file, and the C harness):
# R = Y + 359/256 * V'; G = Y - (88/256 * U' + 183/256 * V'); B = Y + 454/256 * U',
# with U' = U - 128 and V' = V - 128.
LUMA_R, LUMA_G, LUMA_B = 77, 150, 29
CR_R, CR_G, CR_B = 128, -107, -21
CB_R, CB_G, CB_B = -43, -85, 128
DECODE_R_V = 359
DECODE_G_U, DECODE_G_V = 88, 183
DECODE_B_U = 454

FORMATS = ("yuyv", "nv12", "rgb")


def frame_bytes(fmt, width, height):
    """The exact capture buffer size for one frame, or `ValueError` for a bad geometry."""
    if fmt not in FORMATS:
        raise ValueError("unsupported capture format %r (expected yuyv, nv12 or rgb)" % fmt)
    if width < 1 or height < 1:
        raise ValueError("frame geometry must be positive, got %dx%d" % (width, height))
    if fmt in ("yuyv", "nv12") and width % 2:
        raise ValueError("%s requires an even width, got %d" % (fmt, width))
    if fmt == "nv12" and height % 2:
        raise ValueError("nv12 requires an even height, got %d" % height)
    if fmt == "yuyv":
        return width * height * 2
    if fmt == "nv12":
        return width * height + 2 * (width // 2) * (height // 2)
    return width * height * 3


def _floor_shift(value, shift):
    """`value >> shift` with a floor result for negative values (Python's own `>>`)."""
    return value >> shift


def _clip_u8(value):
    return 0 if value < 0 else 255 if value > 255 else value


def decode_pixel(y, u, v):
    """One full-range BT.601 YUV triple to an `(r, g, b)` byte triple."""
    du = u - 128
    dv = v - 128
    r = _clip_u8(y + _floor_shift(DECODE_R_V * dv, 8))
    g = _clip_u8(y - _floor_shift(DECODE_G_U * du + DECODE_G_V * dv, 8))
    b = _clip_u8(y + _floor_shift(DECODE_B_U * du, 8))
    return r, g, b


def decode_to_rgb(data, fmt, width, height):
    """Decode one capture buffer to RGB888: `height * width * 3` row-major bytes."""
    expected = frame_bytes(fmt, width, height)
    if len(data) != expected:
        raise ValueError("%s frame for %dx%d must be %d bytes, got %d"
                         % (fmt, width, height, expected, len(data)))
    if fmt == "rgb":
        return bytes(data)
    out = bytearray(width * height * 3)
    if fmt == "yuyv":
        for y in range(height):
            row = y * width * 2
            for x in range(0, width, 2):
                base = row + x * 2
                u = data[base + 1]
                v = data[base + 3]
                for offset, luma in ((0, data[base]), (1, data[base + 2])):
                    r, g, b = decode_pixel(luma, u, v)
                    at = (y * width + x + offset) * 3
                    out[at], out[at + 1], out[at + 2] = r, g, b
        return bytes(out)
    # NV12: the luma plane is the first width*height bytes; chroma is interleaved at half
    # resolution, one U/V pair per 2x2 block.
    luma_plane = width * height
    for y in range(height):
        for x in range(width):
            chroma = luma_plane + 2 * ((y // 2) * (width // 2) + x // 2)
            r, g, b = decode_pixel(data[y * width + x], data[chroma], data[chroma + 1])
            at = (y * width + x) * 3
            out[at], out[at + 1], out[at + 2] = r, g, b
    return bytes(out)


def downsample_nearest(rgb, width, height, out_width, out_height):
    """Nearest-neighbour resample of an RGB888 raster; integer source-pixel ratios only."""
    if width < 1 or height < 1 or out_width < 1 or out_height < 1:
        raise ValueError("downsample geometry must be positive")
    if len(rgb) != width * height * 3:
        raise ValueError("RGB raster for %dx%d must be %d bytes, got %d"
                         % (width, height, width * height * 3, len(rgb)))
    out = bytearray(out_width * out_height * 3)
    for y in range(out_height):
        src_y = y * height // out_height
        for x in range(out_width):
            src_x = x * width // out_width
            source = (src_y * width + src_x) * 3
            target = (y * out_width + x) * 3
            out[target:target + 3] = rgb[source:source + 3]
    return bytes(out)


def convert_frame(data, fmt, width, height, out_width, out_height):
    """The whole contract: capture buffer -> packed NHWC UINT8 at the model geometry."""
    return downsample_nearest(decode_to_rgb(data, fmt, width, height),
                              width, height, out_width, out_height)


def conversion_rule():
    """The rule as data, for `report.json` and the README."""
    return {
        "decode": ("full-range BT.601 YUV->RGB with integer Q8 coefficients; shifts are "
                   "floor divisions by 256: R=Y+(359*(V-128)>>8), "
                   "G=Y-((88*(U-128)+183*(V-128))>>8), B=Y+(454*(U-128)>>8), "
                   "each clamped to 0..255"),
        "formats": {
            "yuyv": "4:2:2, bytes Y0 U Y1 V per two pixels; the pair shares U and V",
            "nv12": "4:2:0, a width*height Y plane then interleaved U,V at width/2 x height/2",
            "rgb": "RGB888 row-major, three bytes per pixel, used unchanged",
        },
        "downsample": ("nearest neighbour: src_x = x * width // out_width, "
                       "src_y = y * height // out_height (no filtering)"),
        "layout": "packed NHWC UINT8, out_height rows of out_width*3 bytes, no row padding",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert one captured frame to the NPU's "
                                                 "packed NHWC UINT8 input (host-only).")
    parser.add_argument("--frame", required=True, help="recorded capture buffer")
    parser.add_argument("--format", required=True, choices=FORMATS)
    parser.add_argument("--width", required=True, type=int, help="frame width in pixels")
    parser.add_argument("--height", required=True, type=int, help="frame height in pixels")
    parser.add_argument("--out-width", required=True, type=int, help="model input width")
    parser.add_argument("--out-height", required=True, type=int, help="model input height")
    parser.add_argument("--output", required=True, help="packed NHWC UINT8 destination")
    args = parser.parse_args(argv)
    data = Path(args.frame).read_bytes()
    packed = convert_frame(data, args.format, args.width, args.height,
                           args.out_width, args.out_height)
    Path(args.output).write_bytes(packed)
    print("convert frame=%s format=%s %dx%d -> %dx%d bytes=%d sha256=%s"
          % (args.frame, args.format, args.width, args.height, args.out_width,
             args.out_height, len(packed), hashlib.sha256(packed).hexdigest()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
