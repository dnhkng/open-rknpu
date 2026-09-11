"""Read the LUT argument index directly, to close the non-power-of-two blocker.

The accepted `compile --sequence` LUT profile needs
`BASE_WEIGHT_SCALE/weight_scale` to be a power of two. Other bands get the right
whole-bit negative-half gain but leave +-1 differences on ~10% of values because
the hardware's argument -> index grid is not the reference's `round(64x)`.

This probe makes the index observable:

* every stem is emitted twice, once with the real Sigmoid table (so a candidate
  index rule can be scored against the board byte for byte) and once with the
  table replaced by a **linear ramp of slope 128 Q15 units per index**, whose
  output code is an affine readout of the index actually used;
* the input is a full 0..255 code sweep per channel in four 8x8 images, so the
  index is sampled at every input code for every channel.

Development-only: it reads the emitted table and never changes the public
emitter. `allow_mixed_stems`/`negative_gain_override` are passed explicitly so a
non-power-of-two band can be emitted for measurement.
"""
from pathlib import Path
import json
import struct
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.lut import BASE_WEIGHT_SCALE, compile_lut, lut_reference, negative_half_gain
from open_rknpu.model import checksum
from open_rknpu.quantization import quantize
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lut_index_probe"
OUT.mkdir(exist_ok=True)

# (name, stem gain). 1/32 is the accepted control (ratio 1); the rest are bands
# whose BASE/scale ratio is not a power of two.
STEMS = [
    ("ctrl_1_32", 1 / 32),
    ("npot_0_02", 0.02),
    ("npot_0_03", 0.03),
    ("npot_1_48", 1 / 48),
    ("npot_0_024", 0.024),
    ("npot_0_012", 0.012),
]
RAMP_SLOPE = 128  # Q15 units per table index: one output code per index (255 levels).
# A slope-128 output-code readout only spans 256 codes, so each stem is emitted
# with several (negative-half, positive-half) pivots covering the index range its
# input codes reach: bank 0 ramps down from the negative pivot, bank 1 ramps up
# from the positive pivot, so a decoded code gives the index of either half.
RAMP_PIVOTS = ((256, 512), (384, 640), (448, 768), (512, 896))


def stem_model(gain):
    w = (np.eye(3, dtype=np.float32) * np.float32(gain)).reshape(3, 3, 1, 1)
    b = np.zeros(3, np.float32)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node("Sigmoid", ["conv"], ["output"])], "lut_index",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(w, "w"), nh.from_array(b, "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def patch_ramp(binary, slope, neg_pivot, pos_pivot):
    """Replace the LUT table with a sign-aware linear index readout.

    Bank 0 (the negative argument half, indices 0..512) is written as
    `clip((neg_pivot - i)*slope)`, bank 1 (indices 512..1024) as
    `clip((i - pos_pivot)*slope)`, so the output code is an affine readout of the
    index on both sides of zero.
    """
    info = decode_sequence(binary)
    base = 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)
    data = bytearray(binary[base:])
    setup = info["tasks"][0]
    bank = None
    entry = 0
    for i in range(setup["register_count"]):
        offset = setup["command_offset"] + i * 8
        word = struct.unpack_from("<Q", data, offset)[0]
        reg = word & 0xFFFF
        value = word >> 16 & 0xFFFFFFFF
        if reg == 0x4100:
            bank = {0x20000: 0, 0x30000: 1}.get(value)
            entry = 0
            continue
        if reg == 0x4104 and bank is not None:
            index = bank * 512 + entry
            ramp = (neg_pivot - index) if bank == 0 else (index - pos_pivot)
            ramp = int(np.clip(ramp * slope, -32768, 32767)) & 0xFFFF
            struct.pack_into("<Q", data, offset, (word >> 48) << 48 | ramp << 16 | reg)
            entry += 1
    result = bytearray(binary[:base]) + data
    result[80:84] = b"\0" * 4
    struct.pack_into("<I", result, 80, checksum(result))
    return bytes(result)


def table_of(binary):
    """The 1025 Q15 table entries (two 513-entry banks) the LUT task stores."""
    info = decode_sequence(binary)
    base = 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)
    data = binary[base:]
    setup = info["tasks"][0]
    table = np.zeros(1025, np.int64)
    bank = None
    entry = 0
    for i in range(setup["register_count"]):
        word = struct.unpack_from("<Q", data, setup["command_offset"] + i * 8)[0]
        reg = word & 0xFFFF
        value = word >> 16 & 0xFFFFFFFF
        if reg == 0x4100:
            bank = {0x20000: 0, 0x30000: 512}.get(value)
            entry = 0
        elif reg == 0x4104 and bank is not None:
            table[bank + entry] = np.int32(value << 16).astype(np.int32) >> 16
            entry += 1
    return table


def code_images():
    """Four 8x8 RGB images covering all 256 codes on every channel."""
    images = []
    for j in range(4):
        pixels = np.zeros((8, 8, 3), np.uint8)
        for c in range(3):
            codes = (j * 64 + np.arange(64) + c * 85) % 256
            pixels[:, :, c] = codes.reshape(8, 8)
        images.append(pixels)
    return images


def main():
    manifest = []
    images = code_images()
    for sindex, (name, gain) in enumerate(STEMS):
        quant = quantize(np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1) * np.float32(gain),
                         np.zeros(3, np.float32), 1 / 2048, 0, False, 1.0, 128)
        scale = float(quant.weight_scales[0])
        hardware_gain = negative_half_gain(scale)
        variants = [("sigmoid", 0, None)]
        variants += [("ramp", k + 1, pivot) for k, pivot in enumerate(RAMP_PIVOTS)]
        for kind, tag, pivot in variants:
            model = stem_model(gain)
            binary, _ = compile_lut(model, 1, 128, allow_mixed_stems=True,
                                    negative_gain_override=1.0 / hardware_gain)
            if kind == "ramp":
                binary = patch_ramp(binary, RAMP_SLOPE, pivot[0], pivot[1])
            index = sindex * 10 + tag
            (OUT / f"model{index:03}.bin").write_bytes(binary)
            blob = b"".join(image.tobytes() for image in images)
            (OUT / f"input{index:03}.u8").write_bytes(blob)
            reference = b"".join(
                lut_reference(image, np.eye(3) * gain, np.zeros(3), "Sigmoid").tobytes()
                for image in images)
            (OUT / f"expected{index:03}.i8").write_bytes(reference)
            manifest.append({
                "index": index, "name": name, "gain": gain, "table": kind,
                "neg_pivot": pivot[0] if pivot else None,
                "pos_pivot": pivot[1] if pivot else None,
                "weight_scale": scale, "ratio": BASE_WEIGHT_SCALE / scale,
                "hardware_gain": hardware_gain, "table_gain": 1.0 / hardware_gain,
                "declared_scale": 1 / 2048,
                "ramp_slope": RAMP_SLOPE if kind == "ramp" else None, "images": len(images),
            })
            print(index, name, kind, pivot, "ratio %.6g H %.6g"
                  % (BASE_WEIGHT_SCALE / scale, hardware_gain), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built", len(manifest), "index probes in", OUT)


if __name__ == "__main__":
    main()
