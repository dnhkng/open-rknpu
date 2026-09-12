"""SPDX-License-Identifier: MIT

Cookbook 8/8: the supported mutable-parameter workflow (`open_rknpu.mutable`).

Cookbook 3 does the same thing by hand - it finds the region, splices the bytes and
recomputes the checksum. This script uses the API that ships in the package:

    from open_rknpu.mutable import compile_mutable, constant_regions, graft_region

The rule that makes a v4 container safe to update is that the *band lives in the task
program*, not in the region: the multiplier/shift/zero-point registers are part of the
program, so a replacement packed for another band produces a container that runs and
computes the wrong numbers. The workflow is therefore:

1. compile the host model with `mutable_weights=True` and a **pinned** `output_range`;
2. compile the replacement weights for the *same* band (same shape, same pinned range);
3. `graft_region(host, donor)` - which refuses a donor whose `program_bytes` differ;
4. send the grafted container to the board, or keep the host container and call
   `ornpu_set_constant(model, index, region, size)` with the donor region at run time.

On the board the region is read and written whole:

    ornpu_get_constant(model, 0, &constant);   /* byte_offset, bytes, kind */
    ornpu_set_constant(model, 0, region, constant.bytes);

This script proves the host half byte-for-byte: the grafted container is *identical* to
the donor container, and the integer reference shows how the output changes.

Run:
    PYTHONPATH=src python examples/cookbook/08_mutable_api.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hashlib

import numpy as np

from cookbook_common import (build_dir, conv_node, initializer, model_graph, qfrom,
                             tensor_info)
from open_rknpu.mutable import (compile_mutable, constant_payload, graft_region,
                                program_bytes, replace_constant)
from open_rknpu.native import native_input_reference

FOLDER = build_dir("08_mutable_api")
SEED = 80808
IN_CHANNELS, OUT_CHANNELS, KERNEL, SIZE = 4, 6, 3, 8
# Pinning the band is what makes the two compiles interchangeable. Without it each
# compile derives its own analytic band and `graft_region` refuses the donor.
BAND = {"scale": 0.125, "zero_point": -7}
INPUT_SCALE, INPUT_ZERO_POINT = 1.0, 128


def model(seed, mirror):
    """A C4->C6 3x3 Conv that routes to the mutable `native16-input` profile."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.4, 0.4, (OUT_CHANNELS, IN_CHANNELS, KERNEL, KERNEL)).astype(np.float32)
    if mirror:
        weights = weights[:, :, ::-1, ::-1].copy()
    bias = rng.uniform(-0.3, 0.3, OUT_CHANNELS).astype(np.float32)
    node = conv_node("input", "output", "w", "b", KERNEL, pads=[KERNEL // 2] * 4, strides=(1, 1))
    return model_graph([node], "mutable_api",
                       [tensor_info("input", [1, IN_CHANNELS, SIZE, SIZE])],
                       [tensor_info("output", [1, OUT_CHANNELS, SIZE, SIZE])],
                       [initializer("w", weights), initializer("b", bias)])


def digest(data):
    return hashlib.sha256(bytes(data)).hexdigest()[:16]


def main():
    host, host_meta, host_regions = compile_mutable(model(SEED, False), mutable_weights=True,
                                                    input_scale=INPUT_SCALE,
                                                    input_zero_point=INPUT_ZERO_POINT,
                                                    output_range=BAND)
    donor, donor_meta, donor_regions = compile_mutable(model(SEED, True), mutable_weights=True,
                                                       input_scale=INPUT_SCALE,
                                                       input_zero_point=INPUT_ZERO_POINT,
                                                       output_range=BAND)
    print("host  : %d bytes, region %s, sha256 %s" % (len(host), host_regions[0], digest(host)))
    print("donor : %d bytes, region %s, sha256 %s" % (len(donor), donor_regions[0], digest(donor)))

    assert host_regions[0].name == donor_regions[0].name == "conv.parameters"
    assert host_regions[0].kind == 1
    assert program_bytes(host) == program_bytes(donor), "pinned band => identical programs"
    assert constant_payload(host) != constant_payload(donor), "the packed weights differ"
    print("programs identical, packed regions differ (same band, different weights)")

    # Identity: grafting a region into its own container changes nothing.
    assert replace_constant(host, constant_payload(host)) == host

    updated = graft_region(host, donor)
    assert updated == donor, "the grafted container must reproduce the donor byte-for-byte"
    print("graft_region(host, donor) == donor container (%d bytes, sha256 %s)"
          % (len(updated), digest(updated)))

    # What the update does to the numbers: same band, different taps.
    case = np.random.default_rng(SEED + 1).integers(0, 256, (SIZE, SIZE, IN_CHANNELS), dtype=np.uint8)
    before = native_input_reference(case, qfrom(host_meta["quantization"]), INPUT_ZERO_POINT)
    after = native_input_reference(case, qfrom(donor_meta["quantization"]), INPUT_ZERO_POINT)
    changed = int((before != after).sum())
    print("one %dx%dx%d case: %d/%d output codes changed, max |delta| = %d LSB"
          % (SIZE, SIZE, IN_CHANNELS, changed, before.size,
             int(np.abs(before.astype(int) - after.astype(int)).max())))
    assert changed > 0, "the mirrored taps must change the output"

    (FOLDER / "host.bin").write_bytes(host)
    (FOLDER / "donor.bin").write_bytes(donor)
    (FOLDER / "updated.bin").write_bytes(updated)
    (FOLDER / "region.bin").write_bytes(constant_payload(donor))
    (FOLDER / "case.u8").write_bytes(case.tobytes())
    (FOLDER / "before.i8").write_bytes(before.tobytes())
    (FOLDER / "after.i8").write_bytes(after.tobytes())
    print("wrote %s/{host,donor,updated,region}.bin and the reference bytes" % FOLDER)
    print("board: push updated.bin and run it, or call ornpu_set_constant with region.bin "
          "(the host container keeps its program)")


if __name__ == "__main__":
    main()
