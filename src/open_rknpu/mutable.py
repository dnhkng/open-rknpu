"""SPDX-License-Identifier: MIT

Mutable packed parameters: read and replace the named constant regions of a v4 container.

A v4 `ORNPUSEQ` container carries, after the task table, a list of **named packed constant
regions** (`docs/container-format.md`, `docs/container-migration.md`). The runtime exposes
them as `ornpu_get_constant` / `ornpu_set_constant`, and the compiler emits them only when
asked:

* `compile_sequence(model, mutable_weights=True)` emits `conv.parameters` (kind 1) - the
  whole packed weight + bias + per-channel conversion region of the native Conv profile;
* `compile_sequence(model, mutable_constants=True)` emits `mul.factor` (kind 3) - the packed
  INT8 factor of the constant-Mul profile.

The replacement contract is intentionally narrow, and this module enforces it on the host:
a region is replaced **whole**, its length must match the descriptor exactly, and the task
program must stay the same, because the multiplier/shift/zero-point registers are part of
the program and not of the region. A replacement packed for a different band produces a
container that runs but computes the wrong thing, so `program_bytes` is exposed to compare
two containers before swapping a region between them.

    from open_rknpu.mutable import compile_mutable, constant_regions, replace_constant
    binary, meta, regions = compile_mutable("model.onnx", mutable_weights=True)
    print(regions[0])                       # ConstantRegion(name='conv.parameters', kind=1, ...)
    updated = replace_constant(binary, new_region_bytes, name="conv.parameters")

`examples/cookbook/03_mutable_parameters.py` demonstrates the same mechanism end to end and
`examples/cookbook/08_mutable_api.py` uses this API. On the board the equivalent calls are
`ornpu_get_constant` (to learn `byte_offset`/`bytes`/`kind`) and `ornpu_set_constant` (to
copy the replacement into the mapped payload and re-validate).
"""
from dataclasses import dataclass

from .model import checksum, decode as decode_model
from .sequence import decode_sequence, payload_base

# The kinds the loader accepts. Only 1 and 3 are emitted by this compiler; 2 is accepted by
# the runtime's validator but has no producer here, so it is named for completeness only.
KIND_NAMES = {1: "packed Conv parameters", 2: "reserved", 3: "packed multiplication factor"}


@dataclass(frozen=True)
class ConstantRegion:
    """One named packed constant region of a v4 container."""

    name: str
    kind: int
    kind_name: str
    offset: int        # relative to the payload base, as the descriptor stores it
    size: int
    byte_offset: int   # absolute file offset, as `ornpu_constant_info.byte_offset` reports

    def __str__(self):
        return (f"ConstantRegion({self.name!r}, kind={self.kind} [{self.kind_name}], "
                f"offset={self.offset}, size={self.size}, byte_offset={self.byte_offset})")


def container_info(binary):
    """Decode a container (`ORNPUSEQ` or legacy `ORNPUBIN`) and return its metadata."""
    data = bytes(binary)
    if data[:8] == b"ORNPUSEQ":
        return decode_sequence(data)
    return decode_model(data)


def constant_regions(binary):
    """Every mutable constant region of a container, in descriptor order.

    Raises `ValueError` for a legacy (v1/v2) or v5 container: neither has a constant
    descriptor table, so there is nothing to replace. That is a property of the format, not
    a missing implementation.
    """
    info = container_info(binary)
    version = info["format_version"]
    if version == 5:
        raise ValueError("v5 containers have no constant descriptor table; "
                         "mutable parameters require a v4 container")
    if version not in (3, 4):
        raise ValueError("legacy containers have no constant descriptor table; "
                         "mutable parameters require a v4 container")
    base = payload_base(info)
    regions = []
    for entry in info.get("constants") or []:
        kind = int(entry.get("kind", 1))
        regions.append(ConstantRegion(name=str(entry["name"]), kind=kind,
                                      kind_name=KIND_NAMES.get(kind, f"kind-{kind}"),
                                      offset=int(entry["offset"]), size=int(entry["size"]),
                                      byte_offset=base + int(entry["offset"])))
    return tuple(regions)


def _select(regions, name, index):
    if name is not None:
        matches = [region for region in regions if region.name == name]
        if not matches:
            known = ", ".join(repr(region.name) for region in regions) or "none"
            raise ValueError(f"no constant region named {name!r}; this container has {known}")
        return matches[0]
    if not 0 <= index < len(regions):
        raise ValueError(f"constant region index {index} out of range "
                         f"({len(regions)} region(s))")
    return regions[index]


def constant_payload(binary, name=None, index=0):
    """The bytes of one named (or indexed) constant region."""
    data = bytes(binary)
    region = _select(constant_regions(data), name, index)
    return data[region.byte_offset:region.byte_offset + region.size]


def program_bytes(binary):
    """The container up to the first constant region, with the checksum field zeroed.

    Header, task table, descriptors and programs: everything whose values come from the
    model's *band* rather than from the packed region, which is what two containers must
    share before a region may be exchanged. The four checksum bytes are zeroed because the
    checksum covers the region too, so it necessarily differs between two donors.
    """
    data = bytes(binary)
    regions = constant_regions(data)
    prefix = data if not regions else data[:min(region.byte_offset for region in regions)]
    masked = bytearray(prefix)
    masked[80:84] = b"\0" * 4
    return bytes(masked)


def replace_constant(binary, replacement, name=None, index=0):
    """Return a new container with one whole constant region replaced.

    `replacement` must be exactly the region's length. The FNV-1a checksum is recomputed,
    so the result is a container the runtime accepts. This is the host-side equivalent of
    `ornpu_set_constant(model, index, data, size)`.
    """
    data = bytes(binary)
    region = _select(constant_regions(data), name, index)
    payload = bytes(replacement)
    if len(payload) != region.size:
        raise ValueError(f"replacement has {len(payload)} bytes, region {region.name!r} "
                         f"holds {region.size}")
    out = bytearray(data)
    out[region.byte_offset:region.byte_offset + region.size] = payload
    out[80:84] = b"\0" * 4
    out[80:84] = checksum(bytes(out)).to_bytes(4, "little")
    return bytes(out)


def graft_region(host, donor, name=None, index=0):
    """Replace one region of `host` with the same region taken from `donor`.

    This is the safe form of `replace_constant` for a region produced by a second compile:
    it first requires the two containers to share their task program (`program_bytes`), which
    is what makes the packed region and the program's band agree. A donor compiled for a
    different band raises instead of producing a container that runs and computes the wrong
    thing. Pin the band explicitly (`compile_sequence(..., output_range=...)`) when building
    the donor; `examples/cookbook/08_mutable_api.py` shows the whole workflow.
    """
    host_regions = constant_regions(host)
    region = _select(host_regions, name, index)
    donor_regions = constant_regions(donor)
    match = [candidate for candidate in donor_regions if candidate.name == region.name]
    if not match:
        known = ", ".join(repr(candidate.name) for candidate in donor_regions) or "none"
        raise ValueError(f"the donor has no constant region named {region.name!r} "
                         f"(it has {known})")
    host_program, donor_program = program_bytes(host), program_bytes(donor)
    if host_program != donor_program:
        offset = next((position for position, (a, b) in
                       enumerate(zip(host_program, donor_program)) if a != b), None)
        raise ValueError("the containers do not share a task program, so their bands differ "
                         f"(first difference at byte {offset}); compile the donor for the "
                         "host's band (pin output_range) before grafting")
    return replace_constant(host, constant_payload(donor, name=region.name), name=region.name)


def compile_mutable(model, *, mutable_weights=False, mutable_constants=False, **kwargs):
    """Compile a model that carries replaceable regions.

    A thin wrapper over `open_rknpu.scheduler.compile_sequence`: it forwards the mutable
    flags, and fails loudly if the resulting container has no constant table (a profile
    without a mutable form) instead of returning a container the caller cannot update.
    Returns `(binary, meta, regions)`.
    """
    from .scheduler import compile_sequence
    if not mutable_weights and not mutable_constants:
        raise ValueError("compile_mutable needs mutable_weights or mutable_constants")
    binary, meta = compile_sequence(model, mutable_weights=mutable_weights,
                                    mutable_constants=mutable_constants, **kwargs)
    regions = constant_regions(binary)
    if not regions:
        raise ValueError("the compiled profile has no mutable constant region; "
                         "no v4 container was emitted")
    return bytes(binary), meta, regions
