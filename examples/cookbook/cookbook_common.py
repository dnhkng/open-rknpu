"""SPDX-License-Identifier: MIT

Shared helpers for the `examples/cookbook/` scripts.

The cookbook deliberately reuses `examples/primitives/common.py` - the graph
builders, `compile_and_report`, `qfrom`, the deterministic case generator and
the board-suite publishing helpers all come from there, loaded by file path so
this folder never shadows that module. This file only adds the pieces the
cookbook needs on top of it:

* `build_dir` - every script writes its containers under
  `examples/cookbook/build/<script>/` (already covered by `.gitignore`).
* `registers` / `constant_region` / `splice_constant` / `tail_words` - read and
  rewrite a compiled container the way the runtime does, so a script can show
  `ornpu_set_constant` semantics without a board.
* `checksum_masked` - byte comparison of two containers ignoring the checksum
  field, which is what "the programs are identical" means here.
"""
import importlib.util
import struct
import sys
from pathlib import Path

COOKBOOK = Path(__file__).resolve().parent
PRIMITIVES = COOKBOOK.parent / "primitives"
BUILD = COOKBOOK / "build"


def _load_primitives():
    """Load `examples/primitives/common.py` under a private module name."""
    path = PRIMITIVES / "common.py"
    if not path.exists():
        raise FileNotFoundError("examples/primitives/common.py is missing; run from the repository tree")
    spec = importlib.util.spec_from_file_location("_cookbook_primitives_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


primitives = _load_primitives()

# Re-export the primitives helpers the cookbook shares, so a script imports one module.
FLOAT = primitives.FLOAT
assert_int8_reference = primitives.assert_int8_reference
compile_and_report = primitives.compile_and_report
container_info = primitives.container_info
conv_node = primitives.conv_node
deterministic_cases = primitives.deterministic_cases
initializer = primitives.initializer
model_graph = primitives.model_graph
profile_of = primitives.profile_of
qfrom = primitives.qfrom
relu_node = primitives.relu_node
tensor_info = primitives.tensor_info


def build_dir(name):
    """The git-ignored output directory of one cookbook script."""
    folder = BUILD / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# --------------------------------------------------------------------------- #
# Reading a compiled container the way runtime/open_rknpu.c does
# --------------------------------------------------------------------------- #
def _sequence():
    import open_rknpu.sequence as sequence
    return sequence


def registers(binary, info, task_index=0):
    """The register file of one emitted task, as `{register: value}`.

    Register words are `tag << 48 | value << 16 | register`, exactly as
    `runtime/sequence_format.md` documents, so this is a direct read of the
    container rather than a re-derivation of emitter constants.
    """
    base = _sequence().payload_base(info)
    task = info["tasks"][task_index]
    values = {}
    for index in range(task["register_count"]):
        word = struct.unpack_from("<Q", binary, base + task["command_offset"] + index * 8)[0]
        values[word & 0xFFFF] = (word >> 16) & 0xFFFFFFFF
    return values


def constant_descriptor(info, name):
    """The v4/v5 constant descriptor with `name` (raises if the container has none)."""
    for entry in info["constants"]:
        if entry["name"] == name:
            return entry
    raise KeyError("container exposes no constant descriptor named %r" % name)


def constant_region(binary, info, name):
    """`(descriptor, bytes)` for one named packed constant region."""
    descriptor = constant_descriptor(info, name)
    base = _sequence().payload_base(info)
    start = base + descriptor["offset"]
    return descriptor, binary[start:start + descriptor["size"]]


def splice_constant(binary, info, name, replacement):
    """Host-side `ornpu_set_constant`: swap one complete packed region and re-hash.

    The board sequence is `ornpu_open` -> `ornpu_get_constant` (read `byte_offset`
    and `bytes`) -> `ornpu_set_constant(model, index, data, size)` -> `ornpu_run`;
    the runtime copies the bytes into the mapped payload and re-validates the
    container. This function performs the same copy and recomputes the checksum
    with the project's own FNV-1a function, so the result is a valid container.
    """
    descriptor, current = constant_region(binary, info, name)
    if len(replacement) != len(current):
        raise ValueError("replacement has %d bytes, descriptor %r holds %d"
                         % (len(replacement), name, len(current)))
    out = bytearray(binary)
    base = _sequence().payload_base(info)
    start = base + descriptor["offset"]
    out[start:start + len(current)] = replacement
    out[80:84] = b"\0" * 4
    from open_rknpu.model import checksum
    struct.pack_into("<I", out, 80, checksum(bytes(out)))
    _sequence().decode_sequence(bytes(out))
    return bytes(out)


def tail_words(binary, info):
    """`(next-program link, tail control)` for every task descriptor.

    The two tail words are registers `0x10` (link) and `0x14` (control); a serial
    container keeps the terminal `0x28` sentinel and a zero link, while a batched
    container links each task to the successor's fetch amount (`0x40` for a
    126-word task). See `docs/plans/pipelining-plan.md` S1/S8/S10.
    """
    base = _sequence().payload_base(info)
    tails = []
    for task in info["tasks"]:
        start = base + task["command_offset"] + task["register_count"] * 8
        link_word, control_word = struct.unpack_from("<QQ", binary, start)
        tails.append(((link_word >> 16) & 0xFFFFFFFF, (control_word >> 16) & 0xFFFFFFFF))
    return tails


def checksum_masked(data):
    """The container bytes with the four checksum-field bytes zeroed."""
    out = bytearray(data)
    out[80:84] = b"\0" * 4
    return bytes(out)
