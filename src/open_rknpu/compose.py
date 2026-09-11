"""SPDX-License-Identifier: MIT

Stage composer: a declared list of stages becomes one version-5 container.

Every scheduled profile used to assemble its container by hand: it picked arena
offsets, appended program slots at literal addresses, copied constant blocks, wrote
the tensor table and listed the tasks. This module does that in one pass from a
*declaration*:

* each `Stage` names the tensors it `reads` and `writes`, the register words it
  sets (`fields`), the constant blocks it owns, and the address registers it binds
  (`bindings`);
* `compose` orders the tasks topologically with `open_rknpu.liveness`, places every
  internal tensor with the same lifetime allocator, assigns program slots and
  constant blocks in declared order, substitutes the planned addresses into each
  stage's register words, fills the tensor table and encodes the container;
* the declaration is returned in `meta['declared_bindings']`, and
  `derive_bindings` reads the *same* container back from its address registers, so
  an emitter's declared view is checked against the container it actually produced.

The placement policy is explicit and shared: program slots are `(words + 4) * 8`
bytes apart (four terminal words per task) aligned to 64; constant blocks follow in
declared order, each aligned to 64; external inputs start at the payload end
aligned to 4096 and the arena starts after them; internals are placed by liveness;
external outputs follow the last internal; the arena is rounded to 4096.

No vendor artifact, capture or runtime is read; the pass is pure bookkeeping.
"""
from dataclasses import dataclass
import struct

from .liveness import Access, live_intervals, plan, topological_order
from .sequence import ROLE_INPUT, ROLE_INTERNAL, ROLE_OUTPUT
from .pooling import POOL, pool_tag
from .register_profile import REGISTERS
from .sequence import (TERMINAL_CONTROL, amount_control, encode_sequence_v5,
                       payload_base)


def _align(n, a=64):
    return (n + a - 1) // a * a


# --- task families ---------------------------------------------------------
# A family fixes the register word count, the task enable/mask and the register
# defaults the emitter overrides. `reads`/`writes` are the address registers that
# carry tensor offsets, so the derived view can be read back from a container.
@dataclass(frozen=True)
class Family:
    name: str
    words: int
    enable: int
    mask: int
    base: tuple
    reads: tuple = ()
    writes: tuple = ()


NATIVE = Family("native-conv", 126, 29, 768, tuple(REGISTERS), (0x1070,), (0x4020,))
POOL_TASK = Family("pool", 37, 96, 3072,
                   tuple((register, value, pool_tag(register)) for register, value in POOL),
                   (0x701c,), (0x6070,))
ELEMENTWISE = Family("elementwise", 78, 24, 768,
                     tuple(entry for entry in REGISTERS if entry[0] >= 0x4000),
                     (0x5018, 0x5038), (0x4020,))
LUT_SETUP = Family("lut-setup", 1106, 24, 768,
                   tuple(entry for entry in REGISTERS if entry[0] >= 0x4000))
FAMILIES = {family.name: family for family in (NATIVE, POOL_TASK, ELEMENTWISE, LUT_SETUP)}
FAMILY_BY_SIGNATURE = {(family.words, family.enable): family for family in FAMILIES.values()}


@dataclass(frozen=True)
class Binding:
    """One declared address register -> tensor binding."""

    register: int
    tensor: str
    role: str  # 'read' | 'write'


@dataclass(frozen=True)
class ConstantSpec:
    """A payload block owned by a stage; `fill(payload, offset)` writes its bytes."""

    key: str
    size: int
    fill: object


@dataclass(frozen=True)
class Stage:
    """One runtime task: its tensors, its register words and its constants."""

    name: str
    family: str
    reads: tuple
    writes: tuple
    fields: object  # fields(addresses, constants) -> {register: value}
    constants: tuple = ()
    bindings: tuple = ()


@dataclass(frozen=True)
class TensorSpec:
    """A tensor table entry without its offset (the composer plans that)."""

    name: str
    role: str
    layout: str
    shape: tuple
    size: int
    index: int = 0


def _terminal(data, offset, words, enable, link=0, control=0x28):
    """Four-word task tail: `0x10` links the next program, `0x14` its control.

    A **serial** submission runs one task per ioctl, so each tail is terminal
    (`link = 0`, control 0x28). A **batched** submission hands the list to the front
    end, which reaches the next task through `0x10`; the link value is
    payload-relative (the runtime programs `PC_DMA_BASE_ADDR` with the payload base),
    so it is exactly the next program's offset, and the control word says how the
    front end continues: `0x40` inside an engine run, `0x14` from a CNA task to a DPU
    task, `0x28` terminal (measured; docs/plans/pipelining-plan.md S1/S2/S8). A list without
    links runs its first task and times out.
    """
    for index, (register, value, tag) in enumerate(
            ((0x10, link, 0x101), (0x14, control, 0x101), (0, 0, 0x41),
             (8, enable, 0x81))):
        struct.pack_into("<Q", data, offset + (words + index) * 8,
                         tag << 48 | value << 16 | register)


def compose(stages, tensors, *, input_scale=1.0, input_zero_point=0,
            output_scale=1.0, output_zero_point=0, serial=True, reuse=True,
            program_align=64, input_align=4096, arena_align=4096, late_inputs=()):
    """Assemble one v5 container from declared stages and tensor specs.

    Returns `(binary, meta)`. External inputs and outputs are placed by the policy
    documented in the module docstring; every internal tensor is placed by
    `open_rknpu.liveness` from the stages' own read/write sets.
    """
    stages = list(stages)
    tensors = list(tensors)
    if not stages:
        raise ValueError("compose requires at least one stage")
    names = [tensor.name for tensor in tensors]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor name in the tensor table")
    for tensor in tensors:
        if tensor.role not in (ROLE_INPUT, ROLE_INTERNAL, ROLE_OUTPUT):
            raise ValueError("unknown tensor role %s" % tensor.role)
    # Stage names key the program slots, the task table and the declared binding view, so
    # a repeat silently collapses two tasks into one; a repeated register inside a stage
    # silently drops one of its bindings. Both corrupt the container without failing, so
    # they are rejected here rather than in a downstream comparison.
    stage_names = [stage.name for stage in stages]
    repeated_stages = sorted({name for name in stage_names if stage_names.count(name) > 1})
    if repeated_stages:
        raise ValueError("duplicate stage name(s) %s" % ", ".join(repeated_stages))
    for stage in stages:
        registers = [binding.register for binding in stage.bindings]
        repeated = sorted({register for register in registers if registers.count(register) > 1})
        if repeated:
            raise ValueError("stage %s binds register(s) %s twice"
                             % (stage.name, ", ".join("#%x" % register for register in repeated)))
    inputs = [tensor for tensor in tensors if tensor.role == ROLE_INPUT]
    outputs = [tensor for tensor in tensors if tensor.role == ROLE_OUTPUT]
    internals = [tensor for tensor in tensors if tensor.role == ROLE_INTERNAL]
    if not 1 <= len(inputs) <= 8 or not outputs:
        raise ValueError("composer supports one to eight external inputs")
    # Inputs are placed at the payload end in declaration order. A profile whose
    # elementwise operand is read last (the runtime-scale join chain) declares that
    # input as *late*: it is placed after the internals, before the outputs.
    late = set(late_inputs)
    unknown = late - {tensor.name for tensor in inputs}
    if unknown:
        raise ValueError("late_inputs names unknown external input(s) %s" % sorted(unknown))
    early = [tensor for tensor in inputs if tensor.name not in late]
    placed_late = [tensor for tensor in inputs if tensor.name in late]
    known = set(names)
    sizes = {tensor.name: tensor.size for tensor in internals}
    tasks = []
    for stage in stages:
        for name in stage.reads + stage.writes:
            if name not in known:
                raise ValueError("stage %s names unknown tensor %s" % (stage.name, name))
        tasks.append(Access(reads=tuple(stage.reads), writes=tuple(stage.writes)))

    # A batched list chains in execution order: the k-th task links to the (k+1)-th
    # task's program, and its control word names the engine hand-off (S8). Serial lists
    # end every task.
    program_offsets = {}
    cursor = 0
    for stage in stages:
        family = FAMILIES[stage.family]
        program_offsets[stage.name] = _align(cursor, program_align)
        cursor = program_offsets[stage.name] + (family.words + 4) * 8
    constant_offsets = {}
    constant_specs = []
    for stage in stages:
        for spec in stage.constants:
            offset = _align(cursor, program_align)
            constant_offsets[spec.key] = offset
            constant_specs.append((spec, offset))
            cursor = offset + spec.size
    payload_size = _align(cursor, program_align)

    # Externals: inputs at the payload end, outputs after the last internal.
    addresses = {}
    offset = _align(payload_size, input_align)
    for tensor in early:
        addresses[tensor.name] = offset
        offset = _align(offset + tensor.size, program_align)
    arena_start = offset
    defined = tuple(tensor.name for tensor in inputs)
    if reuse:
        order, intervals, internal_offsets = plan(tasks, sizes, start=arena_start, defined=defined)
    else:
        # Fresh slot per internal, in declared order: some profiles avoid arena reuse
        # entirely because a slot written by two different task families returned
        # stale data on the board (see the investigation log).
        order = topological_order(tasks, defined)
        intervals = live_intervals(tasks, order, defined)
        internal_offsets = {}
        cursor = arena_start
        for stage in stages:
            for name in stage.writes:
                if name in sizes and name not in internal_offsets:
                    internal_offsets[name] = cursor
                    cursor = _align(cursor + sizes[name], program_align)
        missing = sorted(set(sizes) - set(internal_offsets))
        if missing:
            raise ValueError("no stage writes internal tensor(s) %s" % missing)
    addresses.update(internal_offsets)
    internal_end = max((internal_offsets[name] + sizes[name] for name in sizes), default=arena_start)
    offset = _align(internal_end, program_align)
    for tensor in placed_late:
        addresses[tensor.name] = offset
        offset = _align(offset + tensor.size, program_align)
    for tensor in outputs:
        addresses[tensor.name] = offset
        offset = _align(offset + tensor.size, program_align)
    arena_bytes = _align(offset, arena_align)
    for external in inputs + outputs:
        begin, size = addresses[external.name], external.size
        for name in sizes:
            other, other_size = internal_offsets[name], sizes[name]
            if begin < other + other_size and other < begin + size:
                raise ValueError("external tensor %s overlaps internal %s"
                                 % (external.name, name))

    # A non-serial submission chains every task through the successor's fetch amount,
    # so the whole list is one job (S8/S10).
    links, controls, runs = {}, {}, []
    if not serial:
        names = [stages[index].name for index in order]
        for position, stage_index in enumerate(order[:-1]):
            stage = stages[stage_index]
            successor = stages[order[position + 1]]
            links[stage.name] = program_offsets[successor.name]
            controls[stage.name] = amount_control(FAMILIES[successor.family].words)
        runs = [names]

    # Fill the payload: stage words in declared order, then constant bytes.
    data = bytearray(payload_size)
    for index, stage in enumerate(stages):
        family = FAMILIES[stage.family]
        values = stage.fields(dict(addresses), dict(constant_offsets))
        base = program_offsets[stage.name]
        for index, (register, default, tag) in enumerate(family.base):
            value = values.get(register, default)
            if not 0 <= value <= 0xffffffff:
                raise ValueError("stage %s register %#x out of range" % (stage.name, register))
            struct.pack_into("<Q", data, base + index * 8,
                             tag << 48 | value << 16 | register)
        if stage.name in links:
            _terminal(data, base, family.words, family.enable,
                      link=links[stage.name], control=controls[stage.name])
        else:
            _terminal(data, base, family.words, family.enable)
    for spec, offset in constant_specs:
        spec.fill(data, offset)

    # Declared order is preserved: the table order is part of the container bytes.
    tensor_table = [dict(name=tensor.name, role=tensor.role, layout=tensor.layout,
                         index=tensor.index, shape=tensor.shape,
                         offset=addresses[tensor.name], size=tensor.size)
                    for tensor in tensors]
    tasks_by_stage = {stage.name: (program_offsets[stage.name],
                                   FAMILIES[stage.family].words,
                                   FAMILIES[stage.family].enable,
                                   FAMILIES[stage.family].mask) for stage in stages}
    tasks_out = [tasks_by_stage[stages[index].name] for index in order]
    binary = encode_sequence_v5(data, tensors=tensor_table, tasks=tasks_out,
                                arena_bytes=arena_bytes, input_scale=input_scale,
                                input_zero_point=input_zero_point,
                                output_scale=output_scale,
                                output_zero_point=output_zero_point, serial=serial)
    declared = [dict(stage=stage.name, family=stage.family,
                     reads={binding.register: binding.tensor for binding in stage.bindings
                            if binding.role == "read"},
                     writes={binding.register: binding.tensor for binding in stage.bindings
                             if binding.role == "write"})
                for stage in stages]
    meta = dict(profile="composed", stages=[stage.name for stage in stages],
                program_offsets=program_offsets, constant_offsets=constant_offsets,
                tensor_offsets={name: addresses[name] for name in addresses},
                tensor_lifetimes={name: list(intervals[name]) for name in sorted(intervals)},
                stage_schedule=[stages[index].name for index in order],
                schedule=[stages[index].writes[0] for index in order],
                declared_bindings=declared, payload_size=payload_size, arena_reuse=reuse,
                submission='batched' if not serial else 'serial',
                engine_runs=[len(run) for run in runs],
                engine_run_stages=["/".join(run) for run in runs],
                arena_bytes=arena_bytes, allocated_bytes=sum(sizes.values()),
                live_bytes=max((internal_offsets[name] + sizes[name] for name in sizes),
                               default=arena_start) - arena_start)
    return binary, meta


# Task enable mask -> NPU engine.
ENGINE_CLASS = {29: "CNA", 96: "DPU", 24: "DPU"}

def engine_runs(data, info):
    """The `(start, count)` runs a non-serial container is submitted as.

    A task whose tail link is zero ends its run; a linked task continues it. The
    runtime derives the same list from the same words (`compute_runs` in
    `runtime/open_rknpu.c`), so a container is submitted as one ioctl per run.
    """
    base = payload_base(info)
    runs, start = [], 0
    for position, task in enumerate(info["tasks"]):
        offset = base + task["command_offset"] + task["register_count"] * 8
        link = (struct.unpack_from("<Q", data, offset)[0] >> 16) & 0xFFFFFFFF
        if link == 0:
            runs.append((start, position - start + 1))
            start = position + 1
    if start < info["task_count"]:
        runs.append((start, info["task_count"] - start))
    return runs


def batched_layout(data, info):
    """Return None when a container is a valid non-serial job, else the reason it is not.

    Measured on RV1103 (docs/plans/pipelining-plan.md S1/S2/S8/S10): a non-serial container is
    submitted as one ioctl per maximal linked run. Inside a run each task links to the
    next program through register `0x10` (payload-relative) with the control measured
    for its engine hand-off - `0x40` CNA->CNA and DPU->DPU, `0x14` CNA->DPU. A task
    whose link is zero ends its run and must be terminal (`0x28`); the last task of the
    container always ends a run. Depth is bounded only by the loader's 64-task table.
    """
    if info.get("serial") is not False:
        return "the container is serial (one task per submission)"
    base = payload_base(info)
    tasks = info["tasks"]
    for position, task in enumerate(tasks):
        offset = base + task["command_offset"] + task["register_count"] * 8
        word0, word1 = struct.unpack_from("<QQ", data, offset)
        link = (word0 >> 16) & 0xFFFFFFFF
        control = (word1 >> 16) & 0xFFFFFFFF
        if position == len(tasks) - 1:
            if link != 0 or control != 0x28:
                return ("the last task must end a run with control 0x28 (found link "
                        "0x%x control 0x%x)" % (link, control))
            continue
        if link == 0:
            if control != TERMINAL_CONTROL:
                return ("task %d ends a run and must be terminal (control 0x%x)"
                        % (position, control))
            continue
        wanted = tasks[position + 1]["command_offset"]
        expected = amount_control(tasks[position + 1]["register_count"])
        if link != wanted or control != expected:
            return ("task %d must link to 0x%x with the successor's fetch amount 0x%x "
                    "(found link 0x%x control 0x%x)"
                    % (position, wanted, expected, link, control))
    return None


def derive_bindings(data, info):
    """Read a v5 container back: each task's address registers -> byte offsets.

    This is the *derived* view: it looks only at the emitted register words and the
    task family, so comparing it with `meta['declared_bindings']` (through
    `check_declared_bindings`) proves that an emitter's declaration matches the
    container it produced. Offsets rather than names are returned because the arena
    deliberately reuses a slot for tensors with disjoint lifetimes, so one offset
    can name several tensors.
    """
    if info.get("format_version") != 5:
        raise ValueError("binding derivation requires a v5 container")
    payload = 112 + info["task_count"] * 16 + info["tensor_count"] * 64
    out = []
    for position, task in enumerate(info["tasks"]):
        family = FAMILY_BY_SIGNATURE.get((task["register_count"], task["enable"]))
        if family is None:
            raise ValueError("unknown task family %s" % ((task["register_count"], task["enable"]),))
        base = payload + task["command_offset"]
        words = {}
        for index in range(task["register_count"]):
            word = struct.unpack_from("<Q", data, base + index * 8)[0]
            words[word & 0xFFFF] = (word >> 16) & 0xFFFFFFFF
        out.append(dict(task=position, family=family.name,
                        reads={register: words.get(register, 0) for register in family.reads},
                        writes={register: words.get(register, 0) for register in family.writes}))
    return out


def check_declared_bindings(declared, derived, tensor_offsets, schedule=None):
    """Raise when a declared stage view and the derived container view disagree.

    `tensor_offsets` maps a declared tensor name to its planned offset and
    `schedule` is the emitted task order of stage names (from a composed meta). Every
    declared read/write register must appear in the derived view with exactly the
    planned offset, and every derived register must be declared.
    """
    by_stage = {entry["stage"]: entry for entry in declared}
    if len(by_stage) != len(declared):
        raise ValueError("declared bindings repeat a stage name")
    if schedule is not None and sorted(schedule) != sorted(by_stage):
        raise ValueError("schedule %s does not match the declared stages" % (schedule,))
    for task in derived:
        if schedule is not None:
            stage = schedule[task["task"]]
        else:
            candidates = [name for name, entry in by_stage.items()
                          if set(entry["reads"]) | set(entry["writes"]) ==
                          set(task["reads"]) | set(task["writes"])]
            if len(candidates) != 1:
                raise ValueError("task %d matches %d declarations" % (task["task"], len(candidates)))
            stage = candidates[0]
        entry = by_stage[stage]
        for role in ("reads", "writes"):
            expected = {}
            for register, tensor in entry[role].items():
                if tensor not in tensor_offsets:
                    raise ValueError("stage %s declares unknown tensor %s" % (stage, tensor))
                expected[register] = tensor_offsets[tensor]
            # A zero address register is unbound, so only bound registers are compared.
            bound = {register: offset for register, offset in task[role].items() if offset}
            if expected != bound:
                raise ValueError("task %d (%s) %s differ: declared %s, container %s"
                                 % (task["task"], stage, role, expected, bound))
    return True
