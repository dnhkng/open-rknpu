"""SPDX-License-Identifier: MIT

Topological ordering, tensor liveness and arena placement for scheduled DAGs.

The emitters in this package describe one runtime task per NPU program and each
task names the tensors it reads and writes. This module turns that description
into an execution order, live intervals and arena offsets, and validates the
result:

* a tensor must be written before it is read (use-before-def is rejected unless
  the name is declared as an external input);
* two tensors may share arena bytes only when their live intervals do not
  conflict. A tensor written and read by the same task is in place on the
  hardware, which is *not* assumed safe, so a task that reads a tensor conflicts
  with that task's own output tensor;
* external (input/output) tensors never overlap internal tensors because the
  version-5 container rejects that overlap, so callers place externals and pass
  only internals to `plan`.

No vendor artifact, capture or runtime is read; the pass is pure bookkeeping.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Access:
    """One task's tensor bindings; names are ordered for deterministic output."""

    reads: tuple
    writes: tuple


def _normalize(tasks):
    if not tasks:
        raise ValueError("schedule requires at least one task")
    normalized = []
    for index, task in enumerate(tasks):
        if isinstance(task, Access):
            normalized.append(task)
            continue
        if not isinstance(task, dict) or set(task) != {"reads", "writes"}:
            raise ValueError("task %d must be Access or {'reads','writes'}" % index)
        reads = tuple(task["reads"])
        writes = tuple(task["writes"])
        if any(not isinstance(name, str) or not name for name in reads + writes):
            raise ValueError("task %d tensor names must be non-empty strings" % index)
        if len(set(writes)) != len(writes):
            raise ValueError("task %d writes a tensor twice" % index)
        normalized.append(Access(reads, writes))
    return normalized


def topological_order(tasks, defined=()):
    """Producer-before-consumer order; ties keep the declared task order.

    `defined` names tensors that already exist (external inputs). They need not
    be produced by a task and never create an edge.
    """
    tasks = _normalize(tasks)
    defined = set(defined)
    writers = {}
    for index, task in enumerate(tasks):
        if set(task.reads) & set(task.writes):
            raise ValueError("task %d reads its own output tensor" % index)
        for name in task.writes:
            if name in writers:
                raise ValueError("tensor %s has more than one producer" % name)
            if name in defined:
                raise ValueError("tensor %s is both defined and written" % name)
            writers[name] = index
    edges = [set() for _ in tasks]
    consumers = [0] * len(tasks)
    for index, task in enumerate(tasks):
        for name in task.reads:
            producer = writers.get(name)
            if producer is None:
                if name in defined:
                    continue
                raise ValueError("tensor %s is read before it is written" % name)
            if producer != index and producer not in edges[index]:
                edges[index].add(producer)
                consumers[index] += 1
    order = []
    ready = [index for index in range(len(tasks)) if not consumers[index]]
    while ready:
        index = ready.pop(0)
        order.append(index)
        for later in range(len(tasks)):
            if index in edges[later]:
                edges[later].discard(index)
                consumers[later] -= 1
                if not consumers[later]:
                    ready.append(later)
        ready.sort()
    if len(order) != len(tasks):
        raise ValueError("task graph contains a cycle")
    return order


def live_intervals(tasks, order=None, defined=()):
    """``{name: (first, last)}`` task positions of definition and final use.

    A written tensor's interval starts at its producer, so a task that reads and
    writes the same buffer produces overlapping intervals and therefore a
    conflict; in-place execution is not assumed to be safe. Tensors in `defined`
    (external inputs) start at -1 so they stay live from before the first task.
    """
    tasks = _normalize(tasks)
    order = topological_order(tasks, defined) if order is None else list(order)
    position = {index: slot for slot, index in enumerate(order)}
    intervals = {}
    for index, task in enumerate(tasks):
        for name in task.writes:
            intervals[name] = (position[index], position[index])
    for name in defined:
        intervals[name] = (-1, -1)
    for index, task in enumerate(tasks):
        slot = position[index]
        for name in task.reads:
            if name not in intervals:
                raise ValueError("tensor %s is read before it is written" % name)
            first, last = intervals[name]
            intervals[name] = (first, max(last, slot))
    return intervals


def conflicts(first, second):
    """True when two live intervals overlap and may not share arena bytes."""
    return not (first[1] < second[0] or second[1] < first[0])


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _overlaps(offset, size, other_offset, other_size):
    return offset < other_offset + other_size and other_offset < offset + size


def allocate(intervals, sizes, start=0, alignment=64):
    """First-fit arena offsets for the names in `sizes`.

    Tensors are placed in definition order. Candidates are the arena start and the
    end of every already placed tensor, which is enough to reproduce the
    two-buffer ping-pong reuse of long chains as well as strict disjoint layouts.
    """
    missing = set(sizes) - set(intervals)
    if missing:
        raise ValueError("missing live intervals for tensors: %s" % ", ".join(sorted(missing)))
    offsets = {}
    for name in sorted(sizes, key=lambda value: (intervals[value][0], intervals[value][1], value)):
        size = int(sizes[name])
        if size <= 0:
            raise ValueError("tensor %s needs a positive size" % name)
        candidates = {start}
        for other, other_offset in offsets.items():
            candidates.add(_align(other_offset + sizes[other], alignment))
        for candidate in sorted(candidates):
            if all(not conflicts(intervals[name], intervals[other])
                   or not _overlaps(candidate, size, other_offset, sizes[other])
                   for other, other_offset in offsets.items()):
                offsets[name] = candidate
                break
        else:
            raise ValueError("no arena placement for tensor %s" % name)
    return offsets


def check_offsets(intervals, sizes, offsets, alignment=64):
    """Validate a fixed layout: aligned and free of live-byte overlap."""
    for name, offset in offsets.items():
        if offset % alignment:
            raise ValueError("tensor %s is not %d-byte aligned" % (name, alignment))
        for other, other_offset in offsets.items():
            if name == other or not conflicts(intervals[name], intervals[other]):
                continue
            if _overlaps(offset, sizes[name], other_offset, sizes[other]):
                raise ValueError("tensors %s and %s overlap while live" % (name, other))
    return True


def disjoint(sizes, offsets):
    """True when no two tensors share a byte, whatever their lifetimes."""
    names = sorted(offsets)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if _overlaps(offsets[name], sizes[name], offsets[other], sizes[other]):
                return False
    return True


def plan(tasks, sizes, start=0, alignment=64, defined=()):
    """Schedule, liveness and placement for one task list.

    Returns ``(order, intervals, offsets)``. Only the tensors named in `sizes`
    are placed; external tensors are positioned by the caller so that the
    version-5 rule (internals never overlap externals) stays explicit.
    """
    tasks = _normalize(tasks)
    order = topological_order(tasks, defined)
    intervals = live_intervals(tasks, order, defined)
    offsets = allocate(intervals, sizes, start=start, alignment=alignment)
    check_offsets(intervals, sizes, offsets, alignment=alignment)
    return order, intervals, offsets
