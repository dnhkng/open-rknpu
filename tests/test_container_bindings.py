"""MIT. Derive each task's tensor bindings from the container and check them.

The emitters build task read/write sets by hand. This test instead *derives* the
bindings from every verified container and checks the invariants that make them safe:

* v5 tensor tables: roles, contiguous external indices, sizes equal to the declared
  layout, offsets inside the arena, and no internal tensor overlapping an external one;
* v5 tasks: for each task family the address registers are known, so each read offset
  must be an external input or a tensor written by an **earlier** task (a topological
  producer/consumer proof), each write must target a declared internal or the output,
  and no task may write an external input;
* v3/v4 containers: the payload, task table and arena bounds all agree.

The check runs over every board-verified model in the repository (a sample per suite is
used when the suite has many) and needs no board.
"""
from pathlib import Path
import math
import struct
import unittest

from open_rknpu.compose import FAMILY_BY_SIGNATURE
from open_rknpu.model import decode as decode_model
from open_rknpu.sequence import (tensor_native_bytes, LAYOUT_NATIVE16,
                                 LAYOUT_PACKED_U8, LAYOUT_PACKED_I8, ROLE_INPUT, ROLE_OUTPUT,
                                 ROLE_INTERNAL)

ROOT = Path(__file__).resolve().parents[1]
# The task families and their address registers come from the composer registry, so
# the derived view and the emitters share one declaration (`open_rknpu.compose`).
FAMILIES = {signature: (family.reads, family.writes)
            for signature, family in FAMILY_BY_SIGNATURE.items()}


def _register_word(data, offset, index):
    return struct.unpack_from('<Q', data, offset + index * 8)[0]


def _check_v5(data, info, where):
    tensors = info['tensors']
    arena = info['arena_bytes']
    assert tensors, where
    roles = {}
    for tensor in tensors:
        name = tensor['name']
        role = tensor['role']
        layout = tensor['layout']
        assert layout in (LAYOUT_PACKED_U8, LAYOUT_NATIVE16, LAYOUT_PACKED_I8), (where, name)
        expected = tensor_native_bytes(layout, tensor['batch'], tensor['height'], tensor['width'],
                                       tensor['channels'])
        assert tensor['bytes'] == expected, (where, name, tensor['bytes'], expected)
        assert 0 <= tensor['byte_offset'] and tensor['byte_offset'] + tensor['bytes'] <= arena, \
            (where, name)
        roles.setdefault(role, []).append(tensor)
    inputs = sorted(entry['index'] for entry in roles.get(ROLE_INPUT, []))
    outputs = sorted(entry['index'] for entry in roles.get(ROLE_OUTPUT, []))
    assert inputs == list(range(len(inputs))) and inputs, (where, inputs)
    assert outputs == list(range(len(outputs))) and outputs, (where, outputs)
    externals = [entry for entry in tensors if entry['role'] in (ROLE_INPUT, ROLE_OUTPUT)]
    internals = [entry for entry in tensors if entry['role'] == ROLE_INTERNAL]
    for external in externals:
        for internal in internals:
            start, size = external['byte_offset'], external['bytes']
            other, other_size = internal['byte_offset'], internal['bytes']
            assert not (start < other + other_size and other < start + size), \
                (where, external['name'], internal['name'])
    external_offsets = {entry['byte_offset']: entry for entry in externals}
    declared = {entry['byte_offset']: entry for entry in internals}
    declared.update(external_offsets)
    payload = 112 + info['task_count'] * 16 + info['tensor_count'] * 64
    produced = {}
    for position, task in enumerate(info['tasks']):
        count, enable = task['register_count'], task['enable']
        assert (count, enable) in FAMILIES, (where, count, enable)
        reads, writes = FAMILIES[(count, enable)]
        base = payload + task['command_offset']
        words = {_register_word(data, base, index) & 65535:
                 (_register_word(data, base, index) >> 16) & 0xffffffff
                 for index in range(count)}
        for reg in reads:
            value = words.get(reg, 0)
            assert value == 0 or value in declared, (where, position, hex(reg), hex(value))
            if value == 0 or declared[value]['role'] == ROLE_INPUT:
                continue
            assert value in produced, (where, position, hex(reg), hex(value))
        for reg in writes:
            value = words.get(reg, 0)
            assert value == 0 or value in declared, (where, position, hex(reg), hex(value))
            if value == 0:
                continue
            entry = declared[value]
            assert entry['role'] != ROLE_INPUT, (where, position, entry['name'])
            produced.setdefault(value, position)


def _check_sequence_legacy(data, info, where):
    """v1-v4 ORNPUSEQ: payload/task table and arena bounds agree."""
    payload = info['payload_bytes']
    arena = info['arena_bytes']
    assert 0 < payload <= 1048576 and payload % 64 == 0, (where, payload)
    assert 0 < arena <= 4194304 and arena % 4096 == 0, (where, arena)
    assert 0 < info['task_count'] <= 64, (where, info['task_count'])
    header = 96 + info['task_count'] * 16 + info['constant_count'] * 40
    assert len(data) == header + payload, (where, len(data), header + payload)
    for task in info['tasks']:
        assert 0 < task['register_count'] <= 2048, (where, task['register_count'])
        assert 0 <= task['command_offset'] and task['command_offset'] + task['register_count'] * 8 <= payload, \
            (where, task)
    for offset in (info['input_offset'], info['output_offset']):
        assert 0 <= offset < arena, (where, offset)


def _check_ornpubin(data, info, where):
    """ORNPUBIN v1/v2: header, single program and arena agree."""
    assert info['format_version'] in (1, 2), (where, info['format_version'])
    assert info['target'] == 'rv1103' and 1 <= info['profile'] <= 8, (where, info)
    assert info['payload_bytes'] > 0 and info['payload_bytes'] % 64 == 0, (where, info['payload_bytes'])
    assert 0 < info['arena_bytes'] <= 4194304 and info['arena_bytes'] % 4096 == 0, (where, info['arena_bytes'])
    assert info['task_count'] >= 1, (where, info['task_count'])
    assert len(data) == 96 + info['payload_bytes'], (where, len(data))
    assert info['input_bytes'] > 0 and info['output_bytes'] > 0, (where, info)
    assert math.isfinite(info['output_scale']) and info['output_scale'] > 0, (where, info)
    assert -128 <= info['output_zero_point'] <= 127, (where, info)


class ContainerBindingTests(unittest.TestCase):
    def test_derived_bindings_hold_for_the_ledger(self):
        models = sorted(ROOT.glob("research/*/model*.bin"))
        self.assertGreater(len(models), 2000)
        checked = 0
        for path in models:
            data = path.read_bytes()
            info = decode_model(data)
            where = str(path.relative_to(ROOT))
            with self.subTest(model=where):
                if info['format_version'] == 5:
                    _check_v5(data, info, where)
                elif data[:8] == b'ORNPUSEQ':
                    _check_sequence_legacy(data, info, where)
                else:
                    _check_ornpubin(data, info, where)
            checked += 1
        self.assertEqual(checked, len(models))

    def test_every_campaign_suite_is_covered(self):
        campaign = ("runtime_scale_suite", "join_chain_suite", "depthwise_join_suite",
                    "pool_join_suite", "mixed_head_suite", "join_scale_suite",
                    "join_residual_suite", "join_dag_suite", "branch_join_suite",
                    "depthwise_chain_suite", "pooled_dag_suite", "pooled_branches_suite",
                    "transpose_k5_dilation_suite")
        for name in campaign:
            with self.subTest(suite=name):
                self.assertTrue((ROOT / "research" / name / "model000.bin").is_file(), name)


if __name__ == "__main__":
    unittest.main()
