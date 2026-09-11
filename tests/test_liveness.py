"""Topological ordering, tensor liveness and arena reuse for scheduled DAGs."""
import unittest
from open_rknpu.liveness import (Access, allocate, check_offsets, conflicts, disjoint,
                                 live_intervals, plan, topological_order)


class LivenessTests(unittest.TestCase):
    def test_diamond_order_and_intervals(self):
        tasks = [Access(("input0",), ("stem",)), Access(("stem",), ("head_a",)),
                 Access(("stem",), ("head_b",)), Access(("head_a", "head_b"), ("output",))]
        self.assertEqual(topological_order(tasks, defined=("input0",)), [0, 1, 2, 3])
        intervals = live_intervals(tasks, defined=("input0",))
        self.assertEqual(intervals, {"input0": (-1, 0), "stem": (0, 2), "head_a": (1, 3),
                                     "head_b": (2, 3), "output": (3, 3)})

    def test_chain_ping_pong_reuse(self):
        # Four layers: o1 -> o2 -> o3 -> out, each task reading its predecessor.
        tasks = [Access(("input0",), ("o1",)), Access(("o1",), ("o2",)),
                 Access(("o2",), ("o3",)), Access(("o3",), ("out",))]
        sizes = {name: 1024 for name in ("o1", "o2", "o3", "out")}
        _, intervals, offsets = plan(tasks, sizes, start=4096, defined=("input0",))
        self.assertEqual(intervals["o1"], (0, 1))
        self.assertEqual(offsets["o1"], 4096)
        self.assertEqual(offsets["o2"], 5120)
        # o3 reuses o1's bytes and out reuses o2's bytes: only two buffers are live.
        self.assertEqual(offsets["o3"], offsets["o1"])
        self.assertEqual(offsets["out"], offsets["o2"])
        self.assertEqual(max(offsets.values()) + 1024 - 4096, 2048)

    def test_in_place_hazard_is_not_reused(self):
        # The consumer writes its own output while reading the producer's buffer.
        tasks = [Access(("input0",), ("a",)), Access(("a",), ("b",))]
        sizes = {"a": 512, "b": 512}
        _, _, offsets = plan(tasks, sizes, start=0, defined=("input0",))
        self.assertEqual(conflicts((0, 0), (0, 1)), True)
        self.assertNotEqual(offsets["a"], offsets["b"])

    def test_disjoint_lifetimes_may_share_bytes(self):
        tasks = [Access(("input0",), ("a",)), Access(("a",), ("b",)), Access(("b",), ("c",))]
        sizes = {"a": 256, "b": 256, "c": 256}
        _, intervals, offsets = plan(tasks, sizes, start=0, defined=("input0",))
        self.assertFalse(conflicts(intervals["a"], intervals["c"]))
        self.assertEqual(offsets["a"], offsets["c"])

    def test_use_before_def_and_cycles_and_duplicates(self):
        with self.assertRaises(ValueError):
            topological_order([Access(("missing",), ("a",))])
        with self.assertRaises(ValueError):
            topological_order([Access(("b",), ("a",)), Access(("a",), ("b",))])
        with self.assertRaises(ValueError):
            topological_order([Access((), ("a",)), Access((), ("a",))])
        with self.assertRaises(ValueError):
            topological_order([{"reads": (), "writes": ("a",), "extra": 1}])
        with self.assertRaises(ValueError):
            topological_order([])
        with self.assertRaises(ValueError):
            topological_order([Access(("a",), ("a",))])

    def test_layout_validation(self):
        intervals = {"a": (0, 0), "b": (1, 1)}
        sizes = {"a": 64, "b": 64}
        self.assertTrue(check_offsets(intervals, sizes, {"a": 0, "b": 64}))
        with self.assertRaises(ValueError):
            check_offsets(intervals, sizes, {"a": 0, "b": 32})
        with self.assertRaises(ValueError):
            check_offsets(intervals, sizes, {"a": 0, "b": 65})
        with self.assertRaises(ValueError):
            allocate({"a": (0, 0)}, {"a": 64, "b": 64})
        with self.assertRaises(ValueError):
            allocate(intervals, {"a": 0, "b": 64})
        self.assertTrue(disjoint(sizes, {"a": 0, "b": 64}))
        self.assertFalse(disjoint(sizes, {"a": 0, "b": 63}))

    def test_external_input_is_predefined(self):
        with self.assertRaises(ValueError):
            topological_order([Access(("input0",), ("a",))])
        order = topological_order([Access(("input0",), ("a",))], defined=("input0",))
        self.assertEqual(order, [0])
        with self.assertRaises(ValueError):
            topological_order([Access((), ("input0",))], defined=("input0",))


if __name__ == "__main__":
    unittest.main()
