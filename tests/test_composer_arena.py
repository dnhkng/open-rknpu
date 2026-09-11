"""SPDX-License-Identifier: MIT

Arena-placement and composer-validation regression tests.

`open_rknpu.liveness` turns a scheduled task list into a topological order, live
intervals and 64-byte-aligned arena offsets, and `open_rknpu.compose` uses the same
allocator to build every version-5 container. A wrong offset is a silent board
corruption, so this module pins the placement contract directly:

* `allocate`/`plan`/`check_offsets`/`disjoint`: 64-byte alignment, no byte sharing
  between conflicting (overlapping) lifetimes, byte sharing for disjoint lifetimes,
  and a planned layout that never exceeds a naive fresh-slot layout;
* a spread of real compiled graphs (a single image Conv, a walk chain with an
  interior pool, a diamond, a standalone depthwise and a depthwise branch join)
  whose container arena always covers `allocated_bytes`, whose live tensor intervals
  never overlap in `tensor_offsets`, and whose externals never overlap an internal;
* the `--reuse-intermediates` / `reuse_intermediates=True` flag never allocates more
  than the plain build and strictly less for a deep chain (with the composer's own
  `reuse` switch tested directly on a declared stage list);
* `compose` rejects duplicate stage names, unknown read/write tensors, duplicate
  register bindings and duplicate tensor names with their specific messages - the
  first two checks were added when the composer silently collapsed repeated stage
  names and dropped repeated bindings instead of failing.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu.compose import (Binding, Stage, TensorSpec, compose)
from open_rknpu.liveness import (Access, allocate, check_offsets, conflicts, disjoint,
                                 live_intervals, plan, topological_order)
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import LAYOUT_NATIVE16, ROLE_INPUT, ROLE_INTERNAL, ROLE_OUTPUT, decode_sequence

ALIGN = 64


def naive_bytes(sizes, alignment=ALIGN):
    """One fresh aligned slot per tensor, in any order: the layout-reuse upper bound."""
    return sum((int(size) + alignment - 1) // alignment * alignment for size in sizes.values())


def extent(offsets, sizes):
    return max(offsets[name] + sizes[name] for name in sizes)


def overlaps(left, left_size, right, right_size):
    return left < right + right_size and right < left + left_size


def chain_schedule(length=6, size=256):
    tasks = [Access(("input0",), ("t0",))]
    tasks += [Access((f"t{index}",), (f"t{index + 1}",)) for index in range(length - 2)]
    tasks += [Access((f"t{length - 2}",), ("out",))]
    sizes = {f"t{index}": size for index in range(length - 1)}
    sizes["out"] = size
    return tasks, sizes


def diamond_schedule():
    tasks = [Access(("input0",), ("a",)), Access(("a",), ("b",)), Access(("a",), ("c",)),
             Access(("b", "c"), ("d",)), Access(("d",), ("out",))]
    sizes = {"a": 128, "b": 256, "c": 192, "d": 64, "out": 256}
    return tasks, sizes


def random_schedule(seed, layers=7):
    """A seeded layered DAG: one input, a fan-out middle, one sink."""
    rng = np.random.default_rng(seed)
    tasks = []
    names = []
    previous = ["input0"]
    for layer in range(layers - 1):
        width = int(rng.integers(1, 4))
        current = [f"t{layer}_{index}" for index in range(width)]
        for name in current:
            reads = tuple(rng.choice(previous, size=int(rng.integers(1, len(previous) + 1)),
                                     replace=False).tolist())
            tasks.append(Access(reads, (name,)))
            names.append(name)
        previous = current
    tasks.append(Access(tuple(previous), ("out",)))
    names.append("out")
    sizes = {name: int(rng.integers(1, 40)) * 16 for name in names}
    return tasks, sizes


def save(model):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "model.onnx"
    onnx.save(model, path)
    return path


def make_model(nodes, initializers, input_shape, output_shape, value_infos=()):
    graph = h.make_graph(nodes, "graph", [h.make_tensor_value_info("input", 1, list(input_shape))],
                         [h.make_tensor_value_info("output", 1, list(output_shape))],
                         list(initializers))
    for name, shape in value_infos:
        graph.value_info.append(h.make_tensor_value_info(name, 1, list(shape)))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def image_conv_graph(seed=1):
    """A single image Conv: the legacy single-Conv container (format 3)."""
    rng = np.random.default_rng(seed)
    return make_model(
        [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
        [nh.from_array(rng.uniform(-.7, .8, (4, 3, 3, 3)).astype(np.float32), "w"),
         nh.from_array(rng.uniform(-2, 2, (4,)).astype(np.float32), "b")],
        [1, 3, 8, 8], [1, 4, 8, 8])


def chain_pool_graph(seed=2, hidden=5):
    """Conv -> Relu -> MaxPool -> Conv -> Relu: the chain walk with an interior pool."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c1"], ["r1"]),
             h.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
             h.make_node("Conv", ["p1", "w2", "b2"], ["c2"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["c2"], ["output"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (hidden, hidden, 1, 1)).astype(np.float32), "w2"),
                    nh.from_array(rng.uniform(-2, 2, (hidden,)).astype(np.float32), "b2")]
    return make_model(nodes, initializers, [1, 3, 8, 8], [1, hidden, 4, 4],
                      value_infos=(("c1", [1, hidden, 8, 8]), ("r1", [1, hidden, 8, 8]),
                                   ("p1", [1, hidden, 4, 4]), ("c2", [1, hidden, 4, 4])))


def diamond_graph(seed=3, hidden=8):
    """Conv -> Relu -> two heads -> Add: the diamond branch join."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["stem"], ["relu1"]),
             h.make_node("Conv", ["relu1", "wa", "ba"], ["head_a"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["relu1", "wb", "bb"], ["head_b"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1]),
             h.make_node("Add", ["head_a", "head_b"], ["output"])]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-4, 4, (hidden,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, 1, 1)).astype(np.float32), "wa"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "ba"),
                    nh.from_array(rng.uniform(-.7, .8, (3, hidden, 3, 3)).astype(np.float32), "wb"),
                    nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "bb")]
    return make_model(nodes, initializers, [1, 3, 8, 8], [1, 3, 8, 8],
                      value_infos=(("relu1", [1, hidden, 8, 8]), ("head_a", [1, 3, 8, 8]),
                                   ("head_b", [1, 3, 8, 8])))


def depthwise_graph(seed=4):
    """A stem Conv followed by a grouped Conv: the standalone depthwise profile."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["stem", "wd", "bd"], ["output"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1], group=3)]
    initializers = [nh.from_array(rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-2, 2, (3,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(-.7, .8, (3, 1, 3, 3)).astype(np.float32), "wd"),
                    nh.from_array(rng.uniform(-2, 2, (3,)).astype(np.float32), "bd")]
    return make_model(nodes, initializers, [1, 3, 8, 8], [1, 3, 8, 8],
                      value_infos=(("stem", [1, 3, 8, 8]),))


def depthwise_join_graph(seed=5, dense_kernel=3, depthwise_kernel=3):
    """Stem Conv -> dense branch + depthwise branch -> Add: placed by liveness."""
    rng = np.random.default_rng(seed)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Relu", ["stem"], ["relu1"]),
             h.make_node("Conv", ["relu1", "wd", "bd"], ["dense"], kernel_shape=[dense_kernel] * 2,
                         pads=[dense_kernel // 2] * 4),
             h.make_node("Conv", ["relu1", "ww", "bw"], ["depthwise"],
                         kernel_shape=[depthwise_kernel] * 2, pads=[depthwise_kernel // 2] * 4,
                         group=3),
             h.make_node("Add", ["dense", "depthwise"], ["output"])]
    initializers = [nh.from_array(rng.uniform(.02, .06, (3, 3, 1, 1)).astype(np.float32), "w1"),
                    nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "b1"),
                    nh.from_array(rng.uniform(.02, .06, (3, 3, dense_kernel, dense_kernel))
                                  .astype(np.float32), "wd"),
                    nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bd"),
                    nh.from_array(rng.uniform(.02, .06, (3, 1, depthwise_kernel, depthwise_kernel))
                                  .astype(np.float32), "ww"),
                    nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bw")]
    return make_model(nodes, initializers, [1, 3, 8, 8], [1, 3, 8, 8],
                      value_infos=(("relu1", [1, 3, 8, 8]), ("dense", [1, 3, 8, 8]),
                                   ("depthwise", [1, 3, 8, 8])))


def deep_chain_graph(layers=((4, 3), (8, 3), (6, 1), (3, 3)), seed=5):
    """A four-plus-layer native chain: the profile that honours intermediate reuse."""
    rng = np.random.default_rng(seed)
    nodes, initializers, previous = [], [], "input"
    for index, (channels, kernel) in enumerate(layers):
        inputs = 3 if index == 0 else layers[index - 1][0]
        initializers += [nh.from_array(rng.uniform(-.7, .8, (channels, inputs, kernel, kernel))
                                       .astype(np.float32), f"w{index}"),
                         nh.from_array(rng.uniform(-2, 2, (channels,)).astype(np.float32), f"b{index}")]
        last = index == len(layers) - 1
        produced = "output" if last else f"c{index}"
        nodes.append(h.make_node("Conv", [previous, f"w{index}", f"b{index}"], [produced],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        previous = produced
        if not last:
            nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
            previous = f"r{index}"
    return make_model(nodes, initializers, [1, 3, 8, 8], [1, layers[-1][0], 8, 8])


class LivenessPlanTests(unittest.TestCase):
    """Alignment, lifetime sharing and the naive-layout upper bound."""

    def test_every_planned_offset_is_sixty_four_byte_aligned(self):
        tasks, sizes = chain_schedule(length=5, size=100)
        order, intervals, offsets = plan(tasks, sizes, defined=("input0",))
        self.assertEqual(sorted(order), list(range(len(tasks))))
        self.assertTrue(check_offsets(intervals, sizes, offsets))
        for name, offset in offsets.items():
            self.assertEqual(offset % ALIGN, 0, name)
            self.assertEqual(intervals[name][0], min(intervals[name][0], intervals[name][1]))
        # A non-64 layout is rejected by check_offsets but a custom alignment is honoured.
        with self.assertRaisesRegex(ValueError, "not 64-byte aligned"):
            check_offsets(intervals, sizes, dict(offsets, t0=offsets["t0"] + 8))
        _, _, odd = plan(tasks, sizes, defined=("input0",), alignment=16)
        self.assertTrue(all(offset % 16 == 0 for offset in odd.values()))

    def test_conflicting_lifetimes_never_share_bytes(self):
        tasks, sizes = diamond_schedule()
        _, intervals, offsets = plan(tasks, sizes, defined=("input0",))
        shared = 0
        names = sorted(sizes)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                same_bytes = overlaps(offsets[left], sizes[left], offsets[right], sizes[right])
                if conflicts(intervals[left], intervals[right]):
                    self.assertFalse(same_bytes, (left, right))
                elif same_bytes:
                    shared += 1
        self.assertGreater(shared, 0, "disjoint lifetimes must be able to share bytes")

    def test_chain_reuses_the_two_buffer_ping_pong(self):
        tasks, sizes = chain_schedule(length=5, size=256)
        _, intervals, offsets = plan(tasks, sizes, defined=("input0",))
        self.assertFalse(conflicts(intervals["t0"], intervals["t2"]))
        self.assertEqual(offsets["t0"], offsets["t2"])
        self.assertTrue(conflicts(intervals["t0"], intervals["t1"]))
        self.assertNotEqual(offsets["t0"], offsets["t1"])
        self.assertLessEqual(extent(offsets, sizes), 3 * 256)

    def test_plan_never_exceeds_a_naive_fresh_slot_layout(self):
        schedules = [chain_schedule(length=6, size=48), diamond_schedule()]
        schedules += [random_schedule(seed) for seed in range(8)]
        for tasks, sizes in schedules:
            with self.subTest(tasks=len(tasks), tensors=len(sizes)):
                _, intervals, offsets = plan(tasks, sizes, defined=("input0",))
                self.assertTrue(check_offsets(intervals, sizes, offsets))
                self.assertLessEqual(extent(offsets, sizes), naive_bytes(sizes))
                # The same bound holds for allocate on a fixed order.
                allocated = allocate(intervals, sizes)
                self.assertLessEqual(extent(allocated, sizes), naive_bytes(sizes))

    def test_disjoint_and_overlap_helpers(self):
        sizes = {"a": 64, "b": 64}
        self.assertTrue(disjoint(sizes, {"a": 0, "b": 64}))
        self.assertFalse(disjoint(sizes, {"a": 0, "b": 63}))
        self.assertTrue(disjoint(sizes, {"a": 0, "b": 128}))
        self.assertTrue(conflicts((0, 1), (1, 2)))
        self.assertFalse(conflicts((0, 0), (1, 1)))

    def test_allocate_rejects_missing_intervals_and_bad_sizes(self):
        intervals = {"a": (0, 0), "b": (1, 1)}
        with self.assertRaisesRegex(ValueError, "missing live intervals"):
            allocate(intervals, {"a": 64, "c": 64})
        with self.assertRaisesRegex(ValueError, "positive size"):
            allocate(intervals, {"a": 64, "b": 0})
        with self.assertRaisesRegex(ValueError, "positive size"):
            allocate(intervals, {"a": 64, "b": -8})

    def test_topological_order_is_producer_first_and_stable(self):
        tasks = [Access(("input0",), ("a",)), Access(("a",), ("b",)), Access(("a",), ("c",)),
                 Access(("b", "c"), ("output",))]
        self.assertEqual(topological_order(tasks, defined=("input0",)), [0, 1, 2, 3])
        self.assertEqual(live_intervals(tasks, defined=("input0",))["a"], (0, 2))


class ComposeValidationTests(unittest.TestCase):
    """`compose` rejects malformed declarations with their specific messages."""

    def tensors(self):
        return [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16),
                TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16),
                TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16)]

    def stages(self, names=("first", "second")):
        def first_fields(addresses, constants):
            return {0x4020: addresses["mid"], 0x5018: addresses["image"], 0x5038: addresses["image"]}

        def second_fields(addresses, constants):
            return {0x4020: addresses["output"], 0x5018: addresses["mid"], 0x5038: addresses["mid"]}

        return [Stage(name=names[0], family="elementwise", reads=("image",), writes=("mid",),
                      fields=first_fields,
                      bindings=(Binding(0x5018, "image", "read"), Binding(0x5038, "image", "read"),
                                Binding(0x4020, "mid", "write"))),
                Stage(name=names[1], family="elementwise", reads=("mid",), writes=("output",),
                      fields=second_fields,
                      bindings=(Binding(0x5018, "mid", "read"), Binding(0x5038, "mid", "read"),
                                Binding(0x4020, "output", "write")))]

    def test_duplicate_stage_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"duplicate stage name\(s\) same"):
            compose(self.stages(names=("same", "same")), self.tensors())

    def test_unknown_read_and_write_tensors_are_rejected(self):
        stages = self.stages()
        bad_read = [Stage(name="only", family="elementwise", reads=("missing",), writes=("mid",),
                          fields=stages[0].fields,
                          bindings=(Binding(0x4020, "mid", "write"),))]
        with self.assertRaisesRegex(ValueError, "stage only names unknown tensor missing"):
            compose(bad_read, self.tensors())
        bad_write = [Stage(name="only", family="elementwise", reads=("image",), writes=("ghost",),
                           fields=stages[0].fields,
                           bindings=(Binding(0x4020, "ghost", "write"),))]
        with self.assertRaisesRegex(ValueError, "stage only names unknown tensor ghost"):
            compose(bad_write, self.tensors())

    def test_duplicate_register_bindings_are_rejected(self):
        stages = self.stages()
        duplicated = [Stage(name="only", family="elementwise", reads=("image",), writes=("mid",),
                            fields=stages[0].fields,
                            bindings=(Binding(0x5018, "image", "read"),
                                      Binding(0x5018, "mid", "write")))]
        with self.assertRaisesRegex(ValueError, r"binds register\(s\) #5018 twice"):
            compose(duplicated, self.tensors())
        # Different registers on the same tensor are legal; only a repeated register is not.
        binary, meta = compose(self.stages(), self.tensors())
        self.assertTrue(binary)

    def test_duplicate_tensor_names_roles_and_input_counts_are_rejected(self):
        duplicate = self.tensors() + [TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 64)]
        with self.assertRaisesRegex(ValueError, "duplicate tensor name"):
            compose(self.stages(), duplicate)
        unknown_role = [TensorSpec("image", "banana", LAYOUT_NATIVE16, (1, 8, 8, 3), 64)]
        with self.assertRaisesRegex(ValueError, "unknown tensor role banana"):
            compose(self.stages(), unknown_role)
        no_input = [TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 64),
                    TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 64)]
        with self.assertRaisesRegex(ValueError, "one to eight external inputs"):
            compose(self.stages(), no_input)
        with self.assertRaisesRegex(ValueError, "compose requires at least one stage"):
            compose([], self.tensors())
        with self.assertRaisesRegex(ValueError, "late_inputs names unknown external input"):
            compose(self.stages(), self.tensors(), late_inputs=("ghost",))

    def test_composer_reuse_flag_shrinks_a_declared_chain(self):
        # A four-stage elementwise chain: t0 and t2 have disjoint lifetimes and may share.
        names = ["t0", "t1", "t2"]
        stages = []
        source = "image"
        for index, name in enumerate(names + ["output"]):
            target = name
            reads = (source,)
            fields = (lambda target=target, source=source:
                      (lambda addresses, constants: {0x4020: addresses[target],
                                                     0x5018: addresses[source],
                                                     0x5038: addresses[source]}))
            stages.append(Stage(name=f"stage{index}", family="elementwise", reads=reads, writes=(target,),
                                fields=fields(),
                                bindings=(Binding(0x5018, source, "read"), Binding(0x4020, target, "write"))))
            source = target
        tensors = [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16)]
        tensors += [TensorSpec(name, ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16)
                    for name in names]
        tensors.append(TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16))
        _, plain = compose(stages, tensors, reuse=False)
        _, reused = compose(stages, tensors, reuse=True)
        self.assertLess(reused["arena_bytes"], plain["arena_bytes"])
        self.assertLessEqual(reused["allocated_bytes"], plain["allocated_bytes"])
        self.assertEqual(reused["tensor_offsets"]["t2"], reused["tensor_offsets"]["t0"])


class CompiledGraphArenaTests(unittest.TestCase):
    """Real containers: allocation bounds, live overlap and reuse."""

    def compile(self, model, **kwargs):
        binary, meta = compile_sequence(save(model), **kwargs)
        return binary, meta, decode_sequence(binary)

    def composer_graphs(self):
        return [("chain_pool", chain_pool_graph()), ("diamond", diamond_graph()),
                ("depthwise_join", depthwise_join_graph())]

    def test_allocated_bytes_never_exceed_the_container_arena(self):
        for name, model in self.composer_graphs():
            with self.subTest(graph=name):
                _, meta, info = self.compile(model)
                self.assertIn("allocated_bytes", meta)
                arena = meta.get("arena_bytes", info["arena_bytes"])
                self.assertLessEqual(meta["allocated_bytes"], arena)
                internal = sum(t["bytes"] for t in info["tensors"] if t["role_name"] == "internal")
                self.assertEqual(meta["allocated_bytes"], internal)

    def test_live_tensors_never_share_bytes_and_stay_inside_the_arena(self):
        for name, model in self.composer_graphs():
            with self.subTest(graph=name):
                _, meta, info = self.compile(model)
                arena = meta.get("arena_bytes", info["arena_bytes"])
                sizes = {t["name"]: t["bytes"] for t in info["tensors"]}
                offsets = meta["tensor_offsets"]
                lifetimes = meta["tensor_lifetimes"]
                self.assertEqual(sorted(offsets), sorted(sizes))
                for tensor, offset in offsets.items():
                    self.assertEqual(offset % ALIGN, 0, tensor)
                    self.assertLessEqual(offset + sizes[tensor], arena, tensor)
                names = sorted(sizes)
                for index, left in enumerate(names):
                    for right in names[index + 1:]:
                        if conflicts(lifetimes[left], lifetimes[right]):
                            self.assertFalse(overlaps(offsets[left], sizes[left],
                                                      offsets[right], sizes[right]), (left, right))
                # The declared layout passes the same validator the planner uses.
                self.assertTrue(check_offsets({key: tuple(value) for key, value in lifetimes.items()},
                                              sizes, offsets))

    def test_external_tensors_never_overlap_an_internal(self):
        for name, model in self.composer_graphs():
            with self.subTest(graph=name):
                _, _, info = self.compile(model)
                externals = [t for t in info["tensors"] if t["role_name"] != "internal"]
                internals = [t for t in info["tensors"] if t["role_name"] == "internal"]
                for external in externals:
                    for internal in internals:
                        self.assertFalse(overlaps(external["byte_offset"], external["bytes"],
                                                  internal["byte_offset"], internal["bytes"]),
                                         (external["name"], internal["name"]))

    def test_format_three_profiles_keep_their_tensors_inside_the_arena(self):
        # The legacy single-Conv and standalone-depthwise profiles have no tensor table;
        # their one external output still has to fit below the declared arena bound.
        for name, model in (("image_conv", image_conv_graph()), ("depthwise", depthwise_graph())):
            with self.subTest(graph=name):
                _, meta, info = self.compile(model)
                self.assertGreaterEqual(meta["arena_bytes"], info["arena_bytes"])
                height, width, channels = info["output_shape_nhwc"][1:]
                surface = ((height * width + 3) // 4) * 64 * ((channels + 15) // 16)
                self.assertEqual(info["output_offset"] % ALIGN, 0)
                self.assertLessEqual(info["output_offset"] + surface, info["arena_bytes"])
                self.assertEqual(info["format_version"], 3)

    def test_reuse_intermediates_never_allocates_more(self):
        graphs = [("image_conv", image_conv_graph()), ("chain_pool", chain_pool_graph()),
                  ("diamond", diamond_graph()), ("depthwise", depthwise_graph()),
                  ("depthwise_join", depthwise_join_graph())]

        def allocation(binary, meta):
            return meta.get("allocated_bytes", decode_sequence(binary)["arena_bytes"])

        for name, model in graphs:
            with self.subTest(graph=name):
                plain, plain_meta, _ = self.compile(model, reuse_intermediates=False)
                reused, reused_meta, _ = self.compile(model, reuse_intermediates=True)
                self.assertLessEqual(allocation(reused, reused_meta), allocation(plain, plain_meta))
        # A four-layer native chain is the profile that actually reuses: strictly less.
        plain, plain_meta, plain_info = self.compile(deep_chain_graph(), reuse_intermediates=False)
        reused, reused_meta, reused_info = self.compile(deep_chain_graph(), reuse_intermediates=True)
        self.assertTrue(reused_meta["reused_intermediates"])
        self.assertLess(reused_info["arena_bytes"], plain_info["arena_bytes"])
        self.assertLessEqual(reused_meta["layers"], plain_meta["layers"])
        # A three-layer chain has only two intermediates, so there is nothing to reuse.
        short = deep_chain_graph(layers=((5, 1), (5, 3), (3, 1)))
        _, short_meta, _ = self.compile(short, reuse_intermediates=True)
        self.assertFalse(short_meta["reused_intermediates"])

    def test_walk_reference_agrees_with_the_allocated_arena(self):
        # The chain-walk meta's tensor offsets are what its integer reference reads.
        binary, meta, info = self.compile(chain_pool_graph())
        self.assertEqual(meta["profile"], "chain-walk")
        self.assertFalse(meta["arena_reuse"])
        self.assertEqual(sorted(meta["tensor_offsets"]), sorted(t["name"] for t in info["tensors"]))
        self.assertGreaterEqual(info["arena_bytes"], max(meta["tensor_offsets"].values()))


if __name__ == "__main__":
    unittest.main()
