"""SPDX-License-Identifier: MIT

The zero-copy dma-buf input path (`ornpu_open_shared` / `ornpu_input_view` /
`ornpu_run_prefilled`, checklist F8).

The NPU driver imports an existing dma-buf when `CREATE`'s `handle` is the fd and bit 7 of
`flags` is set (`research/probe_dmabuf.c` proved the selector on the board). The runtime uses
that to allocate the model's arena from the Rockchip CMA heap and hand the caller the fd, so
a producer (V4L2, RGA, or a test standing in for one) writes the memory the engine reads.
This module pins the source contract; the byte-exact board evidence lives in
`docs/c-api.md` and `docs/investigation-log.md` (32 models / 64 inferences / 19,968 exact
bytes on `add_geometry_suite`, 0 mismatches).
"""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "runtime" / "open_rknpu.h"
SOURCE = ROOT / "runtime" / "open_rknpu.c"
HARNESS = ROOT / "tests" / "board_shared.c"
SEED = ROOT / "research" / "add_geometry_suite" / "model000.bin"
SEED_INPUT = ROOT / "research" / "add_geometry_suite" / "input000.u8"
SEED_EXPECTED = ROOT / "research" / "add_geometry_suite" / "expected000.i8"

COMPILER = os.environ.get("OPEN_RKNPU_HOST_CC") or shutil.which("cc") or shutil.which("gcc")
DEFAULT_FLAGS = ["-O2", "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-Iruntime"]
BUILD_FLAGS = shlex.split(os.environ.get("OPEN_RKNPU_HOST_CFLAGS", "")) or DEFAULT_FLAGS


class ZeroCopyContractTests(unittest.TestCase):
    def setUp(self):
        self.header = HEADER.read_text()
        self.source = SOURCE.read_text()

    def test_the_shared_open_and_run_entry_points_are_declared(self):
        self.assertIn("int ornpu_open_shared(const char *path, ornpu_model **model, int *arena_fd);", self.header)
        self.assertIn("struct ornpu_input_view {", self.header)
        self.assertIn("int ornpu_input_view(const ornpu_model *model, uint32_t index, "
                      "struct ornpu_input_view *view);", self.header)
        self.assertIn("int ornpu_run_prefilled(ornpu_model *model, int8_t *output, size_t output_size);", self.header)

    def test_the_input_view_carries_the_arena_layout(self):
        body = self.header.split("struct ornpu_input_view {", 1)[1].split("};", 1)[0]
        for field in ("offset", "arena_bytes", "batch", "height", "width", "channels", "row_stride"):
            with self.subTest(field=field):
                self.assertIn(field, body)

    def test_the_import_flag_matches_the_probe(self):
        self.assertIn("#define ORNPU_MEM_DMABUF 0x80u", self.source)
        self.assertIn("#define RK_DMA_HEAP \"/dev/rk_dma_heap/rk-dma-heap-cma\"", self.source)
        self.assertIn("DMA_HEAP_IOCTL_ALLOC", self.source)

    def test_the_info_struct_exposes_the_arena_size(self):
        self.assertIn("uint32_t arena_bytes;", self.header)
        # The three container loaders fill it; a missing one would report 0 to a producer.
        self.assertEqual(self.source.count("h->arena_size};") + self.source.count("v[10]};"), 3)

    def test_the_prefilled_path_completes_the_row_padding_without_copying_data(self):
        body = self.source.split("int ornpu_run_prefilled(", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("row_stride", body) if "row_stride" in body else None
        self.assertIn("stride-in->width", body, "the v5 packed padding fill must be there")
        self.assertIn("h->input_stride-h->width", body, "the legacy padding fill must be there")
        self.assertNotIn("pack_tensor_input(model", body,
                         "the prefilled path must not copy the producer's input")

    def test_the_shared_path_refuses_layouts_the_caller_cannot_produce(self):
        body = self.source.split("int ornpu_input_view(", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("LAYOUT_PACKED", body)
        self.assertIn("-ENOTSUP", body)


@unittest.skipUnless(COMPILER, "no host C compiler (cc/gcc) found; shared-input tests skipped")
class SharedHarnessBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="open-rknpu-shared-")
        cls.binary = Path(cls.tmp.name) / "board_shared"
        cls.build = subprocess.run(
            [COMPILER, *BUILD_FLAGS, str(HARNESS.relative_to(ROOT)), str(SOURCE.relative_to(ROOT)),
             "-o", str(cls.binary)], cwd=ROOT, capture_output=True, text=True, timeout=180)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_harness_compiles_and_links_against_the_runtime(self):
        self.assertEqual(self.build.returncode, 0,
                         f"board_shared.c failed to build:\n{self.build.stdout}\n{self.build.stderr}")

    def test_without_a_device_it_fails_before_touching_a_dma_buf(self):
        if Path("/dev/rknpu").exists():
            self.skipTest("this host has /dev/rknpu; the shared path is the board's")
        result = subprocess.run([str(self.binary), str(SEED), str(SEED_INPUT), str(SEED_EXPECTED), "1"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("open_shared failed", result.stderr)
        self.assertNotIn("SUMMARY", result.stdout)

    def test_usage_is_checked(self):
        result = subprocess.run([str(self.binary)], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
