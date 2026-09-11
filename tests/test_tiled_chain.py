"""MIT. Height-strip tiled chain regression (docs/plans/pipelining-plan.md S3/S7).

`compile_tiled_chain` splits an `[Conv, Relu]*(N-1) + [Conv]` chain into T height
strips, one native16 program per (layer, strip), with two intermediate layouts: isolated
double-buffered strip surfaces for 1x1 kernels and one shared double-buffered surface
for the 3x3 case, whose strips read the halo rows the neighbouring strips produce.

Both container families were verified byte-exact on the board against the untiled
chain's own expected bytes (`research/tiled_chain_probe/` for 1x1, re-run over all 16
input cases; `research/tiled_k3_probe/` for K > 1, 4 cases each); these tests pin the
structure, the bounds, the activation/pad registers and the retained evidence.
"""
from pathlib import Path
import json
import struct
import unittest

import onnx
from open_rknpu.chain_n import compile_chain_n
from open_rknpu.compose import batched_layout
from open_rknpu.sequence import decode_sequence
from open_rknpu.tiled_chain import compile_tiled_chain

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "research" / "deep_chain_suite" / "model002.onnx"
PROBE = ROOT / "research" / "tiled_chain_probe"
K3 = ROOT / "research" / "tiled_k3_probe"
K3_MODELS = ("k3", "mixed")


def programs(data):
    """`(register -> value)` per task, plus each task's two link words."""
    info = decode_sequence(data)
    if info["format_version"] == 5:
        base = 112 + 16 * info["task_count"] + 64 * info["tensor_count"]
    else:
        base = 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)
    out = []
    for task in info["tasks"]:
        start = base + task["command_offset"]
        words = [struct.unpack_from("<Q", data, start + i * 8)[0] for i in range(126)]
        tail = [struct.unpack_from("<Q", data, start + (task["register_count"] + i) * 8)[0]
                for i in range(2)]
        out.append(({word & 0xFFFF: (word >> 16) & 0xFFFFFFFF for word in words},
                    [(word >> 16) & 0xFFFFFFFF for word in tail]))
    return out, info


class TiledChainTests(unittest.TestCase):
    def test_strip_structure_and_task_count(self):
        _, reference = compile_chain_n(onnx.load(MODEL))
        for tiles in (1, 2, 4):
            for serial in (True, False):
                data, meta = compile_tiled_chain(onnx.load(MODEL), tiles=tiles, serial=serial)
                info = decode_sequence(data)
                with self.subTest(tiles=tiles, serial=serial):
                    self.assertEqual(info["task_count"], len(reference["quantizations"]) * tiles)
                    self.assertEqual(meta["layers"], len(reference["quantizations"]))
                    self.assertEqual(meta["strip_rows"], 8 // tiles)
                    self.assertEqual(meta["tasks"], info["task_count"])
                    self.assertEqual(meta["layout"], "strip-surfaces")
                    self.assertEqual(meta["kernels"], [1] * meta["layers"])
                    self.assertEqual(info["input_layout"], "native16")
                    self.assertEqual(meta["output_scale"], reference["output_scale"])

    def test_k3_strips_use_a_shared_surface_and_follow_the_layer_activation(self):
        for name in K3_MODELS:
            model = onnx.load(K3 / f"{name}.onnx")
            data, meta = compile_tiled_chain(model, tiles=2)
            entries, info = programs(data)
            with self.subTest(model=name):
                self.assertEqual(meta["layout"], "shared-surfaces")
                self.assertEqual(meta["halo_rows"], 1)
                self.assertEqual(info["task_count"], meta["layers"] * 2)
                # Every program reads the 128-shifted native surface: the pad value the
                # untiled chain's native layers use too (0xff80 = zero point 0).
                for registers, _ in entries:
                    self.assertEqual(registers[0x1184], 0xFF80)
                # The activation registers follow each layer's quantization metadata:
                # `[Conv, Relu]*(N-1) + [Conv]` means every layer but the last carries
                # the graph's Relu (docs/plans/pipelining-plan.md S9).
                relu = [q["relu"] for q in meta["quantizations"]]
                self.assertEqual(relu[0], True)
                self.assertEqual(relu[-1], False)
                for layer, (registers, _) in enumerate(entries):
                    strips = meta["tiles"]
                    self.assertEqual(registers[0x4060],
                                     0x12 if relu[layer // strips] else 0x13)

    def test_batched_tiling_links_every_task_and_is_a_valid_job(self):
        for name in K3_MODELS:
            data, meta = compile_tiled_chain(onnx.load(K3 / f"{name}.onnx"), tiles=2,
                                             serial=False)
            entries, info = programs(data)
            with self.subTest(model=name):
                self.assertFalse(info["serial"])
                self.assertIsNone(batched_layout(data, info))
                for position, (_, (link, control)) in enumerate(entries):
                    if position + 1 < info["task_count"]:
                        self.assertEqual(link, info["tasks"][position + 1]["command_offset"])
                        self.assertEqual(control, 0x40)
                    else:
                        self.assertEqual(link, 0)
                        self.assertEqual(control, 0x28)

    def test_scheduler_and_cli_expose_tiles(self):
        from open_rknpu.scheduler import compile_sequence
        data, meta = compile_sequence(MODEL, tiles=4)
        self.assertEqual(meta["strip_rows"], 2)
        self.assertEqual(decode_sequence(data)["task_count"], meta["layers"] * 4)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            compile_sequence(MODEL, tiles=1)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            compile_sequence(MODEL, tiles=2, expose_intermediates=True)

    def test_bounds_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "1, 2, 4 or 8"):
            compile_tiled_chain(onnx.load(MODEL), tiles=3)
        # The loader's task table holds 64 descriptors, so 16 layers x 8 strips is out.
        with self.assertRaisesRegex(ValueError, "loader table holds 64"):
            compile_tiled_chain(onnx.load(MODEL), tiles=8)
        # A graph outside the chain family still fails chain_n's own validation.
        with self.assertRaisesRegex(ValueError, "native chain requires"):
            compile_tiled_chain(onnx.load(ROOT / "research" / "mnist_pool_suite" /
                                          "model000.onnx"), tiles=2)

    def test_board_evidence_is_retained(self):
        text = (PROBE / "board_results.txt").read_text()
        self.assertIn("16 board cases per variant", text)
        records = json.loads((PROBE / "board_results.json").read_text())
        fixed = [record for record in records if record["variant"] != "pre-s7"]
        self.assertEqual([record["variant"] for record in fixed],
                         ["tiles1", "tiles1_batched", "tiles2", "tiles2_batched",
                          "tiles4", "tiles4_batched"])
        for record in fixed:
            with self.subTest(variant=record["variant"]):
                self.assertTrue(record["passed"])
                self.assertEqual(record["cases"], 16)
                self.assertEqual(record["mismatches"], 0)
        # The container held before the S7 pad/activation fix is the counter-example.
        self.assertIn("pre-s7", [record["variant"] for record in records])
        self.assertTrue((PROBE / "model_tiles2_pre_s7.bin").is_file())
        # Every retained container is exactly what the current emitter writes.
        for entry in json.loads((PROBE / "manifest.json").read_text()):
            tiles = entry["tiles"]
            serial = entry["mode"] == "serial"
            data, meta = compile_tiled_chain(onnx.load(MODEL), tiles=tiles, serial=serial)
            self.assertEqual(data, (PROBE / f"model_{entry['name']}.bin").read_bytes())
            self.assertEqual(meta["arena_bytes"], entry["arena_bytes"])

    def test_k3_board_evidence_is_retained(self):
        manifest = {entry["name"]: entry for entry in
                    json.loads((K3 / "manifest.json").read_text())}
        records = json.loads((K3 / "board_results.json").read_text())
        self.assertEqual(len(records), len(manifest) * 7)     # untiled + 3 tiles x 2 modes
        for record in records:
            entry = manifest[record["model"]]
            variant = record["variant"]
            with self.subTest(model=record["model"], variant=variant):
                self.assertTrue(record["passed"])
                self.assertEqual(record["cases"], entry["cases"])
                self.assertEqual(record["mismatches"], 0)
                if variant == "untiled":
                    continue
                self.assertIn(variant, entry["variants"])
                tiles = int(variant.split("tiles")[1].split("_")[0])
                data, meta = compile_tiled_chain(onnx.load(K3 / f"{record['model']}.onnx"),
                                                 tiles=tiles,
                                                 serial=not variant.endswith("batched"))
                self.assertEqual(meta["tasks"], record["tasks"])
                self.assertEqual(data, (K3 / f"{record['model']}_{variant}.bin").read_bytes())


if __name__ == "__main__":
    unittest.main()
