"""SPDX-License-Identifier: MIT

Primitive 10/11: calibration - analytic versus measured activation bands.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Sequence compilation
now accepts measured calibration ranges ...": single `Conv[/Relu/Clip]` through the
native and scheduled paths, the `Conv-ReLU-Conv` chain per layer, and the
elementwise/Mul profiles; `PRIMITIVE_ROADMAP.md` records that calibrating the
trained MNIST Conv2 output range took the fully offloaded variant from 13/100 to
100/100 on a held-out set). `open_rknpu.calibration.measure` only needs the ONNX
host reference, so it works even for graphs the legacy compiler rejects. Three
selectors share one measurement pass:

* `minmax` - the observed minimum and maximum (one pass, exact for the samples);
* `percentile` - the observed range with the upper `100-percentile`% tail clipped;
* `kl` - a TensorRT-style saturation search that minimizes the KL divergence
  between the full histogram and a 128-level quantization of a candidate range.

This script compiles one `Conv(1x1) -> Relu -> Conv(1x1)` graph four times -
analytic and once per selector - prints the resulting bands and the output scale,
and checks each container against the composed integer reference
(`open_rknpu.quantization.reference` for the first layer, then
`open_rknpu.chain.native_reference`). Calibration changes the *quality*, never the
reference semantics: every variant is asserted byte-exact against the same
integer pipeline. Values outside the measured ranges are clipped, so calibration
can improve or worsen a model - the float column shows that per variant.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/10_calibration/board_io
    adb shell mkdir -p /userdata/open-npu-research/10_calibration
    adb push examples/primitives/build/10_calibration/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/10_calibration/
    adb shell 'cd /userdata/open-npu-research/10_calibration && ./board_io . 4'
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx

from common import (compile_and_report, conv_node, deterministic_cases, initializer, model_graph,
                    print_board_recipe, print_summary, publish_suite, qfrom, read_expected,
                    relu_node, report_checks, tensor_info)
from open_rknpu.calibration import measure
from open_rknpu.chain import native_reference
from open_rknpu.quantization import reference

FOLDER = Path(__file__).resolve().parent / "build" / "10_calibration"
SEED = 111001
HIDDEN = 5


def build():
    """The legacy `Conv -> Relu -> Conv` chain at 8x8 RGB."""
    rng = np.random.default_rng(SEED)
    first_weights = rng.uniform(-0.7, 0.8, (HIDDEN, 3, 1, 1)).astype(np.float32)
    first_bias = rng.uniform(-2, 2, HIDDEN).astype(np.float32)
    second_weights = rng.uniform(-0.7, 0.8, (3, HIDDEN, 1, 1)).astype(np.float32)
    second_bias = rng.uniform(-2, 2, 3).astype(np.float32)
    nodes = [conv_node("input", "c1", "w1", "b1", 1),
             relu_node("c1", "r1"),
             conv_node("r1", "output", "w2", "b2", 1)]
    return model_graph(nodes, "calibration", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, 3, 8, 8])],
                       [initializer("w1", first_weights), initializer("b1", first_bias),
                        initializer("w2", second_weights), initializer("b2", second_bias)])


def calibrated_reference(case, model, meta):
    """First layer through `reference`, then the second through `native_reference`."""
    first = qfrom(meta["first"]["quantization"])
    grid = reference(case, first)
    return native_reference(grid, qfrom(meta["second"]), first.output_zero_point)


def main():
    model = build()
    calibration = np.random.default_rng(SEED + 7).integers(0, 256, (32, 3, 8, 8), dtype=np.uint8)
    cases = deterministic_cases(np.random.default_rng(SEED + 100), 4, 8, 8, 3)

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        model_path = folder / "model.onnx"
        onnx.save(model, model_path)
        calibration_dir = folder / "calibration"
        calibration_dir.mkdir()
        np.save(calibration_dir / "calibration.npy", calibration)
        reports = {
            "minmax": measure(model_path, calibration_dir, method="minmax"),
            "percentile": measure(model_path, calibration_dir, method="percentile", percentile=99.9),
            "kl": measure(model_path, calibration_dir, method="kl", bins=512),
        }

        rows = []
        names = reports["minmax"]["tensors"]
        bands = {}
        analytic_meta = None
        for index, (label, ranges) in enumerate([("analytic", None)] +
                                                [(name, report["ranges"]) for name, report in reports.items()]):
            kwargs = {} if ranges is None else {"calibration_ranges": ranges}
            binary, meta, report = compile_and_report(model, "%s.onnx" % label, label=label, **kwargs)
            report["profile"] = "conv-relu-conv (band %s)" % label
            reference_grid = np.stack([calibrated_reference(case, model, meta) for case in cases])
            publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
            published = read_expected(FOLDER, index).reshape(reference_grid.shape)
            row = dict(report)
            row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                         "quantization.reference + chain.native_reference")
            rows.append(row)
            if ranges is None:
                analytic_meta = meta
                bands[label] = {"output": (float(meta["output_scale"]),
                                           int(meta["output_zero_point"]))}
            else:
                bands[label] = {name: (float(entry["scale"]), int(entry["zero_point"]))
                                for name, entry in ranges.items()}

        # The analytic per-tensor bands come from the two layer quantizations.
        first = analytic_meta["first"]["quantization"]
        bands["analytic"][names[0]] = (float(first["output_scale"]), int(first["output_zero_point"]))
        bands["analytic"][names[1]] = (float(analytic_meta["second"]["output_scale"]),
                                       int(analytic_meta["second"]["output_zero_point"]))

        print("  calibration bands (scale @ zero point):")
        print("    %-10s %-22s %-22s %-22s %-22s" % ("tensor", "analytic", "minmax",
                                                     "percentile", "kl"))
        for name in names:
            cells = []
            for label in ("analytic", "minmax", "percentile", "kl"):
                entry = bands[label].get(name)
                cells.append("%.4f @ %4d" % entry if entry is not None else "-")
            print("    %-10s %-22s %-22s %-22s %-22s" % (name, *cells))

    print_board_recipe(FOLDER)
    print_summary("10_calibration", rows, FOLDER)


if __name__ == "__main__":
    main()
