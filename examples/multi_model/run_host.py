"""SPDX-License-Identifier: MIT

Host-side companion to `examples/multi_model/board.c`.

`board.c` is the measured run: two `ornpu_open` calls, N alternating
inferences, one `ornpu_close` per model. This script is the host rehearsal, so
the lesson can be inspected without the board:

* it reads the published `report.json` and prints the device-memory arithmetic
  of the two lifecycle patterns from `runtime/open_rknpu.c` (every `ornpu_open`
  allocates one 4 KiB task buffer plus one arena, and copies the 8 KiB register
  payload into that arena; `ornpu_close` releases both);
* it replays both models' integer references over the published
  `input_*.u8`/`expected_*.i8` fixtures, which is the same byte-for-byte check
  `board.c` performs on the board.

It needs `build.py` to have run first (same directory), and it never touches
`/dev/rknpu`:

    PYTHONPATH=src python examples/multi_model/build.py
    PYTHONPATH=src python examples/multi_model/run_host.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primitives"))

from common import qfrom
from open_rknpu.chain import native_reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

EXAMPLE = Path(__file__).resolve().parent
OUT = EXAMPLE / "build"
TASK_BUFFER_BYTES = 4096   # runtime/open_rknpu.c: buffers[0]
PAYLOAD_BYTES = 8192       # every container's fixed-size copied register payload


def reference_grid(kind, meta, cases):
    """The profile's integer reference, assembled independently of build.py."""
    if kind == "a":
        quantization = qfrom(meta["quantization"])
        return np.stack([native_input_reference(case, quantization, int(meta["input_zero_point"]),
                                               pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta["conv_dilations"]))
                         for case in cases])
    stem = qfrom(meta["first"]["quantization"])
    layer = qfrom(meta["depthwise"])
    head = qfrom(meta["pointwise"])
    return np.stack([native_reference(depthwise_reference(reference(case, stem), layer,
                                                         stem.output_zero_point),
                                     head, layer.output_zero_point)
                     for case in cases])


def replay(out, kind, report):
    """Recompute one model's reference from its ONNX graph and compare every byte."""
    row = report["models"][kind]
    onnx_path = out / row["onnx"]
    _, meta = compile_sequence(onnx_path, **row["compile"])
    info = decode_sequence((out / ("model_%s.bin" % kind)).read_bytes())
    height, width, channels = info["shape_nhwc"][1:]
    cases = np.fromfile(out / ("input_%s.u8" % kind), dtype=np.uint8)
    cases = cases.reshape(-1, height, width, channels)
    published = np.fromfile(out / ("expected_%s.i8" % kind), dtype=np.int8)
    got = reference_grid(kind, meta, cases)
    if got.shape[1:] != tuple(info["output_shape_nhwc"])[1:]:
        raise AssertionError("model %s: reference shape %s does not match the container %s"
                             % (kind, got.shape, tuple(info["output_shape_nhwc"])))
    differences = got.reshape(-1) != published
    if differences.any():
        raise AssertionError("model %s: %d/%d reference bytes differ"
                             % (kind, int(differences.sum()), published.size))
    per_inference = int(published.size // cases.shape[0])
    print("  replay model=%s cases=%d bytes_per_inference=%d reference=%s"
          % (kind, cases.shape[0], per_inference, row["reference"]))
    return per_inference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inferences", type=int, default=8,
                        help="alternations the board run will use (default: 8)")
    options = parser.parse_args(argv)
    if options.inferences < 1 or options.inferences > 1000000:
        parser.error("--inferences must be between 1 and 1000000")
    report_path = OUT / "report.json"
    if not report_path.is_file():
        print("missing %s; run: PYTHONPATH=src python examples/multi_model/build.py" % report_path)
        return 1
    report = json.loads(report_path.read_text())
    arena = report["arena"]
    schedule = dict(a=(options.inferences + 1) // 2, b=options.inferences // 2)
    churn = sum(schedule[kind] * (TASK_BUFFER_BYTES + report["models"][kind]["arena_bytes"])
                for kind in ("a", "b"))
    payload = options.inferences * PAYLOAD_BYTES
    both_open = arena["both_models_open_bytes"]

    print("arena: model_a=%d B + model_b=%d B = %d B of independent dma allocations "
          "(plus %d B task buffers per model)"
          % (arena["model_a_bytes"], arena["model_b_bytes"], arena["combined_arena_bytes"],
             TASK_BUFFER_BYTES))
    print("pattern open-once-reuse: %d device bytes held for all %d inferences, %d payload bytes "
          "uploaded once" % (both_open, options.inferences, 2 * PAYLOAD_BYTES))
    print("pattern open-close-per-inference: %d device bytes allocated and freed over %d "
          "inferences, %d payload bytes re-uploaded" % (churn, options.inferences, payload))
    print("pattern open-per-inference-no-close: %d device bytes leaked" % churn)

    models = exact = 0
    for kind in ("a", "b"):
        exact += replay(OUT, kind, report) * schedule[kind]
        models += 1
    print("multi_model host replay: %d models, %d inferences, %d exact bytes"
          % (models, options.inferences, exact))
    return 0


if __name__ == "__main__":
    sys.exit(main())
