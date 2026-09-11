"""MIT. Recompile the campaign suites and diff against their published `.bin` artifacts.

The campaign ledger (`research/COVERAGE_EXPANSION_RESULTS.md`) cites 157 byte-identical /
12 drift / 0 error over the suites below. The 12 drifts are pre-existing and documented
there (they are *published* containers, not compiler output, so a recompile is compared
against the artifact to keep the drift visible instead of silently re-baselining it).

    PYTHONPATH=src python3 research/campaign_sweep.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_rknpu.scheduler import compile_sequence

CAMPAIGN = ["runtime_scale_suite", "join_chain_suite", "depthwise_join_suite", "pool_join_suite",
            "mixed_head_suite", "join_scale_suite", "join_residual_suite", "join_dag_suite",
            "branch_join_suite", "transpose_k5_dilation_suite", "diamond_tail_suite",
            "depthwise_chain_suite", "pooled_dag_suite", "pooled_branches_suite"]
EXPECTED_DRIFT = {"depthwise_join_suite/model009.onnx", "depthwise_join_suite/model010.onnx",
                  "mixed_head_suite/model002.onnx", "mixed_head_suite/model005.onnx",
                  "mixed_head_suite/model007.onnx", "mixed_head_suite/model010.onnx",
                  "join_scale_suite/model002.onnx", "join_scale_suite/model005.onnx",
                  "join_scale_suite/model008.onnx", "join_residual_suite/model002.onnx",
                  "join_residual_suite/model005.onnx", "join_residual_suite/model008.onnx"}

same = diff = errors = 0
unexpected = []
for name in CAMPAIGN:
    for path in sorted((ROOT / "research" / name).glob("model*.onnx")):
        artifact = path.with_suffix(".bin")
        if not artifact.exists():
            continue
        key = f"{name}/{path.name}"
        try:
            binary, _ = compile_sequence(str(path))
        except Exception as exc:
            errors += 1
            unexpected.append((key, f"error: {type(exc).__name__}: {exc}"))
            continue
        if bytes(binary) == artifact.read_bytes():
            same += 1
        else:
            diff += 1
            if key not in EXPECTED_DRIFT:
                unexpected.append((key, "new drift"))

print(f"campaign sweep: same={same} diff={diff} err={errors}")
if unexpected:
    for key, why in unexpected:
        print(f"  UNEXPECTED {key}: {why}")
raise SystemExit(0 if not unexpected else 1)
