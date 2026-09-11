"""Run the whole ledger campaign on the board with the right runner per suite.

Every suite is a *format*: from v5 on it is a named-tensor executable and goes
through `run_v5_suite.py` -> `tests/board_io.c`; older suites are streamed one model
at a time through `run_profile_suite.py` -> `tests/board_api.c`. Using the wrong
runner does not just fail, it rewrites the suite's evidence with that failure, so the
mapping is explicit here.

usage: PYTHONPATH=src python research/run_campaign.py [suite ...]
"""
from pathlib import Path
import subprocess
import os
import sys

ROOT = Path(__file__).resolve().parent
V5 = ["runtime_scale_suite", "join_chain_suite", "depthwise_join_suite", "pool_join_suite",
      "mixed_head_suite", "join_scale_suite", "join_residual_suite", "join_dag_suite",
      "branch_join_suite", "depthwise_chain_suite", "pooled_dag_suite",
      "pooled_branches_suite", "diamond_tail_suite", "chain_multi_suite"]
LEGACY = ["transpose_k5_dilation_suite", "native_chain_suite", "chain_reuse_suite",
          "deep_chain_suite"]
# The join-chain fan-outs are v5; the chain suites are v3 through the streaming runner.
RUNNERS = {name: "v5" for name in V5}
RUNNERS.update({name: "profile" for name in LEGACY})


def main():
    wanted = sys.argv[1:] or V5 + LEGACY
    unknown = [name for name in wanted if name not in RUNNERS]
    if unknown:
        raise SystemExit("unknown suite(s): %s" % unknown)
    python = os.environ.get("PYTHON", sys.executable)
    failed = []
    for name in wanted:
        runner = "run_v5_suite.py" if RUNNERS[name] == "v5" else "run_profile_suite.py"
        run = subprocess.run([python, str(ROOT / runner), name], capture_output=True,
                             text=True, timeout=1800)
        summary = [line for line in (run.stdout + run.stderr).splitlines()
                   if line.startswith(("PASS", "FAIL"))]
        print("=== %-28s %s" % (name, summary[-1] if summary else run.stdout.strip()[-80:]),
              flush=True)
        if not summary or summary[-1].startswith("FAIL"):
            failed.append(name)
    print("campaign: %d/%d suites passed" % (len(wanted) - len(failed), len(wanted)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
