"""Run a small prebuilt oracle with capture, saving artifacts on the host."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
ADB = os.environ.get("ADB", "adb")
BOARD = "/userdata/open-npu-research"

def adb(*args, timeout=30):
    result = subprocess.run([ADB, "-s", "498063e3262e55b7", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace"))
    return result.stdout

def run(name):
    if not name.replace("_", "").isalnum():
        raise ValueError("fixture name must contain letters, numbers, underscores")
    model = ROOT / "fixtures" / name / "model.rknn"
    if model.stat().st_size > 65536:
        raise ValueError("small oracle only")
    print(adb("push", str(model), BOARD + "/" + name + ".rknn").decode(), flush=True)
    cmd = "cd {b} && mkdir -p capture_{n} && OPEN_NPU_CAPTURE={b}/capture_{n} LD_LIBRARY_PATH={b} LD_PRELOAD={b}/capture_ioctl.so ./board_probe {n}.rknn".format(b=shlex.quote(BOARD), n=name)
    log = adb("shell", cmd)
    (ROOT / (name + "_capture.log")).write_bytes(log)
    print(adb("pull", BOARD + "/capture_" + name + "/.", str(ROOT / ("capture_" + name))).decode(), flush=True)
    # The host now owns the capture; remove only this experiment's board copies.
    expected = ROOT / ("capture_" + name)
    if not (expected / "run3_submit.bin").exists():
        raise RuntimeError("incomplete capture; board files retained")
    adb("shell", "rm -r {b}/capture_{n} && rm {b}/{n}.rknn".format(b=shlex.quote(BOARD), n=name))
    for line in log.decode(errors="replace").splitlines():
        if line.startswith(("ATTR", "ALLOC", "SUBMIT")):
            print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    run(parser.parse_args().name)
