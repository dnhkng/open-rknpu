"""SPDX-License-Identifier: MIT

Cookbook 5/8: the installed package, not the source checkout.

`examples/cookbook/` normally runs with `PYTHONPATH=src`, so `import open_rknpu`
resolves to this repository - fine for developing the compiler, useless as a
check that a *user* can install and use it. This script uses `importlib` to find
where `open_rknpu` actually comes from:

* if it is the source checkout, it **refuses politely**: it prints the paths, the
  distribution metadata and the two install commands, and probes the installed
  distribution in a child interpreter with `PYTHONPATH` removed;
* otherwise it acts as a smoke test: it compiles a `Conv` with the installed
  package, decodes the container and prints the module path and version.

Install with either command (`pyproject.toml` builds the wheel; the board runtime
sources are shipped inside it under `share/open-rknpu/runtime/`):

    pip install open-rknpu      # from an index, once published
    pip install .               # from this checkout

Then run with the source tree off `PYTHONPATH`:

    python examples/cookbook/05_wheel_installed.py

Run (in the repository, source checkout - this is the refusing path):
    PYTHONPATH=src python examples/cookbook/05_wheel_installed.py
"""
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parents[1] / "src" / "open_rknpu"
FOLDER = HERE / "build" / "05_wheel_installed"
DIST_NAME = "open-rknpu"
SMOKE_FLAG = "--installed-smoke"


def resolved_origin():
    """Where `import open_rknpu` would load from, resolved."""
    spec = importlib.util.find_spec("open_rknpu")
    if spec is None or not spec.origin:
        raise SystemExit("open-rknpu is not importable; install it with `pip install open-rknpu`")
    return Path(spec.origin).resolve()


def is_source_checkout(origin):
    """True when `origin` lives under this repository's `src/open_rknpu`."""
    try:
        origin.relative_to(SOURCE_DIR)
        return True
    except ValueError:
        return False


def distribution_info():
    """Installed-distribution metadata, or None when nothing is installed."""
    try:
        distribution = importlib.metadata.distribution(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = None
    try:
        direct_url = distribution.read_text("direct_url.json")
    except (FileNotFoundError, OSError):
        pass
    return dict(version=distribution.version, path=str(distribution._path), direct_url=direct_url)


def build_conv():
    """A tiny Conv the installed package can compile."""
    import numpy as np
    from onnx import helper as h, numpy_helper as nh
    from onnx import TensorProto
    rng = np.random.default_rng(50505)
    weights = rng.uniform(-0.25, 0.25, (5, 4, 3, 3)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, 5).astype(np.float32)
    node = h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3],
                       pads=[1, 1, 1, 1], strides=[1, 1])
    graph = h.make_graph([node], "wheel_smoke",
                         [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
                         [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 8, 8])],
                         [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def smoke_test():
    """Compile + decode with whatever `open_rknpu` the interpreter imported."""
    import open_rknpu
    from open_rknpu.scheduler import compile_sequence
    from open_rknpu.sequence import decode_sequence
    FOLDER.mkdir(parents=True, exist_ok=True)
    binary, meta = compile_sequence(build_conv())
    info = decode_sequence(binary)
    (FOLDER / "model.bin").write_bytes(binary)
    module_path = Path(open_rknpu.__file__).resolve()
    assert info["task_count"] >= 1, "the installed compiler produced no task"
    assert binary[:8] == b"ORNPUSEQ", "the installed compiler produced an unexpected container"
    return dict(path=str(module_path), version=open_rknpu.__version__, profile=meta.get("profile"),
                tasks=info["task_count"], bytes=len(binary))


def installed_smoke_child():
    """`--installed-smoke`: runs only when PYTHONPATH is clean, prints one status line."""
    origin = resolved_origin()
    if is_source_checkout(origin):
        print("INSTALLED_SMOKE: unavailable reason=child-resolved-source origin=%s" % origin)
        return 0
    missing = [name for name in ("open_rknpu.scheduler", "open_rknpu.sequence")
               if importlib.util.find_spec(name) is None]
    if missing:
        print("INSTALLED_SMOKE: older-build missing=%s origin=%s" % (",".join(missing), origin))
        return 0
    result = smoke_test()
    print("INSTALLED_SMOKE: ok origin=%s version=%s tasks=%d bytes=%d" %
          (origin, result["version"], result["tasks"], result["bytes"]))
    return 0


def probe_installed():
    """Run the child smoke test with the source tree off `sys.path`."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), SMOKE_FLAG],
                               capture_output=True, text=True, env=env)
    line = next((row for row in completed.stdout.splitlines() if row.startswith("INSTALLED_SMOKE:")), None)
    if completed.returncode != 0 or line is None:
        return dict(status="unavailable", detail=(completed.stderr or completed.stdout).strip()[-400:])
    parts = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
    status = line.split()[1].split(":")[0]
    parts["status"] = status
    return parts


def main():
    if SMOKE_FLAG in sys.argv:
        return installed_smoke_child()

    origin = resolved_origin()
    installed = distribution_info()
    if is_source_checkout(origin):
        print("  05_wheel_installed: REFUSING to smoke-test the source checkout")
        print("    importlib resolves open_rknpu -> %s" % origin)
        print("    that is this repository's src/open_rknpu, not an installed distribution")
        if installed is None:
            print("    no '%s' distribution metadata found in this interpreter" % DIST_NAME)
        else:
            print("    installed distribution: %s %s at %s" %
                  (DIST_NAME, installed["version"], installed["path"]))
        print("    install it with:  pip install open-rknpu   # index release")
        print("                      pip install .              # or from this checkout")
        print("    then run with the source tree off PYTHONPATH:  python examples/cookbook/05_wheel_installed.py")
        probe = probe_installed()
        assert probe["status"] in ("ok", "older-build", "unavailable"), probe
        if probe["status"] == "ok":
            assert not is_source_checkout(Path(probe["origin"]).resolve()), probe
            print("    child probe (PYTHONPATH cleared): installed smoke test OK, origin=%s version=%s tasks=%s" %
                  (probe["origin"], probe.get("version"), probe.get("tasks")))
        elif probe["status"] == "older-build":
            assert probe.get("missing"), probe
            print("    child probe (PYTHONPATH cleared): the installed package at %s is an older build missing %s" %
                  (probe.get("origin"), probe["missing"]))
            print("    reinstall with `pip install .` to smoke-test the current compiler")
        else:
            print("    child probe (PYTHONPATH cleared): no usable installed package (%s)" %
                  probe.get("detail", ""))
        print("05_wheel_installed: source checkout refused politely; installed probe status=%s" % probe["status"])
        return 0

    missing = [name for name in ("open_rknpu.scheduler", "open_rknpu.sequence")
               if importlib.util.find_spec(name) is None]
    if missing:
        assert missing
        print("  05_wheel_installed: installed package %s (%s) is an older build" %
              (origin, installed["version"] if installed else "?"))
        print("    it is missing %s, so it cannot compile a Conv/sequence container" % ", ".join(missing))
        print("    reinstall with `pip install open-rknpu` or `pip install .`, then re-run this script")
        print("05_wheel_installed: older installed build reported instead of a false smoke test")
        return 0

    result = smoke_test()
    print("  05_wheel_installed: module=%s version=%s" % (result["path"], result["version"]))
    print("  05_wheel_installed: compiled profile=%s tasks=%d container=%d B -> %s" %
          (result["profile"], result["tasks"], result["bytes"], FOLDER))
    print("05_wheel_installed: installed-package smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
