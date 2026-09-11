"""Fetch the C-only subset of Luckfox's published cross toolchain to the host."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent / "toolchain"
tree = json.loads((root.parent / "toolchain_tree.json").read_text())["tree"]
prefix = "arm-rockchip830-linux-uclibcgnueabihf"
base = "https://raw.githubusercontent.com/LuckfoxTECH/luckfox-pico/main/tools/linux/toolchain/" + prefix + "/"
def selected(path):
    if path.startswith("bin/"):
        return path.split("/")[-1] in [prefix + "-" + s for s in ("gcc", "as", "ld.bfd", "ld", "nm", "strip", "objdump", "readelf")]
    if path.startswith("libexec/"):
        return path.endswith(("/cc1", "/collect2", "/liblto_plugin.so", "/liblto_plugin.so.0", "/liblto_plugin.so.0.0.0"))
    if path.startswith("lib/gcc/"):
        return "/plugin/" not in path and ("/include" in path or path.endswith((".o", "/libgcc.a", "/libgcc_eh.a")))
    if path.startswith(prefix + "/bin/"):
        return path.endswith(("/as", "/ld", "/ld.bfd", "/nm"))
    if "/sysroot/" in path:
        return "/include/" in path and "/c++/" not in path or ("/lib/" in path and any(s in path.split("/")[-1] for s in ("libc.", "libuClibc", "uclibc_nonshared", "ld-uClibc", "libdl", "libpthread", "libm.", "libgcc", "crt")))
    return False

items = [x for x in tree if x["type"] == "blob" and selected(x["path"])]
def fetch(item):
    path = root / item["path"]
    if path.exists() or path.is_symlink():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(base + item["path"], timeout=60).read()
    if item["mode"] == "120000":
        path.symlink_to(data.decode())
    else:
        path.write_bytes(data)
        path.chmod(0o755 if item["mode"] == "100755" else 0o644)

print("Fetching", len(items), "files,", round(sum(x.get("size", 0) for x in items)/1e6, 1), "MB", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for _ in pool.map(fetch, items):
        pass
print("C toolchain ready:", root, flush=True)
