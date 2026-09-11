# Luckfox Pico (RV1103) — Access & Troubleshooting

Practical reference for getting to the board and fixing the things that actually go wrong. Board is USB-connected only (no wifi/ethernet configured), extremely RAM-constrained (33MB total), and running a stock camera/NPU app (`rkipc`) that should stay healthy across sessions.

---

## Two ways in

| Method | Needs network config? | Reliability |
|---|---|---|
| **`adb`** | No — works over raw USB regardless of IP state | **Always try this first.** Survives network breakage entirely. |
| **`ssh root@172.32.0.93`** (alias: `ssh luckfox`) | Yes — both host and board need correct IPs | More convenient when it works, but has two independent failure points (below) |

Confirm the board is even physically present before anything else:
```bash
lsusb -d 2207:0019          # Rockchip rk3xxx — should always show up if board is powered/connected
adb devices -l               # should show one device
```
**Note:** the adb device shows up labeled `product:occam model:Nexus_4 device:mako` — that's not a real Android phone, it's this board (Luckfox's USB gadget firmware mimics a generic Android descriptor). Confirmed by matching the adb serial (`498063e3262e55b7`) against the USB device's `iSerial` — they're identical.

---

## SSH not connecting — the two-part fix

SSH requires **both** sides to have the right IP, and both regenerate on every reboot/replug because the USB gadget MAC (and therefore the host-side interface name) is randomized each time.

### Part 1 — host side
Find the current interface name (it will **not** be the same as last time):
```bash
ip -br link | grep -v -E "^lo|docker|veth|br-|tailscale|ixgbe"
# e.g. enxf27245abff53   DOWN
```
If it's `DOWN` with no IP, the udev rule (`/etc/udev/rules.d/70-luckfox-rndis.rules`) should bring it up automatically on the *next* hotplug — but on the very first plug after boot it commonly misses the race. Fix by hand (**requires sudo — run this yourself in a real terminal, not via an agent/non-interactive shell, since sudo needs a TTY for the password**):
```bash
sudo /usr/local/bin/luckfox-rndis-up.sh <ifname>
```
This assigns `172.32.0.100/24` and brings the link up.

### Part 2 — board side
Even with the host fixed, the board's own `usb0` may not have picked up its IP (same kind of boot-time race). Check and fix over **adb** (doesn't need the network to already work):
```bash
adb shell "ifconfig usb0"                          # look for inet addr — if missing:
adb shell "/etc/init.d/S90usb0config start"         # forces it to 172.32.0.93
```
There's a udev rule on the board too (`/etc/udev/rules.d/70-usb0-ip.rules`) meant to auto-heal this on future `usb0` add/change events, but don't assume it fired — always verify with `ifconfig usb0` if SSH won't connect.

Once both sides are confirmed, `ssh luckfox` (or `ssh root@172.32.0.93`) should just work — key auth already configured, no password.

---

## ⚠️ The one rule that matters most: never fill `/tmp`

`/tmp` is a **16.4MB tmpfs** (RAM-backed). Pushing anything sizeable there (a debugger binary, a large log, anything over a few MB) can overflow it and hang the board's entire USB stack — not just the process, the *board*. This happened once this session pushing a 30MB file; the board became completely unresponsive (USB enumeration failing in a retry loop) until a **physical power cycle**.

- **Good news if it happens:** it's not storage corruption. `/tmp` is wiped by the reboot, rootfs/`/oem`/`/userdata` are untouched, and `rkipc` comes back up cleanly on its own. Confirmed via `df -h` and a fresh `uptime` after recovery.
- **The fix:** if the board stops responding to USB after a push, a real power cycle (not a software reboot command — the board won't be responsive enough to accept one) is the reliable recovery.
- **Prevention:** if you need to get a large file onto the board, use `/oem` or `/userdata` (flash-backed, not tmpfs) — never `/tmp` — and check `df -h` first regardless.

Check current headroom before anything memory-heavy:
```bash
adb shell "free -m; df -h"
```
Typical healthy state: ~1-8MB "available" RAM with `rkipc` running. This is normal, not a problem — but it means there's very little slack for anything else running concurrently.

---

## Don't attach a debugger to `rkipc` directly

`rkipc` runs ~30 threads. Attaching `gdb` to it (even just to set one breakpoint) loads enough debug-symbol/thread-tracking overhead to OOM-kill it on this board — happened once this session. If you need to debug something that requires touching a live process:

1. **Prefer a fresh, small process of your own** (e.g. a short Python script) over attaching to `rkipc`. `strace`'s overhead is much lower than `gdb`'s and is generally safe even on `rkipc` if `-f` thread-following isn't the issue; `gdb` attach is the specific danger.
2. If you must interact with the camera/NPU pipeline itself, **stop `rkipc` cleanly first**, do the work, then restart it — don't work around a live instance.

### Stopping and restarting `rkipc`
```bash
adb shell "killall rkipc"
# takes a few seconds to actually exit (graceful shutdown of ISP/threads) — check:
adb shell "pidof rkipc"     # empty = fully stopped
```
Restart (needs the exact library path and IQ-file flag, or it fails with `can't load library 'librockit.so'`):
```bash
adb shell "export LD_LIBRARY_PATH=/oem/usr/lib:\$LD_LIBRARY_PATH; cd /oem/usr/bin && nohup ./rkipc -a /oem/usr/share/iqfiles > /tmp/rkipc.log 2>&1 &"
sleep 5
adb shell "pidof rkipc; free -m"
```

### If restart fails with `failed to allocate model memory!` / `ENOMEM`
This means the memory/CMA pool is fragmented from repeated crash→restart cycles — `free -m` can look deceptively OK while this is still broken. **Plain process restart won't fix it.** Do a full board reboot instead:
```bash
adb shell "reboot"
# wait ~15-20s, then re-poll for USB:
lsusb -d 2207:0019
```
After reboot, both the host RNDIS IP and board `usb0` IP will need re-fixing (see the two-part SSH fix above) — a fresh boot always re-triggers the same race conditions.

---

## Verifying the camera/NPU are actually healthy

```bash
adb shell "pidof rkipc"                                          # should return a pid
adb shell "mount -t debugfs none /sys/kernel/debug 2>/dev/null; cat /sys/kernel/debug/rknpu/load"   # non-zero % if IVS analytics active
```
**Never read `/sys/kernel/debug/rknpu/volt`** — this board has no NPU voltage regulator wired up, and the kernel driver doesn't null-check that case: reading it reliably segfaults the calling process (a real kernel driver bug, not something you're doing wrong). `version`, `freq`, `load`, `power` are all safe to read.

Full visual confirmation the camera pipeline is alive end-to-end (run from the host, needs SSH/network up):
```bash
ffmpeg -rtsp_transport tcp -i rtsp://172.32.0.93/live/0 -frames:v 1 -update 1 /tmp/check.png
```

---

## Quick reference

```bash
# Is the board even there?
lsusb -d 2207:0019 && adb devices -l

# Fix SSH after any reboot/replug (2 steps, host then board):
ip -br link | grep -v -E "^lo|docker|veth|br-|tailscale|ixgbe"    # find current ifname
sudo /usr/local/bin/luckfox-rndis-up.sh <ifname>                  # host side — run yourself, needs sudo password
adb shell "/etc/init.d/S90usb0config start"                       # board side

# Health check
adb shell "pidof rkipc; free -m; df -h"

# Restart camera service after killing it
adb shell "killall rkipc"; sleep 3
adb shell "export LD_LIBRARY_PATH=/oem/usr/lib:\$LD_LIBRARY_PATH; cd /oem/usr/bin && nohup ./rkipc -a /oem/usr/share/iqfiles > /tmp/rkipc.log 2>&1 &"

# Nuclear option (unresponsive board, or ENOMEM restart failures)
adb shell "reboot"     # or physical power cycle if adb itself is unresponsive
# then redo the SSH fix above once it's back
```
