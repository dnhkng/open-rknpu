# Driver ACTION ioctl probe (2026-09-11)

`research/action_probe.c` exercises the RKNPU driver's `ACTION` ioctl (`_IOWR('r',0,...)`,
`RKNPU_ACTION 0x00`) that `runtime/open_rknpu.c` does not use, so the "unused board
functions" list is measured rather than assumed. Only GET actions are issued.

Build and run (board `498063e3262e55b7`, Luckfox Pico Mini B):

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -Os -std=gnu99 -Wall -Wextra -Werror research/action_probe.c -o /tmp/action_probe
adb -s 498063e3262e55b7 push /tmp/action_probe /userdata/open-npu-research/action_probe
adb -s 498063e3262e55b7 shell "chmod +x /userdata/open-npu-research/action_probe && \
  /userdata/open-npu-research/action_probe"
```

Measured output (driver v0.8.2, `rkipc` running, non-IOMMU mode):

| Action | Code | `rc` | Value | Reading |
| --- | --- | --- | --- | --- |
| `GET_HW_VERSION` | 0 | 0 | `0x54524548` | hardware version magic |
| `GET_DRV_VERSION` | 1 | 0 | `802` | driver v0.8.2 (`MAJOR*10000+MINOR*100+PATCH`) |
| `GET_FREQ` | 2 | 0 | `420000000` | NPU clock 420 MHz |
| `GET_BW_PRIORITY` | 7 | -1 | - | `-EINVAL`: no devfreq/OPP bandwidth control |
| `GET_BW_EXPECT` | 9 | -1 | - | `-EINVAL` |
| `GET_BW_TW` | 11 | -1 | - | `-EINVAL` |
| `GET_DT_WR_AMOUNT` | 14 | 0 | `3396522184` | free-running DT write counter |
| `GET_DT_RD_AMOUNT` | 15 | 0 | `2289273556` | free-running DT read counter |
| `GET_WT_RD_AMOUNT` | 16 | 0 | `2808484212` | free-running WT read counter |
| `GET_TOTAL_RW_AMOUNT` | 17 | 0 | `4200744986` | sum of the three counters |
| `GET_IOMMU_EN` | 18 | 0 | `0` | non-IOMMU mode (matches the device tree) |
| `GET_TOTAL_SRAM_SIZE` | 22 | 0 | `0` | no SRAM pool configured |
| `GET_FREE_SRAM_SIZE` | 23 | 0 | `0` | no SRAM pool configured |

Counter values are a snapshot while the camera service runs; they are cumulative since
boot (the depth of the counters is not documented by the driver).

## GET_VOLT oopses on this board

`GET_VOLT` (action 4) is **not** in the probe: the device tree has no rknpu regulator
(`dmesg`: `dev_pm_opp_set_regulators: no regulator (rknpu) found: -19`), so the driver's
`regulator_get_voltage(rknpu_dev->vdd)` dereferences an ERR_PTR. Running it oopses the
calling process and prints a kernel trace that ends in `regulator_get_voltage` from
`rknpu_ioctl`; the board itself survives (`rkipc` stays alive, later submissions work).
The open runtime never issues it. The same source shows `SET_FREQ` has an empty body in
this driver version, so clock scaling is a no-op even though the action is accepted.

## Consequences for the framework

* version, clock and the three bandwidth counters are the only ACTION results that carry
  information on this board; the counters are a coarse "how much did the NPU read/write"
  metric, not a cycle counter;
* no frequency/voltage control, no hardware reset from userspace, no bandwidth policy, no
  IOMMU and no SRAM residency are available here, so `docs/plans/pipelining-plan.md` S4 cannot be
  closed with a clock-scaling experiment and the runtime has nothing to tune;
* `FENCE_IN`/`FENCE_OUT` stay unavailable (`-EINVAL`, kernel built without
  `CONFIG_ROCKCHIP_RKNPU_FENCE`), so fence-free completion stays the barrier job measured
  in `research/barrier_probe/`.
