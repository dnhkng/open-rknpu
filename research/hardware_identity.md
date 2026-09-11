# Attached board hardware identity

Captured 2026-09-10 over adb from the connected Luckfox board. Every board result
in the ledger was produced on this hardware.

| Item | Value |
| --- | --- |
| Board | Luckfox Pico Mini B |
| SoC `compatible` | `rockchip,rv1103g-38x38-ipc-v10`, `rockchip,rv1103` |
| `/proc/device-tree/model` | `Luckfox Pico Mini B` |
| CPU | ARMv7 rev5 (Cortex-A7), ARM32 uClibc |
| NPU node | `/proc/device-tree/npu@ff660000` |
| **NPU `compatible`** | **`rockchip,rv1106-rknpu`** |
| NPU driver | RKNPU v0.8.2 (`dmesg`: `RKNPU ff660000.npu: Initialized RKNPU driver: v0.8.2`) |
| NPU IOMMU | not present in this device tree (non-IOMMU mode) |
| Serial | `498063e3262e55b7` |

## Consequence for P9

The accelerator block is the shared **RV1106 NPU IP**, so the board's NPU-level
results *are* RV1106 NPU-IP results; no separate RV1106 NPU hardware is required to
exercise that IP. The SoC-level identity is RV1103, so non-NPU conclusions
(CPU, memory map, peripherals, driver image) do not transfer to a distinct RV1106
SoC. Phase P9 therefore records the NPU-IP validation as satisfied by the ledger
and leaves a distinct RV1106 SoC explicitly unclaimed unless such hardware appears.

## Reproduction

```sh
adb shell "cat /proc/device-tree/compatible | tr '\0' '\n'"
adb shell "cat /proc/device-tree/model"
adb shell "cat /proc/device-tree/npu@ff660000/compatible | tr '\0' '\n'"
adb shell "dmesg | grep -i rknpu"
```
