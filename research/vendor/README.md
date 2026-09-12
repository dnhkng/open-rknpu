# Vendor driver sources: referenced, not redistributed

The RV1103 NPU register and job semantics were recovered from three sources: experiments on
the attached board (the bulk of it), Rockchip's public kernel driver, and the vendor
runtime as a black-box oracle. The kernel driver is **GPL-2.0**; the open-rknpu repository
is MIT, so the upstream files are recorded here by provenance and **not** redistributed.

Upstream: `https://github.com/LuckfoxTECH/luckfox-pico`, SDK kernel tree,
`sysdrv/source/kernel/drivers/rknpu/`. Retrieved at commit
`824b817f889c2cbff1d48fcdb18ab494a68f69d1`; per-file sha256 in
[`provenance.json`](provenance.json).

| Upstream file | What it documented for this project |
| --- | --- |
| `rknpu_ioctl.h` | the `SUBMIT`/`MEM_*` ioctl numbers, `struct rknpu_submit` (flags, core mask, fence fd, `subcore_task[]`), the job-flag enums, and the `ACTION` opcode list used by `research/action_probe/` |
| `rknpu_job.c` | how a submission's tasks are walked, when the driver forces `core_mask = CORE0`, and how the job counter/loop field is used |
| `rknpu_drv.c` | the `ACTION` switch (which actions are implemented, which are no-ops) and the `dev_pm_opp`/regulator plumbing that explains the board's missing regulator |
| `rknpu_mem.c` | the memory flags (`CONTIGUOUS`, `NON_CACHEABLE`, `CACHEABLE`, `KERNEL_MAPPING`, `SRAM`, `NBUF`, `DMA32`) and the `SYNC` semantics the runtime relies on |

The facts extracted from these files are written up in
[`docs/container-format.md`](../../docs/container-format.md),
[`docs/registers.md`](../../docs/registers.md),
[`docs/board.md`](../../docs/board.md) and the investigation log
([`docs/investigation-log.md`](../../docs/investigation-log.md)); several probe scripts cite
the driver by filename as the origin of a hypothesis. No vendor code, binary or SDK file is
part of the distribution.

To verify a claim against the upstream source, fetch it yourself at the commit above:

```sh
git clone https://github.com/LuckfoxTECH/luckfox-pico
cd luckfox-pico && git checkout 824b817f889c2cbff1d48fcdb18ab494a68f69d1
ls sysdrv/source/kernel/drivers/rknpu/
```
