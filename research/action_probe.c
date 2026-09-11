/* SPDX-License-Identifier: MIT */
/* Read-only probe of the RKNPU driver ACTION ioctl surface.
 *
 * runtime/open_rknpu.c only uses SUBMIT/MEM_CREATE/MEM_DESTROY/MEM_SYNC. This probe
 * issues the GET actions the driver exposes (versions, clock, IOMMU, bandwidth
 * counters, SRAM size) so the unused capability can be measured instead of assumed.
 *
 * GET_VOLT (action 4) is **not** issued: this board's device tree has no rknpu
 * regulator (`dev_pm_opp_set_regulators: no regulator (rknpu) found: -19`), so the
 * driver's `regulator_get_voltage(rknpu_dev->vdd)` dereferences an ERR_PTR and oopses
 * the calling process (see `research/action_probe/README.md`). The mutating actions
 * (RESET, POWER_ON/OFF, SET_FREQ, SET_VOLT) are likewise not exercised. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct action {
    uint32_t flags;
    uint32_t value;
};
#define RKNPU_ACTION 0x00
#define ACTION _IOWR('r', RKNPU_ACTION, struct action)

static const struct {
    uint32_t code;
    const char *name;
} GETS[] = {
    {0, "GET_HW_VERSION"},   {1, "GET_DRV_VERSION"},     {2, "GET_FREQ"},
    {7, "GET_BW_PRIORITY"},  {9, "GET_BW_EXPECT"},       {11, "GET_BW_TW"},
    {14, "GET_DT_WR_AMOUNT"},{15, "GET_DT_RD_AMOUNT"},   {16, "GET_WT_RD_AMOUNT"},
    {17, "GET_TOTAL_RW_AMOUNT"}, {18, "GET_IOMMU_EN"},
    {22, "GET_TOTAL_SRAM_SIZE"}, {23, "GET_FREE_SRAM_SIZE"},
};

int main(void) {
    int fd = open("/dev/rknpu", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open /dev/rknpu: %s\n", strerror(errno));
        return 1;
    }
    printf("action ioctl probe on /dev/rknpu\n");
    for (size_t i = 0; i < sizeof(GETS) / sizeof(GETS[0]); i++) {
        struct action a = {GETS[i].code, 0};
        errno = 0;
        int rc = ioctl(fd, ACTION, &a);
        int err = errno;
        printf("%-20s rc=%3d value=%10u (0x%08x)", GETS[i].name, rc, a.value, a.value);
        if (rc < 0) printf("  %s", strerror(err));
        printf("\n");
    }
    close(fd);
    return 0;
}
